"""Generate raw SMILE-reseed replicates needed to bootstrap:
  (a) the Feature-branch fidelity ordering in Table 4.7 (all 10 candidate
      models, R=10 reseeds, N=1,024, Ridge surrogate, Wasserstein-1
      response -- same design dehghani_fidelity() already implements), and
  (b) the Geo-branch top-20 overlap counts in Figure 4.8 (the 5 models
      shown there -- xgboost, catboost, lightgbm, gp, dl -- R=10 reseeds,
      N=4,000).

No raw per-reseed values were ever persisted anywhere in this project
before now: the existing R=10 consistency step (verify_metrics_all_models.py)
only ever kept summary mean/std, and ANN/CNN/Transformer were never run
through a reseeded protocol at all (per an earlier explicit "just check
fidelity metrics" scope decision). This script fills that gap directly --
same reseed convention as everywhere else (seeds 42..51, R=10, model
trained once at seed=42, only the SMILE perturbation draw reseeds).

Feature output: outputs/bootstrap_replicates/feature_fidelity_replicates.csv
  (model, seed, r2, weighted_r2) -- one row per (model, seed).
Geo output: outputs/bootstrap_replicates/geo_top20_replicates.csv
  (model, seed, lsoa21cd, rank) -- one row per (model, seed, top-20 LSOA).

Both are written incrementally (appended per model) so a partial/killed run
keeps whatever finished.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eval_by_n_perturb import Y_COL, dehghani_fidelity
from geosmile_hull_twobranch import (
    FORBIDDEN_FEATURE_TOKENS,
    SPATIAL_COLUMNS,
    choose_feature_players,
    geo_smile_coefficients,
    temporal_split,
)
from model_factory import MODEL_NAMES, fit_model, make_model
from response_metric_comparison import feature_pool_multi_response

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "bootstrap_replicates"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"

TRAIN_SEED = 42
RESEEDS = list(range(42, 52))  # R=10, matching this project's established consistency protocol
FEAT_N = 1024
GEO_N = 4000
GEO_TOP_K = 20

GEO_REPLICATE_MODELS = ("xgboost", "catboost", "lightgbm", "gp", "dl")


def _log(msg: str) -> None:
    print(msg, flush=True)


def feature_replicates(model_name: str, *, panel, players, train, validation, test, baseline, X_test,
                        done_seeds: set[int], out_path: Path) -> None:
    model = make_model(model_name, TRAIN_SEED, n_jobs=2)
    fit_model(model_name, model, panel.loc[train, players], panel.loc[train, Y_COL],
              panel.loc[validation, players], panel.loc[validation, Y_COL])
    for seed in RESEEDS:
        if seed in done_seeds:
            _log(f"[{model_name}] Feature seed={seed}: already done, skipping")
            continue
        masks, distances, resp_by_metric = feature_pool_multi_response(model, X_test, players, baseline, FEAT_N, seed)
        fid = dehghani_fidelity(masks, resp_by_metric["wasserstein"], distances)
        row = {"model": model_name, "seed": seed, "r2": fid["R2"], "weighted_r2": fid["Weighted R2"]}
        _log(f"[{model_name}] Feature seed={seed}: R2={fid['R2']:.4f} R2w={fid['Weighted R2']:.4f}")
        append_csv([row], out_path)  # saved immediately -- a later crash keeps this seed's result


def geo_replicates(model_name: str, *, panel, train, validation, test, baseline,
                    done_seeds: set[int], out_path: Path) -> None:
    model = make_model(model_name, TRAIN_SEED, n_jobs=2)
    fit_model(model_name, model, panel.loc[train, SPATIAL_COLUMNS], panel.loc[train, Y_COL],
              panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, Y_COL])
    for seed in RESEEDS:
        if seed in done_seeds:
            _log(f"[{model_name}] Geo seed={seed}: already done, skipping")
            continue
        geo_imp, _diag = geo_smile_coefficients(model, panel, test, baseline, GEO_N, seed)
        geo_imp["lsoa21cd"] = geo_imp["lsoa21cd"].astype(str)
        ranked = geo_imp.sort_values(
            ["geo_abs_coefficient", "lsoa21cd"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        top20 = ranked.head(GEO_TOP_K)
        rows = [{"model": model_name, "seed": seed, "lsoa21cd": lsoa, "rank": rank}
                for rank, lsoa in enumerate(top20["lsoa21cd"], start=1)]
        _log(f"[{model_name}] Geo seed={seed}: top-20 captured")
        append_csv(rows, out_path)  # saved immediately -- a later crash keeps this seed's result


def append_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if path.exists():
        old_df = pd.read_csv(path)
        model_seeds = set(zip(new_df["model"], new_df["seed"]))
        keep_mask = ~old_df.apply(lambda r: (r["model"], r["seed"]) in model_seeds, axis=1)
        df = pd.concat([old_df[keep_mask], new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["feature", "geo"], required=True)
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not PANEL_PATH.exists():
        raise SystemExit(f"Missing {PANEL_PATH}")
    DEST.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    bad = [p for p in players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise SystemExit(f"Spatial leak in Feature players: {bad}")
    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
    X_test = panel.loc[test].reset_index(drop=True)

    if args.mode == "feature":
        out_path = DEST / "feature_fidelity_replicates.csv"
        for model_name in args.models:
            existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["model", "seed"])
            done_seeds = set(existing.loc[existing["model"] == model_name, "seed"])
            if done_seeds == set(RESEEDS):
                _log(f"\n=== [{model_name}] Feature fidelity replicates: all {len(RESEEDS)} seeds already done ===")
                continue
            _log(f"\n=== [{model_name}] Feature fidelity replicates (R={len(RESEEDS)}, {len(done_seeds)} already done) ===")
            feature_replicates(model_name, panel=panel, players=players, train=train,
                                validation=validation, test=test, baseline=baseline, X_test=X_test,
                                done_seeds=done_seeds, out_path=out_path)
    else:
        bad_models = [m for m in args.models if m not in GEO_REPLICATE_MODELS]
        if bad_models:
            raise SystemExit(f"Geo replicates only needed for {GEO_REPLICATE_MODELS}, got {bad_models}")
        out_path = DEST / "geo_top20_replicates.csv"
        for model_name in args.models:
            existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["model", "seed"])
            done_seeds = set(existing.loc[existing["model"] == model_name, "seed"])
            if done_seeds == set(RESEEDS):
                _log(f"\n=== [{model_name}] Geo top-20 replicates: all {len(RESEEDS)} seeds already done ===")
                continue
            _log(f"\n=== [{model_name}] Geo top-20 replicates (R={len(RESEEDS)}, {len(done_seeds)} already done) ===")
            geo_replicates(model_name, panel=panel, train=train, validation=validation,
                           test=test, baseline=baseline, done_seeds=done_seeds, out_path=out_path)

    _log("\nDone.")


if __name__ == "__main__":
    main()
