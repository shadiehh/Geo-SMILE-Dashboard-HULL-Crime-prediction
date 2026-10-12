"""Feature-branch fidelity (Ridge surrogate, N=1,024, seed=42, Wasserstein
response) for all 10 candidate models, with crime_rolling_mean_6 -- GP's
single dominant Feature predictor (|coef|=4.853, ~9.3x the runner-up) --
excluded from BOTH the base model's inputs and GeoSMILE's Feature player
set. Answers: does the established Feature-fidelity ranking across models
depend on this one feature's outsized presence, or is it robust to
removing it?

Same design as every other Feature fidelity number in this project
(dehghani_fidelity, kernel-weighted Ridge, 60/20/20 mask-pool split) --
only the player list changes (29 features instead of 30). Each model is
refit from scratch on the reduced feature set (fast: this is the same
per-model training cost paid everywhere else in this project, not the
expensive part).

Writes outputs/feature_fidelity_excl_dominant/summary.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eval_by_n_perturb import Y_COL, dehghani_fidelity
from geosmile_hull_twobranch import FORBIDDEN_FEATURE_TOKENS, SPATIAL_COLUMNS, choose_feature_players, temporal_split
from model_factory import MODEL_NAMES, fit_model, make_model
from response_metric_comparison import feature_pool_multi_response

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "feature_fidelity_excl_dominant"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"

SMILE_SEED = 42
FEAT_N = 1024


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exclude", nargs="*", default=[],
                    help="Feature player(s) to drop from both the base model and GeoSMILE's Feature branch "
                         "(default: none, i.e. the full 30-feature baseline run through this same script).")
    p.add_argument("--out-name", default=None,
                    help="Output CSV filename (default: summary.csv with no exclusions, "
                         "summary_excl_<features>.csv otherwise).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    all_players = choose_feature_players(panel)
    bad = [p for p in all_players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise SystemExit(f"Spatial leak: {bad}")
    missing = [f for f in args.exclude if f not in all_players]
    if missing:
        raise SystemExit(f"Not found in player list: {missing}")
    players = [p for p in all_players if p not in args.exclude]
    if args.exclude:
        _log(f"Excluding {args.exclude}: {len(all_players)} -> {len(players)} Feature players")
    else:
        _log(f"No exclusions: baseline run with all {len(all_players)} Feature players")

    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
    X_test = panel.loc[test].reset_index(drop=True)

    rows = []
    for model_name in MODEL_NAMES:
        _log(f"\n=== [{model_name}] ===")
        model = make_model(model_name, SMILE_SEED, n_jobs=2)
        fit_model(model_name, model, panel.loc[train, players], panel.loc[train, Y_COL],
                  panel.loc[validation, players], panel.loc[validation, Y_COL])
        masks, distances, resp_by_metric = feature_pool_multi_response(model, X_test, players, baseline, FEAT_N, SMILE_SEED)
        fid = dehghani_fidelity(masks, resp_by_metric["wasserstein"], distances)
        rows.append({
            "model": model_name, "n_features": len(players), "r2": fid["R2"], "weighted_r2": fid["Weighted R2"],
            "mae": fid["MAE"], "rmse": fid["RMSE"], "best_kernel": fid["best_kernel"],
        })
        _log(f"[{model_name}] Feature fidelity ({len(players)} features): R2={fid['R2']:.4f} R2w={fid['Weighted R2']:.4f} kernel={fid['best_kernel']}")

    df = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    if args.out_name:
        out_name = args.out_name
    elif args.exclude:
        out_name = f"summary_excl_{'_'.join(args.exclude)}.csv"
    else:
        out_name = "summary_baseline_30feat.csv"
    out_path = DEST / out_name
    df.to_csv(out_path, index=False)
    _log(f"\nDone. {out_path}")
    _log(df.to_string(index=False))


if __name__ == "__main__":
    main()
