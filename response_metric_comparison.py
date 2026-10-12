"""Compare GeoSMILE's response metric -- currently Wasserstein-1 between the
full original vs. perturbed prediction distributions -- against two LIME-style
point-wise prediction-difference metrics: mean absolute difference (L1) and
mean squared difference (L2). Motivated by GP's Geo-branch fidelity being
weaker than its Feature-branch fidelity despite GP being the strongest
predictor overall: is that a property of the Wasserstein response metric, or
does it persist under a simpler LIME-like target too?

Sweeps perturbation count N on both branches to separate "the response metric
matters" from "the perturbation budget matters": Geo at N in {2000, 4000,
6000, 8000} (bracketing the project's current N=4000), Feature at N in {512,
1024, 2048, 4096} (bracketing the project's current N=1024). The Feature
sweep is specifically aimed at MLP/DL's collapsed Feature fidelity: is it a
hard ceiling (like Geo's ~0.95 plateau turned out to be for every model) or
fixable with a bigger mask budget?

All three response metrics are computed from the SAME predict-loop per mask/
perturbation (one black-box .predict() call generates one (original, perturbed)
prediction pair, from which Wasserstein/L1/L2 are all cheaply derived) --
trying three metrics costs no more than trying one.

Scope: fidelity ONLY (R2/MAE/RMSE/Weighted R2/WMAE/WRMSE via the SAME
dehghani_fidelity() used everywhere else in this project -- it operates on a
plain (masks, responses, distances) tuple and has no idea how "responses" was
computed, so it is reused completely unmodified). Does NOT recompute ATT
accuracy/faithfulness/stability/consistency for these variants -- that would
multiply the already-large cost of this sweep several-fold for no benefit to
the specific question being asked here.

Separate-models architecture (two models per candidate, as used everywhere in
this project except the dedicated combined-model scripts). Same R2>0.7
predictive gate as elsewhere (outputs/eval_by_model/predictive_performance.csv)
-- a branch that fails is skipped entirely, consistent with every other script
in this project.

Writes outputs/response_metric_comparison/response_metric_comparison_summary.csv.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from eval_by_n_perturb import Y_COL, dehghani_fidelity
from geosmile_hull_twobranch import (
    FORBIDDEN_FEATURE_TOKENS,
    ID_COLUMNS,
    SPATIAL_COLUMNS,
    choose_feature_players,
    mask_pool,
    temporal_split,
)
from model_factory import MODEL_NAMES, fit_model, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "response_metric_comparison"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"
PREDICTIVE_CSV = OUT / "eval_by_model" / "predictive_performance.csv"

FEAT_N_GRID = (512, 1024, 2048, 4096)
GEO_N_GRID = (2000, 4000, 6000, 8000)
SMILE_SEED = 42
R2_GATE = 0.7


def _log(msg: str) -> None:
    print(msg, flush=True)


def feature_pool_multi_response(model, reference, players, baselines, n_masks, seed):
    """Mirrors smile_global_feature's masking loop, but returns responses
    under THREE metrics computed from the same predict-loop instead of just
    Wasserstein.
    """
    rng = np.random.default_rng(seed)
    masks = mask_pool(n_masks, len(players), rng)
    original = np.asarray(model.predict(reference[players]), float)
    n = len(masks)
    resp_w, resp_l1, resp_l2 = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, mask in enumerate(masks):
        perturbed = reference[players].copy()
        for j in np.flatnonzero(mask == 0):
            perturbed[players[j]] = baselines[players[j]]
        pred = np.asarray(model.predict(perturbed), float)
        resp_w[i] = wasserstein_distance(original, pred)
        diff = original - pred
        resp_l1[i] = np.mean(np.abs(diff))
        resp_l2[i] = np.mean(diff ** 2)
    distances = 1.0 - masks.mean(axis=1)
    return masks, distances, {"wasserstein": resp_w, "l1_mean_abs_diff": resp_l1, "l2_mean_sq_diff": resp_l2}


def geo_pool_multi_response(model, panel, test_mask, baselines, n_perturb, seed, min_sel=5, max_sel=10):
    """Mirrors geo_smile_coefficients' masking loop, three response metrics."""
    frame = panel.loc[test_mask, ID_COLUMNS + SPATIAL_COLUMNS].copy().reset_index(drop=True)
    x = frame[SPATIAL_COLUMNS].copy()
    original = np.asarray(model.predict(x), float)
    lsoa_ids = frame["lsoa21cd"].astype(str).to_numpy()
    unique_lsoas = np.array(sorted(set(lsoa_ids.tolist())), dtype=object)
    n_lsoas = len(unique_lsoas)
    index_by_lsoa = {code: np.flatnonzero(lsoa_ids == code) for code in unique_lsoas}
    rng = np.random.default_rng(seed)
    M = np.zeros((n_perturb, n_lsoas), dtype=np.int8)
    resp_w, resp_l1, resp_l2 = np.zeros(n_perturb), np.zeros(n_perturb), np.zeros(n_perturb)
    baseline_xy = baselines[SPATIAL_COLUMNS].to_numpy(dtype=float)
    for j in range(n_perturb):
        n_sel = int(rng.integers(min_sel, max_sel + 1))
        sel = rng.choice(n_lsoas, size=n_sel, replace=False)
        M[j, sel] = 1
        perturbed = x.copy()
        affected = []
        for idx in sel:
            rows = index_by_lsoa[unique_lsoas[idx]]
            perturbed.iloc[rows, :] = baseline_xy
            affected.extend(rows.tolist())
        affected = np.asarray(affected, dtype=int)
        pert_pred = np.asarray(model.predict(perturbed), float)
        resp_w[j] = wasserstein_distance(original[affected], pert_pred[affected])
        diff = original[affected] - pert_pred[affected]
        resp_l1[j] = np.mean(np.abs(diff))
        resp_l2[j] = np.mean(diff ** 2)
    distances = M.sum(axis=1).astype(float) / n_lsoas
    return M, distances, {"wasserstein": resp_w, "l1_mean_abs_diff": resp_l1, "l2_mean_sq_diff": resp_l2}


def fidelity_row(masks, responses, distances) -> dict:
    fid = dehghani_fidelity(masks, responses, distances)
    return {
        "r2": fid["R2"], "weighted_r2": fid["Weighted R2"],
        "mae": fid["MAE"], "wmae": fid["WMAE"],
        "rmse": fid["RMSE"], "wrmse": fid["WRMSE"],
        "best_kernel": fid["best_kernel"],
    }


def evaluate_model(model_name: str, *, panel, players, train, validation, test, baseline, X_test,
                    n_jobs: int, do_feature: bool, do_geo: bool) -> list[dict]:
    _log(f"\n=== [{model_name}] === Feature: {'run' if do_feature else 'SKIPPED'}, Geo: {'run' if do_geo else 'SKIPPED'}")
    t0 = time.time()
    rows: list[dict] = []

    if do_feature:
        model_feat = make_model(model_name, SMILE_SEED, n_jobs=n_jobs)
        fit_model(model_name, model_feat, panel.loc[train, players], panel.loc[train, Y_COL],
                  panel.loc[validation, players], panel.loc[validation, Y_COL])
        for n in FEAT_N_GRID:
            _log(f"[{model_name}] Feature pool (n={n}, all 3 response metrics)...")
            masks, distances, resp_by_metric = feature_pool_multi_response(model_feat, X_test, players, baseline, n, SMILE_SEED)
            for metric_name, responses in resp_by_metric.items():
                r = fidelity_row(masks, responses, distances)
                rows.append({"model": model_name, "branch": "Feature", "n": n, "response_metric": metric_name, **r})
                _log(f"[{model_name}] Feature(n={n})/{metric_name}: R2={r['r2']:.4f} MAE={r['mae']:.4f} RMSE={r['rmse']:.4f}")

    if do_geo:
        model_geo = make_model(model_name, SMILE_SEED, n_jobs=n_jobs)
        fit_model(model_name, model_geo, panel.loc[train, SPATIAL_COLUMNS], panel.loc[train, Y_COL],
                  panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, Y_COL])
        for n in GEO_N_GRID:
            _log(f"[{model_name}] Geo pool (n={n}, all 3 response metrics)...")
            M, distances, resp_by_metric = geo_pool_multi_response(model_geo, panel, test, baseline, n, SMILE_SEED)
            for metric_name, responses in resp_by_metric.items():
                r = fidelity_row(M, responses, distances)
                rows.append({"model": model_name, "branch": "Geo", "n": n, "response_metric": metric_name, **r})
                _log(f"[{model_name}] Geo(n={n})/{metric_name}: R2={r['r2']:.4f} MAE={r['mae']:.4f} RMSE={r['rmse']:.4f}")

    _log(f"[{model_name}] done in {(time.time() - t0) / 60:.1f} min")
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--branches", nargs="+", choices=["Feature", "Geo"], default=["Feature", "Geo"],
                    help="Restrict which branch(es) to (re)compute -- e.g. --branches Feature to add a "
                         "Feature N-sweep without redoing an already-computed (and expensive) Geo sweep.")
    p.add_argument("--force", nargs="+", default=[], choices=MODEL_NAMES,
                    help="Run both branches for these models regardless of the R2>0.7 predictive guard "
                         "(explicit override, e.g. to inspect a branch that narrowly fails the gate).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not PANEL_PATH.exists():
        raise SystemExit(f"Missing {PANEL_PATH}")
    if not PREDICTIVE_CSV.exists():
        raise SystemExit(f"Missing {PREDICTIVE_CSV} -- run eval_by_model.py first")
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

    predictive = pd.read_csv(PREDICTIVE_CSV)

    all_rows: list[dict] = []
    for model_name in args.models:
        pred_feat = predictive[(predictive["model"] == model_name) & (predictive["branch"] == "Feature")]
        pred_geo = predictive[(predictive["model"] == model_name) & (predictive["branch"] == "Geo")]
        feat_r2 = float(pred_feat.iloc[0]["r2"]) if not pred_feat.empty else float("nan")
        geo_r2 = float(pred_geo.iloc[0]["r2"]) if not pred_geo.empty else float("nan")
        forced = model_name in args.force
        do_feature = "Feature" in args.branches and (forced or (np.isfinite(feat_r2) and feat_r2 > R2_GATE))
        do_geo = "Geo" in args.branches and (forced or (np.isfinite(geo_r2) and geo_r2 > R2_GATE))
        if forced:
            _log(f"[{model_name}] FORCED: running both branches regardless of R2 guard "
                 f"(Feature={feat_r2:.3f}, Geo={geo_r2:.3f})")
        if not do_feature and not do_geo:
            _log(f"\n=== [{model_name}] === SKIPPED ENTIRELY (both branches at/below R2={R2_GATE}, or excluded by --branches)")
            continue
        all_rows.extend(evaluate_model(
            model_name, panel=panel, players=players, train=train, validation=validation, test=test,
            baseline=baseline, X_test=X_test, n_jobs=2, do_feature=do_feature, do_geo=do_geo,
        ))

    new_df = pd.DataFrame(all_rows)
    out_path = DEST / "response_metric_comparison_summary.csv"
    if out_path.exists() and not new_df.empty:
        old_df = pd.read_csv(out_path)
        # Drop old rows for exactly the (model, branch) combos just recomputed; keep everything else.
        recomputed = set(zip(new_df["model"], new_df["branch"]))
        keep_mask = ~old_df.apply(lambda r: (r["model"], r["branch"]) in recomputed, axis=1)
        df = pd.concat([old_df[keep_mask], new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(out_path, index=False)
    _log(f"\nDone. {len(df)} rows ({len(new_df)} new/recomputed). Summary: {out_path}")


if __name__ == "__main__":
    main()
