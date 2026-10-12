"""Try alternative SMILE surrogate regressors -- not just Ridge -- and report
prediction R2/MAE/RMSE for each (base model, branch, surrogate) combination.

Every fidelity number reported elsewhere in this project comes from fitting a
single kernel-weighted Ridge regression on the mask/perturbation -> Wasserstein
-response pool. That is a deliberate SMILE design choice for the *attribution*
step (the reported "coefficient" per feature/LSOA IS that Ridge fit's
.coef_), but it is not the only reasonable choice for measuring *fidelity* --
how well can any surrogate reconstruct the black-box model's masking
response? A neural network's masking response could plausibly be more
non-linear than a boosted tree's, in which case a linear surrogate would
under-fit it regardless of how good the underlying black-box model is. This
script tests that directly.

For each candidate model (same R2>0.7 gate as verify_metrics_all_models.py),
the Feature/Geo mask->response pool is generated ONCE per branch -- the
expensive part, calling the black-box model's .predict() up to N times --
by reusing smile_global_feature()/geo_smile_coefficients() unmodified. Three
surrogates are then fit on that SAME pool and scored on the SAME held-out
test split (split_masks, seed 42, matching dehghani_fidelity elsewhere):
  - Ridge, kernel-weighted (the incumbent SMILE surrogate; reuses
    eval_by_n_perturb.dehghani_fidelity() unmodified for exact continuity
    with every other fidelity number reported in this project).
  - RandomForestRegressor (unweighted).
  - HistGradientBoostingRegressor (unweighted).
Kernel-based sample weighting is Ridge/SMILE-specific with no established
convention for tree ensembles here, so the two tree surrogates are fit
without it -- a plainer, still-fair comparison of "can this regressor learn
the mask -> response mapping at all."

IMPORTANT: swapping in a non-linear surrogate would break GeoSMILE's own
attribution reporting -- "importance" throughout this pipeline IS the Ridge
surrogate's .coef_, and a Random Forest / gradient-boosted surrogate has no
equivalent linear coefficient. This script answers a narrower question (can
*some* surrogate fit the mask -> response relationship well) and does not
produce, or attempt to produce, alternative importance rankings.

Writes outputs/surrogate_comparison/surrogate_comparison_summary.csv and
outputs/surrogate_comparison/<model>_<branch>_pool.npz (the raw pool, so
further surrogates can be tried later without re-running the black-box model).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from eval_by_n_perturb import Y_COL, dehghani_fidelity, feature_responses
from geosmile_hull_twobranch import (
    FORBIDDEN_FEATURE_TOKENS,
    SPATIAL_COLUMNS,
    choose_feature_players,
    geo_smile_coefficients,
    smile_global_feature,
    temporal_split,
)
from hull_dehghani_eval import split_masks
from model_factory import MODEL_NAMES, fit_model, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "surrogate_comparison"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"
PREDICTIVE_CSV = OUT / "eval_by_model" / "predictive_performance.csv"

FEAT_N_MASKS = 1024
GEO_N_PERTURB = 4000
SMILE_SEED = 42
SPLIT_SEED = 42
R2_GATE = 0.7


def _log(msg: str) -> None:
    print(msg, flush=True)


def score(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def try_surrogates(masks: np.ndarray, responses: np.ndarray, distances: np.ndarray) -> dict:
    tr, va, te = split_masks(len(masks), SPLIT_SEED)
    trva = np.concatenate([tr, va])
    results = {}

    ridge_fid = dehghani_fidelity(masks, responses, distances)
    results["ridge_kernel_weighted"] = {
        "r2": ridge_fid["R2"], "mae": ridge_fid["MAE"], "rmse": ridge_fid["RMSE"],
        "best_kernel": ridge_fid["best_kernel"],
    }

    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=SMILE_SEED, n_jobs=-1)
    rf.fit(masks[trva], responses[trva])
    results["random_forest"] = score(responses[te], rf.predict(masks[te]))

    hgb = HistGradientBoostingRegressor(max_depth=4, random_state=SMILE_SEED)
    hgb.fit(masks[trva], responses[trva])
    results["hist_gradient_boosting"] = score(responses[te], hgb.predict(masks[te]))

    return results


def evaluate_model(model_name: str, *, panel, players, train, validation, test, baseline, X_test,
                    n_jobs: int, do_feature: bool, do_geo: bool) -> list[dict]:
    _log(f"\n=== [{model_name}] === Feature: {'run' if do_feature else 'SKIPPED'}, Geo: {'run' if do_geo else 'SKIPPED'}")
    t0 = time.time()
    rows = []

    if do_feature:
        model_feat = make_model(model_name, SMILE_SEED, n_jobs=n_jobs)
        fit_model(model_name, model_feat, panel.loc[train, players], panel.loc[train, Y_COL],
                  panel.loc[validation, players], panel.loc[validation, Y_COL])
        _log(f"[{model_name}] generating Feature mask pool (n={FEAT_N_MASKS})...")
        _, _, feat_masks, feat_all_pert, feat_original = smile_global_feature(
            model_feat, X_test[players], players, baseline, FEAT_N_MASKS, SMILE_SEED
        )
        feat_resp = feature_responses(feat_masks, feat_all_pert, feat_original)
        feat_dist = 1.0 - feat_masks.mean(axis=1)
        np.savez(DEST / f"{model_name}_feature_pool.npz", masks=feat_masks, responses=feat_resp, distances=feat_dist)
        _log(f"[{model_name}] Feature: trying surrogates...")
        for surrogate, m in try_surrogates(feat_masks, feat_resp, feat_dist).items():
            rows.append({"model": model_name, "branch": "Feature", "surrogate": surrogate, **m})
            _log(f"[{model_name}] Feature/{surrogate}: R2={m['r2']:.4f} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f}")

    if do_geo:
        model_geo = make_model(model_name, SMILE_SEED, n_jobs=n_jobs)
        fit_model(model_name, model_geo, panel.loc[train, SPATIAL_COLUMNS], panel.loc[train, Y_COL],
                  panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, Y_COL])
        _log(f"[{model_name}] generating Geo mask pool (n={GEO_N_PERTURB})...")
        _, _, geo_M, geo_resp, geo_dist = geo_smile_coefficients(
            model_geo, panel, test, baseline, GEO_N_PERTURB, SMILE_SEED, return_pool=True
        )
        np.savez(DEST / f"{model_name}_geo_pool.npz", masks=geo_M, responses=geo_resp, distances=geo_dist)
        _log(f"[{model_name}] Geo: trying surrogates...")
        for surrogate, m in try_surrogates(geo_M, geo_resp, geo_dist).items():
            rows.append({"model": model_name, "branch": "Geo", "surrogate": surrogate, **m})
            _log(f"[{model_name}] Geo/{surrogate}: R2={m['r2']:.4f} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f}")

    _log(f"[{model_name}] done in {(time.time() - t0) / 60:.1f} min")
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--force", nargs="+", default=[], choices=MODEL_NAMES,
                    help="Run both branches for these models regardless of the R2>0.7 predictive guard.")
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
        do_feature = forced or (np.isfinite(feat_r2) and feat_r2 > R2_GATE)
        do_geo = forced or (np.isfinite(geo_r2) and geo_r2 > R2_GATE)
        if forced:
            _log(f"\n=== [{model_name}] === FORCED: running both branches regardless of R2 guard "
                 f"(Feature={feat_r2:.3f}, Geo={geo_r2:.3f})")
        if not do_feature and not do_geo:
            _log(f"\n=== [{model_name}] === SKIPPED ENTIRELY (both branches at/below R2={R2_GATE})")
            continue
        all_rows.extend(evaluate_model(
            model_name, panel=panel, players=players, train=train, validation=validation, test=test,
            baseline=baseline, X_test=X_test, n_jobs=args.n_jobs, do_feature=do_feature, do_geo=do_geo,
        ))

    new_df = pd.DataFrame(all_rows)
    summary_path = DEST / "surrogate_comparison_summary.csv"
    if summary_path.exists():
        old_df = pd.read_csv(summary_path)
        df = pd.concat([old_df[~old_df["model"].isin(args.models)], new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(summary_path, index=False)
    _log(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()
