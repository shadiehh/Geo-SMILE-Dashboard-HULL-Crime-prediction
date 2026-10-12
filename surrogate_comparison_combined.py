"""Same surrogate comparison as surrogate_comparison.py (Ridge vs. Random
Forest vs. HistGradientBoosting on the SAME mask -> response pool), but using
the COMBINED-model architecture instead of two separately-trained models: ONE
model per candidate, trained on all 32 inputs (30 Feature players + 2
coordinates), giving one predictive R2/MAE/RMSE per model. GeoSMILE then
explains that SAME model along both branches (combined_feature_smile holds
coordinates fixed; combined_geo_smile holds Feature players fixed) -- see
combined_model_fidelity.py.

Gate: a model is fit and explained by GeoSMILE only if ITS OWN combined
predictive R2 exceeds 0.7 (one shared gate for both branches now, since there
is only one model and one R2 -- not a separate Feature/Geo gate as in the
two-separate-models version).

Writes outputs/surrogate_comparison_combined/{predictive_performance,
surrogate_comparison_summary}.csv.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from combined_model_fidelity import combined_feature_smile, combined_geo_smile
from eval_by_n_perturb import Y_COL, dehghani_fidelity
from geosmile_hull_twobranch import (
    FORBIDDEN_FEATURE_TOKENS,
    SPATIAL_COLUMNS,
    choose_feature_players,
    metrics,
    temporal_split,
)
from hull_dehghani_eval import split_masks
from model_factory import MODEL_NAMES, fit_model, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "surrogate_comparison_combined"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"

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


def evaluate_model(model_name: str, *, panel, players, columns, train, validation, test, baseline,
                    X_test, n_jobs: int) -> tuple[dict, list[dict]]:
    t0 = time.time()
    model = make_model(model_name, SMILE_SEED, n_jobs=n_jobs)
    fit_model(model_name, model, panel.loc[train, columns], panel.loc[train, Y_COL],
              panel.loc[validation, columns], panel.loc[validation, Y_COL])
    pred = model.predict(X_test)
    predictive = metrics(panel.loc[test, Y_COL], pred)
    _log(f"[{model_name}] combined predictive R2={predictive['r2']:.4f} MAE={predictive['mae']:.4f} RMSE={predictive['rmse']:.4f}")

    do_run = predictive["r2"] > R2_GATE
    rows = []
    if not do_run:
        _log(f"[{model_name}] SKIPPED: combined R2={predictive['r2']:.4f} <= {R2_GATE}")
        return predictive, rows

    _log(f"[{model_name}] generating Feature mask pool (n={FEAT_N_MASKS}, coordinates held fixed)...")
    _, _, feat_masks, feat_resp, feat_dist = combined_feature_smile(model, X_test, players, baseline, FEAT_N_MASKS, SMILE_SEED)
    np.savez(DEST / f"{model_name}_feature_pool.npz", masks=feat_masks, responses=feat_resp, distances=feat_dist)
    for surrogate, m in try_surrogates(feat_masks, feat_resp, feat_dist).items():
        rows.append({"model": model_name, "branch": "Feature", "surrogate": surrogate, **m})
        _log(f"[{model_name}] Feature/{surrogate}: R2={m['r2']:.4f} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f}")

    _log(f"[{model_name}] generating Geo mask pool (n={GEO_N_PERTURB}, Feature players held fixed)...")
    _, _, geo_M, geo_resp, geo_dist = combined_geo_smile(model, panel, test, baseline, players, GEO_N_PERTURB, SMILE_SEED)
    np.savez(DEST / f"{model_name}_geo_pool.npz", masks=geo_M, responses=geo_resp, distances=geo_dist)
    for surrogate, m in try_surrogates(geo_M, geo_resp, geo_dist).items():
        rows.append({"model": model_name, "branch": "Geo", "surrogate": surrogate, **m})
        _log(f"[{model_name}] Geo/{surrogate}: R2={m['r2']:.4f} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f}")

    _log(f"[{model_name}] done in {(time.time() - t0) / 60:.1f} min")
    return predictive, rows


def main() -> None:
    if not PANEL_PATH.exists():
        raise SystemExit(f"Missing {PANEL_PATH}")
    DEST.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    bad = [p for p in players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise SystemExit(f"Spatial leak in Feature players: {bad}")
    columns = players + SPATIAL_COLUMNS
    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, columns].median()
    X_test = panel.loc[test, columns].reset_index(drop=True)

    pred_rows, all_rows = [], []
    for model_name in MODEL_NAMES:
        _log(f"\n=== [{model_name}] ===")
        predictive, rows = evaluate_model(
            model_name, panel=panel, players=players, columns=columns, train=train, validation=validation,
            test=test, baseline=baseline, X_test=X_test, n_jobs=2,
        )
        pred_rows.append({"model": model_name, **predictive})
        all_rows.extend(rows)

    pd.DataFrame(pred_rows).to_csv(DEST / "predictive_performance.csv", index=False)
    pd.DataFrame(all_rows).to_csv(DEST / "surrogate_comparison_summary.csv", index=False)
    _log(f"\nDone. {DEST}")


if __name__ == "__main__":
    main()
