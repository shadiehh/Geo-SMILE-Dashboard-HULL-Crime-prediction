"""Combined-model architecture (ONE model per candidate, trained on the
26-feature version -- crime_rolling_mean_6 kept as the sole crime-history
predictor, crime_lag_1/2/3 and crime_rolling_mean_3 dropped -- plus the 2
spatial coordinates, 28 inputs total), giving one unified predictive R2 per
model. GeoSMILE then explains that SAME model along both branches:
Feature-branch masking holds coordinates fixed at true values; Geo-branch
masking holds the 26 features fixed at true values (combined_feature_smile/
combined_geo_smile from combined_model_fidelity.py, reused unmodified).

Fidelity for both branches is dehghani_fidelity() on the raw
(masks, responses, distances) tuple each SMILE pass returns -- the same
function/convention used for every other fidelity number in this project,
not the kernel-selection diagnostic's own test_r2 (a different quantity).

Writes outputs/feature_fidelity_excl_dominant/combined_model_26feat_summary.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from combined_model_fidelity import combined_feature_smile, combined_geo_smile
from eval_by_n_perturb import Y_COL, dehghani_fidelity
from geosmile_hull_twobranch import FORBIDDEN_FEATURE_TOKENS, SPATIAL_COLUMNS, choose_feature_players, metrics, temporal_split
from model_factory import MODEL_NAMES, fit_model, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "feature_fidelity_excl_dominant"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"

SMILE_SEED = 42
FEAT_N = 1024
GEO_N = 4000
EXCLUDE = ["crime_lag_1", "crime_lag_2", "crime_lag_3", "crime_rolling_mean_3"]


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    all_players = choose_feature_players(panel)
    bad = [p for p in all_players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise SystemExit(f"Spatial leak: {bad}")
    players = [p for p in all_players if p not in EXCLUDE]
    _log(f"Excluding {EXCLUDE}: {len(all_players)} -> {len(players)} Feature players (+ 2 coordinates = "
         f"{len(players) + len(SPATIAL_COLUMNS)} combined inputs)")
    columns = players + SPATIAL_COLUMNS

    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, columns].median()
    X_test = panel.loc[test, columns].reset_index(drop=True)
    X_test_full = panel.loc[test].reset_index(drop=True)

    rows = []
    for model_name in MODEL_NAMES:
        _log(f"\n=== [{model_name}] ===")
        model = make_model(model_name, SMILE_SEED, n_jobs=2)
        fit_model(model_name, model, panel.loc[train, columns], panel.loc[train, Y_COL],
                  panel.loc[validation, columns], panel.loc[validation, Y_COL])
        pred = model.predict(X_test)
        predictive = metrics(panel.loc[test, Y_COL], pred)
        _log(f"[{model_name}] unified predictive R2={predictive['r2']:.4f} MAE={predictive['mae']:.4f} RMSE={predictive['rmse']:.4f}")

        _, _, feat_masks, feat_resp, feat_dist = combined_feature_smile(model, X_test, players, baseline, FEAT_N, SMILE_SEED)
        feat_fid = dehghani_fidelity(feat_masks, feat_resp, feat_dist)
        _log(f"[{model_name}] Feature fidelity: R2={feat_fid['R2']:.4f} kernel={feat_fid['best_kernel']}")

        _, _, geo_M, geo_resp, geo_dist = combined_geo_smile(model, panel, test, baseline, players, GEO_N, SMILE_SEED)
        geo_fid = dehghani_fidelity(geo_M, geo_resp, geo_dist)
        _log(f"[{model_name}] Geo fidelity: R2={geo_fid['R2']:.4f} kernel={geo_fid['best_kernel']}")

        rows.append({
            "model": model_name,
            "predictive_r2_unified": predictive["r2"], "predictive_mae": predictive["mae"], "predictive_rmse": predictive["rmse"],
            "feature_fidelity_r2": feat_fid["R2"], "feature_fidelity_weighted_r2": feat_fid["Weighted R2"], "feature_kernel": feat_fid["best_kernel"],
            "geo_fidelity_r2": geo_fid["R2"], "geo_fidelity_weighted_r2": geo_fid["Weighted R2"], "geo_kernel": geo_fid["best_kernel"],
        })

    df = pd.DataFrame(rows).sort_values("predictive_r2_unified", ascending=False).reset_index(drop=True)
    out_path = DEST / "combined_model_26feat_summary.csv"
    df.to_csv(out_path, index=False)
    _log(f"\nDone. {out_path}")
    _log(df.to_string(index=False))


if __name__ == "__main__":
    main()
