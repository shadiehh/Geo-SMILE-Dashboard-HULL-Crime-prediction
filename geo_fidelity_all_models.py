"""Geo-branch fidelity (Ridge surrogate, N=4,000, seed=42, Wasserstein
response) for all 10 candidate models -- the Geo companion to
feature_fidelity_excl_dominant.py's Feature-branch tables.

Note: the Geo branch's predictive model and GeoSMILE masking only ever use
the two spatial coordinates (SPATIAL_COLUMNS) as input -- it never sees any
of the 30 Feature players, so unlike the Feature-branch experiments this
script has no "N-feature version": there is exactly one Geo-branch input
set, unaffected by any Feature-player exclusion tried elsewhere in this
project.

Writes outputs/feature_fidelity_excl_dominant/geo_summary_all_models.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eval_by_n_perturb import Y_COL, dehghani_fidelity
from geosmile_hull_twobranch import SPATIAL_COLUMNS, choose_feature_players, temporal_split
from model_factory import MODEL_NAMES, fit_model, make_model
from response_metric_comparison import geo_pool_multi_response

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "feature_fidelity_excl_dominant"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"

SMILE_SEED = 42
GEO_N = 4000


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()

    rows = []
    for model_name in MODEL_NAMES:
        _log(f"\n=== [{model_name}] ===")
        model = make_model(model_name, SMILE_SEED, n_jobs=2)
        fit_model(model_name, model, panel.loc[train, SPATIAL_COLUMNS], panel.loc[train, Y_COL],
                  panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, Y_COL])
        M, distances, resp_by_metric = geo_pool_multi_response(model, panel, test, baseline, GEO_N, SMILE_SEED)
        fid = dehghani_fidelity(M, resp_by_metric["wasserstein"], distances)
        rows.append({
            "model": model_name, "r2": fid["R2"], "weighted_r2": fid["Weighted R2"],
            "mae": fid["MAE"], "rmse": fid["RMSE"], "best_kernel": fid["best_kernel"],
        })
        _log(f"[{model_name}] Geo fidelity: R2={fid['R2']:.4f} R2w={fid['Weighted R2']:.4f} kernel={fid['best_kernel']}")

    df = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    out_path = DEST / "geo_summary_all_models.csv"
    df.to_csv(out_path, index=False)
    _log(f"\nDone. {out_path}")
    _log(df.to_string(index=False))


if __name__ == "__main__":
    main()
