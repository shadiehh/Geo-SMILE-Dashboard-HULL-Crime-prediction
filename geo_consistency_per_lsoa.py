"""Per-LSOA ATT consistency for the XGBoost area-data Geo branch (N=4,000).

ATT consistency (Dehghani): run-to-run variance and SD of each attribution
coefficient over independently seeded perturbation pools, same fitted model and
settings. hull_geo_area_data_eval.consistency stores only the summary; this
reruns the same five seeds (100-104) and keeps every LSOA's mean, variance, SD
and relative SD, flagged by expert-region membership, so the SD of the leading
and of the expert-region LSOAs can be reported.

Checks the summary against outputs/geo_area_data_eval/n_4000/metrics.json.
Writes outputs/geo_area_data_eval/n_4000/consistency_per_lsoa.csv. Not causal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from geo_area_data import GEO_SCORE, area_geo_coefficients
from geosmile_hull_twobranch import SPATIAL_COLUMNS, choose_feature_players, temporal_split
from hull_geo_area_data_eval import CONSISTENCY_SEEDS, PANEL_PATH, SEED, Y_COL
from model_factory import fit_model, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "geo_area_data_eval" / "n_4000"
REF_PATH = ROOT / "outputs" / "geo_area_data_models" / "top50_maps" / "highlighted_regions_lsoas.csv"
N = 4000


def main() -> None:
    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
    model = make_model("xgboost", SEED, n_jobs=4)
    fit_model("xgboost", model, panel.loc[train, players], panel.loc[train, Y_COL],
              panel.loc[validation, players], panel.loc[validation, Y_COL])

    runs = []
    for s in CONSISTENCY_SEEDS:
        coef, _ = area_geo_coefficients(model, panel, test, baseline, players, N, s)
        runs.append(coef.set_index("lsoa21cd")[GEO_SCORE].rename(f"seed_{s}"))
        print(f"seed {s} done", flush=True)
    mat = pd.concat(runs, axis=1)
    out = pd.DataFrame({
        "mean_coefficient": mat.mean(axis=1),
        "variance": mat.var(axis=1, ddof=1),
    })
    out["sd"] = np.sqrt(out["variance"])
    out["sd_pct_of_abs_mean"] = 100 * out["sd"] / out["mean_coefficient"].abs()
    out["rank_by_mean"] = out["mean_coefficient"].rank(ascending=False, method="first").astype(int)
    ref = set(pd.read_csv(REF_PATH)["lsoa21cd"].astype(str))
    out["expert_region"] = out.index.isin(ref).astype(int)
    out = out.join(mat).sort_values("rank_by_mean")
    out.index.name = "lsoa21cd"
    out.to_csv(OUT / "consistency_per_lsoa.csv")

    published = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))["consistency"]
    for key, val in (("mean_variance", out["variance"].mean()), ("mean_sd", out["sd"].mean()), ("max_sd", out["sd"].max())):
        print(f"{key}: recomputed {val:.6f}  published {published[key]:.6f}")


if __name__ == "__main__":
    main()
