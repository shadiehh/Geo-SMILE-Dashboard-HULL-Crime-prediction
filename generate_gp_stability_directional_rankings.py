"""Full per-direction Feature and Geo SMILE rankings for GP's cardinal-
neighbour stability check (30-feature version) -- the raw rankings that
verify_metrics_all_models.py's att_stability_asymmetric() computes
internally but never saves (only the summary Jaccard/Spearman comparison
survives in results.json). Needed to build the "Hull-only baseline vs.
West addition" style stability figure and the four directional Geo top-20
maps.

Predictive models are trained ONCE on Hull-only data (matching the
established protocol: "predictive models remain trained on Hull only; only
the SMILE reference is expanded"); only the SMILE explanation reruns on
each direction's expanded reference. Same N/seed as everywhere else in this
project (Feature N=1,024, Geo N=4,000, seed=42).

Writes outputs/hull_chapter4_gp_assets/stability/{feature,geo}_{baseline,
north,east,south,west}.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from compute_att_stability import DIRECTIONS
from eval_by_n_perturb import Y_COL, load_cached_neighbors
from geosmile_hull_twobranch import (
    FORBIDDEN_FEATURE_TOKENS,
    SPATIAL_COLUMNS,
    choose_feature_players,
    geo_smile_coefficients,
    smile_global_feature,
    temporal_split,
)
from hull_boundary_neighbors import expand_panel
from model_factory import fit_model, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "hull_chapter4_gp_assets" / "stability"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"

SMILE_SEED = 42
FEAT_N = 1024
GEO_N = 4000


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    bad = [p for p in players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise SystemExit(f"Spatial leak: {bad}")
    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
    X_test = panel.loc[test].reset_index(drop=True)
    hull_lsoas = sorted(panel["lsoa21cd"].astype(str).unique())
    neighbors, extra_panel = load_cached_neighbors()
    extra_by_dir = {
        str(row.direction): extra_panel[extra_panel["lsoa21cd"].astype(str) == str(row.lsoa21cd)].copy()
        for row in neighbors.itertuples()
    }

    _log("Fitting GP Feature and Geo models (Hull-only train data)...")
    model_feat = make_model("gp", SMILE_SEED, n_jobs=2)
    fit_model("gp", model_feat, panel.loc[train, players], panel.loc[train, Y_COL],
              panel.loc[validation, players], panel.loc[validation, Y_COL])
    model_geo = make_model("gp", SMILE_SEED, n_jobs=2)
    fit_model("gp", model_geo, panel.loc[train, SPATIAL_COLUMNS], panel.loc[train, Y_COL],
              panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, Y_COL])

    _log("Baseline (Hull-only) Feature + Geo SMILE...")
    base_feat, _, _, _, _ = smile_global_feature(model_feat, X_test[players], players, baseline, FEAT_N, SMILE_SEED)
    base_feat.to_csv(DEST / "feature_baseline.csv", index=False)
    base_geo, _ = geo_smile_coefficients(model_geo, panel, test, baseline, GEO_N, SMILE_SEED)
    base_geo["lsoa21cd"] = base_geo["lsoa21cd"].astype(str)
    base_geo.to_csv(DEST / "geo_baseline.csv", index=False)

    for direction in DIRECTIONS:
        _log(f"\n=== [{direction}] ===")
        extra_rows = extra_by_dir[direction]
        code = str(extra_rows["lsoa21cd"].iloc[0])
        expanded = expand_panel(panel, extra_rows)
        _, _, exp_test = temporal_split(expanded)
        X_exp = expanded.loc[exp_test].reset_index(drop=True)

        _log(f"[{direction}] Feature SMILE (expanded reference, extra LSOA {code})...")
        feat_imp, _, _, _, _ = smile_global_feature(model_feat, X_exp, players, baseline, FEAT_N, SMILE_SEED)
        feat_imp.to_csv(DEST / f"feature_{direction}.csv", index=False)

        _log(f"[{direction}] Geo SMILE (expanded reference, extra LSOA {code})...")
        geo_imp, _ = geo_smile_coefficients(model_geo, expanded, exp_test, baseline, GEO_N, SMILE_SEED, sample_lsoas=hull_lsoas)
        geo_imp["lsoa21cd"] = geo_imp["lsoa21cd"].astype(str)
        geo_imp.to_csv(DEST / f"geo_{direction}.csv", index=False)
        _log(f"[{direction}] done (extra LSOA {code})")

    _log("\nDone.")


if __name__ == "__main__":
    main()
