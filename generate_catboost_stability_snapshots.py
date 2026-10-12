"""CatBoost baseline + 4-directional stability snapshots, in the exact
format make_feature_stability_pptx.py / make_feature_stability_schematic.py
expect (compute_att_stability.save_stability_snapshot's directory layout),
so that existing, richer PPTX/schematic machinery can be reused for
CatBoost instead of rebuilt. Same cached cardinal-neighbour LSOAs as the
existing XGBoost run (eval_by_n_perturb.load_cached_neighbors), but at
THIS project's current design (Feature N=1,024, Geo N=4,000 -- the old
compute_att_stability.py module still hardcodes the superseded N=2,000 Geo
budget and geosmile_hull_twobranch.make_model's XGBoost-only construction,
neither of which apply here).

Writes outputs/att_stability_catboost/{baseline,north,east,south,west}/
(feature_importance.csv, geo_coefficients.csv, feature_importance_ranking.png,
geo_top20_map.png, snapshot_meta.json).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from compute_att_stability import compare_attributions, hull_row_pred_spearman, save_stability_snapshot
from eval_by_n_perturb import Y_COL, load_cached_neighbors
from geosmile_hull_twobranch import FORBIDDEN_FEATURE_TOKENS, SPATIAL_COLUMNS, choose_feature_players, geo_smile_coefficients, smile_global_feature, temporal_split
from hull_boundaries import load_boundaries
from hull_boundary_neighbors import expand_panel
from hull_expert_topk import FEATURE_TOP_K, GEO_TOP_K
from model_factory import fit_model, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "att_stability_catboost"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"

MODEL_NAME = "catboost"
SMILE_SEED = 42
FEAT_N = 1024
GEO_N = 4000
DIRECTIONS = ("north", "east", "south", "west")


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
    boundaries, _meta = load_boundaries(hull_lsoas)
    name_lookup = dict(zip(panel["lsoa21cd"].astype(str), panel["lsoa21nm"]))
    neighbors, extra_panel = load_cached_neighbors()
    extra_by_dir = {
        str(row.direction): extra_panel[extra_panel["lsoa21cd"].astype(str) == str(row.lsoa21cd)].copy()
        for row in neighbors.itertuples()
    }

    _log(f"Fitting {MODEL_NAME} Feature and Geo models (Hull-only)...")
    model_feat = make_model(MODEL_NAME, SMILE_SEED, n_jobs=2)
    fit_model(MODEL_NAME, model_feat, panel.loc[train, players], panel.loc[train, Y_COL],
              panel.loc[validation, players], panel.loc[validation, Y_COL])
    model_geo = make_model(MODEL_NAME, SMILE_SEED, n_jobs=2)
    fit_model(MODEL_NAME, model_geo, panel.loc[train, SPATIAL_COLUMNS], panel.loc[train, Y_COL],
              panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, Y_COL])

    _log("Baseline (Hull-only) Feature + Geo SMILE...")
    base_feat, _, _, _, _ = smile_global_feature(model_feat, X_test[players], players, baseline, FEAT_N, SMILE_SEED)
    base_geo, _ = geo_smile_coefficients(model_geo, panel, test, baseline, GEO_N, SMILE_SEED)
    base_geo["lsoa21cd"] = base_geo["lsoa21cd"].astype(str)
    base_feat_top = set(base_feat.sort_values(["abs_coefficient", "feature"], ascending=[False, True], kind="mergesort").head(FEATURE_TOP_K)["feature"])
    base_geo_top = set(base_geo.sort_values(["geo_abs_coefficient", "lsoa21cd"], ascending=[False, True], kind="mergesort").head(GEO_TOP_K)["lsoa21cd"])
    save_stability_snapshot(DEST / "baseline", label="Hull-only baseline", feat_imp=base_feat, geo_imp=base_geo,
                             boundaries=boundaries, name_lookup=name_lookup)

    for direction in DIRECTIONS:
        _log(f"\n=== [{direction}] ===")
        extra_rows = extra_by_dir[direction]
        code = str(extra_rows["lsoa21cd"].iloc[0])
        name = str(extra_rows["lsoa21nm"].iloc[0])
        expanded = expand_panel(panel, extra_rows)
        _, _, exp_test = temporal_split(expanded)
        X_exp = expanded.loc[exp_test].reset_index(drop=True)

        feat_imp, _, _, _, _ = smile_global_feature(model_feat, X_exp, players, baseline, FEAT_N, SMILE_SEED)
        geo_imp, _ = geo_smile_coefficients(model_geo, expanded, exp_test, baseline, GEO_N, SMILE_SEED, sample_lsoas=hull_lsoas)
        geo_imp["lsoa21cd"] = geo_imp["lsoa21cd"].astype(str)

        f_pred, _ = hull_row_pred_spearman(model_feat, X_test, X_exp, players)
        g_pred, _ = hull_row_pred_spearman(model_geo, X_test, X_exp, SPATIAL_COLUMNS)

        cmp = compare_attributions(base_feat, feat_imp, base_geo, geo_imp,
                                    base_feat_top=base_feat_top, base_geo_top=base_geo_top, added_lsoa=code)
        meta = {
            "direction": direction, "added_lsoa": code, "added_name": name,
            "feature_jaccard": cmp["feature_jaccard"], "feature_spearman": cmp["feature_spearman"],
            "geo_jaccard": cmp["geo_jaccard"], "geo_spearman": cmp["geo_spearman"],
            "geo_extra_rank": cmp["geo_extra_rank"], "geo_extra_in_top20": cmp["geo_extra_in_top20"],
            "feature_pred_spearman": f_pred, "geo_pred_spearman": g_pred,
        }
        save_stability_snapshot(DEST / direction, label=f"{direction.title()} addition", feat_imp=feat_imp, geo_imp=geo_imp,
                                 boundaries=boundaries, name_lookup=name_lookup, meta=meta)
        _log(f"[{direction}] added {code} ({name}): feature_jaccard={cmp['feature_jaccard']:.3f} "
             f"geo_jaccard={cmp['geo_jaccard']:.3f} geo_extra_rank={cmp['geo_extra_rank']}")

    _log("\nDone.")


if __name__ == "__main__":
    main()
