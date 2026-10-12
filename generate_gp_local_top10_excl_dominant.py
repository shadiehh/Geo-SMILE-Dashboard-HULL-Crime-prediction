"""GP-only: local (per-LSOA) Feature SMILE coefficients for the top-10
global drivers under the 29-feature version (crime_rolling_mean_6 excluded
from both the base model and GeoSMILE, per the fidelity-robustness check in
feature_fidelity_excl_dominant.py). Reuses the same
smile_global_feature/local_feature_coefficients pipeline as
generate_gp_chapter_assets.py, just with the reduced player list and top-10
(not top-6) panels.

Writes outputs/feature_fidelity_excl_dominant/{feature_global_importance_gp_29feat.csv,
feature_local_coefficients_gp_29feat.csv, local_top10_drivers_gp_29feat.png}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from eval_by_n_perturb import Y_COL
from geosmile_hull_twobranch import (
    FORBIDDEN_FEATURE_TOKENS,
    SPATIAL_COLUMNS,
    choose_feature_players,
    local_feature_coefficients,
    smile_global_feature,
    temporal_split,
)
from hull_boundaries import hull_outline, load_boundaries
from model_factory import fit_model, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "feature_fidelity_excl_dominant"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"

SMILE_SEED = 42
FEAT_N = 1024
TOP_K_LOCAL = 10
EXCLUDED_FEATURE = "crime_rolling_mean_6"

BLUE, PINK = "#1E88E5", "#FF0051"
INK, INK_MUTED = "#12171C", "#57646D"
DIVERGING_CMAP = plt.cm.colors.LinearSegmentedColormap.from_list("geosmile_diverging", [BLUE, "#FFFFFF", PINK])
MAP_DPI = 220


def _log(msg: str) -> None:
    print(msg, flush=True)


def shared_symmetric_vmax(values: np.ndarray, pct: float = 98.0) -> tuple[float, bool]:
    vmax = max(float(np.percentile(np.abs(values), pct)), 1e-9)
    return vmax, bool(np.any(np.abs(values) > vmax))


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    all_players = choose_feature_players(panel)
    bad = [p for p in all_players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise SystemExit(f"Spatial leak: {bad}")
    players = [p for p in all_players if p != EXCLUDED_FEATURE]
    _log(f"Excluding {EXCLUDED_FEATURE!r}: {len(all_players)} -> {len(players)} Feature players")

    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
    X_test = panel.loc[test].reset_index(drop=True)
    hull_lsoas = sorted(panel["lsoa21cd"].astype(str).unique())
    boundaries, _meta = load_boundaries(hull_lsoas)

    _log("Fitting GP Feature model (29 features) and running Feature SMILE (N=1024, seed=42)...")
    model_feat = make_model("gp", SMILE_SEED, n_jobs=2)
    fit_model("gp", model_feat, panel.loc[train, players], panel.loc[train, Y_COL],
              panel.loc[validation, players], panel.loc[validation, Y_COL])
    importance, feat_diag, masks, all_pert, original = smile_global_feature(
        model_feat, X_test[players], players, baseline, FEAT_N, SMILE_SEED
    )
    importance.to_csv(DEST / "feature_global_importance_gp_29feat.csv", index=False)
    _log(f"Fidelity kernel selected: {feat_diag['best_kernel']}")

    _log("Computing local (per-LSOA) Feature coefficients...")
    local_feat = local_feature_coefficients(
        masks, all_pert, original, players, X_test["lsoa21cd"].to_numpy(), feat_diag["best_kernel"], SMILE_SEED
    )
    names = panel[["lsoa21cd", "lsoa21nm"]].drop_duplicates("lsoa21cd")
    local_feat = names.merge(local_feat, on="lsoa21cd", how="right")
    local_feat.to_csv(DEST / "feature_local_coefficients_gp_29feat.csv", index=False)

    top10 = importance.reindex(importance["abs_coefficient"].sort_values(ascending=False).index).head(TOP_K_LOCAL)
    top10_features = top10["feature"].tolist()
    _log(f"Top-{TOP_K_LOCAL} global drivers (29-feature GP): {top10_features}")
    _log(top10.to_string(index=False))

    vmax, clipped = shared_symmetric_vmax(local_feat[top10_features].to_numpy())
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    for ax, feat in zip(axes.ravel(), top10_features):
        merged = boundaries.merge(local_feat[["lsoa21cd", feat]], on="lsoa21cd", how="inner", validate="one_to_one")
        plot_gdf = merged[merged.geometry.notna() & ~merged.geometry.is_empty].copy()
        plot_gdf["geometry"] = plot_gdf.geometry.buffer(0)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        plot_gdf.plot(column=feat, cmap=DIVERGING_CMAP, norm=norm, edgecolor="#FFFFFF", linewidth=0.08, ax=ax)
        import geopandas as gpd
        gpd.GeoSeries([hull_outline(plot_gdf)], crs=plot_gdf.crs).plot(ax=ax, color="none", edgecolor=INK, linewidth=0.9)
        ax.set_axis_off()
        ax.set_title(feat, fontsize=10, fontweight="bold", color=INK)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING_CMAP, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, orientation="horizontal", location="bottom")
    cbar.set_label(
        "Local |SMILE coefficient| (signed), shared scale" + (" (98th-pct clipped)" if clipped else ""),
        fontsize=9, color=INK_MUTED,
    )
    fig.suptitle(
        f"GP (29 features, {EXCLUDED_FEATURE} excluded): local Feature SMILE coefficients, top-10 global drivers",
        fontsize=13, fontweight="bold", color=INK,
    )
    out_path = DEST / "local_top10_drivers_gp_29feat.png"
    fig.savefig(out_path, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _log(f"Saved: {out_path}")
    _log("\nDone.")


if __name__ == "__main__":
    main()
