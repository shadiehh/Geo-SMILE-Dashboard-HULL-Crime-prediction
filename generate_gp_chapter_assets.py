"""Generate the GP-specific assets needed for the GP application chapter
that are not already saved anywhere: local (per-LSOA) Feature SMILE
coefficients for the top-6 global drivers, the full signed Geo map (all 168
LSOAs, not just the top-20), and concentration/entropy diagnostics.

Reuses geosmile_hull_twobranch.smile_global_feature/local_feature_coefficients
(model-agnostic, unchanged) with a fresh GP fit -- the ONLY reason this needs
a fresh SMILE pass at all is that verify_metrics_all_models.py never
persisted the raw (masks, all_pert, original) arrays local_feature_coefficients
needs; the resulting global importance is expected to reproduce
outputs/verify_metrics_all_models/gp/feature_importance.csv exactly (same
seed, same N, same model config) as a sanity check.

The Geo signed map reuses already-saved data (outputs/top_maps/tables/
geo_ranking_gp.csv) -- no recomputation there, just a new plotting routine
adapted from make_visuals_hull.py's signed diverging-colormap style (that
script itself is legacy/unrunnable against the current two-branch data, but
its plot_map() visual convention -- TwoSlopeNorm, blue-white-pink diverging
colormap, symmetric percentile clipping -- is what Figure 2 in the XGBoost
chapter used, so this reproduces that convention rather than inventing a
new one).

Writes:
  outputs/hull_chapter4_gp_assets/feature_local_coefficients_gp.csv
  outputs/hull_chapter4_gp_assets/geo_signed_full_map_gp.png
  outputs/hull_chapter4_gp_assets/local_top6_drivers_gp.png
  outputs/hull_chapter4_gp_assets/diagnostics_gp.json
"""

from __future__ import annotations

import json
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
DEST = OUT / "hull_chapter4_gp_assets"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"

SMILE_SEED = 42
FEAT_N = 1024
TOP_K_LOCAL = 6

BLUE, PINK = "#1E88E5", "#FF0051"
INK, INK_MUTED = "#12171C", "#57646D"
DIVERGING_CMAP = plt.cm.colors.LinearSegmentedColormap.from_list(
    "geosmile_diverging", [BLUE, "#FFFFFF", PINK]
)
MAP_DPI = 220


def _log(msg: str) -> None:
    print(msg, flush=True)


def hoyer_sparsity(x: np.ndarray) -> float:
    n = len(x)
    l1, l2 = np.sum(np.abs(x)), np.linalg.norm(x, 2)
    if l2 < 1e-12:
        return float("nan")
    return (np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1)


def gini(x: np.ndarray) -> float:
    x = np.sort(np.abs(x))
    n = len(x)
    if x.sum() < 1e-12:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def shannon_entropy(x: np.ndarray) -> float:
    p = np.abs(x) / (np.abs(x).sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def shared_symmetric_vmax(values: np.ndarray, pct: float = 98.0) -> tuple[float, bool]:
    vmax = float(np.percentile(np.abs(values), pct))
    vmax = max(vmax, 1e-9)
    clipped = bool(np.any(np.abs(values) > vmax))
    return vmax, clipped


def plot_signed_map(gdf, column: str, title: str, path: Path, vmax: float, cbar_label: str, footer: str) -> None:
    plot_gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    plot_gdf["geometry"] = plot_gdf.geometry.buffer(0)
    outline = hull_outline(plot_gdf)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(8.5, 9.0))
    plot_gdf.plot(column=column, cmap=DIVERGING_CMAP, norm=norm, edgecolor="#FFFFFF", linewidth=0.12, ax=ax, zorder=2)
    import geopandas as gpd
    gpd.GeoSeries([outline], crs=plot_gdf.crs).plot(ax=ax, color="none", edgecolor=INK, linewidth=1.1, zorder=4)
    ax.set_axis_off()
    minx, miny, maxx, maxy = plot_gdf.total_bounds
    pad_x, pad_y = 0.03 * (maxx - minx), 0.03 * (maxy - miny)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=8)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.68, pad=0.015)
    cbar.set_label(cbar_label, fontsize=9, color=INK_MUTED)
    fig.text(0.5, 0.012, footer, ha="center", fontsize=7.0, color=INK_MUTED, wrap=True)
    fig.savefig(path, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _log(f"Saved: {path}")


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
    name_lookup = dict(zip(panel["lsoa21cd"].astype(str), panel["lsoa21nm"]))
    boundaries, _meta = load_boundaries(hull_lsoas)

    _log("Fitting GP Feature model and running fresh Feature SMILE (N=1024, seed=42)...")
    model_feat = make_model("gp", SMILE_SEED, n_jobs=2)
    fit_model("gp", model_feat, panel.loc[train, players], panel.loc[train, Y_COL],
              panel.loc[validation, players], panel.loc[validation, Y_COL])
    importance, feat_diag, masks, all_pert, original = smile_global_feature(
        model_feat, X_test[players], players, baseline, FEAT_N, SMILE_SEED
    )
    saved_importance = pd.read_csv(OUT / "verify_metrics_all_models" / "gp" / "feature_importance.csv")
    reproduces = np.allclose(
        importance.set_index("feature").loc[players, "coefficient"],
        saved_importance.set_index("feature").loc[players, "coefficient"],
        atol=1e-6,
    )
    _log(f"Global Feature coefficient reproduction check (vs. saved verify_metrics_all_models): {reproduces}")

    _log("Computing local (per-LSOA) Feature coefficients...")
    local_feat = local_feature_coefficients(
        masks, all_pert, original, players, X_test["lsoa21cd"].to_numpy(), feat_diag["best_kernel"], SMILE_SEED
    )
    names = panel[["lsoa21cd", "lsoa21nm"]].drop_duplicates("lsoa21cd")
    local_feat = names.merge(local_feat, on="lsoa21cd", how="right")
    local_feat.to_csv(DEST / "feature_local_coefficients_gp.csv", index=False)

    top6 = importance.reindex(importance["abs_coefficient"].sort_values(ascending=False).index).head(TOP_K_LOCAL)
    top6_features = top6["feature"].tolist()
    _log(f"Top-{TOP_K_LOCAL} global drivers: {top6_features}")

    vmax, clipped = shared_symmetric_vmax(local_feat[top6_features].to_numpy())
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, feat in zip(axes.ravel(), top6_features):
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
    cbar.set_label("Local |SMILE coefficient| (signed), shared scale" + (" (98th-pct clipped)" if clipped else ""), fontsize=9, color=INK_MUTED)
    fig.suptitle("GP: local Feature SMILE coefficients for the top-6 global drivers", fontsize=13, fontweight="bold", color=INK)
    local_map_path = DEST / "local_top6_drivers_gp.png"
    fig.savefig(local_map_path, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _log(f"Saved: {local_map_path}")

    _log("Building full signed Geo map (all 168 LSOAs)...")
    geo_ranking = pd.read_csv(OUT / "top_maps" / "tables" / "geo_ranking_gp.csv")
    geo_ranking["lsoa21cd"] = geo_ranking["lsoa21cd"].astype(str)
    geo_map_gdf = boundaries.merge(geo_ranking, on="lsoa21cd", how="inner", validate="one_to_one")
    geo_vmax, geo_clipped = shared_symmetric_vmax(geo_map_gdf["geo_coefficient"].to_numpy())
    plot_signed_map(
        geo_map_gdf, "geo_coefficient",
        "GP: Geo branch signed baseline-ablation effect",
        DEST / "geo_signed_full_map_gp.png", geo_vmax,
        "Signed baseline-ablation effect (location)",
        f"GeoSMILE Geo branch, N=4,000, seed=42. Model: GP. "
        f"{'Colour scale clipped at 98th percentile of |effect|. ' if geo_clipped else ''}"
        "Positive = original coordinates raise prediction vs. training-median replacement; negative = lower. Not causal.",
    )

    _log("Computing concentration/entropy diagnostics...")
    feat_abs = importance["abs_coefficient"].to_numpy()
    geo_abs = geo_ranking["geo_abs_coefficient"].to_numpy()
    diagnostics = {
        "feature": {"hoyer": hoyer_sparsity(feat_abs), "gini": gini(feat_abs), "entropy": shannon_entropy(feat_abs), "n_nonzero": int(np.sum(feat_abs > 1e-12)), "n_total": len(feat_abs)},
        "geo": {"hoyer": hoyer_sparsity(geo_abs), "gini": gini(geo_abs), "entropy": shannon_entropy(geo_abs), "n_nonzero": int(np.sum(geo_abs > 1e-12)), "n_total": len(geo_abs)},
        "feature_global_reproduction_check": bool(reproduces),
        "top6_global_drivers": top6_features,
    }
    (DEST / "diagnostics_gp.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    _log(json.dumps(diagnostics, indent=2))
    _log(f"\nDone. Assets in {DEST}")


if __name__ == "__main__":
    main()
