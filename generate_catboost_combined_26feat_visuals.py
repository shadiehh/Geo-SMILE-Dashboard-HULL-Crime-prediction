"""Full visualization suite for CatBoost -- the combined-model, 26-feature
architecture's joint winner on both Feature fidelity (0.998) and Geo
fidelity (0.948), per combined_model_26feat_fidelity.py.

combined_feature_smile() (combined_model_fidelity.py) doesn't retain the
per-observation, per-mask prediction array (all_pert) that
geosmile_hull_twobranch.local_feature_coefficients() needs, so this script
inlines an equivalent Feature-branch masking loop that does keep it --
otherwise identical to combined_feature_smile (coordinates held fixed at
true values throughout, N=1,024, seed=42) -- rather than changing that
shared function's return signature and risking the other scripts that call it.

Produces, all for CatBoost under the 26-feature (crime_rolling_mean_6 kept,
crime_lag_1/2/3 and crime_rolling_mean_3 dropped) combined architecture:
  - Global Feature importance (top-10 bar chart)
  - Local per-LSOA Feature coefficients, top-10 drivers (panel map)
  - Geo branch: full signed map (all 168 LSOAs)
  - Geo branch: top-20 |coefficient| map

Writes outputs/feature_fidelity_excl_dominant/catboost_combined_26feat/*.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from combined_model_fidelity import combined_geo_smile
from compute_att_stability import save_feature_importance_ranking, save_geo_rank_map
from eval_by_n_perturb import Y_COL
from geosmile_hull_twobranch import (
    FORBIDDEN_FEATURE_TOKENS,
    SPATIAL_COLUMNS,
    apply_kernel,
    choose_feature_players,
    local_feature_coefficients,
    mask_pool,
    temporal_split,
)
from hull_boundaries import hull_outline, load_boundaries
from model_factory import fit_model, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "feature_fidelity_excl_dominant" / "catboost_combined_26feat"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"

MODEL_NAME = "catboost"
SMILE_SEED = 42
FEAT_N = 1024
GEO_N = 4000
TOP_K_LOCAL = 10
EXCLUDE = ["crime_lag_1", "crime_lag_2", "crime_lag_3", "crime_rolling_mean_3"]

BLUE, PINK = "#1E88E5", "#FF0051"
INK, INK_MUTED = "#12171C", "#57646D"
DIVERGING_CMAP = plt.cm.colors.LinearSegmentedColormap.from_list("geosmile_diverging", [BLUE, "#FFFFFF", PINK])
MAP_DPI = 220


def _log(msg: str) -> None:
    print(msg, flush=True)


def combined_feature_smile_with_pert(model, reference_full, players, baselines, n_masks, seed):
    """Same as combined_model_fidelity.combined_feature_smile, but also
    retains all_pert (needed for local_feature_coefficients)."""
    rng = np.random.default_rng(seed)
    masks = mask_pool(n_masks, len(players), rng)
    original = np.asarray(model.predict(reference_full), float)
    responses = np.zeros(len(masks), dtype=float)
    all_pert = np.zeros((len(masks), len(reference_full)), dtype=np.float32)
    for i, mask in enumerate(masks):
        perturbed = reference_full.copy()
        for j in np.flatnonzero(mask == 0):
            perturbed[players[j]] = baselines[players[j]]
        pred = np.asarray(model.predict(perturbed), float)
        all_pert[i] = pred
        responses[i] = wasserstein_distance(original, pred)
    distances = 1.0 - masks.mean(axis=1)
    order = rng.permutation(len(masks))
    a, b = int(0.6 * len(order)), int(0.8 * len(order))
    tr, va, te = order[:a], order[a:b], order[b:]
    sigma = max(float(np.median(distances[tr])), 1 / len(players))
    scores, fitted = {}, {}
    for kernel in ("rbf", "laplace", "inverse"):
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(masks[tr], responses[tr], sample_weight=apply_kernel(distances[tr], kernel, sigma))
        scores[kernel] = r2_score(responses[va], surrogate.predict(masks[va])) if len(va) > 1 else np.nan
        fitted[kernel] = surrogate
    best = max({k: v for k, v in scores.items() if np.isfinite(v)}, key=scores.get, default="rbf")
    surrogate = fitted[best]
    importance = pd.DataFrame(
        {"feature": players, "coefficient": surrogate.coef_, "abs_coefficient": np.abs(surrogate.coef_)}
    )
    diagnostics = {"best_kernel": best}
    return importance, diagnostics, masks, all_pert, original


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
    players = [p for p in all_players if p not in EXCLUDE]
    columns = players + SPATIAL_COLUMNS
    _log(f"26-feature combined columns: {len(players)} features + {len(SPATIAL_COLUMNS)} coordinates")

    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, columns].median()
    X_test = panel.loc[test, columns].reset_index(drop=True)
    X_test_lsoa = panel.loc[test, "lsoa21cd"].reset_index(drop=True)
    hull_lsoas = sorted(panel["lsoa21cd"].astype(str).unique())
    name_lookup = dict(zip(panel["lsoa21cd"].astype(str), panel["lsoa21nm"]))
    boundaries, _meta = load_boundaries(hull_lsoas)

    _log(f"Fitting {MODEL_NAME} (combined, 28 inputs)...")
    model = make_model(MODEL_NAME, SMILE_SEED, n_jobs=2)
    fit_model(MODEL_NAME, model, panel.loc[train, columns], panel.loc[train, Y_COL],
              panel.loc[validation, columns], panel.loc[validation, Y_COL])

    _log("Feature-branch SMILE (coordinates held fixed, N=1024)...")
    importance, feat_diag, masks, all_pert, original = combined_feature_smile_with_pert(
        model, X_test, players, baseline, FEAT_N, SMILE_SEED
    )
    importance.to_csv(DEST / "feature_global_importance.csv", index=False)
    save_feature_importance_ranking(
        importance, DEST / "feature_top10.png",
        title=f"{MODEL_NAME} (combined, 26 features): Feature SMILE ranking",
        footer=f"Combined-model architecture (26 features + 2 coordinates), N={FEAT_N}, seed=42. "
               f"Coordinates held fixed at true values throughout. Top-10 highlighted.",
    )
    _log(f"Saved: {DEST / 'feature_top10.png'}")

    _log("Computing local (per-LSOA) Feature coefficients...")
    local_feat = local_feature_coefficients(
        masks, all_pert, original, players, X_test_lsoa.to_numpy(), feat_diag["best_kernel"], SMILE_SEED
    )
    names = panel[["lsoa21cd", "lsoa21nm"]].drop_duplicates("lsoa21cd")
    local_feat = names.merge(local_feat, on="lsoa21cd", how="right")
    local_feat.to_csv(DEST / "feature_local_coefficients.csv", index=False)

    top10 = importance.reindex(importance["abs_coefficient"].sort_values(ascending=False).index).head(TOP_K_LOCAL)
    top10_features = top10["feature"].tolist()
    _log(f"Top-{TOP_K_LOCAL} global drivers: {top10_features}")
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
    cbar.set_label("Local |SMILE coefficient| (signed), shared scale" + (" (98th-pct clipped)" if clipped else ""), fontsize=9, color=INK_MUTED)
    fig.suptitle(f"{MODEL_NAME} (combined, 26 features): local Feature SMILE coefficients, top-10 drivers", fontsize=13, fontweight="bold", color=INK)
    local_path = DEST / "local_top10_drivers.png"
    fig.savefig(local_path, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _log(f"Saved: {local_path}")

    _log("Geo-branch SMILE (26 features held fixed, N=4000)...")
    geo_coef, geo_diag, *_ = combined_geo_smile(model, panel, test, baseline, players, GEO_N, SMILE_SEED)
    geo_coef["lsoa21cd"] = geo_coef["lsoa21cd"].astype(str)
    geo_coef.to_csv(DEST / "geo_coefficients.csv", index=False)
    _log(f"Geo kernel selected: {geo_diag['best_kernel']}")

    geo_vmax, geo_clipped = shared_symmetric_vmax(geo_coef["geo_coefficient"].to_numpy())
    geo_gdf = boundaries.merge(geo_coef, on="lsoa21cd", how="inner", validate="one_to_one")
    fig, ax = plt.subplots(figsize=(8.5, 9.0))
    plot_gdf = geo_gdf[geo_gdf.geometry.notna() & ~geo_gdf.geometry.is_empty].copy()
    plot_gdf["geometry"] = plot_gdf.geometry.buffer(0)
    norm = TwoSlopeNorm(vmin=-geo_vmax, vcenter=0.0, vmax=geo_vmax)
    plot_gdf.plot(column="geo_coefficient", cmap=DIVERGING_CMAP, norm=norm, edgecolor="#FFFFFF", linewidth=0.12, ax=ax, zorder=2)
    import geopandas as gpd
    gpd.GeoSeries([hull_outline(plot_gdf)], crs=plot_gdf.crs).plot(ax=ax, color="none", edgecolor=INK, linewidth=1.1, zorder=4)
    ax.set_axis_off()
    minx, miny, maxx, maxy = plot_gdf.total_bounds
    pad_x, pad_y = 0.03 * (maxx - minx), 0.03 * (maxy - miny)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")
    ax.set_title(f"{MODEL_NAME} (combined, 26 features): Geo signed baseline-ablation effect", fontsize=12.5, fontweight="bold", color=INK, pad=8)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.68, pad=0.015)
    cbar.set_label("Signed baseline-ablation effect (location)", fontsize=9, color=INK_MUTED)
    fig.text(0.5, 0.012,
              f"Combined-model architecture, Geo N={GEO_N}, seed=42. 26 features held fixed at true values throughout. "
              f"{'Colour scale clipped at 98th percentile. ' if geo_clipped else ''}Not causal.",
              ha="center", fontsize=7.0, color=INK_MUTED, wrap=True)
    signed_path = DEST / "geo_signed_full_map.png"
    fig.savefig(signed_path, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _log(f"Saved: {signed_path}")

    top20_path = DEST / "geo_top20.png"
    save_geo_rank_map(
        geo_coef, boundaries, top20_path,
        title=f"{MODEL_NAME} (combined, 26 features): Geo SMILE top-20",
        footer=f"Combined-model architecture, Geo N={GEO_N}, seed=42. 26 features held fixed at true values.",
        name_lookup=name_lookup,
    )
    _log(f"Saved: {top20_path}")
    _log("\nDone.")


if __name__ == "__main__":
    main()
