"""Thesis-quality LSOA maps for Hull GeoSMILE baseline-ablation effects."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from hull_boundaries import hull_outline, load_boundaries, save_boundary_report

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
MAP_DIR = OUT_DIR / "maps"
DIFF_DIR = MAP_DIR / "differences"
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = ROOT / "data"

FEATURE_LABELS = {
    "crime_lag_1": "Crime count (1-month lag)",
    "crime_lag_2": "Crime count (2-month lag)",
    "crime_lag_3": "Crime count (3-month lag)",
    "crime_rolling_mean_3": "Crime rolling mean (3 months)",
    "crime_rolling_mean_6": "Crime rolling mean (6 months)",
    "month_sin": "Seasonality (sin)",
    "month_cos": "Seasonality (cos)",
}

BLUE, PINK = "#1E88E5", "#FF0051"
INK, INK_MUTED = "#12171C", "#57646D"
DIVERGING_CMAP = plt.cm.colors.LinearSegmentedColormap.from_list(
    "geosmile_diverging", [BLUE, "#FFFFFF", PINK]
)
MAP_DPI = 300
FIG_SIZE = (8.5, 9.0)


def load_test_period() -> str:
    summary = json.loads((OUT_DIR / "run_summary.json").read_text(encoding="utf-8"))
    start, end = summary["test_period"]
    return f"{start} to {end}"


def shared_symmetric_limit(*arrays: np.ndarray, pct: float = 98) -> tuple[float, bool]:
    combined = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    combined = combined[np.isfinite(combined)]
    vmax = float(np.percentile(np.abs(combined), pct))
    vmax = max(vmax, 1e-9)
    clipped = bool(np.any(np.abs(combined) > vmax))
    return vmax, clipped


def aggregate_effects(path: Path, value_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.groupby("lsoa21cd", as_index=False)[value_columns].mean(numeric_only=True)


def _add_north_arrow(ax, x, y, size=0.045):
    # Arrow points from text (xytext) to tip (xy); tip must be above for north.
    ax.annotate(
        "N",
        xy=(x, y + size),
        xytext=(x, y),
        arrowprops=dict(facecolor=INK, width=2, headwidth=8, headlength=8),
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=INK,
        xycoords="axes fraction",
        textcoords="axes fraction",
    )


def _add_scale_bar(ax, gdf: gpd.GeoDataFrame, length_m: float = 2000):
    bounds = gdf.total_bounds
    x0 = bounds[0] + 0.04 * (bounds[2] - bounds[0])
    y0 = bounds[1] + 0.04 * (bounds[3] - bounds[1])
    ax.plot([x0, x0 + length_m], [y0, y0], color=INK, linewidth=3, solid_capstyle="butt", zorder=10)
    ax.text(x0 + length_m / 2, y0 + 120, f"{length_m / 1000:.0f} km", ha="center", va="bottom", fontsize=8, color=INK)


def plot_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    fname: Path,
    vmax: float,
    boundary_meta: dict,
    clip_note: str = "",
    cbar_label: str = "Signed baseline-ablation effect (predicted crimes)",
) -> None:
    plot_gdf = gdf.to_crs("EPSG:27700")
    # Drop empty/tiny artefacts; keep only valid polygons for display.
    plot_gdf = plot_gdf[plot_gdf.geometry.notna() & ~plot_gdf.geometry.is_empty].copy()
    plot_gdf["geometry"] = plot_gdf.geometry.buffer(0)
    outline = hull_outline(plot_gdf)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plot_gdf.plot(
        column=column,
        cmap=DIVERGING_CMAP,
        norm=norm,
        edgecolor="#FFFFFF",
        linewidth=0.12,
        ax=ax,
        zorder=2,
    )
    gpd.GeoSeries([outline], crs=plot_gdf.crs).plot(
        ax=ax, color="none", edgecolor=INK, linewidth=1.1, zorder=4
    )
    ax.set_axis_off()
    minx, miny, maxx, maxy = plot_gdf.total_bounds
    pad_x = 0.03 * (maxx - minx)
    pad_y = 0.03 * (maxy - miny)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=8)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.68, pad=0.015)
    cbar.set_label(cbar_label, fontsize=9, color=INK_MUTED)
    _add_scale_bar(ax, plot_gdf)
    _add_north_arrow(ax, 0.93, 0.12)
    test_period = load_test_period()
    source = (
        "ONS LSOA 2021 BGC (EPSG:27700)"
        if boundary_meta.get("source") != "voronoi_approximate"
        else "Approximate Voronoi geography"
    )
    warning = boundary_meta.get("warning", "")
    footer = (
        f"Test period: {test_period}. Source: {source}. "
        "Positive = original feature value raises prediction vs training median replacement; "
        "negative = lowers prediction. Not a causal effect."
    )
    if clip_note:
        footer = f"{clip_note} {footer}"
    if warning or boundary_meta.get("source") == "voronoi_approximate":
        footer = f"WARNING: Approximate Voronoi geography — not official ONS LSOAs. {footer}"
        fig.text(
            0.5,
            0.955,
            "Approximate Voronoi geography",
            ha="center",
            fontsize=10,
            color="#B00020",
            fontweight="bold",
        )
    fig.text(0.5, 0.012, footer, ha="center", fontsize=7.0, color=INK_MUTED, wrap=True)
    fig.savefig(fname, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {fname}")


def main() -> None:
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    DIFF_DIR.mkdir(parents=True, exist_ok=True)

    geo_raw = pd.read_csv(OUT_DIR / "geo_location_effects.csv")
    geo_col = (
        "location_baseline_ablation_effect"
        if "location_baseline_ablation_effect" in geo_raw.columns
        else "location_effect"
    )
    geo = geo_raw.groupby("lsoa21cd", as_index=False)[geo_col].mean(numeric_only=True)
    geo = geo.rename(columns={geo_col: "location_effect"})
    feat_a = aggregate_effects(OUT_DIR / "feature_a_local_effects.csv", list(FEATURE_LABELS))
    feat_b = aggregate_effects(OUT_DIR / "feature_b_local_effects.csv", list(FEATURE_LABELS))

    lsoa_codes = sorted(set(geo["lsoa21cd"]) | set(feat_a["lsoa21cd"]) | set(feat_b["lsoa21cd"]))
    boundaries, boundary_meta = load_boundaries(lsoa_codes)
    save_boundary_report(boundary_meta, DATA_DIR / "boundary_validation.json")

    geo_map = boundaries.merge(geo, on="lsoa21cd", how="inner", validate="one_to_one")
    geo_vmax, geo_clipped = shared_symmetric_limit(geo_map["location_effect"].to_numpy())
    plot_map(
        geo_map,
        "location_effect",
        "Hull — Geo branch signed baseline-ablation effect",
        MAP_DIR / "geo_location_sensitivity.png",
        geo_vmax,
        boundary_meta,
        clip_note="Colour scale clipped at combined 98th percentile of |effect|." if geo_clipped else "",
        cbar_label="Signed baseline-ablation effect (location)",
    )

    manifest = {"geo": ["geo_location_sensitivity.png"], "feature_a": [], "feature_b": [], "differences": []}
    for feature, label in FEATURE_LABELS.items():
        a_vals = feat_a[feature].to_numpy()
        b_vals = feat_b[feature].to_numpy()
        vmax, clipped = shared_symmetric_limit(a_vals, b_vals, pct=98)
        clip_note = (
            f"Shared symmetric limit = 98th percentile of |effect| ({vmax:.2f}). Values beyond this clip to full saturation."
            if clipped
            else f"Shared symmetric limit = 98th percentile of |effect| ({vmax:.2f})."
        )
        for branch, table, prefix, branch_title in (
            ("feature_a", feat_a, "feature_a", "Feature-A (no coordinates in model)"),
            ("feature_b", feat_b, "feature_b", "Feature-B (coordinates in prediction only)"),
        ):
            merged = boundaries.merge(table[["lsoa21cd", feature]], on="lsoa21cd", how="inner", validate="one_to_one")
            fname = MAP_DIR / f"{prefix}_{feature}.png"
            plot_map(
                merged,
                feature,
                f"Hull — {label}\n{branch_title}",
                fname,
                vmax,
                boundary_meta,
                clip_note=clip_note,
            )
            manifest[branch].append(fname.name)

        diff = feat_b[["lsoa21cd", feature]].merge(
            feat_a[["lsoa21cd", feature]], on="lsoa21cd", suffixes=("_b", "_a")
        )
        diff["difference_b_minus_a"] = diff[f"{feature}_b"] - diff[f"{feature}_a"]
        diff_map = boundaries.merge(diff[["lsoa21cd", "difference_b_minus_a"]], on="lsoa21cd", how="inner")
        dvmax, dclipped = shared_symmetric_limit(diff_map["difference_b_minus_a"].to_numpy(), pct=98)
        dclip = f"Difference scale clipped at 98th percentile (±{dvmax:.2f})." if dclipped else ""
        plot_map(
            diff_map,
            "difference_b_minus_a",
            f"Hull — {label}\nFeature-B minus Feature-A baseline-ablation effect",
            DIFF_DIR / f"difference_{feature}.png",
            dvmax,
            boundary_meta,
            clip_note=dclip,
            cbar_label="Difference in signed baseline-ablation effect",
        )
        manifest["differences"].append(f"differences/difference_{feature}.png")

    (MAP_DIR / "map_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nMaps written to {MAP_DIR.resolve()}")


if __name__ == "__main__":
    main()
