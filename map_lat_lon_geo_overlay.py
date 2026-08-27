"""Overlay map of GeoSensitivity, latitude effect, and longitude effect.

These three layers are distinct quantities and remain independently scaled.
The RGB overlay is a visualisation composite, not a fourth GeoSMILE object.

Red   = |GeoSensitivity|          (Geo-branch location-masking SMILE)
Green = |latitude feature effect| (Feature SMILE latitude player)
Blue  = |longitude feature effect|(Feature SMILE longitude player)

Each channel is divided by its own 98th percentile of absolute value.
Not causal.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle

from hull_boundaries import hull_outline, load_boundaries

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "maps_overlay"
GEO_CSV = OUT / "geo_coefficients.csv"
LOC_CSV = OUT / "feature_with_location" / "location_local_coefficients.csv"
SUMMARY = OUT / "run_summary_twobranch.json"

BLUE, PINK = "#1E88E5", "#FF0051"
INK, INK_MUTED = "#12171C", "#57646D"
DIVERGING = plt.cm.colors.LinearSegmentedColormap.from_list(
    "geosmile_diverging", [BLUE, "#FFFFFF", PINK]
)
GEO_RGB = np.array([1.0, 0.0, 0.31])
LAT_RGB = np.array([0.12, 0.62, 0.22])
LON_RGB = np.array([0.12, 0.40, 0.75])
RANK_CMAP = plt.cm.colors.LinearSegmentedColormap.from_list(
    "geosmile_rank", ["#8A0033", "#FF0051", "#FFD0DC"]
)
BASE_FILL = "#E8ECF0"
MAP_DPI = 300
TOP_K = 50


def _json_safe(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def unit_scale(values: np.ndarray, pct: float = 98) -> tuple[np.ndarray, float]:
    abs_vals = np.abs(np.asarray(values, dtype=float))
    finite = abs_vals[np.isfinite(abs_vals)]
    vmax = max(float(np.percentile(finite, pct)), 1e-9)
    return np.clip(abs_vals / vmax, 0.0, 1.0), vmax


def overlay_channels(
    geo: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    pct: float = 98,
) -> dict:
    """Independently scale |effects| to [0, 1] and build RGB + dominance."""
    r, geo_p98 = unit_scale(geo, pct)
    g, lat_p98 = unit_scale(latitude, pct)
    b, lon_p98 = unit_scale(longitude, pct)
    rgb = np.column_stack([r, g, b])
    stacked = np.column_stack([r, g, b])
    labels = np.array(["geo_sensitivity", "latitude_effect", "longitude_effect"])
    dominant = labels[np.argmax(stacked, axis=1)]
    ties = (np.ptp(stacked, axis=1) < 1e-12) & (stacked.max(axis=1) > 0)
    dominant = dominant.astype(object)
    dominant[ties] = "tie"
    dominant[stacked.max(axis=1) == 0] = "none"
    return {
        "red": r,
        "green": g,
        "blue": b,
        "rgb": rgb,
        "dominant": dominant,
        "geo_p98": geo_p98,
        "lat_p98": lat_p98,
        "lon_p98": lon_p98,
    }


def load_overlay_table() -> pd.DataFrame:
    geo = pd.read_csv(GEO_CSV)
    loc = pd.read_csv(LOC_CSV)
    frame = geo.merge(
        loc[
            [
                "lsoa21cd",
                "lsoa21nm",
                "centroid_latitude",
                "centroid_longitude",
                "centroid_latitude_abs",
                "centroid_longitude_abs",
            ]
        ],
        on="lsoa21cd",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_loc"),
    )
    if "lsoa21nm_loc" in frame.columns:
        frame = frame.drop(columns=["lsoa21nm_loc"])
    channels = overlay_channels(
        frame["geo_coefficient"].to_numpy(),
        frame["centroid_latitude"].to_numpy(),
        frame["centroid_longitude"].to_numpy(),
    )
    frame["geo_unit"] = channels["red"]
    frame["lat_unit"] = channels["green"]
    frame["lon_unit"] = channels["blue"]
    frame["dominant_layer"] = channels["dominant"]
    frame["overlay_r"] = channels["red"]
    frame["overlay_g"] = channels["green"]
    frame["overlay_b"] = channels["blue"]
    scales = {
        "geo_p98": channels["geo_p98"],
        "lat_p98": channels["lat_p98"],
        "lon_p98": channels["lon_p98"],
    }
    frame = add_top_k_ranks(frame, k=TOP_K)
    frame.attrs["scales"] = scales
    return frame


def _stable_abs_rank(frame: pd.DataFrame, value: pd.Series) -> np.ndarray:
    """Rank 1 = largest |value|. Ties broken by lsoa21cd so each list has exactly k."""
    order = pd.DataFrame(
        {
            "lsoa21cd": frame["lsoa21cd"].astype(str),
            "abs_value": np.abs(value.to_numpy(dtype=float)),
        }
    )
    order = order.sort_values(
        ["abs_value", "lsoa21cd"], ascending=[False, True], kind="mergesort"
    )
    order["rank"] = np.arange(1, len(order) + 1)
    return order.set_index("lsoa21cd")["rank"].reindex(frame["lsoa21cd"].astype(str)).to_numpy()


def add_top_k_ranks(frame: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """Top-k |GeoSensitivity| is the importance set. Lat/lon overlay only on that set."""
    out = frame.copy()
    out["geo_rank_abs"] = _stable_abs_rank(out, out["geo_coefficient"])
    out["lat_rank_abs"] = _stable_abs_rank(out, out["centroid_latitude"])
    out["lon_rank_abs"] = _stable_abs_rank(out, out["centroid_longitude"])
    out["in_top50_geo"] = out["geo_rank_abs"] <= k
    out["in_top50_lat"] = out["lat_rank_abs"] <= k
    out["in_top50_lon"] = out["lon_rank_abs"] <= k
    out["overlay_lat_on_geo"] = out["in_top50_geo"] & out["in_top50_lat"]
    out["overlay_lon_on_geo"] = out["in_top50_geo"] & out["in_top50_lon"]
    out["n_coord_overlays"] = (
        out["overlay_lat_on_geo"].astype(int) + out["overlay_lon_on_geo"].astype(int)
    )
    out["n_top50_lists"] = (
        out["in_top50_geo"].astype(int)
        + out["in_top50_lat"].astype(int)
        + out["in_top50_lon"].astype(int)
    )
    out["geo_rank_shade"] = np.where(out["in_top50_geo"], out["geo_rank_abs"], np.nan)
    ranks = out[["geo_rank_abs", "lat_rank_abs", "lon_rank_abs"]].to_numpy(dtype=float)
    masks = out[["in_top50_geo", "in_top50_lat", "in_top50_lon"]].to_numpy()
    best = np.where(masks, ranks, np.inf).min(axis=1)
    out["best_top50_rank"] = np.where(np.isfinite(best), best, np.nan)
    return out


def _test_period() -> str:
    if not SUMMARY.exists():
        return "published test window"
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    start, end = payload["test_period"]
    return f"{start[:10]} to {end[:10]}"


def _add_north_arrow(ax, x=0.93, y=0.12, size=0.045):
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
    ax.plot([x0, x0 + length_m], [y0, y0], color=INK, linewidth=3, solid_capstyle="butt")
    ax.text(
        x0 + length_m / 2,
        y0 + 120,
        f"{length_m / 1000:.0f} km",
        ha="center",
        va="bottom",
        fontsize=8,
        color=INK,
    )


def _prepare_map(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    plot_gdf = gdf.to_crs("EPSG:27700")
    plot_gdf = plot_gdf[plot_gdf.geometry.notna() & ~plot_gdf.geometry.is_empty].copy()
    plot_gdf["geometry"] = plot_gdf.geometry.buffer(0)
    return plot_gdf


def _finish_axes(ax, plot_gdf: gpd.GeoDataFrame, title: str) -> None:
    outline = hull_outline(plot_gdf)
    gpd.GeoSeries([outline], crs=plot_gdf.crs).plot(
        ax=ax, color="none", edgecolor=INK, linewidth=1.0
    )
    ax.set_axis_off()
    minx, miny, maxx, maxy = plot_gdf.total_bounds
    ax.set_xlim(minx - 0.03 * (maxx - minx), maxx + 0.03 * (maxx - minx))
    ax.set_ylim(miny - 0.03 * (maxy - miny), maxy + 0.03 * (maxy - miny))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    _add_scale_bar(ax, plot_gdf)
    _add_north_arrow(ax)


def plot_signed(ax, plot_gdf: gpd.GeoDataFrame, column: str, title: str, vmax: float) -> None:
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    plot_gdf.plot(
        column=column,
        cmap=DIVERGING,
        norm=norm,
        edgecolor="white",
        linewidth=0.12,
        ax=ax,
    )
    _finish_axes(ax, plot_gdf, title)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.62, pad=0.02)
    cbar.set_label("Signed coefficient", fontsize=8, color=INK_MUTED)


def plot_rgb_overlay(ax, plot_gdf: gpd.GeoDataFrame, title: str) -> None:
    colours = list(map(tuple, plot_gdf[["overlay_r", "overlay_g", "overlay_b"]].to_numpy()))
    plot_gdf.plot(color=colours, edgecolor="white", linewidth=0.12, ax=ax)
    _finish_axes(ax, plot_gdf, title)
    _rgb_legend(ax)


def _rgb_legend(ax) -> None:
    items = [
        (GEO_RGB, "Red  |GeoSensitivity|"),
        (LAT_RGB, "Green  |latitude effect|"),
        (LON_RGB, "Blue  |longitude effect|"),
    ]
    x0, y0 = 0.04, 0.22
    box = FancyBboxPatch(
        (x0 - 0.015, y0 - 0.15),
        0.46,
        0.18,
        boxstyle="round,pad=0.01",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="#D0D5DA",
        linewidth=0.6,
        zorder=6,
    )
    ax.add_patch(box)
    for i, (rgb, label) in enumerate(items):
        y = y0 - i * 0.048
        ax.add_patch(
            Rectangle(
                (x0, y),
                0.045,
                0.032,
                transform=ax.transAxes,
                facecolor=tuple(rgb),
                edgecolor="none",
                zorder=7,
            )
        )
        ax.text(
            x0 + 0.055,
            y + 0.014,
            label,
            transform=ax.transAxes,
            fontsize=7.5,
            color=INK,
            va="center",
            zorder=7,
        )


def plot_dominance(ax, plot_gdf: gpd.GeoDataFrame) -> None:
    colour_map = {
        "geo_sensitivity": tuple(GEO_RGB),
        "latitude_effect": tuple(LAT_RGB),
        "longitude_effect": tuple(LON_RGB),
        "tie": (0.75, 0.75, 0.75),
        "none": (0.92, 0.92, 0.92),
    }
    colours = [colour_map[str(v)] for v in plot_gdf["dominant_layer"]]
    plot_gdf.plot(color=colours, edgecolor="white", linewidth=0.12, ax=ax)
    _finish_axes(
        ax,
        plot_gdf,
        "Dominant independently scaled |layer|",
    )
    labels = [
        (GEO_RGB, "GeoSensitivity largest"),
        (LAT_RGB, "Latitude effect largest"),
        (LON_RGB, "Longitude effect largest"),
    ]
    x0, y0 = 0.04, 0.20
    for i, (rgb, label) in enumerate(labels):
        ax.add_patch(
            Rectangle(
                (x0, y0 - i * 0.05),
                0.045,
                0.032,
                transform=ax.transAxes,
                facecolor=tuple(rgb),
                edgecolor="none",
            )
        )
        ax.text(
            x0 + 0.055,
            y0 - i * 0.05 + 0.014,
            label,
            transform=ax.transAxes,
            fontsize=8,
            color=INK,
            va="center",
        )


def plot_top50_rank(ax, plot_gdf: gpd.GeoDataFrame, k: int = TOP_K) -> None:
    from matplotlib.colors import Normalize

    base = plot_gdf.copy()
    geo_set = plot_gdf[plot_gdf["in_top50_geo"]].copy()
    lat_on_geo = plot_gdf[plot_gdf["overlay_lat_on_geo"] & ~plot_gdf["overlay_lon_on_geo"]]
    lon_on_geo = plot_gdf[plot_gdf["overlay_lon_on_geo"] & ~plot_gdf["overlay_lat_on_geo"]]
    both_on_geo = plot_gdf[plot_gdf["overlay_lat_on_geo"] & plot_gdf["overlay_lon_on_geo"]]
    base.plot(color=BASE_FILL, edgecolor="white", linewidth=0.12, ax=ax)
    if not geo_set.empty:
        geo_set.plot(
            column="geo_rank_shade",
            cmap=RANK_CMAP,
            norm=Normalize(vmin=1, vmax=k),
            edgecolor="white",
            linewidth=0.12,
            ax=ax,
        )
    if not lat_on_geo.empty:
        lat_on_geo.plot(
            ax=ax,
            facecolor="none",
            edgecolor=INK,
            linewidth=0.7,
            hatch="////",
        )
    if not lon_on_geo.empty:
        lon_on_geo.plot(
            ax=ax,
            facecolor="none",
            edgecolor=INK,
            linewidth=0.7,
            hatch="\\\\\\\\",
        )
    if not both_on_geo.empty:
        both_on_geo.plot(
            ax=ax,
            facecolor="none",
            edgecolor=INK,
            linewidth=1.35,
            hatch="xxxx",
        )
    _finish_axes(
        ax,
        plot_gdf,
        f"Top {k} |GeoSensitivity|, overlaid with latitude and longitude",
    )
    sm = plt.cm.ScalarMappable(cmap=RANK_CMAP, norm=Normalize(vmin=1, vmax=k))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.62, pad=0.02)
    cbar.set_label(
        f"|GeoSensitivity| rank (1 = highest). Lat/lon overlay only on this set.",
        fontsize=7.5,
        color=INK_MUTED,
    )
    ax.text(
        0.04,
        0.14,
        "Grey = not in GeoSensitivity top 50.\n"
        "Hatch / = also top-50 latitude.\n"
        "Hatch \\ = also top-50 longitude.\n"
        "Cross-hatch + thick outline = both.",
        transform=ax.transAxes,
        fontsize=7.2,
        color=INK,
        va="bottom",
    )


def _footer(scales: dict) -> str:
    return (
        f"Test period: {_test_period()}. Source: ONS LSOA 2021 BGC. "
        f"|GeoSensitivity| p98={scales['geo_p98']:.3f}; "
        f"|latitude| p98={scales['lat_p98']:.3f}; "
        f"|longitude| p98={scales['lon_p98']:.3f}. "
        "RGB channels are independently scaled. Three distinct quantities. Not causal."
    )


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    table = load_overlay_table()
    scales = table.attrs["scales"]
    boundaries, _meta = load_boundaries(table["lsoa21cd"].astype(str).tolist())
    gdf = boundaries.merge(table, on="lsoa21cd", how="inner", validate="one_to_one")
    if len(gdf) != 168:
        raise SystemExit(f"Expected 168 LSOAs in overlay map, got {len(gdf)}")
    plot_gdf = _prepare_map(gdf)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 13.2))
    plot_signed(
        axes[0, 0],
        plot_gdf,
        "geo_coefficient",
        "GeoSensitivity (signed)",
        scales["geo_p98"],
    )
    plot_signed(
        axes[0, 1],
        plot_gdf,
        "centroid_latitude",
        "Latitude feature effect (signed)",
        scales["lat_p98"],
    )
    plot_signed(
        axes[1, 0],
        plot_gdf,
        "centroid_longitude",
        "Longitude feature effect (signed)",
        scales["lon_p98"],
    )
    plot_rgb_overlay(axes[1, 1], plot_gdf, "RGB overlay of independently scaled |effects|")
    fig.text(0.5, 0.012, _footer(scales), ha="center", fontsize=7.2, color=INK_MUTED)
    four_path = DEST / "lat_lon_geo_overlay_four_panel.png"
    fig.savefig(four_path, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 9.2))
    plot_rgb_overlay(ax, plot_gdf, "Hull — GeoSensitivity / latitude / longitude overlay")
    fig.text(0.5, 0.012, _footer(scales), ha="center", fontsize=7.2, color=INK_MUTED)
    rgb_path = DEST / "lat_lon_geo_overlay_rgb.png"
    fig.savefig(rgb_path, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 9.2))
    plot_dominance(ax, plot_gdf)
    fig.text(0.5, 0.012, _footer(scales), ha="center", fontsize=7.2, color=INK_MUTED)
    dom_path = DEST / "lat_lon_geo_overlay_dominance.png"
    fig.savefig(dom_path, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 9.2))
    plot_top50_rank(ax, plot_gdf)
    fig.text(
        0.5,
        0.012,
        (
            f"Test period: {_test_period()}. Source: ONS LSOA 2021 BGC. "
            f"Importance set = top {TOP_K} |GeoSensitivity|. "
            "Latitude and longitude overlay only where they also rank in their "
            "own top 50. Shade = GeoSensitivity rank. Lat/lon-only LSOAs are "
            "not treated as important. One sequential colour. Not causal."
        ),
        ha="center",
        fontsize=7.2,
        color=INK_MUTED,
    )
    rank_path = DEST / "lat_lon_geo_top50_rank.png"
    fig.savefig(rank_path, dpi=MAP_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    export_cols = [
        "lsoa21cd",
        "lsoa21nm",
        "geo_coefficient",
        "geo_abs_coefficient",
        "centroid_latitude",
        "centroid_longitude",
        "geo_unit",
        "lat_unit",
        "lon_unit",
        "dominant_layer",
        "overlay_r",
        "overlay_g",
        "overlay_b",
        "geo_rank_abs",
        "lat_rank_abs",
        "lon_rank_abs",
        "in_top50_geo",
        "in_top50_lat",
        "in_top50_lon",
        "overlay_lat_on_geo",
        "overlay_lon_on_geo",
        "n_coord_overlays",
        "n_top50_lists",
        "geo_rank_shade",
        "best_top50_rank",
    ]
    csv_path = DEST / "lat_lon_geo_overlay.csv"
    table[export_cols].to_csv(csv_path, index=False)

    counts = table["dominant_layer"].value_counts().to_dict()
    manifest = {
        "n_lsoas": int(len(table)),
        "scales": {k: _json_safe(v) for k, v in scales.items()},
        "dominant_counts": {str(k): int(v) for k, v in counts.items()},
        "top50": {
            "k": TOP_K,
            "importance_rule": "must be in top-k |GeoSensitivity|",
            "geo_top50": int(table["in_top50_geo"].sum()),
            "geo_and_lat": int(table["overlay_lat_on_geo"].sum()),
            "geo_and_lon": int(table["overlay_lon_on_geo"].sum()),
            "geo_and_both_coord": int(
                (table["overlay_lat_on_geo"] & table["overlay_lon_on_geo"]).sum()
            ),
            "lat_or_lon_only_excluded_from_importance": int(
                ((table["in_top50_lat"] | table["in_top50_lon"]) & ~table["in_top50_geo"]).sum()
            ),
            "shade": "|GeoSensitivity| rank; lat/lon overlay only on that set",
        },
        "definition": (
            "RGB overlay of independently 98th-percentile-scaled absolute "
            "GeoSensitivity, latitude feature effect, and longitude feature "
            "effect. Top-50 rank map uses one sequential colour. "
            "Not a merged GeoSMILE quantity. Not causal."
        ),
        "files": {
            "four_panel": four_path.name,
            "rgb_overlay": rgb_path.name,
            "dominance": dom_path.name,
            "top50_rank": rank_path.name,
            "table": csv_path.name,
        },
    }
    (DEST / "overlay_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved overlay maps to {DEST}")
    print(json.dumps(manifest["top50"], indent=2))


if __name__ == "__main__":
    main()
