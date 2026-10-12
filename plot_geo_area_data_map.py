"""Two-panel Chapter-4 Geo map for the area-data design (N=4,000).

(a) signed Geo-SMILE coefficient on ONE sequential scale (higher = the model
    is more sensitive to that LSOA's data; the sign marks above/below-average
    sensitivity, not a direction of effect, so no diverging midpoint is used);
(b) exact prediction shift when the LSOA's data is replaced by the Hull median
    (crimes/month), with the expert-highlighted regions (58 LSOAs) outlined.

Reads outputs/geo_area_data_eval/geo_coefficients_N4000.csv.
Writes outputs/geo_area_data_eval/geo_area_data_map_N4000.{pdf,png}. Not causal.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize

from geosmile_hull_twobranch import INK, SEQUENTIAL, shared_limit
from hull_boundaries import hull_outline, load_boundaries

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "outputs" / "geo_area_data_eval"


def main() -> None:
    coef = pd.read_csv(DEST / "geo_coefficients_N4000.csv", dtype={"lsoa21cd": str})
    boundaries, _ = load_boundaries(sorted(coef["lsoa21cd"]))
    gdf = boundaries[["lsoa21cd", "geometry"]].merge(coef, on="lsoa21cd").to_crs("EPSG:27700")
    outline = hull_outline(gdf)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    panels = [
        ("geo_coefficient", "(a) Geo-SMILE geographic sensitivity coefficient",
         "Geo-SMILE geographic sensitivity coefficient", True),
        ("exact_w1", "(b) Single-LSOA Wasserstein prediction shift", "W1, crimes per month", False),
    ]
    for ax, (col, title, label, signed) in zip(axes, panels):
        vmax = shared_limit(gdf[col].to_numpy())
        # Signed beta: sequential from its minimum (least sensitive) upwards.
        norm = Normalize(float(gdf[col].min()), vmax) if signed else Normalize(0, vmax)
        cmap = SEQUENTIAL
        gdf.plot(column=col, cmap=cmap, norm=norm, edgecolor="#d9d9d9", linewidth=0.12, ax=ax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.01, extend="max").set_label(label, fontsize=9)
        if outline is not None:
            gpd.GeoSeries([outline], crs=gdf.crs).plot(ax=ax, facecolor="none", edgecolor=INK, linewidth=0.9)
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK)
        ax.set_axis_off()
    # Expert-highlighted regions (select_highlighted_region_lsoas.py), the Geo accuracy reference.
    regions = pd.read_csv(ROOT / "outputs" / "geo_area_data_models" / "top50_maps" / "highlighted_regions_lsoas.csv",
                          dtype={"lsoa21cd": str})[["lsoa21cd", "region"]]
    ref = gdf.merge(regions, on="lsoa21cd").dissolve(by="region")
    ref.boundary.plot(ax=axes[1], color=INK, linewidth=1.3)
    fig.text(
        0.5,
        0.02,
        "Geo branch explains the 30-feature XGBoost model; masking an LSOA replaces all of its inputs with the "
        "training median. N=4,000 perturbations, seed 42. Panel (b): black outlines = expert-highlighted regions (58 LSOAs). "
        "Both panels use one sequential scale (darker = more sensitive), capped at the 98th percentile. Model sensitivity, not causal effects.",
        ha="center",
        fontsize=7.5,
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(DEST / f"geo_area_data_map_N4000.{ext}", dpi=250, facecolor="white")
    plt.close(fig)
    print(f"Saved {DEST / 'geo_area_data_map_N4000.pdf'}")
    plain_map(gdf, outline)


def plain_map(gdf: gpd.GeoDataFrame, outline) -> None:
    """Single-panel Geo map with no annotations, outlines of case areas or callouts."""
    values = gdf["geo_coefficient"].to_numpy()
    norm = Normalize(float(values.min()), shared_limit(values))
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    gdf.plot(column="geo_coefficient", cmap=SEQUENTIAL, norm=norm, edgecolor="#d9d9d9", linewidth=0.12, ax=ax)
    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.01, extend="max").set_label(
        "Geo-SMILE geographic sensitivity coefficient", fontsize=9)
    if outline is not None:
        gpd.GeoSeries([outline], crs=gdf.crs).plot(ax=ax, facecolor="none", edgecolor=INK, linewidth=0.9)
    ax.set_title("Hull — Geo-SMILE geographic sensitivity coefficient", fontsize=12, fontweight="bold", color=INK)
    ax.set_axis_off()
    fig.text(0.5, 0.02,
             "Area-data design: masking an LSOA replaces all of its inputs with the training median; N=4,000, seed 42. "
             "Lighter = lower, darker = higher relative contribution to the coalition W1 response (capped at the 98th "
             "percentile); the sign is not a direction of prediction change. Not causal effects.",
             ha="center", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(DEST / f"geo_area_data_map_N4000_plain.{ext}", dpi=250, facecolor="white")
    plt.close(fig)
    print(f"Saved {DEST / 'geo_area_data_map_N4000_plain.pdf'}")


if __name__ == "__main__":
    main()
