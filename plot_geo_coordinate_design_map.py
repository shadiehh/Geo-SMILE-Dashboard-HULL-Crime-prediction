"""Maps for the Seattle/Chicago-matched Hull Geo design (coordinate-only masking, N=4,000).

Figure 1, same layout as plot_geo_area_data_map.py:
  (a) signed Geo-SMILE coefficient, one sequential scale (sign is not a direction of change);
  (b) single-LSOA Wasserstein prediction shift when only that LSOA's coordinates move to the
      training-median point, with the expert-highlighted regions (58 LSOAs) outlined.
Figure 2: published area-data GeoSensitivity (a) beside the coordinate-only design (b), each on
its own scale, top 5 of each labelled by rank.

Reads outputs/geo_design_checks/coordinate_masking_eval/geo_coefficients_N4000.csv
(geo_coordinate_design_eval.py) and outputs/geo_area_data_eval/geo_coefficients_N4000.csv.
Writes geo_coordinate_map_N4000.{pdf,png} and geo_coordinate_vs_area_data_N4000.{pdf,png}
to outputs/geo_design_checks/coordinate_masking_eval/. Not causal.
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
DEST = ROOT / "outputs" / "geo_design_checks" / "coordinate_masking_eval"
COORD_CSV = DEST / "geo_coefficients_N4000.csv"
AREA_CSV = ROOT / "outputs" / "geo_area_data_eval" / "geo_coefficients_N4000.csv"
REGIONS_CSV = ROOT / "outputs" / "geo_area_data_models" / "top50_maps" / "highlighted_regions_lsoas.csv"
NOTE = ("Coordinate-only design (as Seattle/Chicago): the XGBoost model uses the 30 predictors plus centroid "
        "longitude/latitude; masking an LSOA moves only its coordinates to the training-median point, all other "
        "inputs kept. N=4,000, seed 42. ")


def frame(path: Path) -> gpd.GeoDataFrame:
    coef = pd.read_csv(path, dtype={"lsoa21cd": str})
    boundaries, _ = load_boundaries(sorted(coef["lsoa21cd"]))
    return boundaries[["lsoa21cd", "geometry"]].merge(coef, on="lsoa21cd").to_crs("EPSG:27700")


def shade(fig, ax, gdf, col: str, label: str, signed: bool, outline) -> None:
    vmax = shared_limit(gdf[col].to_numpy())
    norm = Normalize(float(gdf[col].min()), vmax) if signed else Normalize(0, vmax)
    gdf.plot(column=col, cmap=SEQUENTIAL, norm=norm, edgecolor="#d9d9d9", linewidth=0.12, ax=ax)
    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.01, extend="max").set_label(label, fontsize=9)
    if outline is not None:
        gpd.GeoSeries([outline], crs=gdf.crs).plot(ax=ax, facecolor="none", edgecolor=INK, linewidth=0.9)
    ax.set_axis_off()


def label_top(ax, gdf, col: str, k: int = 5) -> None:
    for rank, (_, row) in enumerate(gdf.nlargest(k, col).iterrows(), start=1):
        pt = row.geometry.representative_point()
        ax.annotate(str(rank), (pt.x, pt.y), ha="center", va="center", fontsize=8, fontweight="bold", color="white",
                    bbox=dict(boxstyle="circle,pad=0.2", facecolor=INK, edgecolor="none"))


def save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(DEST / f"{stem}.{ext}", dpi=250, facecolor="white")
    plt.close(fig)
    print(f"Saved {DEST / stem}.png")


def main() -> None:
    coord = frame(COORD_CSV)
    outline = hull_outline(coord)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    shade(fig, axes[0], coord, "geo_coefficient", "Geo-SMILE geographic sensitivity coefficient", True, outline)
    axes[0].set_title("(a) Geo-SMILE geographic sensitivity coefficient\n(coordinate-only masking)",
                      fontsize=12, fontweight="bold", color=INK)
    shade(fig, axes[1], coord, "exact_w1", "W1, crimes per month", False, outline)
    axes[1].set_title("(b) Single-LSOA Wasserstein prediction shift\n(coordinates moved to the median point)",
                      fontsize=12, fontweight="bold", color=INK)
    regions = pd.read_csv(REGIONS_CSV, dtype={"lsoa21cd": str})[["lsoa21cd", "region"]]
    coord.merge(regions, on="lsoa21cd").dissolve(by="region").boundary.plot(ax=axes[1], color=INK, linewidth=1.3)
    label_top(axes[0], coord, "geo_coefficient")
    fig.text(0.5, 0.02, NOTE + "Numbers mark the top 5 by signed coefficient. Panel (b): black outlines = "
             "expert-highlighted regions (58 LSOAs). One sequential scale per panel (darker = more sensitive), "
             "capped at the 98th percentile; the sign is not a direction of prediction change. Not causal effects.",
             ha="center", fontsize=7.5, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "geo_coordinate_map_N4000")

    area = frame(AREA_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    for ax, gdf, title in [
        (axes[0], area, "(a) Area-data design (published)\nall 30 inputs of an LSOA masked"),
        (axes[1], coord, "(b) Coordinate-only design (Seattle/Chicago-matched)\nonly longitude/latitude masked"),
    ]:
        shade(fig, ax, gdf, "geo_coefficient", "Signed Geo-SMILE coefficient", True, outline)
        label_top(ax, gdf, "geo_coefficient")
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK)
    rho = area.set_index("lsoa21cd")["geo_coefficient"].corr(
        coord.set_index("lsoa21cd")["geo_coefficient"], method="spearman")
    fig.text(0.5, 0.02, f"Both: XGBoost, N=4,000, seed 42; signed coefficient, each panel on its own sequential "
             f"scale capped at the 98th percentile; numbers mark each design's top 5. Spearman between the two "
             f"rankings = {rho:.2f}. Mean W1 coalition response: 9.67 (a) vs 0.32 (b) crimes per month. "
             "Model sensitivity, not causal effects.", ha="center", fontsize=7.5, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "geo_coordinate_vs_area_data_N4000")


if __name__ == "__main__":
    main()
