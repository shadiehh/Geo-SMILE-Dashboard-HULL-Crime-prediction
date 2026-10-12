"""Two compact maps for the City Planner card (Chapter 6).

  planner_geo_map        signed GeoSensitivity beta, red sequential scale (the only red map),
                         top-ranked LSOA outlined and named.
  planner_built_map      share of each LSOA's local Feature importance from the 12
                         built-environment players, teal sequential scale; top areas named.

Compact layout: no title or footer (the card carries the text), colourbar inside the frame,
short name labels next to their areas, tight crop.

Reads  outputs/geo_area_data_eval/geo_coefficients_N4000.csv,
       outputs/stakeholder_narratives/planner_built_share.csv (plot_planner_built_share.py),
       data/hull_lsoa_place_names.csv.
Writes outputs/stakeholder_narratives/planner_{geo,built}_map.{pdf,png}. Not causal.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

from geosmile_hull_twobranch import INK, SEQUENTIAL  # noqa: E402
from hull_boundaries import hull_outline, load_boundaries  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "outputs" / "stakeholder_narratives"
TEAL = LinearSegmentedColormap.from_list("planner_teal", ["#F4F8F9", "#7FB7C2", "#1F4E5F"])
FIG_WIDTH = 4.6  # inches; height follows the shared map extent
PAD = 0.03       # fraction of the city extent left blank around the outline
N_BUILT_LABELS = 5
CENTRE = "E01033104"  # Kingston upon Hull 029E, rank 1 GeoSensitivity


def base(values: pd.DataFrame) -> gpd.GeoDataFrame:
    boundaries, _ = load_boundaries(sorted(values["lsoa21cd"]))
    gdf = boundaries[["lsoa21cd", "geometry"]].merge(values, on="lsoa21cd").to_crs("EPSG:27700")
    assert len(gdf) == 168
    places = pd.read_csv(ROOT / "data" / "hull_lsoa_place_names.csv", dtype={"lsoa21cd": str})
    return gdf.merge(places[["lsoa21cd", "locality_label"]], on="lsoa21cd", how="left")


def frame(gdf: gpd.GeoDataFrame, column: str, cmap, norm, label: str, fmt=None, extend="neither"):
    # Fixed axes over the whole figure and a shared extent, so both maps print at the same
    # scale side by side; labels are kept inside the frame instead of growing the crop.
    x0, y0, x1, y1 = gdf.total_bounds
    px, py = PAD * (x1 - x0), PAD * (y1 - y0)
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * (y1 - y0 + 2 * py) / (x1 - x0 + 2 * px)))
    ax = fig.add_axes([0, 0, 1, 1])
    gdf.plot(column=column, cmap=cmap, norm=norm, edgecolor="#d9d9d9", linewidth=0.1, ax=ax)
    outline = hull_outline(gdf)
    if outline is not None:
        gpd.GeoSeries([outline], crs=gdf.crs).plot(ax=ax, facecolor="none", edgecolor=INK, linewidth=0.7)
    ax.set_xlim(x0 - px, x1 + px)
    ax.set_ylim(y0 - py, y1 + py)
    ax.set_axis_off()
    # Colourbar inside the empty south-east corner of the frame.
    cax = ax.inset_axes([0.70, 0.07, 0.27, 0.035])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    bar = fig.colorbar(sm, cax=cax, orientation="horizontal", format=fmt, extend=extend)
    bar.ax.tick_params(labelsize=6, length=2, pad=1)
    bar.set_label(label, fontsize=6.5, labelpad=2)
    return fig, ax


def outline(ax, gdf: gpd.GeoDataFrame, code: str):
    row = gdf.loc[gdf["lsoa21cd"] == code].iloc[0]
    gpd.GeoSeries([row.geometry], crs=gdf.crs).plot(ax=ax, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=4)
    return row


def label(ax, geom, text: str, dx: float, dy: float) -> None:
    """Short name label offset by (dx, dy) points from the area, with a tiny leader."""
    p = geom.representative_point()
    ax.plot(p.x, p.y, "o", ms=2.2, color=INK, zorder=5)
    ax.annotate(text, xy=(p.x, p.y), xytext=(dx, dy), textcoords="offset points",
                fontsize=6.5, color=INK, ha="left" if dx >= 0 else "right", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.5, shrinkA=0, shrinkB=1.5))


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(DEST / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {DEST / name}.pdf / .png")


def geo_map() -> None:
    coef = pd.read_csv(ROOT / "outputs" / "geo_area_data_eval" / "geo_coefficients_N4000.csv",
                       dtype={"lsoa21cd": str})[["lsoa21cd", "geo_coefficient"]]  # signed beta only
    coef["rank"] = coef["geo_coefficient"].rank(ascending=False, method="first").astype(int)
    gdf = base(coef)
    beta = gdf["geo_coefficient"].to_numpy()
    norm = Normalize(float(beta.min()), float(np.percentile(beta, 98)))
    fig, ax = frame(gdf, "geo_coefficient", SEQUENTIAL, norm, r"Signed GeoSensitivity $\beta$", extend="max")
    assert gdf.loc[gdf["rank"] == 1, "lsoa21cd"].item() == CENTRE
    top = outline(ax, gdf, CENTRE)
    label(ax, top.geometry, f"Old Town (029E)\nrank 1 · β = {top['geo_coefficient']:+.1f}", 24, -16)
    save(fig, "planner_geo_map")


# Label offsets (points) for the top built-share areas, keyed by LSOA; chosen to stay
# inside the frame and clear of each other.
BUILT_OFFSETS = {
    "E01012838": (8, 14),     # Longhill 010E
    "E01012795": (-10, -14),  # Avenues 019B
    "E01012849": (-8, -30),   # Bilton Grange 017C
    "E01012856": (-10, -22),  # Sculcoates 024C
    "E01012918": (-10, 14),   # Orchard Park 008G
}


def built_map() -> None:
    share = pd.read_csv(DEST / "planner_built_share.csv", dtype={"lsoa21cd": str})
    gdf = base(share[["lsoa21cd", "built_share", "dominant_built_feature", "rank"]])
    norm = Normalize(0, float(gdf["built_share"].max()))
    fig, ax = frame(gdf, "built_share", TEAL, norm, "Built-environment share",
                    fmt=lambda v, _: f"{100 * v:.0f}%")
    top = gdf.nsmallest(N_BUILT_LABELS, "rank")
    assert set(top["lsoa21cd"]) == set(BUILT_OFFSETS), sorted(top["lsoa21cd"])
    for _, row in top.iterrows():
        dx, dy = BUILT_OFFSETS[row["lsoa21cd"]]
        label(ax, row.geometry, f"{row['locality_label']} {100 * row['built_share']:.0f}%\n"
                                f"{row['dominant_built_feature']}", dx, dy)
    # Same city-centre outline and offset as map 01, linking the two panels.
    centre = outline(ax, gdf, CENTRE)
    label(ax, centre.geometry, f"Old Town {100 * centre['built_share']:.0f}%\n"
                               f"{centre['dominant_built_feature']}", 24, -16)
    save(fig, "planner_built_map")


if __name__ == "__main__":
    geo_map()
    built_map()
