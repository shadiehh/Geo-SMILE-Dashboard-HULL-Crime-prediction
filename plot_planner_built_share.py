"""City Planner evidence: where the model's local explanation leans on the built environment.

For every LSOA, the share of its total local Feature importance (sum of |local coefficient|
over all 30 players) that comes from the 12 built-environment players, plus the built
feature that contributes most. The top areas are annotated by place name.

Reads  outputs/feature_local_coefficients.csv, data/hull_lsoa_place_names.csv.
Writes outputs/stakeholder_narratives/planner_built_share.csv
       outputs/stakeholder_narratives/planner_built_share_map{,_plain}.{pdf,png}
Local Feature attributions of the fitted model; not causal effects.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from geosmile_hull_twobranch import INK, SEQUENTIAL  # noqa: E402
from hull_boundaries import hull_outline, load_boundaries  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "outputs" / "stakeholder_narratives"
N_LABELS = 5

BUILT = {
    "lsoa_area_km2": "area",
    "nightlife_count": "nightlife", "nightlife_per_km2": "nightlife",
    "retail_count": "retail", "retail_per_km2": "retail",
    "transit_stop_count": "transit stops", "transit_stop_per_km2": "transit stops",
    "road_length_km": "roads", "road_length_km_per_km2": "roads",
    "intersection_count": "intersections", "intersection_per_km2": "intersections",
    "green_space_pct": "green space",
}


def built_share() -> pd.DataFrame:
    coef = pd.read_csv(ROOT / "outputs" / "feature_local_coefficients.csv").set_index("lsoa21cd")
    mag = coef.drop(columns="lsoa21nm").abs()
    assert mag.shape[1] == 30 and set(BUILT) <= set(mag.columns)
    built = mag[list(BUILT)]
    # Dominant built feature, with count and density variants of one feature pooled.
    by_feature = built.T.groupby(BUILT).sum().T
    places = pd.read_csv(ROOT / "data" / "hull_lsoa_place_names.csv").set_index("lsoa21cd")
    out = pd.DataFrame({
        "lsoa21nm": coef["lsoa21nm"],
        "locality": places["locality_label"].reindex(coef.index),
        "total_local_importance": mag.sum(axis=1),
        "built_importance": built.sum(axis=1),
        "dominant_built_feature": by_feature.idxmax(axis=1),
        "dominant_built_share": by_feature.max(axis=1) / mag.sum(axis=1),
    })
    out["built_share"] = out["built_importance"] / out["total_local_importance"]
    out = out.sort_values("built_share", ascending=False)
    out["rank"] = range(1, len(out) + 1)
    return out


def draw(gdf: gpd.GeoDataFrame, annotate: bool) -> None:
    norm = Normalize(0, float(gdf["built_share"].max()))
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    gdf.plot(column="built_share", cmap=SEQUENTIAL, norm=norm, edgecolor="#d9d9d9", linewidth=0.12, ax=ax)
    outline = hull_outline(gdf)
    if outline is not None:
        gpd.GeoSeries([outline], crs=gdf.crs).plot(ax=ax, facecolor="none", edgecolor=INK, linewidth=0.9)
    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=norm)
    sm.set_array([])
    bar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.01, format=lambda v, _: f"{100 * v:.0f}%")
    bar.set_label("Share of local importance from built-environment features", fontsize=9)
    ax.set_title("Hull — where the model leans on the built environment", fontsize=12,
                 fontweight="bold", color=INK)
    ax.set_axis_off()
    if annotate:
        top = gdf.nsmallest(N_LABELS, "rank")
        slots = [(0.02, 0.97), (0.02, 0.62), (0.02, 0.14), (0.38, 0.14), (0.70, 0.97), (0.70, 0.14)]
        ax.apply_aspect()
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        # Greedy nearest free slot (in axes fraction) so leader lines stay short and uncrossed.
        free, placed = list(slots), []
        for _, row in top.iterrows():
            p = row.geometry.representative_point()
            fx, fy = (p.x - x0) / (x1 - x0), (p.y - y0) / (y1 - y0)
            best = min(free, key=lambda s: (s[0] + 0.12 - fx) ** 2 + (s[1] - 0.04 - fy) ** 2)
            free.remove(best)
            placed.append((row, best))
        for row, xy in placed:
            p = row.geometry.representative_point()
            ax.plot(p.x, p.y, "o", ms=3.5, color=INK, zorder=5)
            ax.annotate(f"{row['locality']} ({row['lsoa21nm'].split()[-1]})\n"
                        f"{100 * row['built_share']:.0f}% built · {row['dominant_built_feature']}",
                        xy=(p.x, p.y), xycoords="data", xytext=xy, textcoords="axes fraction",
                        fontsize=8, color=INK, va="top", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cfd6dc", alpha=0.94),
                        arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.8, shrinkB=3))
    fig.text(0.5, 0.02,
             "Share of each LSOA's total local Feature importance (|local coefficient| summed over all 30 players) "
             "that comes from the 12 built-environment players: area, nightlife, retail, transit stops, roads, "
             "intersections, green space. Labels name the top areas and their leading built feature. "
             "Model attributions, not causal effects or planning targets.",
             ha="center", fontsize=7, wrap=True, color="#57646D")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    stem = DEST / f"planner_built_share_map{'' if annotate else '_plain'}"
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=250, facecolor="white")
    plt.close(fig)
    print(f"Saved {stem}.pdf / .png")


def main() -> None:
    table = built_share()
    table.to_csv(DEST / "planner_built_share.csv", index_label="lsoa21cd")
    boundaries, _ = load_boundaries(sorted(table.index))
    gdf = boundaries[["lsoa21cd", "geometry"]].merge(table, left_on="lsoa21cd", right_index=True).to_crs("EPSG:27700")
    assert len(gdf) == 168
    print(table.head(10)[["lsoa21nm", "locality", "built_share", "dominant_built_feature"]].round(3).to_string())
    print(f"\ncity-wide built share {table['built_importance'].sum() / table['total_local_importance'].sum():.3f}, "
          f"median {table['built_share'].median():.3f}")
    print("dominant built feature among top 20:", table.head(20)["dominant_built_feature"].value_counts().to_dict())
    try:
        from esda.moran import Moran
        from libpysal.weights import Queen
        w = Queen.from_dataframe(gdf, use_index=False)
        w.transform = "r"
        m = Moran(gdf["built_share"].to_numpy(), w, permutations=999)
        print(f"Moran's I {m.I:.3f}, p = {m.p_sim:.3f}")
    except ImportError:
        print("esda/libpysal not installed; Moran's I skipped")
    draw(gdf, annotate=True)
    draw(gdf, annotate=False)


if __name__ == "__main__":
    main()
