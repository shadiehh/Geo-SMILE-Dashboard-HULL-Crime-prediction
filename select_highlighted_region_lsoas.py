"""LSOAs inside the expert-highlighted Hull regions, and their overlap with each
model's area-data top 50.

The regions were hand-drawn on Google Maps screenshots and traced here as
lon/lat polygons (georeferenced on street landmarks: The Deep, MKM Stadium,
Hedon Road roundabouts / Marfleet Lane, the "Hull" and "Bransholme" labels).
Tracing error is roughly 100-300 m, so LSOAs with small overlaps are borderline.

Selection rule: an LSOA is in a region if >= MIN_FRAC of its area, or
>= MIN_KM2 of it, lies inside the drawn region (regions in STRICT_MIN_FRAC need a
majority of the LSOA inside instead). The same set drives the list,
the check map and the per-model overlap table.

Reads outputs/geo_area_data_models/top50_maps/geo_top50_{ranks,consensus}.csv
(from map_area_data_top50.py). Writes to the same folder:
  highlighted_regions_lsoas.csv
  highlighted_regions_overlap_by_model.csv (top 50), ..._top42.csv
  highlighted_regions_check.png   (street basemap if tiles are reachable)
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from shapely.geometry import Polygon
from shapely.ops import unary_union

from hull_boundaries import load_boundaries
from hull_expert_topk import geo_expert_binary_frame

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "outputs" / "geo_area_data_models" / "top50_maps"
PANEL_PATH = ROOT / "outputs" / "hull_lsoa_month_panel_with_covariates.csv"
MIN_FRAC = 0.20
MIN_KM2 = 0.10
# Per-region override (user revision 4): the northern region keeps only LSOAs lying mostly
# inside the drawn boundary -- majority of area, no absolute-area exception.
STRICT_MIN_FRAC = {"Bransholme-Sutton-Stoneferry": 0.50}
MODELS = ("xgboost", "lightgbm", "catboost", "svr", "gp", "mlp", "dl", "ann", "cnn", "transformer")

# Each region is one or more traced shapes (lists of lon/lat vertices), unioned.
REGIONS = {
    "Sculcoates-City Centre-Dairycoates": [[
        (-0.36001, 53.76742), (-0.34956, 53.76707), (-0.34894, 53.76352), (-0.34371, 53.76387),
        (-0.34125, 53.75322), (-0.34525, 53.75322), (-0.34648, 53.74754), (-0.33418, 53.74772),
        (-0.32865, 53.74683), (-0.32988, 53.7408), (-0.33418, 53.74009), (-0.34464, 53.73867),
        (-0.35509, 53.73689), (-0.35755, 53.73547), (-0.37477, 53.72908), (-0.37846, 53.73299),
        (-0.37416, 53.73796), (-0.36063, 53.74364), (-0.36309, 53.75074), (-0.36001, 53.75606),
        (-0.35755, 53.75997), (-0.36001, 53.76281),
    ]],
    # user revision 3: replaces the earlier Bransholme region, its north-east exclusion and the
    # Beverley Road-Oak Road strip. Two drawings merged into one shape: a loop from Raich Carter
    # Way round Bransholme, down the River Hull to Clough Road; and a block between Stoneferry
    # Road, Sutton Road and East Park.
    "Bransholme-Sutton-Stoneferry": [
        [
            (-0.36512, 53.79377), (-0.35764, 53.79382), (-0.3479, 53.79262), (-0.33977, 53.79444),
            (-0.33165, 53.79646), (-0.32515, 53.7968), (-0.32191, 53.7955), (-0.31541, 53.79117),
            (-0.3081, 53.7854), (-0.30688, 53.78131), (-0.31054, 53.77963), (-0.31541, 53.77771),
            (-0.32191, 53.77386), (-0.32678, 53.77049), (-0.33003, 53.76833), (-0.33409, 53.76833),
            (-0.33815, 53.76929), (-0.34302, 53.77146), (-0.3479, 53.77434), (-0.34912, 53.77819),
            (-0.34928, 53.78131), (-0.35196, 53.78165), (-0.35846, 53.78204), (-0.3617, 53.7854),
            (-0.36333, 53.79021),
        ],
        [
            (-0.31652, 53.77605), (-0.31079, 53.77432), (-0.30265, 53.77389), (-0.30265, 53.76741),
            (-0.31235, 53.7671), (-0.31631, 53.76339), (-0.32934, 53.76339), (-0.32934, 53.76957),
            (-0.32726, 53.76944),
        ],
    ],
    "Southcoates-Marfleet": [[
        (-0.31018, 53.7599), (-0.30468, 53.761), (-0.28081, 53.76119), (-0.26551, 53.7599),
        (-0.25449, 53.75698), (-0.25419, 53.75368), (-0.25572, 53.75021), (-0.25939, 53.75002),
        (-0.2753, 53.75295), (-0.29305, 53.75551), (-0.30835, 53.75551), (-0.31018, 53.75771),
    ]],
}
TOP_KS = (50, 42)  # model top-k cut-offs for the overlap tables
COLORS = {
    "Sculcoates-City Centre-Dairycoates": "#FF0051",
    "Bransholme-Sutton-Stoneferry": "#1E88E5",
    "Southcoates-Marfleet": "#0D7377",
}


def select(boundaries: gpd.GeoDataFrame) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    regions = gpd.GeoDataFrame(
        {"region": list(REGIONS)}, geometry=[unary_union([Polygon(r) for r in shapes]) for shapes in REGIONS.values()], crs=4326
    ).to_crs(27700)
    g = boundaries[["lsoa21cd", "lsoa21nm", "geometry"]].to_crs(27700)
    g["lsoa_km2"] = g.area / 1e6
    x = gpd.overlay(g, regions, how="intersection")
    x["inside_km2"] = x.area / 1e6
    x["overlap_fraction"] = x["inside_km2"] / x["lsoa_km2"]
    # an LSOA touching two regions goes to the one it overlaps most
    x = x.sort_values("inside_km2", ascending=False).drop_duplicates("lsoa21cd")
    x["selected"] = (x["overlap_fraction"] >= MIN_FRAC) | (x["inside_km2"] >= MIN_KM2)
    for region, frac in STRICT_MIN_FRAC.items():
        in_region = x["region"] == region
        x.loc[in_region, "selected"] = x.loc[in_region, "overlap_fraction"] >= frac
    return pd.DataFrame(x.drop(columns="geometry")), regions


def check_map(boundaries, table, regions, path) -> None:
    g = boundaries[["lsoa21cd", "geometry"]].to_crs(3857).merge(table, on="lsoa21cd", how="left")
    fig, ax = plt.subplots(figsize=(11, 11))
    g.boundary.plot(ax=ax, color="#555", linewidth=0.4, zorder=2)
    for region, color in COLORS.items():
        sel = g[(g["region"] == region) & (g["selected"] == True)]  # noqa: E712 -- NaN-safe
        sel.plot(ax=ax, color=color, alpha=0.45, edgecolor="#333", linewidth=0.5, zorder=3)
    part = g[g["selected"] == False]  # noqa: E712
    part.plot(ax=ax, facecolor="none", edgecolor="#999", hatch="///", linewidth=0.3, zorder=3)
    regions.to_crs(3857).boundary.plot(ax=ax, color="black", linewidth=1.6, linestyle="--", zorder=4)
    for _, row in g[g["selected"] == True].iterrows():  # noqa: E712
        p = row.geometry.representative_point()
        ax.text(p.x, p.y, row["lsoa21cd"][-5:], fontsize=5.5, ha="center", va="center", zorder=5)
    try:
        import contextily as cx

        cx.add_basemap(ax, source=cx.providers.Esri.WorldStreetMap, zoom=13, alpha=0.8)
    except Exception as exc:  # offline: boundaries alone still show the selection
        print(f"basemap skipped: {exc}")
    handles = [Patch(color=c, alpha=0.45, label=f"{r} (selected)") for r, c in COLORS.items()]
    handles += [
        Patch(facecolor="none", edgecolor="#999", hatch="///", label="Touches region, below rule"),
        plt.Line2D([], [], color="black", linestyle="--", label="Traced highlighted region"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"LSOAs in highlighted regions (n={int(table['selected'].sum())}); rule: "
        f">= {MIN_FRAC:.0%} of LSOA area or >= {MIN_KM2} km2 inside"
        + "".join(f"; {r}: >= {v:.0%} inside" for r, v in STRICT_MIN_FRAC.items())
        + ". Labels = last 5 of LSOA code.",
        fontsize=10,
    )
    ax.set_axis_off()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def overlap_table(sel: pd.DataFrame, ranks: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per model: how many selected LSOAs fall in that model's top k (k <= 50)."""
    rows = []
    for m in MODELS:
        top = ranks[(ranks["model"] == m) & (ranks["rank"] <= k)]
        rank = dict(zip(top["lsoa21cd"], top["rank"]))
        hit = sel[sel["lsoa21cd"].isin(rank)]
        row = {"model": m}
        for region in REGIONS:
            row[f"{region} (of {int((sel['region'] == region).sum())})"] = int((hit["region"] == region).sum())
        row[f"total (of {len(sel)})"] = len(hit)
        row[f"share_of_model_top{k}_pct"] = round(100 * len(hit) / k, 1)
        row["in_model_top20"] = int(sum(r <= 20 for r in map(rank.get, hit["lsoa21cd"])))
        row["mean_rank_of_hits"] = round(float(pd.Series([rank[c] for c in hit["lsoa21cd"]]).mean()), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    codes = sorted(pd.read_csv(PANEL_PATH, usecols=["lsoa21cd"])["lsoa21cd"].astype(str).unique())
    boundaries, _ = load_boundaries(codes)
    table, regions = select(boundaries)
    check_map(boundaries, table, regions, DEST / "highlighted_regions_check.png")

    ranks = pd.read_csv(DEST / "geo_top50_ranks.csv")
    consensus = pd.read_csv(DEST / "geo_top50_consensus.csv")[["lsoa21cd", "n_models"]]
    sel = table[table["selected"]].merge(consensus, on="lsoa21cd", how="left").fillna({"n_models": 0})
    sel["expert_top20"] = geo_expert_binary_frame(sel["lsoa21cd"].tolist())["expert_top10_reference"].values
    sel = sel.rename(columns={"n_models": "models_with_lsoa_in_top50"}).astype({"models_with_lsoa_in_top50": int})
    sel = sel.sort_values(["region", "overlap_fraction"], ascending=[True, False])
    cols = ["region", "lsoa21cd", "lsoa21nm", "overlap_fraction", "inside_km2", "lsoa_km2",
            "models_with_lsoa_in_top50", "expert_top20"]
    sel[cols].round(3).to_csv(DEST / "highlighted_regions_lsoas.csv", index=False)

    print(sel.groupby("region").size().to_string())
    for k in TOP_KS:
        overlap = overlap_table(sel, ranks, k)
        suffix = "" if k == 50 else f"_top{k}"
        overlap.to_csv(DEST / f"highlighted_regions_overlap_by_model{suffix}.csv", index=False)
        print(f"\n--- overlap with each model's top {k} ---")
        print(overlap.to_string(index=False))


if __name__ == "__main__":
    main()
