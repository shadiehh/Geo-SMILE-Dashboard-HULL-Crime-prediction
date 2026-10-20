"""Geo accuracy against the expert map v3 (expert map v2 with hand-drawn extensions).

The user extended three regions by hand on a scaled crop of expert map v2 (expert_map_v3.png,
1377 x 871): Orchard Park, the City centre / Old Town / Beverley Road corridor, and one region joining
Preston Road and Greatfield down to the shore. The other regions are unchanged. The crop maps onto the
v2 image linearly (scale 0.636, offset +107 px in x; fitted on four polygon vertices), so the traced
outlines reuse the v2 georeferencing (geo_expert_map_v2_accuracy.affine). Each extended region is the
union of its v2 polygon and the drawn outline.

Reference LSOAs: >= 20% of area or >= 0.1 km2 inside (primary) and >= 50% (strict). Accuracy for
top 20 / 30 / 42 of every design and of crime level. Writes
outputs/hull_geo_design_review/09_expert_map_v3/. Not causal.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

import geo_expert_map_v2_accuracy as v2
from geosmile_hull_twobranch import INK, temporal_split
from hull_aligned_design_eval import geo_accuracy
from hull_boundaries import hull_outline, load_boundaries

DEST = v2.OUT / "hull_geo_design_review" / "09_expert_map_v3"
SCALE, DX, DY = 0.636, 107.0, 0.0      # v3 crop pixel = SCALE * v2 pixel + (DX, DY)
DRAWN = {   # hand-drawn outlines, v3 crop pixels
    "Orchard Park": [(540, 198), (600, 212), (640, 240), (663, 270), (660, 300), (645, 320), (632, 345), (628, 368),
                     (600, 378), (560, 380), (500, 377), (450, 370), (410, 355), (390, 330), (395, 300), (408, 278),
                     (440, 260), (470, 248), (500, 235), (520, 215)],
    "City centre, Old Town and Beverley Road": [(665, 476), (700, 478), (725, 486), (760, 510), (785, 540), (810, 570),
                     (830, 590), (840, 615), (848, 640), (845, 680), (835, 705), (820, 715), (795, 728), (765, 732),
                     (735, 728), (715, 712), (700, 698), (680, 690), (655, 682), (648, 672), (640, 640), (632, 625),
                     (618, 600), (600, 575), (597, 540), (602, 505), (625, 490)],
    "Preston Road and Greatfield": [(858, 615), (900, 600), (940, 598), (980, 585), (1020, 565), (1060, 545), (1100, 520),
                     (1140, 505), (1190, 495), (1230, 485), (1260, 482), (1295, 490), (1330, 515), (1350, 540),
                     (1357, 580), (1350, 610), (1330, 622), (1300, 640), (1285, 660), (1280, 685), (1265, 700),
                     (1240, 707), (1180, 714), (1100, 717), (1000, 716), (920, 712), (870, 705), (855, 690),
                     (850, 660), (852, 635)],
}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    A, _ = v2.affine()
    to_bng = lambda pts: Polygon([tuple(np.array([x, y, 1.0]) @ A) for x, y in pts])
    crop_to_v2 = lambda pts: [((x - DX) / SCALE, (y - DY) / SCALE) for x, y in pts]
    base = {k: to_bng(p) for k, p in v2.REGIONS.items()}
    drawn = {k: to_bng(crop_to_v2(p)) for k, p in DRAWN.items()}
    regions = {
        "Bransholme": base["Bransholme"],
        "Longhill and Bilton Grange": base["Longhill and Bilton Grange"],
        "Hessle Road corridor": base["Hessle Road corridor"],
        "Orchard Park": unary_union([base["Orchard Park"], drawn["Orchard Park"]]),
        "City centre, Old Town and Beverley Road": unary_union([base["City centre, Old Town and Beverley Road"],
                                                                drawn["City centre, Old Town and Beverley Road"]]),
        "Preston Road and Greatfield": unary_union([base["Preston Road"], base["Greatfield"], drawn["Preston Road and Greatfield"]]),
    }
    reg = gpd.GeoDataFrame({"region": list(regions)}, geometry=list(regions.values()), crs=27700)
    reg.to_file(DEST / "expert_regions_v3.geojson", driver="GeoJSON")

    panel = pd.read_csv(v2.OUT / "hull_lsoa_month_panel_with_covariates.csv")
    panel["month"] = pd.to_datetime(panel["month"])
    b, _ = load_boundaries(sorted(panel["lsoa21cd"].astype(str).unique()))
    g = b[["lsoa21cd", "geometry"]].to_crs(27700)
    g["lsoa_km2"] = g.area / 1e6
    x = gpd.overlay(g, gpd.GeoDataFrame(geometry=[reg.union_all()], crs=27700), how="intersection")
    x["inside_km2"] = x.area / 1e6
    ov = x.groupby("lsoa21cd")["inside_km2"].sum().to_frame().join(g.set_index("lsoa21cd")["lsoa_km2"])
    ov["frac"] = ov["inside_km2"] / ov["lsoa_km2"]
    rules = {"primary (>=20% or >=0.1 km2)": set(ov[(ov.frac >= 0.2) | (ov.inside_km2 >= 0.1)].index),
             "strict (>=50%)": set(ov[ov.frac >= 0.5].index)}
    ov.assign(**{f"in_{k.split()[0]}": ov.index.isin(s) for k, s in rules.items()}).to_csv(DEST / "lsoa_overlap_v3.csv")

    _, _, test = temporal_split(panel)
    crime = panel.loc[test].groupby("lsoa21cd")["crime_count"].mean()
    designs = {k: pd.read_csv(p, dtype={"lsoa21cd": str})[["lsoa21cd", "geo_coefficient"]] for k, p in v2.DESIGNS.items()}
    designs["Crime level (baseline)"] = crime.rename("geo_coefficient").rename_axis("lsoa21cd").reset_index()
    rows = []
    for rn, ref in rules.items():
        for name, coef in designs.items():
            r = {"rule": rn, "reference": len(ref), "design": name}
            for k in (20, 30, 42):
                a = geo_accuracy(coef, ref, k)
                r[f"top{k}"], r[f"lift{k}"], r[f"p{k}"] = a["intersection"], a["lift"], a["p_hypergeom"]
            r["auroc"] = geo_accuracy(coef, ref)["att_auroc"]
            rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(DEST / "accuracy_v3.csv", index=False)

    ref = rules["primary (>=20% or >=0.1 km2)"]
    fig, ax = plt.subplots(figsize=(10, 8))
    g.plot(ax=ax, color="#f2f2f2", edgecolor="#cccccc", linewidth=0.3)
    g[g.lsoa21cd.isin(ref)].plot(ax=ax, color="#F28CB1", edgecolor="#cccccc", linewidth=0.3)
    reg.boundary.plot(ax=ax, color="#B5003C", linewidth=1.5)
    gpd.GeoSeries([hull_outline(g)], crs=27700).plot(ax=ax, facecolor="none", edgecolor=INK, linewidth=0.9)
    for _, r in reg.iterrows():
        c = r.geometry.representative_point()
        ax.annotate(r["region"], (c.x, c.y), ha="center", fontsize=7, color="#7a0026", fontweight="bold")
    ax.set_axis_off()
    ax.set_title(f"Expert map v3 (hand-extended), georeferenced: {len(ref)} reference LSOAs", fontsize=12,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(DEST / "expert_map_v3_check.png", dpi=200, facecolor="white")
    plt.close(fig)
    for rn, s in table.groupby("rule", sort=False):
        n = s["reference"].iloc[0]
        print(f"\n== {rn}: {n} LSOAs ({n / 168:.0%}); chance top20/30/42 = {20 * n / 168:.1f}/{30 * n / 168:.1f}/{42 * n / 168:.1f}")
        for _, r in s.iterrows():
            print(f"  {r.design:45s} top20 {r.top20:2d} (p={r.p20:.3g})  top30 {r.top30:2d} (p={r.p30:.3g})  "
                  f"top42 {r.top42:2d} (p={r.p42:.3g})  AUROC {r.auroc:.3f}")


if __name__ == "__main__":
    main()
