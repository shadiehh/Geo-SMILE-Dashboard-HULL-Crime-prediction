"""Geo accuracy against the second (schematic) expert map, for every Hull Geo design.

The expert map (expert_map_v2.png) is schematic: seven reference regions drawn on an approximate city
outline with no coordinates. Its polygons were traced in image pixels (2000 x 1499 display scale) and
georeferenced to British National Grid with a least-squares affine fit on five control points (the
city's west, east, north and south-west extremes and the River Hull mouth); residuals are reported.
An LSOA belongs to the reference if >= 20% of its area or >= 0.1 km2 lies inside a traced region
(the uniform Chapter-4 rule); 10/50/75% rules are reported as robustness.

Accuracy = share of each design's top 20 (signed coefficient; descending) inside the reference, with
chance, lift, one-sided hypergeometric p and ATT-AUROC. Crime-level ranking as a baseline.
Writes outputs/hull_geo_design_review/08_expert_map_v2/. Not causal.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from geosmile_hull_twobranch import INK, temporal_split
from hull_aligned_design_eval import geo_accuracy
from hull_boundaries import hull_outline, load_boundaries

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "hull_geo_design_review" / "08_expert_map_v2"

# image (x, y) -> BNG (easting, northing)
CONTROL = {
    "west extreme": ((150, 860), (504129, 429396)),
    "east extreme": ((1900, 980), (516045, 430176)),
    "north extreme": ((1060, 88), (510967, 436533)),
    "south-west extreme": ((262, 1245), (504412, 425954)),
    "River Hull mouth": ((1068, 1132), (510212, 428340)),
}
REGIONS = {
    "Orchard Park": [(553, 405), (680, 315), (797, 347), (812, 476), (706, 544), (553, 515)],
    "Bransholme": [(1058, 98), (1200, 92), (1330, 188), (1432, 288), (1356, 517), (1187, 593), (1088, 535), (1044, 325)],
    "Longhill and Bilton Grange": [(1525, 458), (1646, 458), (1724, 540), (1817, 682), (1755, 772), (1603, 755), (1532, 618)],
    "Greatfield": [(1652, 805), (1815, 790), (1862, 899), (1784, 975), (1661, 949)],
    "Preston Road": [(1414, 882), (1574, 861), (1611, 975), (1505, 1034), (1420, 998)],
    "City centre, Old Town and Beverley Road": [(826, 761), (926, 755), (1003, 847), (1056, 962), (1118, 1053), (1133, 1123),
                                                (1030, 1146), (965, 1140), (958, 1060), (928, 990), (857, 915), (797, 838)],
    "Hessle Road corridor": [(258, 1236), (528, 1123), (645, 1028), (778, 1034), (958, 1087), (965, 1158), (698, 1208), (285, 1318)],
}
DESIGNS = {
    "Area-data (published)": OUT / "geo_area_data_eval" / "n_4000" / "geo_coefficients.csv",
    "Coordinates -> median point (aligned)": OUT / "geo_aligned_design" / "xgboost" / "geo_coefficients_N4000.csv",
    "Coordinates -> median point (lon/lat model)": OUT / "geo_design_checks" / "coordinate_masking_eval" / "geo_coefficients_N4000.csv",
    "Random donor, absolute (K=10)": OUT / "hull_geo_design_review" / "06_random_donor" / "geo_coefficients_N4000.csv",
    "Random donor, log scale (K=100)": OUT / "hull_geo_design_review" / "07_random_donor_ksweep" / "geo_coefficients_K100.csv",
}


def affine():
    src = np.array([[x, y, 1.0] for (x, y), _ in CONTROL.values()])
    dst = np.array([d for _, d in CONTROL.values()], float)
    A, *_ = np.linalg.lstsq(src, dst, rcond=None)
    resid = np.hypot(*(src @ A - dst).T)
    return A, dict(zip(CONTROL, resid.round(0)))


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    A, resid = affine()
    to_bng = lambda pts: [tuple(np.array([x, y, 1.0]) @ A) for x, y in pts]
    regions = gpd.GeoDataFrame({"region": list(REGIONS)}, geometry=[Polygon(to_bng(p)) for p in REGIONS.values()], crs=27700)
    regions.to_file(DEST / "expert_regions_v2.geojson", driver="GeoJSON")

    panel = pd.read_csv(OUT / "hull_lsoa_month_panel_with_covariates.csv")
    panel["month"] = pd.to_datetime(panel["month"])
    codes = sorted(panel["lsoa21cd"].astype(str).unique())
    b, _ = load_boundaries(codes)
    g = b[["lsoa21cd", "geometry"]].to_crs(27700)
    g["lsoa_km2"] = g.area / 1e6
    x = gpd.overlay(g, regions, how="intersection")
    x["inside_km2"] = x.area / 1e6
    ov = x.groupby("lsoa21cd").agg(inside_km2=("inside_km2", "sum"), region=("region", "first")).join(g.set_index("lsoa21cd")["lsoa_km2"])
    ov["frac"] = ov["inside_km2"] / ov["lsoa_km2"]
    rules = {">=20% or >=0.1 km2 (primary)": set(ov[(ov.frac >= 0.2) | (ov.inside_km2 >= 0.1)].index),
             ">=10%": set(ov[ov.frac >= 0.1].index), ">=50%": set(ov[ov.frac >= 0.5].index), ">=75%": set(ov[ov.frac >= 0.75].index)}
    ref = rules[">=20% or >=0.1 km2 (primary)"]
    ov.assign(in_reference=ov.index.isin(ref)).to_csv(DEST / "lsoa_overlap_v2.csv")

    _, _, test = temporal_split(panel)
    crime = panel.loc[test].groupby("lsoa21cd")["crime_count"].mean()
    designs = {k: pd.read_csv(v, dtype={"lsoa21cd": str})[["lsoa21cd", "geo_coefficient"]] for k, v in DESIGNS.items()}
    designs["Crime level (baseline)"] = crime.rename("geo_coefficient").rename_axis("lsoa21cd").reset_index()
    rows, rob = [], []
    for name, coef in designs.items():
        a = geo_accuracy(coef, ref)
        rows.append({"design": name, "top20_inside": a["intersection"], "accuracy": a["accuracy"], "chance": a["chance"],
                     "lift": a["lift"], "p_hypergeom": a["p_hypergeom"], "att_auroc": a["att_auroc"],
                     "top42_inside": geo_accuracy(coef, ref, 42)["intersection"]})
        for rn, s in rules.items():
            r = geo_accuracy(coef, s)
            rob.append({"design": name, "rule": rn, "reference_size": len(s), "top20_inside": r["intersection"], "p": r["p_hypergeom"]})
    table, robt = pd.DataFrame(rows), pd.DataFrame(rob)
    table.to_csv(DEST / "accuracy_v2.csv", index=False)
    robt.to_csv(DEST / "accuracy_v2_rule_robustness.csv", index=False)
    old = set(pd.read_csv(OUT / "geo_area_data_models" / "top50_maps" / "highlighted_regions_lsoas.csv")["lsoa21cd"].astype(str))
    meta = {"control_point_residuals_m": resid, "reference_size": len(ref), "per_region": ov.loc[list(ref)].groupby("region").size().to_dict(),
            "overlap_with_first_expert_map": {"first_map_size": len(old), "shared": len(ref & old)}}
    (DEST / "georeferencing.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")

    # check map
    fig, ax = plt.subplots(figsize=(10, 8))
    g.plot(ax=ax, color="#f2f2f2", edgecolor="#cccccc", linewidth=0.3)
    g[g.lsoa21cd.isin(ref)].plot(ax=ax, color="#F28CB1", edgecolor="#cccccc", linewidth=0.3)
    regions.boundary.plot(ax=ax, color="#B5003C", linewidth=1.6)
    gpd.GeoSeries([hull_outline(g)], crs=27700).plot(ax=ax, facecolor="none", edgecolor=INK, linewidth=0.9)
    for _, r in regions.iterrows():
        c = r.geometry.representative_point()
        ax.annotate(r["region"], (c.x, c.y), ha="center", fontsize=7, color="#7a0026", fontweight="bold")
    ax.set_axis_off()
    ax.set_title(f"Second expert map, georeferenced: {len(ref)} reference LSOAs (pink)", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.5, 0.02, "Red outlines: traced regions after affine fit (control-point residuals "
             + ", ".join(f"{k} {v:.0f} m" for k, v in resid.items()) + "). Pink: LSOAs with >= 20% or >= 0.1 km2 inside.",
             ha="center", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(DEST / "expert_map_v2_check.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(json.dumps(meta, indent=1, default=float))
    print(table.round(4).to_string(index=False))
    print(robt.pivot(index="design", columns="rule", values="top20_inside").to_string())


if __name__ == "__main__":
    main()
