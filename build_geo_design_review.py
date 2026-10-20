"""Collect every Hull Geo-design check (results + maps) into outputs/hull_geo_design_review/.

Copies the outputs of geo_crime_history_check.py, geo_coordinate_design_check.py,
geo_coordinate_design_eval.py (+ plot_geo_coordinate_design_map.py), geo_coordinate_reference_check.py
and hull_aligned_design_eval.py, and draws the maps that did not exist yet:
  * reference-point check: signed Geo map for each reference point;
  * aligned design: Geo map (signed coefficient + single-LSOA shift with expert regions), Geo map
    beside the published area-data map, and the six leading local Feature drivers.
Originals stay where their scripts write them. Not causal.

    python build_geo_design_review.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

from geosmile_hull_twobranch import DIVERGING, FEATURE_LABELS, INK, SEQUENTIAL, shared_limit
from hull_boundaries import hull_outline, load_boundaries

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "hull_geo_design_review"
COPIES = {
    "01_crime_history_check": OUT / "geo_design_checks" / "crime_history_check",
    "02_coordinate_lonlat_check": OUT / "geo_design_checks" / "coordinate_masking_check",
    "03_coordinate_lonlat_eval": OUT / "geo_design_checks" / "coordinate_masking_eval",
    "04_reference_point_check": OUT / "geo_design_checks" / "coordinate_reference_check",
    "05_aligned_design": OUT / "geo_aligned_design",
}
REGIONS_CSV = OUT / "geo_area_data_models" / "top50_maps" / "highlighted_regions_lsoas.csv"
AREA_CSV = OUT / "geo_area_data_eval" / "geo_coefficients_N4000.csv"


def geoframe(values: pd.DataFrame) -> gpd.GeoDataFrame:
    b, _ = load_boundaries(sorted(values["lsoa21cd"].astype(str)))
    return b[["lsoa21cd", "geometry"]].merge(values, on="lsoa21cd").to_crs("EPSG:27700")


def shade(fig, ax, gdf, col, label, outline, *, start_at_min=True, diverging=False, top=0):
    vals = gdf[col].to_numpy()
    if diverging:
        lim = shared_limit(vals)
        norm, cmap, ext = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim), DIVERGING, "both"
    else:
        norm, cmap, ext = Normalize(float(vals.min()) if start_at_min else 0.0, shared_limit(vals)), SEQUENTIAL, "max"
    gdf.plot(column=col, cmap=cmap, norm=norm, edgecolor="#d9d9d9", linewidth=0.12, ax=ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.01, extend=ext).set_label(label, fontsize=8)
    if outline is not None:
        gpd.GeoSeries([outline], crs=gdf.crs).plot(ax=ax, facecolor="none", edgecolor=INK, linewidth=0.8)
    for rank, (_, row) in enumerate(gdf.nlargest(top, col).iterrows(), start=1):
        pt = row.geometry.representative_point()
        ax.annotate(str(rank), (pt.x, pt.y), ha="center", va="center", fontsize=7, fontweight="bold", color="white",
                    bbox=dict(boxstyle="circle,pad=0.2", facecolor=INK, edgecolor="none"))
    ax.set_axis_off()


def _copy_unless_locked(src, dst):
    """Copy a file, leaving the existing copy if it is open in another program."""
    try:
        return shutil.copy2(src, dst)
    except PermissionError:
        print("skipped (open elsewhere):", dst)
        return dst


def save(fig, folder: Path, stem: str, note: str) -> None:
    fig.text(0.5, 0.015, note, ha="center", fontsize=7.5, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        try:
            fig.savefig(folder / f"{stem}.{ext}", dpi=220, facecolor="white")
        except PermissionError:
            print("skipped (open elsewhere):", folder / f"{stem}.{ext}")
    plt.close(fig)
    print("map:", folder / f"{stem}.png")


def reference_maps(folder: Path) -> None:
    coef = pd.read_csv(folder / "coefficients_by_reference.csv", index_col=0)
    summ = pd.read_csv(folder / "reference_point_summary.csv").set_index("reference")
    gdf = geoframe(coef.rename_axis("lsoa21cd").reset_index())
    outline = hull_outline(gdf)
    refs = list(coef.columns)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    pts = gpd.GeoSeries(gpd.points_from_xy(summ["lon"], summ["lat"]), crs=4326).to_crs(27700)
    for ax, ref in zip(axes.ravel(), refs):
        shade(fig, ax, gdf, ref, "signed coefficient", outline, top=5)
        p = pts.iloc[list(summ.index).index(ref)]
        if gdf.total_bounds[0] - 8000 < p.x < gdf.total_bounds[2] + 8000:
            ax.plot(p.x, p.y, marker="*", color="#1E88E5", markersize=16, markeredgecolor="white", zorder=6)
        s = summ.loc[ref]
        ax.set_title(f"{ref.replace('_', ' ')}\nρ with median point {s['rho_with_median_point']:.2f}; "
                     f"{int(s['top20_in_expert_regions'])}/20 in regions", fontsize=10, color=INK)
    axes.ravel()[-1].set_axis_off()
    save(fig, folder, "reference_point_maps",
         "Coordinate-only Geo branch (longitude/latitude model, N=4,000, seed 42): masked LSOAs' coordinates moved to "
         "the reference point (blue star where inside the frame). Signed coefficient, own scale per panel, top 5 numbered. "
         "Not causal effects.")


def aligned_maps(folder: Path) -> None:
    geo = pd.read_csv(folder / "xgboost" / "geo_coefficients_N4000.csv", dtype={"lsoa21cd": str})
    gdf = geoframe(geo)
    outline = hull_outline(gdf)
    note = ("Aligned design: XGBoost on 30 predictors + BNG easting/northing; Geo masking moves only an LSOA's "
            "coordinates to the training-median point. N=4,000, seed 42. ")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    shade(fig, axes[0], gdf, "geo_coefficient", "Geo-SMILE geographic sensitivity coefficient", outline, top=5)
    axes[0].set_title("(a) Geo-SMILE geographic sensitivity coefficient", fontsize=12, fontweight="bold", color=INK)
    shade(fig, axes[1], gdf, "exact_w1", "W1, crimes per month", outline, start_at_min=False)
    axes[1].set_title("(b) Single-LSOA Wasserstein prediction shift", fontsize=12, fontweight="bold", color=INK)
    regions = pd.read_csv(REGIONS_CSV, dtype={"lsoa21cd": str})[["lsoa21cd", "region"]]
    gdf.merge(regions, on="lsoa21cd").dissolve(by="region").boundary.plot(ax=axes[1], color=INK, linewidth=1.3)
    save(fig, folder, "geo_map_N4000", note + "Top 5 numbered in (a); (b) black outlines = expert regions (58 LSOAs). "
         "Sequential scales capped at the 98th percentile; sign is not a direction of change. Not causal effects.")

    area = geoframe(pd.read_csv(AREA_CSV, dtype={"lsoa21cd": str})[["lsoa21cd", "geo_coefficient"]])
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    for ax, g, title in [(axes[0], area, "(a) Published area-data design\nall 30 inputs of an LSOA masked"),
                         (axes[1], gdf, "(b) Aligned design (as Seattle/Chicago)\nonly coordinates masked")]:
        shade(fig, ax, g, "geo_coefficient", "Signed Geo-SMILE coefficient", outline, top=5)
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK)
    rho = area.set_index("lsoa21cd")["geo_coefficient"].corr(gdf.set_index("lsoa21cd")["geo_coefficient"], method="spearman")
    save(fig, folder, "geo_aligned_vs_area_data_N4000",
         f"Both XGBoost, N=4,000, seed 42; each panel on its own scale, top 5 numbered. Spearman between the rankings "
         f"= {rho:.2f}. Mean coalition W1: 9.67 (a) vs 0.30 (b) crimes per month. Not causal effects.")

    local = pd.read_csv(folder / "xgboost" / "feature_local_coefficients_N4000.csv")
    local = local.drop(columns=[c for c in local.columns if c.startswith("Unnamed")]).groupby("lsoa21cd").mean().reset_index()
    top6 = pd.read_csv(folder / "xgboost" / "feature_global_importance_N4000.csv")["feature"].head(6).tolist()
    lg = geoframe(local)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, f in zip(axes.ravel(), top6):
        shade(fig, ax, lg, f, "local coefficient (signed)", outline, diverging=True)
        ax.set_title(FEATURE_LABELS.get(f, f), fontsize=12, fontweight="bold", color=INK)
    save(fig, folder, "feature_local_top6_N4000", note.replace("Geo masking", "Feature masking keeps true coordinates; Geo masking")
         + "Six leading global Feature drivers; signed local coefficients (original minus perturbed prediction), "
         "diverging scale. Not causal effects.")


def random_donor_maps(folder: Path) -> None:
    rd = pd.read_csv(folder / "geo_coefficients_N4000.csv", dtype={"lsoa21cd": str})
    gdf = geoframe(rd)
    outline = hull_outline(gdf)
    med = geoframe(pd.read_csv(OUT / "geo_aligned_design" / "xgboost" / "geo_coefficients_N4000.csv",
                               dtype={"lsoa21cd": str})[["lsoa21cd", "geo_coefficient"]])
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.2))
    shade(fig, axes[0], med, "geo_coefficient", "signed coefficient", outline, top=5)
    axes[0].set_title("(a) Median-point coordinate masking", fontsize=12, fontweight="bold", color=INK)
    shade(fig, axes[1], gdf, "geo_coefficient", "signed coefficient", outline, top=5)
    axes[1].set_title("(b) Random-donor coordinate masking", fontsize=12, fontweight="bold", color=INK)
    shade(fig, axes[2], gdf, "exact_w1", "W1, crimes per month", outline, start_at_min=False)
    regions = pd.read_csv(REGIONS_CSV, dtype={"lsoa21cd": str})[["lsoa21cd", "region"]]
    gdf.merge(regions, on="lsoa21cd").dissolve(by="region").boundary.plot(ax=axes[2], color=INK, linewidth=1.3)
    axes[2].set_title("(c) Random-donor single-LSOA shift\n(mean over 167 donor locations)", fontsize=12,
                      fontweight="bold", color=INK)
    save(fig, folder, "random_donor_map_N4000",
         "Aligned design (XGBoost on 30 predictors + BNG easting/northing), N=4,000, seed 42. (a) masked coordinates moved to "
         "the training-median point; (b, c) moved to random other Hull LSOAs (10 draws per perturbation). Top 5 numbered; "
         "(c) black outlines = expert regions. Own sequential scale per panel. Not causal effects.")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, src in COPIES.items():
        dst = DEST / name
        shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=_copy_unless_locked)
    reference_maps(DEST / "04_reference_point_check")
    aligned_maps(DEST / "05_aligned_design")
    if (DEST / "06_random_donor" / "geo_coefficients_N4000.csv").exists():
        random_donor_maps(DEST / "06_random_donor")


if __name__ == "__main__":
    main()
