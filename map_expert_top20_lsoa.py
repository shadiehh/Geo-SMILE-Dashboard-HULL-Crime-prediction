"""Expert top-20 LSOA maps for Hull GeoSMILE evaluation v3.

Uses official ONS LSOA 2021 geometry via hull_boundaries. Join key: lsoa21cd.
Highlights exactly the expert Geo top-20 set; compares to GeoSMILE top-20.
"""

from __future__ import annotations

import gc
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from hull_boundaries import hull_outline, load_boundaries
from hull_expert_alignment import DEFAULT_ANNOTATIONS, load_annotations, resolve_reference_rating
from hull_expert_topk import (
    EXPERT_DISPLAY_NOTE,
    GEO_TOP_K,
    select_expert_map_top20,
    select_expert_top_geo,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
COEF_PATH = OUT / "geo_coefficients.csv"
ANN_PATH = DEFAULT_ANNOTATIONS

BLUE, PINK = "#1E88E5", "#FF0051"
INK, INK_MUTED = "#12171C", "#57646D"
MUTED_FILL = "#E8ECF0"
OVERLAP = "#0D7377"
SEQ_EXPERT = plt.cm.colors.LinearSegmentedColormap.from_list(
    "expert_score", ["#FFFFFF", BLUE, "#0D47A1"]
)
FIG_SIZE = (8.5, 9.0)
MAP_DPI = 220


def load_expert_geo_frame() -> pd.DataFrame:
    _, geo_ann = load_annotations(ANN_PATH)
    geo_ann = geo_ann.copy()
    geo_ann["lsoa21cd"] = geo_ann["lsoa21cd"].astype(str)
    geo_ann["final_expert_rating"] = resolve_reference_rating(
        geo_ann["expert_relevance_1to5"], geo_ann.get("human_override_1to5")
    )
    geo_ann["expert_relevance_score"] = pd.to_numeric(
        geo_ann["expert_relevance_score"], errors="coerce"
    )
    if "persistent_crime_proxy" in geo_ann.columns:
        geo_ann["persistent_crime_proxy"] = pd.to_numeric(
            geo_ann["persistent_crime_proxy"], errors="coerce"
        )
    return geo_ann


def load_geosmile_top20() -> set[str]:
    coef = pd.read_csv(COEF_PATH)
    coef["lsoa21cd"] = coef["lsoa21cd"].astype(str)
    order = coef.sort_values(
        ["geo_abs_coefficient", "lsoa21cd"], ascending=[False, True], kind="mergesort"
    )
    return set(order.head(GEO_TOP_K)["lsoa21cd"].astype(str))


def _prepare_plot_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    plot = gdf.to_crs("EPSG:27700").copy()
    plot = plot[plot.geometry.notna() & ~plot.geometry.is_empty]
    plot["geometry"] = plot.geometry.buffer(0).simplify(5.0, preserve_topology=True)
    return plot


def _draw_outline(ax, plot_gdf: gpd.GeoDataFrame) -> None:
    gpd.GeoSeries([hull_outline(plot_gdf)], crs=plot_gdf.crs).plot(
        ax=ax, color="none", edgecolor=INK, linewidth=1.0, zorder=4
    )


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=MAP_DPI, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    if path.suffix.lower() == ".png":
        pdf = path.with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.15, facecolor="white")
        print(f"Saved: {pdf}")
    plt.close(fig)
    plt.close("all")
    gc.collect()
    print(f"Saved: {path}")


def _map_score_column(gdf: gpd.GeoDataFrame) -> str:
    if "persistent_crime_proxy" in gdf.columns and gdf["persistent_crime_proxy"].notna().any():
        return "persistent_crime_proxy"
    return "expert_relevance_score"


def plot_overview(gdf: gpd.GeoDataFrame, top20: pd.DataFrame, path: Path) -> None:
    plot = _prepare_plot_gdf(gdf)
    top_ids = set(top20["lsoa21cd"].astype(str))
    selected = plot["lsoa21cd"].astype(str).isin(top_ids)
    score_col = _map_score_column(plot)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plot.plot(ax=ax, color=MUTED_FILL, edgecolor="white", linewidth=0.12, zorder=1)
    vmax = max(float(plot.loc[selected, score_col].max()), 1e-9)
    norm = Normalize(vmin=0.0, vmax=vmax)
    plot.loc[selected].plot(
        column=score_col,
        cmap=SEQ_EXPERT,
        norm=norm,
        edgecolor=INK,
        linewidth=1.1,
        ax=ax,
        zorder=3,
    )
    _draw_outline(ax, plot)
    for _, row in plot.loc[selected].iterrows():
        pt = row.geometry.representative_point()
        ax.text(
            pt.x,
            pt.y,
            f"{int(row['expert_rank'])}",
            ha="center",
            va="center",
            fontsize=5.0,
            fontweight="bold",
            color=INK,
            zorder=5,
        )
    ax.set_axis_off()
    ax.set_title(
        f"Top {GEO_TOP_K} Hull LSOAs by Prediction-Oriented Expert Reference\n"
        "Fixed expert top-20 set; independent of GeoSMILE attribution scores",
        fontsize=11,
        fontweight="bold",
        color=INK,
        pad=8,
    )
    sm = plt.cm.ScalarMappable(cmap=SEQ_EXPERT, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.68, pad=0.015)
    cbar.set_label(f"{score_col} (top {GEO_TOP_K} only)", fontsize=9, color=INK_MUTED)
    fig.text(
        0.5,
        0.02,
        EXPERT_DISPLAY_NOTE + " ONS LSOA 2021 BGC. Not GeoSMILE-selected. Not causal.",
        ha="center",
        fontsize=6.5,
        color=INK_MUTED,
        wrap=True,
    )
    _save(fig, path)


def plot_comparison(gdf: gpd.GeoDataFrame, path: Path) -> dict:
    plot = _prepare_plot_gdf(gdf)
    exp = plot["expert_map_top20"] == 1
    geo = plot["geosmile_top20"] == 1
    both = exp & geo
    exp_only = exp & ~geo
    geo_only = geo & ~exp
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plot.plot(ax=ax, color=MUTED_FILL, edgecolor="white", linewidth=0.12, zorder=1)
    if geo_only.any():
        plot.loc[geo_only].plot(ax=ax, color=PINK, edgecolor="white", linewidth=0.35, zorder=2)
    if exp_only.any():
        plot.loc[exp_only].plot(ax=ax, color=BLUE, edgecolor="white", linewidth=0.35, zorder=2)
    if both.any():
        plot.loc[both].plot(ax=ax, color=OVERLAP, edgecolor="white", linewidth=0.45, zorder=3)
    _draw_outline(ax, plot)
    n_overlap = int(both.sum())
    union = int((exp | geo).sum())
    jaccard = n_overlap / union if union else 0.0
    ax.set_axis_off()
    ax.set_title(
        f"Expert top-{GEO_TOP_K} vs GeoSMILE top-{GEO_TOP_K} LSOAs\n"
        f"Overlap={n_overlap}; Jaccard={jaccard:.3f}",
        fontsize=12,
        fontweight="bold",
        color=INK,
        pad=8,
    )
    legend = [
        Patch(facecolor=OVERLAP, edgecolor="white", label=f"Both (n={n_overlap})"),
        Patch(facecolor=PINK, edgecolor="white", label=f"GeoSMILE only (n={int(geo_only.sum())})"),
        Patch(facecolor=BLUE, edgecolor="white", label=f"Expert only (n={int(exp_only.sum())})"),
        Patch(facecolor=MUTED_FILL, edgecolor="white", label="Neither"),
    ]
    ax.legend(handles=legend, loc="lower left", frameon=True, fontsize=8)
    fig.text(
        0.5,
        0.02,
        f"GeoSMILE top {GEO_TOP_K} by |geo coefficient|. " + EXPERT_DISPLAY_NOTE,
        ha="center",
        fontsize=6.5,
        color=INK_MUTED,
    )
    _save(fig, path)
    return {
        "overlap": n_overlap,
        "jaccard": jaccard,
        "expert_only": sorted(plot.loc[exp_only, "lsoa21cd"].astype(str)),
        "geosmile_only": sorted(plot.loc[geo_only, "lsoa21cd"].astype(str)),
        "both": sorted(plot.loc[both, "lsoa21cd"].astype(str)),
    }


def write_interactive(gdf: gpd.GeoDataFrame, path: Path, *, comparison: bool = False) -> None:
    gdf4326 = gdf.to_crs("EPSG:4326")
    center = [float(gdf4326.geometry.centroid.y.mean()), float(gdf4326.geometry.centroid.x.mean())]
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    def style_fn(feature):
        props = feature["properties"]
        if comparison:
            e = int(props.get("expert_map_top20") or 0)
            g = int(props.get("geosmile_top20") or 0)
            if e and g:
                color = OVERLAP
            elif e:
                color = BLUE
            elif g:
                color = PINK
            else:
                color = MUTED_FILL
            weight = 1.5 if (e or g) else 0.3
            return {"fillColor": color, "color": "#ffffff", "weight": weight, "fillOpacity": 0.75}
        top = int(props.get("expert_map_top20") or 0)
        if top:
            return {"fillColor": BLUE, "color": INK, "weight": 1.5, "fillOpacity": 0.75}
        return {"fillColor": MUTED_FILL, "color": "#ffffff", "weight": 0.3, "fillOpacity": 0.45}

    fields = [
        "lsoa21cd",
        "lsoa21nm",
        "expert_rank",
        "final_expert_rating",
        "expert_relevance_score",
        "persistent_crime_proxy",
        "confidence",
        "expected_crime_context",
        "crime_domain_dependence",
        "prediction_selection_basis",
        "rationale",
        "expert_map_top20",
        "geosmile_top20",
        "geo_abs_coefficient",
    ]
    fields = [f for f in fields if f in gdf4326.columns]
    folium.GeoJson(
        gdf4326.__geo_interface__,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=fields),
        popup=folium.GeoJsonPopup(fields=fields, aliases=fields),
    ).add_to(m)
    m.get_root().html.add_child(
        folium.Element(f"<div style='padding:6px;font-size:12px'>{EXPERT_DISPLAY_NOTE}</div>")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(path))
    print(f"Saved: {path}")


def run() -> dict[str, Path]:
    if not ANN_PATH.exists():
        raise SystemExit(f"Missing annotations: {ANN_PATH}")
    if not COEF_PATH.exists():
        raise SystemExit(f"Missing geo coefficients: {COEF_PATH}")

    geo_ann = load_expert_geo_frame()
    ranked = select_expert_top_geo(geo_ann, k=GEO_TOP_K)
    top20 = select_expert_map_top20(geo_ann)
    if len(top20) != GEO_TOP_K:
        raise SystemExit(f"Expected exactly {GEO_TOP_K} map LSOAs, got {len(top20)}")

    coef = pd.read_csv(COEF_PATH)
    coef["lsoa21cd"] = coef["lsoa21cd"].astype(str)
    geosmile_top20 = load_geosmile_top20()

    combined = ranked.merge(
        coef[["lsoa21cd", "geo_coefficient", "geo_abs_coefficient"]],
        on="lsoa21cd",
        how="inner",
        validate="one_to_one",
    )
    combined["expert_map_top20"] = combined["lsoa21cd"].isin(top20["lsoa21cd"]).astype(int)
    combined["geosmile_top20"] = combined["lsoa21cd"].isin(geosmile_top20).astype(int)
    order = combined.sort_values(
        ["geo_abs_coefficient", "lsoa21cd"], ascending=[False, True], kind="mergesort"
    )
    combined.loc[order.index, "geosmile_rank"] = np.arange(1, len(order) + 1)
    combined["geosmile_rank"] = combined["geosmile_rank"].astype(int)

    codes = combined["lsoa21cd"].tolist()
    boundaries, meta = load_boundaries(codes)
    if meta.get("matched_lsoas") != 168:
        searched = [
            ROOT / "data" / "hull_lsoa_2021_official.geojson",
            ROOT / "data" / "hull_lsoa_boundaries.geojson",
        ]
        raise SystemExit(
            "Geometry incomplete for Hull LSOAs. "
            f"matched={meta.get('matched_lsoas')} source={meta.get('source')}. "
            f"Searched: {[str(p) for p in searched]}"
        )
    missing_top = set(top20["lsoa21cd"]) - set(boundaries["lsoa21cd"].astype(str))
    if missing_top:
        raise SystemExit(f"Top-{GEO_TOP_K} expert LSOAs missing geometry: {sorted(missing_top)}")

    value_cols = [
        c
        for c in [
            "lsoa21cd",
            "lsoa21nm",
            "final_expert_rating",
            "expert_relevance_score",
            "persistent_crime_proxy",
            "expert_rank",
            "prediction_expert_rank",
            "expert_top10_reference",
            "expert_top20_reference",
            "expert_map_top20",
            "city_centre_priority",
            "confidence",
            "expected_crime_context",
            "crime_domain_dependence",
            "prediction_selection_basis",
            "rationale",
            "geo_coefficient",
            "geo_abs_coefficient",
            "geosmile_rank",
            "geosmile_top20",
        ]
        if c in combined.columns
    ]
    gdf = boundaries[["lsoa21cd", "geometry"]].merge(
        combined[value_cols], on="lsoa21cd", how="inner", validate="one_to_one"
    )

    overview = OUT / "geosmile_hull_expert_top20_lsoa_overview.png"
    comparison = OUT / "geosmile_hull_expert_vs_geosmile_top20.png"
    interactive = OUT / "geosmile_hull_expert_top20_lsoa_interactive.html"
    comparison_html = OUT / "geosmile_hull_expert_vs_geosmile_top20.html"
    data_csv = OUT / "geosmile_hull_expert_top20_lsoa_map_data.csv"

    plot_overview(gdf, top20, overview)
    stats = plot_comparison(gdf, comparison)
    write_interactive(gdf, interactive, comparison=False)
    write_interactive(gdf, comparison_html, comparison=True)

    map_data = combined.sort_values("expert_rank").copy()
    map_data["in_expert_map_top20"] = map_data["expert_map_top20"]
    map_data["overlap_class"] = np.where(
        (map_data["expert_map_top20"] == 1) & (map_data["geosmile_top20"] == 1),
        "both",
        np.where(
            map_data["expert_map_top20"] == 1,
            "expert_only",
            np.where(map_data["geosmile_top20"] == 1, "geosmile_only", "neither"),
        ),
    )
    map_data.to_csv(data_csv, index=False)
    print(f"Saved: {data_csv}")
    print(
        f"Map overlap: {stats['overlap']}/{GEO_TOP_K} Jaccard={stats['jaccard']:.3f} "
        f"both={stats['both']}"
    )

    return {
        "overview_png": overview,
        "overview_pdf": overview.with_suffix(".pdf"),
        "interactive_html": interactive,
        "comparison_png": comparison,
        "comparison_html": comparison_html,
        "map_data_csv": data_csv,
        "overlap": stats["overlap"],
        "jaccard": stats["jaccard"],
    }


if __name__ == "__main__":
    run()
