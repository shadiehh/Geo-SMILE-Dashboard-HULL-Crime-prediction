"""Model vs expert top-k LSOA ranking maps (Hull GeoSMILE).

Produces:
  - Geo model top-10 / top-20 by |geo coefficient|
  - Expert top-10 / top-20 by reference rating / expert importance rank
  - Side-by-side comparisons and overlap maps

Expert ranking:
  reference_rating = human_override_1to5 if non-null else expert_relevance_1to5
  Rank descending by reference_rating; ties broken by lsoa21cd ascending (mergesort).
  Final shared maps use the expert top-20 set from map_expert_top20.py.

Geo ranking:
  Rank descending by geo_abs_coefficient; ties broken by lsoa21cd ascending.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from hull_boundaries import hull_outline, load_boundaries
from hull_expert_alignment import resolve_reference_rating

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
MAP_DIR = OUT_DIR / "maps_twobranch" / "ranking"
COEF_PATH = OUT_DIR / "geo_coefficients.csv"
MERGED_EXPERT_PATH = OUT_DIR / "geosmile_hull_expert_alignment_merged_geo.csv"
EXPERT_XLSX = ROOT / "geosmile_hull_single_expert_annotations_v1.xlsx"

BLUE, PINK = "#1E88E5", "#FF0051"
INK, INK_MUTED = "#12171C", "#57646D"
MUTED_FILL = "#E8ECF0"
OVERLAP = "#0D7377"
SEQ_GEO = plt.cm.colors.LinearSegmentedColormap.from_list(
    "geosmile_abs", ["#FFFFFF", PINK, "#8A0033"]
)
SEQ_EXPERT = plt.cm.colors.LinearSegmentedColormap.from_list(
    "expert_rating", ["#FFFFFF", BLUE, "#0D47A1"]
)
FIG_SIZE = (8.5, 9.0)
PAIR_SIZE = (14.0, 8.0)
MAP_DPI = 220

TIE_BREAK_RULE = "Ties broken by lsoa21cd ascending (stable mergesort)."


def load_test_period() -> str:
    for name in ("run_summary_twobranch.json", "run_summary.json"):
        path = OUT_DIR / name
        if path.exists():
            summary = json.loads(path.read_text(encoding="utf-8"))
            period = summary.get("test_period")
            if isinstance(period, (list, tuple)) and len(period) == 2:
                return f"{period[0]} to {period[1]}"
    return "unknown"


def rank_by(frame: pd.DataFrame, value_col: str, rank_col: str) -> pd.DataFrame:
    """Higher value -> rank 1; ties -> lsoa21cd ascending."""
    out = frame.copy()
    order = out.sort_values([value_col, "lsoa21cd"], ascending=[False, True], kind="mergesort")
    out.loc[order.index, rank_col] = np.arange(1, len(order) + 1)
    out[rank_col] = out[rank_col].astype(int)
    return out.sort_values(rank_col, kind="mergesort").reset_index(drop=True)


def load_geo_ranked() -> pd.DataFrame:
    coef = pd.read_csv(COEF_PATH)
    need = {"lsoa21cd", "geo_coefficient", "geo_abs_coefficient"}
    missing = need - set(coef.columns)
    if missing:
        raise SystemExit(f"{COEF_PATH} missing columns: {sorted(missing)}")
    if "lsoa21nm" not in coef.columns:
        coef["lsoa21nm"] = coef["lsoa21cd"]
    return rank_by(coef, "geo_abs_coefficient", "geo_importance_rank")


def load_expert_ranked() -> tuple[pd.DataFrame, str]:
    """Return ranked expert table and a short source note."""
    if MERGED_EXPERT_PATH.exists():
        merged = pd.read_csv(MERGED_EXPERT_PATH)
        # Prefer stored expert_rank from the evaluation merge when present —
        # this is the ranking used in alignment outputs and yields the
        # documented Geo∩Expert top-20 overlap.
        if "expert_rank" in merged.columns:
            cols = ["lsoa21cd", "expert_rank"]
            if "lsoa21nm" in merged.columns:
                cols.insert(1, "lsoa21nm")
            rating_col = next(
                (
                    c
                    for c in (
                        "final_expert_rating",
                        "primary_reference_rating",
                        "reference_rating",
                    )
                    if c in merged.columns
                ),
                None,
            )
            if rating_col:
                cols.append(rating_col)
            out = merged[cols].copy()
            if "lsoa21nm" not in out.columns:
                out["lsoa21nm"] = out["lsoa21cd"]
            if rating_col:
                out = out.rename(columns={rating_col: "reference_rating"})
            else:
                out["reference_rating"] = pd.NA
            out["expert_importance_rank"] = pd.to_numeric(out["expert_rank"], errors="coerce").astype(int)
            out = out.drop(columns=["expert_rank"])
            ranked = out.sort_values("expert_importance_rank", kind="mergesort").reset_index(drop=True)
            return ranked, (
                f"merged CSV ({MERGED_EXPERT_PATH.name}; stored expert_rank)"
            )

        rating_col = next(
            (
                c
                for c in (
                    "final_expert_rating",
                    "primary_reference_rating",
                    "reference_rating",
                )
                if c in merged.columns
            ),
            None,
        )
        if rating_col is None:
            raise SystemExit(f"{MERGED_EXPERT_PATH} lacks reference rating columns")
        cols = ["lsoa21cd", rating_col]
        if "lsoa21nm" in merged.columns:
            cols.insert(1, "lsoa21nm")
        out = merged[cols].copy()
        if "lsoa21nm" not in out.columns:
            out["lsoa21nm"] = out["lsoa21cd"]
        out = out.rename(columns={rating_col: "reference_rating"})
        out["reference_rating"] = pd.to_numeric(out["reference_rating"], errors="coerce").astype(int)
        ranked = rank_by(out, "reference_rating", "expert_importance_rank")
        return ranked, f"merged CSV ({MERGED_EXPERT_PATH.name}; rating={rating_col})"

    if not EXPERT_XLSX.exists():
        raise SystemExit(
            f"Need {MERGED_EXPERT_PATH.name} or {EXPERT_XLSX.name} for expert ratings"
        )
    ann = pd.read_excel(EXPERT_XLSX, sheet_name="LSOA_Annotations")
    need = {"lsoa21cd", "expert_relevance_1to5"}
    missing = need - set(ann.columns)
    if missing:
        raise SystemExit(f"LSOA_Annotations missing columns: {sorted(missing)}")
    ann["reference_rating"] = resolve_reference_rating(
        ann["expert_relevance_1to5"], ann.get("human_override_1to5")
    )
    keep = ["lsoa21cd", "reference_rating"]
    if "lsoa21nm" in ann.columns:
        keep.insert(1, "lsoa21nm")
    ranked = rank_by(ann[keep].copy(), "reference_rating", "expert_importance_rank")
    if "lsoa21nm" not in ranked.columns:
        ranked["lsoa21nm"] = ranked["lsoa21cd"]
    return ranked, f"xlsx sheet LSOA_Annotations ({EXPERT_XLSX.name})"


def _prepare_plot_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    plot = gdf.to_crs("EPSG:27700").copy()
    plot = plot[plot.geometry.notna() & ~plot.geometry.is_empty]
    # Light simplify reduces Agg memory without changing map readability.
    plot["geometry"] = plot.geometry.buffer(0).simplify(5.0, preserve_topology=True)
    return plot


def _draw_outline(ax, plot_gdf: gpd.GeoDataFrame) -> None:
    gpd.GeoSeries([hull_outline(plot_gdf)], crs=plot_gdf.crs).plot(
        ax=ax, color="none", edgecolor=INK, linewidth=1.0, zorder=4
    )


def _label_ranks(ax, subset: gpd.GeoDataFrame, rank_col: str, *, color=INK, fontsize=6.5) -> None:
    for _, row in subset.iterrows():
        pt = row.geometry.representative_point()
        ax.text(
            pt.x,
            pt.y,
            str(int(row[rank_col])),
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=color,
            zorder=5,
        )


def _footer(fig, note: str, *, y: float = 0.015) -> None:
    footer = (
        f"{note} Test: {load_test_period()}. ONS LSOA 2021 BGC. "
        f"{TIE_BREAK_RULE} Not causal."
    )
    fig.text(0.5, y, footer, ha="center", fontsize=6.5, color=INK_MUTED)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=MAP_DPI, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    plt.close("all")
    gc.collect()
    print(f"Saved: {path}")


def _draw_added_boundaries(ax, plot: gpd.GeoDataFrame, added_codes: set[str]) -> None:
    """Stroke added LSOAs so they stay visible even when they are not in the top-k fill."""
    extra = plot[plot["lsoa21cd"].astype(str).isin(added_codes)]
    if extra.empty:
        return
    extra.plot(
        ax=ax,
        facecolor="#FFE082",
        edgecolor="#E65100",
        linewidth=2.0,
        zorder=6,
    )
    for _, row in extra.iterrows():
        pt = row.geometry.representative_point()
        ax.annotate(
            "added",
            (pt.x, pt.y),
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="#BF360C",
            zorder=7,
        )


def plot_topk_value_map(
    gdf: gpd.GeoDataFrame,
    *,
    rank_col: str,
    value_col: str,
    top_n: int,
    title: str,
    cbar_label: str,
    cmap,
    path: Path,
    footer_note: str,
    added_codes: list[str] | set[str] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_colorbar: bool = True,
) -> None:
    added = {str(c) for c in (added_codes or []) if str(c)}
    selected = gdf[rank_col] <= top_n
    plot = _prepare_plot_gdf(gdf)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plot.plot(ax=ax, color=MUTED_FILL, edgecolor="white", linewidth=0.12, zorder=1)
    if value_col == "reference_rating":
        norm = Normalize(vmin=1.0, vmax=5.0)
    elif vmax is not None:
        # Caller-supplied shared scale (e.g. a small-multiple panel comparing
        # several maps) -- overrides the default per-map auto-scaling below,
        # which would otherwise make relative intensity across maps misleading.
        norm = Normalize(vmin=vmin if vmin is not None else 0.0, vmax=vmax)
    else:
        auto_vmax = max(float(plot.loc[selected, value_col].max()), 1e-9)
        norm = Normalize(vmin=0.0, vmax=auto_vmax)
    plot.loc[selected].plot(
        column=value_col,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.35,
        ax=ax,
        zorder=3,
    )
    hull_plot = plot[~plot["lsoa21cd"].astype(str).isin(added)] if added else plot
    _draw_outline(ax, hull_plot if not hull_plot.empty else plot)
    if added:
        _draw_added_boundaries(ax, plot, added)
    _label_ranks(ax, plot.loc[selected], rank_col)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.68, pad=0.015)
        cbar.set_label(cbar_label, fontsize=9, color=INK_MUTED)
    _footer(fig, footer_note)
    _save(fig, path)


def _plot_highlight_panel(
    ax,
    plot: gpd.GeoDataFrame,
    mask: pd.Series,
    *,
    color: str,
    rank_col: str,
    title: str,
) -> None:
    plot.plot(ax=ax, color=MUTED_FILL, edgecolor="white", linewidth=0.12, zorder=1)
    plot.loc[mask].plot(ax=ax, color=color, edgecolor="white", linewidth=0.35, zorder=3)
    _draw_outline(ax, plot)
    label_color = "white" if color == PINK or color == OVERLAP else INK
    _label_ranks(ax, plot.loc[mask], rank_col, color=label_color)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, pad=6)


def plot_side_by_side(gdf: gpd.GeoDataFrame, *, top_n: int, path: Path) -> None:
    plot = _prepare_plot_gdf(gdf)
    geo_mask = plot["geo_importance_rank"] <= top_n
    exp_mask = plot["expert_importance_rank"] <= top_n
    fig, axes = plt.subplots(1, 2, figsize=PAIR_SIZE)
    _plot_highlight_panel(
        axes[0],
        plot,
        geo_mask,
        color=PINK,
        rank_col="geo_importance_rank",
        title=f"GeoSMILE model - top {top_n}\nby |geo coefficient|",
    )
    _plot_highlight_panel(
        axes[1],
        plot,
        exp_mask,
        color=BLUE,
        rank_col="expert_importance_rank",
        title=f"Expert - top {top_n}\nby importance rank",
    )
    n_overlap = int((geo_mask & exp_mask).sum())
    fig.suptitle(
        f"Hull - Model vs expert top-{top_n} LSOAs  (overlap = {n_overlap})",
        fontsize=13,
        fontweight="bold",
        color=INK,
        y=0.98,
    )
    _footer(
        fig,
        f"Left: GeoSMILE |geo|. Right: expert reference. Overlap: {n_overlap}/{top_n}.",
        y=0.02,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _save(fig, path)


def plot_overlap(gdf: gpd.GeoDataFrame, *, top_n: int, path: Path) -> int:
    plot = _prepare_plot_gdf(gdf)
    geo_mask = plot["geo_importance_rank"] <= top_n
    exp_mask = plot["expert_importance_rank"] <= top_n
    both = geo_mask & exp_mask
    geo_only = geo_mask & ~exp_mask
    exp_only = exp_mask & ~geo_mask

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plot.plot(ax=ax, color=MUTED_FILL, edgecolor="white", linewidth=0.12, zorder=1)
    if geo_only.any():
        plot.loc[geo_only].plot(ax=ax, color=PINK, edgecolor="white", linewidth=0.3, zorder=2)
    if exp_only.any():
        plot.loc[exp_only].plot(ax=ax, color=BLUE, edgecolor="white", linewidth=0.3, zorder=2)
    if both.any():
        plot.loc[both].plot(ax=ax, color=OVERLAP, edgecolor="white", linewidth=0.4, zorder=3)
    _draw_outline(ax, plot)

    for mask, kind in ((both, "both"), (geo_only, "geo"), (exp_only, "exp")):
        for _, row in plot.loc[mask].iterrows():
            pt = row.geometry.representative_point()
            if kind == "both":
                label = f"G{int(row['geo_importance_rank'])}/E{int(row['expert_importance_rank'])}"
                color, fs = "white", 5.5
            elif kind == "geo":
                label = f"G{int(row['geo_importance_rank'])}"
                color, fs = INK, 6.0
            else:
                label = f"E{int(row['expert_importance_rank'])}"
                color, fs = INK, 6.0
            ax.text(
                pt.x, pt.y, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=color, zorder=5,
            )

    n_overlap = int(both.sum())
    ax.set_axis_off()
    ax.set_title(
        f"Hull - Top-{top_n} overlap (model intersect expert)\nOverlap = {n_overlap} LSOAs",
        fontsize=12,
        fontweight="bold",
        color=INK,
        pad=8,
    )
    legend = [
        Patch(facecolor=OVERLAP, edgecolor="white", label=f"Both (n={n_overlap})"),
        Patch(facecolor=PINK, edgecolor="white", label=f"Model only (n={int(geo_only.sum())})"),
        Patch(facecolor=BLUE, edgecolor="white", label=f"Expert only (n={int(exp_only.sum())})"),
        Patch(facecolor=MUTED_FILL, edgecolor="white", label="Neither"),
    ]
    ax.legend(handles=legend, loc="lower left", frameon=True, fontsize=8)
    _footer(fig, f"Expert vs GeoSMILE |geo| top-{top_n}. G=model, E=expert.")
    _save(fig, path)
    return n_overlap


def print_table(df: pd.DataFrame, n: int, *, kind: str) -> None:
    if kind == "geo":
        cols = ["geo_importance_rank", "lsoa21cd", "lsoa21nm", "geo_coefficient", "geo_abs_coefficient"]
        print(f"\n=== Model/Geo top {n} by |geo coefficient| ===")
        for row in df.head(n)[cols].itertuples(index=False):
            print(
                f"  {int(row.geo_importance_rank):2d}. {row.lsoa21cd}  {row.lsoa21nm}  "
                f"coef={row.geo_coefficient:+.4f}  |coef|={row.geo_abs_coefficient:.4f}"
            )
    else:
        cols = ["expert_importance_rank", "lsoa21cd", "lsoa21nm", "reference_rating"]
        print(f"\n=== Expert top {n} by reference_rating ===")
        for row in df.head(n)[cols].itertuples(index=False):
            print(
                f"  {int(row.expert_importance_rank):2d}. {row.lsoa21cd}  {row.lsoa21nm}  "
                f"rating={int(row.reference_rating)}"
            )


def main() -> None:
    if not COEF_PATH.exists():
        raise SystemExit(f"Missing geo coefficients: {COEF_PATH}")

    geo = load_geo_ranked()
    expert, expert_source = load_expert_ranked()
    print(f"Expert source: {expert_source}")
    print(f"Tie-break / labelling: {TIE_BREAK_RULE}")

    combined = geo.merge(
        expert[["lsoa21cd", "reference_rating", "expert_importance_rank"]],
        on="lsoa21cd",
        how="inner",
        validate="one_to_one",
    )
    codes = combined["lsoa21cd"].tolist()
    boundaries, meta = load_boundaries(codes)
    print(f"Boundaries: {meta.get('source')} ({meta.get('matched_lsoas')} LSOAs)")

    value_cols = [
        "lsoa21cd",
        "lsoa21nm",
        "geo_coefficient",
        "geo_abs_coefficient",
        "geo_importance_rank",
        "reference_rating",
        "expert_importance_rank",
    ]
    gdf = boundaries[["lsoa21cd", "geometry"]].merge(
        combined[value_cols], on="lsoa21cd", how="inner", validate="one_to_one"
    )

    MAP_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "geo_top10": MAP_DIR / "geo_top10.png",
        "geo_top20": MAP_DIR / "geo_top20.png",
        "expert_top10": MAP_DIR / "expert_top10.png",
        "expert_top20": MAP_DIR / "expert_top20.png",
        "comparison_top10": MAP_DIR / "comparison_top10.png",
        "comparison_top20": MAP_DIR / "comparison_top20.png",
        "overlap_top10": MAP_DIR / "overlap_top10.png",
        "overlap_top20": MAP_DIR / "overlap_top20.png",
    }

    for n, key in ((10, "geo_top10"), (20, "geo_top20")):
        plot_topk_value_map(
            gdf,
            rank_col="geo_importance_rank",
            value_col="geo_abs_coefficient",
            top_n=n,
            title=f"Hull - GeoSMILE top {n} LSOAs\nby |geo coefficient|",
            cbar_label="|Geo SMILE coefficient| (top only)",
            cmap=SEQ_GEO,
            path=paths[key],
            footer_note=f"Model geo ranking, ranks 1-{n}.",
        )

    for n, key in ((10, "expert_top10"), (20, "expert_top20")):
        plot_topk_value_map(
            gdf,
            rank_col="expert_importance_rank",
            value_col="reference_rating",
            top_n=n,
            title=f"Hull - Expert top {n}\nby reference rating",
            cbar_label="Expert reference rating (1-5)",
            cmap=SEQ_EXPERT,
            path=paths[key],
            footer_note=(
                f"Expert ranking, ranks 1-{n}. "
                "reference_rating = override if set else expert_relevance_1to5."
            ),
        )

    plot_side_by_side(gdf, top_n=10, path=paths["comparison_top10"])
    plot_side_by_side(gdf, top_n=20, path=paths["comparison_top20"])
    n10 = plot_overlap(gdf, top_n=10, path=paths["overlap_top10"])
    n20 = plot_overlap(gdf, top_n=20, path=paths["overlap_top20"])

    table_path = MAP_DIR / "geo_expert_importance_ranks.csv"
    combined.sort_values("geo_importance_rank").to_csv(table_path, index=False)
    print(f"Saved: {table_path}")

    geo_sorted = combined.sort_values("geo_importance_rank")
    exp_sorted = combined.sort_values("expert_importance_rank")
    print_table(geo_sorted, 10, kind="geo")
    print_table(geo_sorted, 20, kind="geo")
    print_table(exp_sorted, 10, kind="expert")
    print_table(exp_sorted, 20, kind="expert")

    geo10 = set(geo_sorted.head(10)["lsoa21cd"])
    geo20 = set(geo_sorted.head(20)["lsoa21cd"])
    exp10 = set(exp_sorted.head(10)["lsoa21cd"])
    exp20 = set(exp_sorted.head(20)["lsoa21cd"])
    print("\n=== Overlap ===")
    print(f"  Top-10 overlap: {n10} / 10  -> {sorted(geo10 & exp10)}")
    print(f"  Top-20 overlap: {n20} / 20  -> {sorted(geo20 & exp20)}")

    print("\nOutput image paths:")
    for key, path in paths.items():
        print(f"  {key}: {path.resolve()}")


if __name__ == "__main__":
    main()
