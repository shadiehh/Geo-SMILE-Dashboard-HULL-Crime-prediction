"""Top-50 Geo maps for all ten models under the area-data design.

Area-data design (geo_area_data.py): the Geo branch explains the same
30-feature model as the Feature branch; masking an LSOA replaces ALL of its
inputs with the training median, so each coefficient measures how sensitive the
model is to the data it takes from that location. Rankings use the SIGNED
coefficient (geo_score) -- |coefficient| folds the ordering in this design.

Reads the cached per-model pools from hull_geo_area_data_models.py
(outputs/geo_area_data_models/<model>.json, N=4,000, seed 42); nothing is refit.

Writes outputs/geo_area_data_models/top50_maps/:
  geo_top50_<model>.png       per-model map, top 50 coloured by rank (signed coefficient
                              range in footer), rank labels, expert-highlighted regions outlined
  geo_top50_all_models.png    2x5 panel, top 50 coloured by rank (comparable
                              across models whose coefficient scales differ)
  geo_top50_consensus.png     number of models placing each LSOA in its top 50
  geo_top50_ranks.csv         long table: model, rank, lsoa21cd, name, coefficient
  geo_top50_consensus.csv     per-LSOA count of models + mean rank

Not causal.

    python map_area_data_top50.py [--top-k 50]
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.lines import Line2D

from hull_boundaries import load_boundaries
from hull_expert_topk import geo_expert_binary_frame
from map_geo_importance_ranks import (
    INK,
    INK_MUTED,
    MUTED_FILL,
    _draw_outline,
    _label_ranks,
    _prepare_plot_gdf,
    _save,
    rank_by,
)
from model_factory import MODEL_NAMES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SRC = OUT / "geo_area_data_models"
DEST = SRC / "top50_maps"
REGIONS_PATH = DEST / "highlighted_regions_lsoas.csv"  # from select_highlighted_region_lsoas.py
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"
RANK_CMAP = plt.cm.colors.LinearSegmentedColormap.from_list("rank", ["#8A0033", "#FF0051", "#FFD1DC"])
DESIGN_NOTE = (
    "Area-data design: masking an LSOA replaces ALL 30 model inputs with the training median. "
    "N=4,000 perturbations, seed 42. Ranked by signed Geo-SMILE coefficient."
)
COUNT_CMAP = plt.cm.colors.LinearSegmentedColormap.from_list("count", ["#FFD1DC", "#FF0051", "#8A0033"])


def _footer(fig, note: str, y: float = 0.02) -> None:
    text = f"{note} Test period only. ONS LSOA 2021 BGC. Ties broken by lsoa21cd ascending. Not causal."
    fig.text(0.5, y, textwrap.fill(text, 150), ha="center", va="bottom", fontsize=6.5, color=INK_MUTED)


def load_models() -> dict[str, dict]:
    out = {}
    for name in MODEL_NAMES:
        path = SRC / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"[{name}] missing {path} -- run hull_geo_area_data_models.py")
    return out


def ranked_coefficients(res: dict) -> pd.DataFrame:
    coef = pd.DataFrame(res["coefficients"])
    coef["lsoa21cd"] = coef["lsoa21cd"].astype(str)
    return rank_by(coef, "geo_coefficient", "rank")


SHOW_REGIONS = True  # --no-region-outlines turns the expert-region outlines off


def _expert_outline(ax, plot: gpd.GeoDataFrame, expert_codes: set[str]) -> None:
    """Outline the expert regions as dissolved areas, not LSOA by LSOA."""
    if not SHOW_REGIONS:
        return
    ref = plot[plot["lsoa21cd"].isin(expert_codes)]
    if not ref.empty:
        merged = gpd.GeoSeries([ref.geometry.buffer(25).union_all().buffer(-25)], crs=ref.crs)  # closes slivers left by simplify
        merged.boundary.plot(ax=ax, color=INK, linewidth=1.6, zorder=4.5)


def _legend(ax, top_k: int) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=MUTED_FILL, label=f"Outside top {top_k}")]
    if SHOW_REGIONS:
        handles.insert(0, Line2D([], [], color=INK, linewidth=1.3, label="Expert-highlighted regions"))
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=7,
        frameon=False,
    )


def model_map(plot, ranked, res, expert_codes, top_k, path) -> None:
    gdf = plot.merge(ranked, on="lsoa21cd", how="inner")
    top = gdf[gdf["rank"] <= top_k]
    norm = BoundaryNorm(np.arange(0.5, top_k + 1.5, 1), RANK_CMAP.N)
    fig, ax = plt.subplots(figsize=(8.5, 9.0))
    gdf.plot(ax=ax, color=MUTED_FILL, edgecolor="white", linewidth=0.12, zorder=1)
    top.plot(column="rank", cmap=RANK_CMAP, norm=norm, edgecolor="white", linewidth=0.35, ax=ax, zorder=3)
    _expert_outline(ax, gdf, expert_codes)
    _draw_outline(ax, gdf)
    _label_ranks(ax, top, "rank", fontsize=5.5)
    overlap20 = int(ranked.head(20)["lsoa21cd"].isin(expert_codes).sum())
    overlap_k = int(top["lsoa21cd"].isin(expert_codes).sum())
    n_neg = int((top["geo_coefficient"] < 0).sum())
    ax.set_title(
        f"{res['model']}: area-data sensitivity, top {top_k} LSOAs\n"
        f"predictive R2={res['predictive_r2']:.3f} | in expert regions: {overlap20} of top 20"
        + (f", {overlap_k} of top {top_k}" if top_k != 20 else ""),
        fontsize=11,
        fontweight="bold",
        color=INK,
        pad=8,
    )
    ax.set_axis_off()
    _legend(ax, top_k)
    sm = plt.cm.ScalarMappable(cmap=RANK_CMAP, norm=Normalize(1, top_k))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.68, pad=0.015)
    cbar.set_label(f"Rank within top {top_k} (1 = most sensitive)", fontsize=9, color=INK_MUTED)
    cbar.set_ticks([1] + list(range(5, top_k + 1, 5)))  # whole ranks only
    cbar.ax.invert_yaxis()
    note = DESIGN_NOTE + (
        f" Signed coefficient range in top {top_k}: "
        f"{top['geo_coefficient'].min():.2f} to {top['geo_coefficient'].max():.2f}."
    )
    if n_neg:
        note += f" {n_neg} of the top {top_k} have negative (below-average) coefficients."
    _footer(fig, note)
    _save(fig, path)


def panel_map(plot, ranked_by_model, results, expert_codes, top_k, path) -> None:
    names = list(ranked_by_model)
    ncols = 5
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    bounds = np.arange(0.5, top_k + 1.5, 1)
    norm = BoundaryNorm(bounds, RANK_CMAP.N)
    for ax, name in zip(axes, names):
        ranked = ranked_by_model[name]
        gdf = plot.merge(ranked, on="lsoa21cd", how="inner")
        top = gdf[gdf["rank"] <= top_k]
        gdf.plot(ax=ax, color=MUTED_FILL, edgecolor="white", linewidth=0.1, zorder=1)
        top.plot(column="rank", cmap=RANK_CMAP, norm=norm, edgecolor="white", linewidth=0.25, ax=ax, zorder=3)
        _expert_outline(ax, gdf, expert_codes)
        _draw_outline(ax, gdf)
        overlap = int(top["lsoa21cd"].isin(expert_codes).sum())
        ax.set_title(
            f"{name}  (R2={results[name]['predictive_r2']:.3f})\nin expert regions: {overlap}/{top_k}",
            fontsize=10,
            fontweight="bold",
            color=INK,
        )
        ax.set_axis_off()
    for ax in axes[len(names):]:
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=RANK_CMAP, norm=Normalize(1, top_k))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.6, pad=0.01, aspect=30)
    cbar.set_label(f"Rank within model top {top_k} (1 = most sensitive)", fontsize=9, color=INK_MUTED)
    cbar.ax.invert_yaxis()
    fig.suptitle(
        f"Hull - top {top_k} LSOAs by area-data sensitivity, all models", fontsize=14, fontweight="bold", color=INK
    )
    outline_note = f" Black outlines = expert-highlighted regions ({len(expert_codes)} LSOAs)." if SHOW_REGIONS else ""
    _footer(fig, DESIGN_NOTE + outline_note, y=0.01)
    _save(fig, path)


def consensus_map(plot, consensus, expert_codes, n_models, top_k, path) -> None:
    gdf = plot.merge(consensus, on="lsoa21cd", how="left").fillna({"n_models": 0})
    hit = gdf[gdf["n_models"] > 0]
    bounds = np.arange(0.5, n_models + 1.5, 1)
    norm = BoundaryNorm(bounds, COUNT_CMAP.N)
    fig, ax = plt.subplots(figsize=(8.5, 9.0))
    gdf.plot(ax=ax, color=MUTED_FILL, edgecolor="white", linewidth=0.12, zorder=1)
    hit.plot(column="n_models", cmap=COUNT_CMAP, norm=norm, edgecolor="white", linewidth=0.35, ax=ax, zorder=3)
    _expert_outline(ax, gdf, expert_codes)
    _draw_outline(ax, gdf)
    hit = hit.assign(n_label=hit["n_models"].astype(int))
    _label_ranks(ax, hit, "n_label", fontsize=5.5)
    n_all = int((gdf["n_models"] == n_models).sum())
    ax.set_title(
        f"Model agreement: top {top_k} area-data sensitivity\n"
        f"{n_all} LSOAs in the top {top_k} of all {n_models} models",
        fontsize=11,
        fontweight="bold",
        color=INK,
        pad=8,
    )
    ax.set_axis_off()
    _legend(ax, top_k)
    sm = plt.cm.ScalarMappable(cmap=COUNT_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.68, pad=0.015, ticks=range(1, n_models + 1))
    cbar.set_label(f"Number of models with LSOA in top {top_k}", fontsize=9, color=INK_MUTED)
    _footer(fig, DESIGN_NOTE + " Labels = model count.")
    _save(fig, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--no-region-outlines", action="store_true", help="omit the expert-region outlines")
    args = p.parse_args()
    global SHOW_REGIONS
    SHOW_REGIONS = not args.no_region_outlines
    top_k = args.top_k

    results = load_models()
    if not results:
        raise SystemExit("No cached area-data model results found.")
    panel = pd.read_csv(PANEL_PATH, usecols=["lsoa21cd", "lsoa21nm"]).drop_duplicates("lsoa21cd")
    name_lookup = dict(zip(panel["lsoa21cd"].astype(str), panel["lsoa21nm"]))
    codes = sorted(name_lookup)
    boundaries, _ = load_boundaries(codes)
    plot = _prepare_plot_gdf(boundaries[["lsoa21cd", "geometry"]].drop_duplicates("lsoa21cd"))
    expert = geo_expert_binary_frame(codes)
    top20_codes = set(expert.loc[expert["expert_top10_reference"] == 1, "lsoa21cd"])
    # outlines and title counts use the expert-highlighted regions when they exist
    expert_codes = set(pd.read_csv(REGIONS_PATH)["lsoa21cd"].astype(str)) if REGIONS_PATH.exists() else top20_codes

    DEST.mkdir(parents=True, exist_ok=True)
    ranked_by_model, rows = {}, []
    for name, res in results.items():
        ranked = ranked_coefficients(res)
        ranked_by_model[name] = ranked
        top = ranked.head(top_k)
        rows.append(
            top.assign(
                model=name,
                lsoa21nm=top["lsoa21cd"].map(name_lookup),
                expert_reference=top["lsoa21cd"].isin(top20_codes).astype(int),
                expert_region=top["lsoa21cd"].isin(expert_codes).astype(int),
            )[["model", "rank", "lsoa21cd", "lsoa21nm", "geo_coefficient", "expert_reference", "expert_region"]]
        )
        model_map(plot, ranked, res, expert_codes, top_k, DEST / f"geo_top{top_k}_{name}.png")

    long = pd.concat(rows, ignore_index=True)
    long.to_csv(DEST / f"geo_top{top_k}_ranks.csv", index=False)
    consensus = (
        long.groupby("lsoa21cd")
        .agg(n_models=("model", "size"), mean_rank_when_top=("rank", "mean"))
        .reset_index()
        .assign(
            lsoa21nm=lambda d: d["lsoa21cd"].map(name_lookup),
            expert_reference=lambda d: d["lsoa21cd"].isin(top20_codes).astype(int),
            expert_region=lambda d: d["lsoa21cd"].isin(expert_codes).astype(int),
        )
        .sort_values(["n_models", "mean_rank_when_top", "lsoa21cd"], ascending=[False, True, True])
    )
    consensus.to_csv(DEST / f"geo_top{top_k}_consensus.csv", index=False)

    panel_map(plot, ranked_by_model, results, expert_codes, top_k, DEST / f"geo_top{top_k}_all_models.png")
    consensus_map(plot, consensus, expert_codes, len(results), top_k, DEST / f"geo_top{top_k}_consensus.png")

    summary = long.groupby("model")["expert_region"].sum().rename(f"in_expert_regions_of_top{top_k}")
    print(summary.to_string())
    print(f"LSOAs in top {top_k} of all {len(results)} models: {int((consensus['n_models'] == len(results)).sum())}")


if __name__ == "__main__":
    main()
