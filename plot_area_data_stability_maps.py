"""Chapter-4 Geo stability maps for the area-data design (XGBoost, N=4,000, seed 42).

Reruns the cardinal-neighbour stability experiment of hull_geo_area_data_eval.stability,
saving the full coefficient vector of each run, and draws:
  stability_four_directions_area_data.{png,pdf}  2x2 top-20 maps, one per addition
  stability_base_vs_west_area_data.{png,pdf}     Hull-only baseline vs west addition,
                                                 fixed inverse kernel (as in the chapter)
Top-20 LSOAs are labelled by rank and coloured by the signed Geo-SMILE geographic
sensitivity coefficient on one sequential scale; the added LSOA is shaded and outlined.
Coefficients are cached in outputs/geo_area_data_eval/stability_runs/, so re-plotting
does not rerun SMILE unless --force.

    python plot_area_data_stability_maps.py [--force]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import FancyArrowPatch, Patch

from compute_att_stability import DIRECTIONS
from eval_by_n_perturb import load_cached_neighbors
from geo_area_data import GEO_SCORE, area_geo_coefficients
from geosmile_hull_twobranch import SPATIAL_COLUMNS, choose_feature_players, temporal_split
from hull_boundaries import load_boundaries
from hull_boundary_neighbors import NEIGHBOR_GEOJSON, expand_panel
from hull_geo_area_data_eval import PANEL_PATH, SEED, Y_COL, ranked
from map_geo_importance_ranks import INK, INK_MUTED, MUTED_FILL, _draw_outline, _label_ranks, _prepare_plot_gdf
from model_factory import fit_model, make_model

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "outputs" / "geo_area_data_eval"
RUNS = DEST / "stability_runs"
N = 4000
K = 20
CMAP = LinearSegmentedColormap.from_list("geo_sens", ["#FFE3EA", "#FF0051", "#8A0033"])
ADDED_FILL, ADDED_EDGE = "#FFE08A", "#E8590C"
LABEL = "Geo-SMILE geographic sensitivity coefficient"


def compute(force: bool) -> None:
    """Base (adaptive and fixed inverse kernel), four additions, and west under the fixed kernel."""
    jobs = {"base": (None, None), "base_inverse": (None, "inverse"), "west_inverse": ("west", "inverse")}
    jobs.update({d: (d, None) for d in DIRECTIONS})
    todo = {k: v for k, v in jobs.items() if force or not (RUNS / f"{k}.csv").exists()}
    if not todo:
        return
    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
    hull_lsoas = sorted(panel["lsoa21cd"].astype(str).unique())
    neighbors, extra_panel = load_cached_neighbors()
    model = make_model("xgboost", SEED, n_jobs=4)
    fit_model("xgboost", model, panel.loc[train, players], panel.loc[train, Y_COL],
              panel.loc[validation, players], panel.loc[validation, Y_COL])
    RUNS.mkdir(parents=True, exist_ok=True)
    for name, (direction, kernel) in todo.items():
        if direction is None:
            data, data_test = panel, test
        else:
            code = str(neighbors.loc[neighbors["direction"] == direction, "lsoa21cd"].iloc[0])
            data = expand_panel(panel, extra_panel[extra_panel["lsoa21cd"].astype(str) == code])
            _, _, data_test = temporal_split(data)
        coef, _ = area_geo_coefficients(model, data, data_test, baseline, players, N, SEED,
                                        sample_lsoas=hull_lsoas, kernel=kernel)
        ranked(coef).to_csv(RUNS / f"{name}.csv", index=False)
        print(f"  {name}: saved", flush=True)


def load_run(name: str) -> pd.DataFrame:
    return pd.read_csv(RUNS / f"{name}.csv", dtype={"lsoa21cd": str})


def draw(ax, hull, extra, run, added: str | None, norm, title: str, bounds) -> None:
    hull.plot(ax=ax, color=MUTED_FILL, edgecolor="white", linewidth=0.12, zorder=1)
    top = hull.merge(run[run["geo_rank"] <= K], on="lsoa21cd")
    top.plot(ax=ax, column=GEO_SCORE, cmap=CMAP, norm=norm, edgecolor="white", linewidth=0.3, zorder=3)
    _draw_outline(ax, hull)
    _label_ranks(ax, top, "geo_rank", fontsize=7)
    if added:
        a = extra[extra["lsoa21cd"] == added]
        a.plot(ax=ax, color=ADDED_FILL, edgecolor=ADDED_EDGE, linewidth=1.4, zorder=4)
        hit = run[run["lsoa21cd"] == added].iloc[0]
        # label just outside the added polygon, on the side away from Hull's centre
        c, h = a.geometry.iloc[0].centroid, hull.union_all().centroid
        minx, miny, maxx, maxy = a.total_bounds
        above = c.y >= h.y
        ax.text(c.x, maxy + 500 if above else miny - 500, f"added: rank {int(hit['geo_rank'])}",
                ha="center", va="bottom" if above else "top", fontsize=9, fontweight="bold", color=ADDED_EDGE,
                zorder=6, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():  # panel frame
        sp.set_edgecolor("#9AA0A6")
        sp.set_linewidth(1.0)
    ax.set_aspect("equal")
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])


def extent(*frames, pad: float = 700.0, label_room: float = 1300.0) -> tuple[float, float, float, float]:
    b = np.array([f.total_bounds for f in frames])
    return b[:, 0].min() - pad, b[:, 1].min() - pad - label_room, b[:, 2].max() + pad, b[:, 3].max() + pad + label_room


def panel_size(bounds, width: float) -> tuple[float, float]:
    return width, width * (bounds[3] - bounds[1]) / (bounds[2] - bounds[0])


def colourbar(fig, norm, rect) -> None:
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=fig.add_axes(rect))
    cb.set_ticks([0, 1, 2, 5, 10, 15, 20])
    cb.set_label(LABEL + " (top 20)", fontsize=9, color=INK_MUTED)


def legend(fig) -> None:
    fig.legend(handles=[Patch(facecolor=MUTED_FILL, label=f"Outside top {K}"),
                        Patch(facecolor=ADDED_FILL, edgecolor=ADDED_EDGE, label="Added neighbouring LSOA")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    compute(args.force)

    base = load_run("base")
    hull_codes = sorted(pd.read_csv(PANEL_PATH, usecols=["lsoa21cd"])["lsoa21cd"].astype(str).unique())
    boundaries, _ = load_boundaries(hull_codes)
    hull = _prepare_plot_gdf(boundaries[["lsoa21cd", "geometry"]].drop_duplicates("lsoa21cd"))
    extra = _prepare_plot_gdf(gpd.read_file(NEIGHBOR_GEOJSON)[["lsoa21cd", "direction", "geometry"]])
    added = dict(zip(extra["direction"], extra["lsoa21cd"]))

    runs = {d: load_run(d) for d in DIRECTIONS}
    vmax = max(r.loc[r["geo_rank"] <= K, GEO_SCORE].max() for r in runs.values())
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)  # one coefficient dominates; sqrt keeps the rest legible

    bounds = extent(hull, extra)  # same frame in every panel
    pw, ph = panel_size(bounds, 6.2)
    fig, axes = plt.subplots(2, 2, figsize=(2 * pw + 1.3, 2 * ph + 1.6))
    fig.subplots_adjust(left=0.01, right=1 - 1.3 / (2 * pw + 1.3), top=1 - 1.0 / (2 * ph + 1.6),
                        bottom=0.6 / (2 * ph + 1.6), wspace=0.03, hspace=0.09)
    for ax, d in zip(axes.ravel(), DIRECTIONS):
        r = runs[d]
        kept = len(set(r[r["geo_rank"] <= K]["lsoa21cd"]) & set(base.head(K)["lsoa21cd"]))
        draw(ax, hull, extra, r, added[d], norm, f"{d.capitalize()} addition: {kept}/20 baseline top 20 retained", bounds)
    colourbar(fig, norm, [1 - 1.0 / (2 * pw + 1.3), 0.25, 0.015, 0.5])
    legend(fig)
    fig.suptitle("Geo-SMILE top-20 ranking under four cardinal neighbouring-LSOA additions\n"
                 "XGBoost, area-data design, N=4,000, seed 42", fontsize=13, fontweight="bold", color=INK)
    for ext in ("png", "pdf"):
        fig.savefig(DEST / f"stability_four_directions_area_data.{ext}", dpi=220, facecolor="white")
    plt.close(fig)

    b, w = load_run("base_inverse"), load_run("west_inverse")
    vmax = max(x.loc[x["geo_rank"] <= K, GEO_SCORE].max() for x in (b, w))
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    bounds = extent(hull, extra[extra["direction"] == "west"])
    pw, ph = panel_size(bounds, 6.6)
    W, H = 2 * pw + 1.9, ph + 1.5
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    fig.subplots_adjust(left=0.01, right=1 - 1.3 / W, top=1 - 1.0 / H, bottom=0.5 / H, wspace=0.09)
    draw(axes[0], hull, extra, b, None, norm, "Hull-only baseline\nGeo-SMILE top 20 (signed coefficient)", bounds)
    draw(axes[1], hull, extra, w, added["west"], norm, "West addition\nGeo-SMILE top 20 (signed coefficient)", bounds)
    x0, x1 = axes[0].get_position().x1, axes[1].get_position().x0
    fig.add_artist(FancyArrowPatch((x0 + 0.004, 0.5), (x1 - 0.004, 0.5), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=20, lw=2.5, color="#666666"))
    colourbar(fig, norm, [1 - 1.0 / W, 0.2, 0.012, 0.6])
    legend(fig)
    kept = len(set(b.head(K)["lsoa21cd"]) & set(w[w["geo_rank"] <= K]["lsoa21cd"]))
    fig.suptitle(f"West-neighbour addition under a fixed inverse kernel: {kept}/20 baseline top 20 retained",
                 fontsize=12, fontweight="bold", color=INK)
    for ext in ("png", "pdf"):
        fig.savefig(DEST / f"stability_base_vs_west_area_data.{ext}", dpi=220, facecolor="white")
    plt.close(fig)
    for d in DIRECTIONS:
        r = runs[d]
        print(d, "added rank", int(r.loc[r["lsoa21cd"] == added[d], "geo_rank"].iloc[0]))
    print("west (inverse) added rank", int(w.loc[w["lsoa21cd"] == added["west"], "geo_rank"].iloc[0]))
    print(f"Saved {DEST / 'stability_four_directions_area_data.png'} and stability_base_vs_west_area_data.png")


if __name__ == "__main__":
    main()
