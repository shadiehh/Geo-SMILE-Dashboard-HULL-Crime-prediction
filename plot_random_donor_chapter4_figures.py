"""Chapter-4 figures for the random-donor Hull design that the evaluation run does not draw.

  geo_map_random_donor                    (a) coefficient, top 10 numbered and named; (b) single-LSOA W1
  geo_sign_random_donor                   signed coefficient, diverging at 0; top 10 and bottom 5 named
  geo_clusters_random_donor               signed coefficient with LISA-defined clusters A-E and callouts
  geo_top20_random_donor                  top-20 Geo-SMILE LSOAs (signed coefficient), labelled by rank,
                                          with the expert-highlighted regions outlined
  stability_four_directions_random_donor  2x2 top-20 maps, one per cardinal neighbouring-LSOA addition
  stability_base_vs_west_random_donor     Hull-only baseline vs west addition, fixed inverse kernel
  feature_stability_west_random_donor     Feature top 10, Hull-only baseline vs west addition

geo_random_donor_eval.stability only records summary metrics, so the stability runs are repeated here
with the same model, seed (42), N=4,000 and K=50, and their full coefficient vectors cached in
outputs/geo_random_donor_design/stability_runs/ (rerun only with --force). The rerun is checked
against xgboost/n_4000.json. Figures go to outputs/geo_random_donor_design/report/, and all
Chapter-4 random-donor figures are copied under their chapter names to
outputs/geo_random_donor_design/chapter4_figs/. Not causal.

    python plot_random_donor_chapter4_figures.py [--force]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "geo_random_donor_design"
RUNS = DEST / "stability_runs"
REPORT = DEST / "report"
CHAPTER = DEST / "chapter4_figs"
COEF_CSV = DEST / "xgboost" / "geo_coefficients_N4000.csv"
FEAT_CSV = OUT / "geo_aligned_design" / "xgboost" / "feature_global_importance_N4000.csv"
REGIONS_CSV = OUT / "geo_area_data_models" / "top50_maps" / "highlighted_regions_lsoas.csv"
PLACES_CSV = ROOT / "data" / "hull_lsoa_place_names.csv"
N, KD, TOP, FTOP = 4000, 50, 20, 10
DIRS = ("north", "east", "south", "west")
CMAP = LinearSegmentedColormap.from_list("geo_sens", ["#FFE3EA", "#FF0051", "#8A0033"])
ADDED_FILL, ADDED_EDGE = "#FFE08A", "#E8590C"
LABEL = "Geo-SMILE geographic sensitivity coefficient"


# ---- compute -------------------------------------------------------------------------------------
def compute(force: bool) -> None:
    geo_jobs = {"base_inverse": (None, "inverse"), "west_inverse": ("west", "inverse")}
    geo_jobs.update({d: (d, None) for d in DIRS})
    todo = {k: v for k, v in geo_jobs.items() if force or not (RUNS / f"{k}.csv").exists()}
    feat_todo = force or not (RUNS / "feature_west.csv").exists()
    if not todo and not feat_todo:
        return
    import geo_random_donor_eval as rd
    import hull_geo_area_data_eval as ev
    from eval_by_n_perturb import load_cached_neighbors
    from geosmile_hull_twobranch import smile_global_feature, temporal_split
    from hull_aligned_design_eval import Aligned
    from hull_boundary_neighbors import expand_panel

    panel, cols, split, hull, names, refs, rules = rd.setup()
    model = rd.fit("xgboost", panel, cols, split)
    neighbors, extra = load_cached_neighbors()
    code = {d: str(neighbors.loc[neighbors["direction"] == d, "lsoa21cd"].iloc[0]) for d in DIRS}
    RUNS.mkdir(parents=True, exist_ok=True)
    for name, (d, kernel) in todo.items():
        if d is None:
            ctx = rd.Ctx(model, panel, cols, hull)
        else:
            exp = expand_panel(panel, extra[extra["lsoa21cd"].astype(str) == code[d]])
            ctx = rd.Ctx(model, exp, cols, hull, [code[d]])
        coef, *_ = ctx.pool(N, rd.SEED, KD, kernel)
        ev.ranked(coef).to_csv(RUNS / f"{name}.csv", index=False)
        rd.log(f"  {name}: saved")
    if feat_todo:
        players = cols[:-2]
        train = split[0]
        exp = expand_panel(panel, extra[extra["lsoa21cd"].astype(str) == code["west"]])
        _, _, et = temporal_split(exp)
        X_exp = exp.loc[et].reset_index(drop=True)
        w = Aligned(model, players).set_coords(X_exp)
        imp, *_ = smile_global_feature(w, X_exp, players, panel.loc[train, players].median(), N, rd.SEED)
        imp.to_csv(RUNS / "feature_west.csv", index=False)
        rd.log("  feature_west: saved")


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RUNS / f"{name}.csv", dtype={"lsoa21cd": str})


def check(base: pd.DataFrame) -> None:
    """The rerun must reproduce the stability metrics recorded by geo_random_donor_eval."""
    rec = json.loads((DEST / "xgboost" / "n_4000.json").read_text())
    btop = set(base.head(TOP)["lsoa21cd"])
    for d, ref in list(rec["stability"]["directions"].items()) + [("west_inverse", rec["fixed_kernel_west"])]:
        r = load(d)
        b = load("base_inverse") if d == "west_inverse" else base
        m = b[["lsoa21cd", "geo_score"]].merge(r[["lsoa21cd", "geo_score"]], on="lsoa21cd")
        rho = spearmanr(m["geo_score_x"], m["geo_score_y"])[0]
        rank = int(r.loc[r["lsoa21cd"] == ref["lsoa21cd"], "geo_rank"].iloc[0])
        kept = len(set(b.head(TOP)["lsoa21cd"]) & set(r[r["geo_rank"] <= TOP]["lsoa21cd"]))
        ok = abs(rho - ref["spearman"]) < 1e-6 and rank == ref["added_rank"] and kept == ref["top20_retained"]
        print(f"  check {d:13s} rho {rho:.4f} (json {ref['spearman']:.4f})  added rank {rank} "
              f"(json {ref['added_rank']})  kept {kept}  {'OK' if ok else 'MISMATCH'}")
    _ = btop


# ---- print-size styling ----------------------------------------------------------------------------
# Every figure is drawn at the width it is printed in the thesis (\textwidth = 6.3 in, A4 with 1 in
# margins, times the \includegraphics width used in chapter4.tex), so LaTeX does not rescale it and the
# font sizes below are the printed sizes beside 12 pt body text.
TW = 6.3
INK, MUTED, MUTED_FILL = "#12171C", "#57646D", "#E8ECF0"
RED, BLUE = "#E8174F", "#1E6FD0"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "legend.fontsize": 10, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
RANK_FS = 7.5


def save(fig, stem: str) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(REPORT / f"{stem}.{ext}", dpi=300, facecolor="white")
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"Saved {stem} ({w:.2f} x {h:.2f} in)")


# ---- maps ----------------------------------------------------------------------------------------
def frames():
    from hull_boundaries import load_boundaries
    from hull_boundary_neighbors import NEIGHBOR_GEOJSON
    from map_geo_importance_ranks import _prepare_plot_gdf
    coef = pd.read_csv(COEF_CSV, dtype={"lsoa21cd": str})
    b, _ = load_boundaries(sorted(coef["lsoa21cd"]))
    hull = _prepare_plot_gdf(b[["lsoa21cd", "geometry"]].drop_duplicates("lsoa21cd"))
    extra = _prepare_plot_gdf(gpd.read_file(NEIGHBOR_GEOJSON)[["lsoa21cd", "direction", "geometry"]])
    return coef, hull, extra


def outline(ax, hull) -> None:
    from map_geo_importance_ranks import _draw_outline
    _draw_outline(ax, hull)


def ranks(ax, gdf, col: str, fs: float = RANK_FS) -> None:
    import matplotlib.patheffects as pe
    for _, row in gdf.iterrows():
        pt = row.geometry.representative_point()
        ax.text(pt.x, pt.y, str(int(row[col])), ha="center", va="center", fontsize=fs, fontweight="bold",
                color=INK, zorder=7, path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])


def regions(ax, hull, lw=1.1) -> None:
    reg = pd.read_csv(REGIONS_CSV, dtype={"lsoa21cd": str})[["lsoa21cd", "region"]]
    hull.merge(reg, on="lsoa21cd").dissolve(by="region").boundary.plot(ax=ax, color=INK, linewidth=lw,
                                                                       linestyle="--", zorder=6)


def frame_axes(ax, title: str, bounds, fs=11) -> None:
    ax.set_title(title, fontsize=fs, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#9AA0A6"); sp.set_linewidth(0.8)
    ax.set_aspect("equal")
    ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])


def extent(*fr, pad=500.0, room=900.0):
    b = np.array([f.total_bounds for f in fr])
    return b[:, 0].min() - pad, b[:, 1].min() - pad - room, b[:, 2].max() + pad, b[:, 3].max() + pad + room


def norm_for(*runs):
    vmax = max(r.loc[r["geo_rank"] <= TOP, "geo_score"].max() for r in runs)
    return PowerNorm(gamma=0.5, vmin=0, vmax=vmax)  # one LSOA dominates; sqrt keeps the rest legible


def hbar(fig, norm, rect, label, cmap=CMAP, ticks=(0, 0.1, 0.25, 0.5, 1.0)) -> None:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=fig.add_axes(rect), orientation="horizontal")
    cb.set_ticks([t for t in ticks if norm.vmin <= t <= norm.vmax])
    cb.ax.tick_params(labelsize=9)
    cb.set_label(label, fontsize=10, color=INK)


def draw_top(ax, hull, extra, run, added, norm, title, bounds, fs=11, rank_fs=5.0) -> None:
    hull.plot(ax=ax, color=MUTED_FILL, edgecolor="white", linewidth=0.1, zorder=1)
    top = hull.merge(run[run["geo_rank"] <= TOP], on="lsoa21cd")
    top.plot(ax=ax, column="geo_score", cmap=CMAP, norm=norm, edgecolor="white", linewidth=0.25, zorder=3)
    outline(ax, hull)
    ranks(ax, top, "geo_rank", rank_fs)
    if added:
        a = extra[extra["lsoa21cd"] == added]
        a.plot(ax=ax, color=ADDED_FILL, edgecolor=ADDED_EDGE, linewidth=1.1, zorder=4)
        hit = run[run["lsoa21cd"] == added].iloc[0]
        c, h = a.geometry.iloc[0].centroid, hull.union_all().centroid
        _, miny, _, maxy = a.total_bounds
        above = c.y >= h.y
        ax.text(c.x, maxy + 300 if above else miny - 300, f"rank {int(hit['geo_rank'])}",
                ha="center", va="bottom" if above else "top", fontsize=10, fontweight="bold", color=ADDED_EDGE,
                zorder=8, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    frame_axes(ax, title, bounds, fs)


def legend_patches(fig, y, added=True) -> None:
    h = [Patch(facecolor=MUTED_FILL, label=f"Outside top {TOP}")]
    if added:
        h.append(Patch(facecolor=ADDED_FILL, edgecolor=ADDED_EDGE, label="Added neighbouring LSOA"))
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(h), frameon=False, fontsize=10)


def place_names() -> pd.Series:
    p = pd.read_csv(PLACES_CSV, dtype={"lsoa21cd": str})
    return p.set_index("lsoa21cd")["locality_label"]


def rank_key(rows: pd.DataFrame, names: pd.Series, per_line: int = 4) -> str:
    """'1 Old Town 029E: 1.07' entries, per_line to a line."""
    items = [f"{r} {names.get(c, '?')} {n.split()[-1]}: {v:.2f}".replace("-", "−")
             for r, c, n, v in zip(rows["r"], rows["lsoa21cd"], rows["lsoa21nm"], rows["geo_coefficient"])]
    return "\n".join("   ".join(items[i:i + per_line]) for i in range(0, len(items), per_line))


def geo_labels(ax, north=(0.06, 0.86)) -> None:
    """Humber label and north arrow, in axes fractions of the Hull extent used by every map."""
    ax.text(0.72, 0.1, "Humber Estuary", ha="center", transform=ax.transAxes, fontsize=7.5, style="italic",
            color=MUTED, zorder=8)
    ax.annotate("N", xy=(north[0], north[1] + 0.11), xytext=north, xycoords="axes fraction", ha="center", va="center",
                fontsize=8, fontweight="bold", color=INK, zorder=8,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, shrinkA=6, shrinkB=0))


def circled(ax, gdf, col: str, face: str, fs: float = 7) -> None:
    for _, row in gdf.iterrows():
        pt = row.geometry.representative_point()
        ax.annotate(str(row[col]), (pt.x, pt.y), ha="center", va="center", fontsize=fs, fontweight="bold",
                    color="white", bbox=dict(boxstyle="circle,pad=0.15", facecolor=face, edgecolor="none"), zorder=7)


def expert_legend(ax, loc: str) -> None:
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.1, ls="--", label="Expert-highlighted\nareas (map 1)")],
              loc=loc, frameon=False, fontsize=7.5)


def plot_geo_map(coef, hull) -> None:
    """(a) signed coefficient, top 10 numbered and named; (b) single-LSOA shift with the expert regions."""
    from matplotlib.colors import Normalize
    from geosmile_hull_twobranch import SEQUENTIAL, shared_limit
    W = TW
    bounds = extent(hull, room=0.0)
    ph = (W / 2 - 0.1) * (bounds[3] - bounds[1]) / (bounds[2] - bounds[0])
    H = ph + 1.75
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.3 / H, bottom=1.45 / H, wspace=0.04)
    g = hull.merge(coef, on="lsoa21cd")
    for ax, col, title, lab, lo in ((axes[0], "geo_coefficient", "(a) GeoSensitivity coefficient", "Coefficient", None),
                                    (axes[1], "exact_w1", "(b) Single-LSOA $W_1$ shift", "$W_1$, crimes per month", 0.0)):
        v = g[col].to_numpy()
        norm = Normalize(float(v.min()) if lo is None else lo, shared_limit(v))
        g.plot(ax=ax, column=col, cmap=SEQUENTIAL, norm=norm, edgecolor="#d9d9d9", linewidth=0.1)
        outline(ax, hull)
        frame_axes(ax, title, bounds)
        for sp in ax.spines.values():
            sp.set_visible(False)
        pos = ax.get_position()
        sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=norm); sm.set_array([])
        cb = fig.colorbar(sm, cax=fig.add_axes([pos.x0 + 0.08 * pos.width, 1.1 / H, 0.84 * pos.width, 0.1 / H]),
                          orientation="horizontal", extend="max")
        cb.ax.tick_params(labelsize=9); cb.set_label(lab, fontsize=10)
        geo_labels(ax)
    top = g.nlargest(10, "geo_coefficient").assign(r=range(1, 11))
    circled(axes[0], top, "r", INK)
    regions(axes[1], hull)
    expert_legend(axes[1], "upper right")
    fig.text(0.5, 0.38 / H, "Top 10 in (a), coefficient\n" + rank_key(top, place_names()), ha="center", va="center",
             fontsize=7.5, color=INK, linespacing=1.35)
    save(fig, "geo_map_random_donor")


def plot_geo_sign(coef, hull) -> None:
    """Signed GeoSensitivity coefficient on a symmetric diverging scale centred at zero; top 10 and bottom 5 named.

    The sign is relative to the surrogate intercept: positive = relocating the LSOA moves the coalition W1
    above average, negative = it dilutes it. It is not a direction of predicted crime change."""
    from matplotlib.colors import Normalize
    from geosmile_hull_twobranch import DIVERGING, shared_limit
    W = 0.85 * TW
    bounds = extent(hull, room=0.0)
    pw = W - 1.15
    ph = pw * (bounds[3] - bounds[1]) / (bounds[2] - bounds[0])
    H = ph + 1.75
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0.01, right=pw / W + 0.01, top=1 - 0.1 / H, bottom=1.65 / H)
    g = hull.merge(coef, on="lsoa21cd")
    v = g["geo_coefficient"].to_numpy()
    lim = shared_limit(v)  # symmetric, so a small negative is not drawn as strongly as a large positive
    norm = Normalize(-lim, lim)
    g.plot(ax=ax, column="geo_coefficient", cmap=DIVERGING, norm=norm, edgecolor="#d9d9d9", linewidth=0.1, zorder=1)
    outline(ax, hull)
    regions(ax, hull)
    frame_axes(ax, "", bounds)
    for sp in ax.spines.values():
        sp.set_visible(False)
    geo_labels(ax)
    expert_legend(ax, "upper right")
    top = g.nlargest(10, "geo_coefficient").assign(r=range(1, 11))
    low = g.nsmallest(5, "geo_coefficient").assign(r=[f"−{i}" for i in range(1, 6)])
    circled(ax, top, "r", INK)
    circled(ax, low, "r", "#0B4F9C", fs=6)
    npos, nneg = int((v > 0).sum()), int((v < 0).sum())
    ax.text(0.02, 0.02, f"{npos} positive, {nneg} negative", transform=ax.transAxes, fontsize=8, color=INK)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, cax=fig.add_axes([pw / W + 0.04, 0.42, 0.025, 0.48]), extend="max")
    cb.set_ticks([-0.2, -0.1, 0, 0.1, 0.2]); cb.ax.tick_params(labelsize=8)
    cb.set_label("Signed GeoSensitivity coefficient", fontsize=9)
    names = place_names()
    fig.text(0.5, 1.05 / H, "Most positive (dark circles)\n" + rank_key(top, names, 3), ha="center", va="center",
             fontsize=7, color=INK, linespacing=1.35)
    fig.text(0.5, 0.3 / H, "Most negative (blue circles)\n" + rank_key(low, names, 3), ha="center", va="center",
             fontsize=7, color="#0B4F9C", linespacing=1.35)
    save(fig, "geo_sign_random_donor")


HIGH_C, LOW_C = "#9C1C6B", "#1E6FD0"


def geo_clusters(g: gpd.GeoDataFrame) -> dict:
    """Cluster LSOA sets for the annotated map: local Moran's I (queen, 999 permutations, seed 42, p < 0.05)
    plus the top-10 corridor west of the River Hull. Each is a boolean mask over g."""
    from esda.moran import Moran_Local
    from libpysal.weights import Queen
    w = Queen.from_dataframe(g, use_index=False)
    w.transform = "r"
    lm = Moran_Local(g["geo_coefficient"].to_numpy(), w, permutations=999, seed=42)
    sig = lm.p_sim < 0.05
    hh, ll = sig & (lm.q == 1), sig & (lm.q == 3)
    loc = g["locality"]
    c = g.geometry.centroid
    x0, y0, x1, y1 = g.total_bounds
    cx, cy = (c.x - x0) / (x1 - x0), (c.y - y0) / (y1 - y0)
    sculcoates = (g["geo_rank"] == 3).to_numpy()
    corridor = (g["geo_rank"] <= 10) & (cx < 0.5) & ~sculcoates
    return {"A": (hh | corridor.to_numpy()) & ~sculcoates,
            "B": ((g["geo_rank"] <= 20) & (cx > 0.55)).to_numpy(),
            "C": sculcoates,
            "D": ll & loc.isin(["Bransholme", "Kingswood"]).to_numpy(),
            "E": ll & loc.isin(["Garden Village", "Avenues"]).to_numpy()}


def plot_geo_clusters(coef, hull) -> None:
    """Signed GeoSensitivity map with data-defined clusters outlined and explained in callout boxes."""
    import textwrap
    from matplotlib.colors import Normalize
    from geosmile_hull_twobranch import SEQUENTIAL, shared_limit
    g = hull.merge(coef, on="lsoa21cd")
    g["locality"] = g["lsoa21cd"].map(place_names())
    cl = geo_clusters(g)
    med = float(g["exact_w1"].median())
    st = {k: dict(n=int(m.sum()), b=g.loc[m, "geo_coefficient"].mean(), w=g.loc[m, "exact_w1"].median(),
                  ranks=sorted(g.loc[m, "geo_rank"].astype(int))) for k, m in cl.items()}

    def rk(r):
        return ", ".join(map(str, r))

    text = {
        "A": ("A: peak corridor", HIGH_C,
              f"Old Town, Stepney and Dairycoates, west of the River Hull: the main high belt "
              f"({st['A']['n']} LSOAs, mean β {st['A']['b']:.2f}, ranks 1, 2, 4–7). Relocating one "
              f"shifts its predictions by about {st['A']['w']:.1f} crimes per month (city median {med:.2f})."),
        "B": ("B: eastern secondary", HIGH_C,
              f"Marfleet, Greatfield and Summergangs (ranks {rk(st['B']['ranks'])}). High, but secondary to "
              f"the western corridor and not a significant LISA cluster."),
        "C": ("C: isolated high", HIGH_C,
              f"One Sculcoates LSOA by the river (rank 3, β {st['C']['b']:.2f}, shift "
              f"{st['C']['w']:.1f} crimes per month) among near-zero neighbours."),
        "D": ("D: northern low", LOW_C,
              f"Bransholme and Kingswood: significant Low–Low cluster ({st['D']['n']} LSOAs, mean "
              f"β {st['D']['b']:.3f}). Relocation barely shifts the prediction "
              f"(about {st['D']['w']:.1f} crimes per month)."),
        "E": ("E: low pockets", LOW_C,
              f"Garden Village and the Avenues: Low–Low pockets right beside the high cores "
              f"(mean β {st['E']['b']:.3f}, shift about {st['E']['w']:.2f} crimes per month)."),
    }
    # (x, top y) of each box in figure fractions; left column boxes end at x + BOXW
    BOXW = 0.215
    boxes = {"C": (0.01, 0.96), "E": (0.01, 0.66), "A": (0.01, 0.36),
             "D": (0.775, 0.96), "B": (0.775, 0.62)}

    W, H = TW, 5.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.2, 0.17, 0.6, 0.82])
    v = g["geo_coefficient"].to_numpy()
    norm = Normalize(float(v.min()), shared_limit(v))
    g.plot(ax=ax, column="geo_coefficient", cmap=SEQUENTIAL, norm=norm, edgecolor="#d9d9d9", linewidth=0.1)
    outline(ax, hull)
    frame_axes(ax, "", extent(hull, room=0.0))
    for sp in ax.spines.values():
        sp.set_visible(False)
    geo_labels(ax, north=(0.93, 0.03))
    import matplotlib.patheffects as pe
    for k, m in cl.items():
        col = text[k][1]
        # closing (grow then shrink) merges neighbouring fragments into one smooth outline
        shape = g.loc[m].geometry.union_all().buffer(600).buffer(-380).simplify(80)
        gpd.GeoSeries([shape], crs=g.crs).boundary.plot(ax=ax, color=col, linewidth=1.2, linestyle="--", zorder=6)
        parts = list(getattr(shape, "geoms", [shape]))
        part = max(parts, key=lambda p: p.area)
        bx, by = boxes[k]
        right = bx > 0.5
        for q in parts:  # letter every piece, on the side facing the box
            px, py = (max if right else min)(q.exterior.coords, key=lambda p: p[0])
            ax.text(px + (200 if right else -200), py, k, ha="left" if right else "right", va="center",
                    fontsize=9 if q is part else 7.5, fontweight="bold", color=col, zorder=8,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        tgt = part.representative_point()
        ax.annotate("", xy=(tgt.x, tgt.y), xycoords="data", xytext=(bx if right else bx + BOXW, by - 0.06),
                    textcoords="figure fraction", zorder=9,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, shrinkA=0, shrinkB=1, mutation_scale=7))
        title, _, body = text[k]
        body = body.replace(" -0", " −0")
        fig.text(bx, by, title, fontsize=7.5, fontweight="bold", color=col, va="top")
        fig.text(bx + 0.006, by - 0.04, textwrap.fill(body, 28 if right else 30), fontsize=6.3, color=INK, va="top",
                 linespacing=1.2, bbox=dict(boxstyle="square,pad=0.35", fc="white", ec=col, lw=0.7))
    npos, nneg = int((v > 0).sum()), int((v < 0).sum())
    fig.text(0.78, 0.33, textwrap.fill(
        f"{npos} LSOAs positive, {nneg} negative. Higher β: relocating the area raises the coalition "
        f"W1 more. Negative β: the area dilutes that response; it does not mean predicted crime falls.", 29),
        fontsize=6.3, color=MUTED, va="top", linespacing=1.2)
    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, cax=fig.add_axes([0.32, 0.125, 0.36, 0.02]), orientation="horizontal", extend="max")
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label("Signed GeoSensitivity coefficient", fontsize=8.5)
    fig.text(0.5, 0.012, "Outlines: significant local Moran's I clusters (queen contiguity, 999 permutations, "
             "p < 0.05) and the top-10 corridor. Not causal.", ha="center", fontsize=6.3, color=MUTED)
    save(fig, "geo_clusters_random_donor")


def plot_top20(coef, hull) -> None:
    W = 0.85 * TW
    bounds = extent(hull, room=0.0)
    pw = W - 0.9
    ph = pw * (bounds[3] - bounds[1]) / (bounds[2] - bounds[0])
    H = ph + 0.75
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0.01, right=pw / W + 0.01, top=1 - 0.1 / H, bottom=0.65 / H)
    norm = norm_for(coef)
    draw_top(ax, hull, None, coef, None, norm, "", bounds, rank_fs=7.0)
    regions(ax, hull)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, cax=fig.add_axes([pw / W + 0.04, 0.2, 0.025, 0.6]))
    cb.set_ticks([0, 0.1, 0.25, 0.5, 1.0]); cb.ax.tick_params(labelsize=9)
    cb.set_label("GeoSensitivity coefficient (top 20)", fontsize=10)
    fig.legend(handles=[Patch(facecolor=MUTED_FILL, label=f"Outside top {TOP}"),
                        Line2D([], [], color=INK, lw=1.1, ls="--", label="Expert-highlighted areas")],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=10)
    save(fig, "geo_top20_random_donor")


def plot_four(base, hull, extra, added) -> None:
    runs = {d: load(d) for d in DIRS}
    norm = norm_for(*runs.values())
    bounds = extent(hull, extra)
    W = 0.96 * TW
    pw = (W - 0.1) / 2
    ph = pw * (bounds[3] - bounds[1]) / (bounds[2] - bounds[0])
    H = 2 * ph + 2.05
    fig, axes = plt.subplots(2, 2, figsize=(W, H))
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - 0.3 / H, bottom=1.55 / H, wspace=0.03, hspace=0.12)
    btop = set(base.head(TOP)["lsoa21cd"])
    for ax, d in zip(axes.ravel(), DIRS):
        r = runs[d]
        kept = len(set(r[r["geo_rank"] <= TOP]["lsoa21cd"]) & btop)
        draw_top(ax, hull, extra, r, added[d], norm, f"{d.capitalize()}: {kept}/20 retained", bounds)
    hbar(fig, norm, [0.2, 0.95 / H, 0.6, 0.1 / H], "GeoSensitivity coefficient (top 20)")
    legend_patches(fig, 0.0)
    save(fig, "stability_four_directions_random_donor")


def plot_west(hull, extra, added) -> None:
    b, w = load("base_inverse"), load("west_inverse")
    norm = norm_for(b, w)
    bounds = extent(hull, extra[extra["direction"] == "west"])
    W = 0.96 * TW
    pw = (W - 0.35) / 2
    ph = pw * (bounds[3] - bounds[1]) / (bounds[2] - bounds[0])
    H = ph + 1.85
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - 0.32 / H, bottom=1.5 / H, wspace=0.11)
    draw_top(axes[0], hull, extra, b, None, norm, "Hull-only baseline", bounds)
    kept = len(set(b.head(TOP)["lsoa21cd"]) & set(w[w["geo_rank"] <= TOP]["lsoa21cd"]))
    draw_top(axes[1], hull, extra, w, added["west"], norm, f"West addition: {kept}/20 retained", bounds)
    x0, x1 = axes[0].get_position().x1, axes[1].get_position().x0
    yc = (axes[0].get_position().y0 + axes[0].get_position().y1) / 2
    fig.add_artist(FancyArrowPatch((x0 + 0.003, yc), (x1 - 0.003, yc), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=14, lw=1.8, color="#666666"))
    hbar(fig, norm, [0.2, 0.92 / H, 0.6, 0.1 / H], "GeoSensitivity coefficient (top 20), fixed inverse kernel")
    legend_patches(fig, 0.0)
    save(fig, "stability_base_vs_west_random_donor")


def plot_feature_local(hull) -> None:
    from matplotlib.colors import TwoSlopeNorm
    from geosmile_hull_twobranch import DIVERGING, FEATURE_LABELS, shared_limit
    folder = OUT / "geo_aligned_design" / "xgboost"
    local = pd.read_csv(folder / "feature_local_coefficients_N4000.csv")
    local = local.drop(columns=[c for c in local.columns if c.startswith("Unnamed")]).groupby("lsoa21cd").mean().reset_index()
    top6 = pd.read_csv(FEAT_CSV)["feature"].head(6).tolist()
    g = hull.merge(local, on="lsoa21cd")
    bounds = extent(hull, room=0.0, pad=200.0)
    W = TW
    pw = (W - 0.1) / 3
    ph = pw * (bounds[3] - bounds[1]) / (bounds[2] - bounds[0])
    H = 2 * (ph + 0.95) + 0.45
    fig, axes = plt.subplots(2, 3, figsize=(W, H))
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.32 / H, bottom=1.0 / H, wspace=0.05, hspace=0.62)
    for i, (ax, f) in enumerate(zip(axes.ravel(), top6)):
        lim = shared_limit(g[f].to_numpy())
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
        g.plot(ax=ax, column=f, cmap=DIVERGING, norm=norm, edgecolor="#d9d9d9", linewidth=0.1)
        outline(ax, hull)
        frame_axes(ax, f"{i + 1}. {FEATURE_LABELS.get(f, f)}", bounds, fs=10.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        pos = ax.get_position()
        sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=norm); sm.set_array([])
        cb = fig.colorbar(sm, cax=fig.add_axes([pos.x0 + 0.1 * pos.width, pos.y0 - 0.3 / H, 0.8 * pos.width, 0.08 / H]),
                          orientation="horizontal", extend="both")
        t = float(f"{0.8 * lim:.1g}")
        cb.set_ticks([-t, 0, t]); cb.ax.set_xticklabels([f"{-t:g}", "0", f"{t:g}"], fontsize=9)
    fig.text(0.5, 0.06 / H, "Signed local coefficient (original minus perturbed prediction)", ha="center",
             fontsize=10, color=INK)
    save(fig, "feature_local_random_donor")


# ---- charts --------------------------------------------------------------------------------------
def plot_feature_global() -> None:
    from geosmile_hull_twobranch import FEATURE_LABELS
    imp = pd.read_csv(FEAT_CSV).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(0.8 * TW, 3.7))
    y = np.arange(len(imp))
    ax.barh(y, imp["abs_coefficient"], color=RED, height=0.7)
    ax.set_yticks(y, [FEATURE_LABELS.get(f, f) for f in imp["feature"]])
    for yy, v in zip(y, imp["abs_coefficient"]):
        ax.text(v * 1.08, yy, f"{v:.3f}", va="center", fontsize=10, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(0.06, 12)
    ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.set_xlabel("|Global Feature-SMILE coefficient| (log scale)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "feature_global_random_donor")


def plot_metrics() -> None:
    """3x2 diagnostics across perturbation budgets; the shaded band marks the N=4,000 reporting design."""
    import string
    NS = (1000, 2000, 3000, 4000, 5000, 7500, 10000)
    fx = {n: json.loads((OUT / "geo_aligned_design" / "xgboost" / f"n_{n}.json").read_text())["feature"] for n in NS}
    gx = {n: json.loads((DEST / "xgboost" / f"n_{n}.json").read_text()) for n in NS}
    panels = [
        ("Fidelity $R^2$", lambda d: d["fidelity"]["R2"], lambda d: d["fidelity"]["R2"], False),
        ("ATT accuracy", lambda d: d["accuracy"]["intersection"] / 10, lambda d: d["accuracy"]["intersection"] / 20, False),
        ("Intervention $\\Delta W_1$",lambda d: d["intervention"]["top_minus_random"],
         lambda d: d["intervention"]["top_minus_random"], False),
        ("Stability Jaccard", lambda d: d["stability"]["jaccard_mean"], lambda d: d["stability"]["jaccard_mean"], False),
        ("Consistency mean variance", lambda d: d["consistency"]["mean_variance"],
         lambda d: d["consistency"]["mean_variance"], True),
        ("Consistency mean SD", lambda d: d["consistency"]["mean_sd"], lambda d: d["consistency"]["mean_sd"], False),
    ]
    GEO_C, FEAT_C, BAND = "#D81B60", "#1565C0", "#EEF1F6"
    fig, axes = plt.subplots(3, 2, figsize=(0.98 * TW, 7.4))
    for k, (ax, (title, ff, gf, logy)) in enumerate(zip(axes.ravel(), panels)):
        fy, gy = [ff(fx[n]) for n in NS], [gf(gx[n]) for n in NS]
        ax.axvspan(3600, 4450, color=BAND, zorder=0)
        ax.plot(NS, gy, "-o", color=GEO_C, mfc="white", mew=1.3, ms=4.5, lw=1.6, zorder=3, label="Geo (random donor)")
        ax.plot(NS, fy, "-o", color=FEAT_C, mfc="white", mew=1.3, ms=4.5, lw=1.6, zorder=3, label="Feature")
        i = NS.index(4000)
        ax.plot(4000, gy[i], "o", color=GEO_C, ms=5, zorder=4)
        ax.plot(4000, fy[i], "o", color=FEAT_C, ms=5, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks([1000, 2000, 4000, 10000], ["1k", "2k", "4k", "10k"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"({string.ascii_lowercase[k]}) {title}", fontsize=11, loc="left", fontweight="bold")
        ax.grid(True, axis="y", color="#E3E6EA", lw=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#9AA0A6")
        ax.tick_params(colors="#3A4249", length=3)
    axes[0, 1].set_ylim(0.4, 1.0)
    axes[0, 1].text(0.97, 0.06, "Feature: top-10 overlap\nGeo: top 20 in expert areas", transform=axes[0, 1].transAxes,
                    ha="right", va="bottom", fontsize=8.5, color=MUTED, linespacing=1.15)
    axes[1, 1].set_ylim(0.88, 1.01)
    for ax in axes[2]:
        ax.set_xlabel("Perturbations $N$ (log scale)", fontsize=10)
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(Patch(facecolor=BAND, edgecolor="none")); l.append("Reported design ($N$ = 4,000)")
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.2, w_pad=1.4)
    save(fig, "metrics_by_n_random_donor")


# ---- Feature stability cards ---------------------------------------------------------------------
# Card layout: rank 1 is the large central card, ranks 2-10 sit around it (slot = rank). Colour =
# thesis category; shade = |coefficient| band, shared by all categories and both panels.
SHADES = {
    "Crime / temporal": ["#9DBBD9", "#6A9BD1", "#3584D4", "#1F6BB5", "#1F3864"],
    "Economic / labour & tenure": ["#C39BD3", "#A86BD8", "#8E3AD6", "#7419B8", "#5B0F95"],
    "Social / demographic": ["#DDEBD5", "#C4DFB2", "#A8D08D", "#548235", "#375623"],
    "Transport / accessibility": ["#FFC6EC", "#FF9AC8", "#FF6699", "#D93545", "#C00000"],
    "Built environment / activity": ["#FCE4C4", "#F9C98A", "#F4A64A", "#E07B1A", "#A9520B"],
    "Environmental": ["#D5F0EC", "#A3DDD4", "#5FC2B4", "#2A9D8F", "#1D6F65"],
}
NAVY = "#1F3864"
BANDS = (0.12, 0.19, 0.30, 1.0)
# (x centre, y centre, width, height) for ranks 2..10; rank 1 is CENTRE
SLOTS = [(2.45, 0.05, 1.6, 1.2), (1.65, 1.45, 1.7, 0.95), (-0.05, 1.55, 1.55, 0.95), (-1.65, 1.25, 1.55, 0.95),
         (-2.45, -0.05, 1.45, 1.0), (-2.35, -1.3, 1.55, 0.95), (-0.8, -1.25, 1.45, 0.95), (0.75, -1.45, 1.45, 1.05),
         (2.25, -1.45, 1.45, 1.05)]
CENTRE = (0.0, 0.0, 4.4, 2.4)


def card_colour(cat: str, v: float) -> tuple[str, str]:
    pal = SHADES.get(cat, ["#E0E0E0", "#C0C0C0", "#A0A0A0", "#707070", "#404040"])
    i = int(np.searchsorted(BANDS, v, side="right"))
    return pal[i], ("white" if i >= 2 else INK)


def card_panel(ax, ranked: pd.DataFrame, cats: dict, title: str) -> None:
    import textwrap
    from matplotlib.patches import Rectangle
    from geosmile_hull_twobranch import FEATURE_LABELS
    for i, row in ranked.head(FTOP).iterrows():
        f, v = row["feature"], row["abs_coefficient"]
        x, y, w, h = CENTRE if i == 0 else SLOTS[i - 1]
        face, ink = (NAVY, "white") if i == 0 else card_colour(cats.get(f, ""), v)
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=face, edgecolor="white", lw=1.0,
                               zorder=1 if i == 0 else 3))
        lab = FEATURE_LABELS.get(f, f).replace(" %", " %").replace("mean ", "mean ")
        name = textwrap.fill(f"{i + 1}. {lab}", 14 if w < 1.6 else 17).replace(" ", " ")
        tx, fs = (-0.3, 11) if i == 0 else (x, 9.5)
        ax.text(tx, y + 0.06, name, ha="center", va="bottom", fontsize=fs, color=ink, zorder=4, linespacing=1.05)
        ax.text(tx, y + 0.0, f"{v:.2f}", ha="center", va="top", fontsize=fs + 1, color=ink, fontweight="bold", zorder=4)
    ax.set_xlim(-3.3, 3.3); ax.set_ylim(-2.1, 2.1)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(title, fontsize=12, color=NAVY, pad=2)


def plot_feature_west() -> None:
    from matplotlib.patches import Rectangle
    from make_feature_stability_schematic import load_feature_categories
    rank = lambda f: f.sort_values(["abs_coefficient", "feature"], ascending=[False, True]).reset_index(drop=True)
    base, west = rank(pd.read_csv(FEAT_CSV)), rank(load("feature_west"))
    cats = load_feature_categories()
    both = pd.concat([base.head(FTOP), west.head(FTOP)])
    kept = len(set(base.head(FTOP)["feature"]) & set(west.head(FTOP)["feature"]))
    rho = spearmanr(base.set_index("feature")["abs_coefficient"],
                    west.set_index("feature")["abs_coefficient"].loc[base["feature"]])[0]
    # stacked vertically so the cards are large enough for 9.5-11 pt text at 0.94 \textwidth
    W = 0.94 * TW
    ph = W * 4.2 / 6.6
    H = 2 * ph + 1.55
    fig = plt.figure(figsize=(W, H), facecolor="white")
    a0 = fig.add_axes([0, (1.35 + ph) / H, 1, ph / H]); a1 = fig.add_axes([0, 1.05 / H, 1, ph / H])
    card_panel(a0, base, cats, "Hull-only baseline")
    card_panel(a1, west, cats, f"West addition (E01013112): {kept}/10 retained, Spearman {rho:.4f}")
    present = [c for c in SHADES if c in set(both["feature"].map(cats))]
    lg = fig.add_axes([0.02, 0.05 / H, 0.96, 0.85 / H]); lg.set_axis_off(); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
    for n, c in enumerate(present):
        cx = 0.01 + n * 0.335
        lg.text(cx, 0.62, c.replace(" & tenure", "\n& tenure").replace(" / activity", "\n/ activity").replace(" / temporal", "\n/ temporal"), fontsize=9.5, fontweight="bold",
                color=INK, va="bottom")
        for k, col in enumerate(SHADES[c]):
            lg.add_patch(Rectangle((cx + k * 0.06, 0.3), 0.058, 0.24, fc=col, ec="white", lw=0.6))
    lg.text(0.5, 0.0, "Shade = |coefficient| band: <0.12, 0.12–0.19, 0.19–0.30, 0.30–1.0, ≥1.0",
            ha="center", fontsize=9, color=MUTED)
    save(fig, "feature_stability_west_random_donor")
    print(f"  feature west: kept {kept}/10, rho {rho:.4f} (aligned-design json: 0.9996)")


def copy_chapter() -> None:
    CHAPTER.mkdir(parents=True, exist_ok=True)
    stems = ("feature_global_random_donor", "geo_map_random_donor", "geo_sign_random_donor", "geo_clusters_random_donor", "feature_local_random_donor",
             "geo_top20_random_donor", "stability_four_directions_random_donor",
             "stability_base_vs_west_random_donor", "feature_stability_west_random_donor", "metrics_by_n_random_donor")
    for s in stems:
        for ext in ("png", "pdf"):
            shutil.copy2(REPORT / f"{s}.{ext}", CHAPTER / f"{s}.{ext}")
    shutil.copy2(REGIONS_CSV.parent / "highlighted_regions_check.png", CHAPTER / "highlighted_regions_check.png")
    print(f"Copied {len(stems)} figures (png + pdf) and highlighted_regions_check.png to {CHAPTER}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    compute(a.force)
    coef, hull, extra = frames()
    base = coef.sort_values("geo_rank").reset_index(drop=True)
    check(base)
    added = dict(zip(extra["direction"], extra["lsoa21cd"]))
    plot_feature_global()
    plot_geo_map(coef, hull)
    plot_geo_sign(coef, hull)
    plot_geo_clusters(coef, hull)
    plot_feature_local(hull)
    plot_top20(base, hull)
    plot_four(base, hull, extra, added)
    plot_west(hull, extra, added)
    plot_feature_west()
    plot_metrics()
    copy_chapter()


if __name__ == "__main__":
    main()
