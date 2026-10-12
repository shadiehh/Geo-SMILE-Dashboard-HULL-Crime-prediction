"""Chapter 6 stakeholder-narrative figures for Police, Manager, Researcher and Governance.

Same visual language as plot_lsoa_feature_importance.py (resident) and
plot_planner_comparison.py (planner). Evidence is the published area-data design:
XGBoost, N=4,000, seed 42; Feature local coefficients from feature_local_coefficients.csv.

Writes to outputs/stakeholder_narratives/:
  police_E01033104_rm6_reliance_map.png     |local six-month-average coefficient| across Hull
  police_E01033104_crime_timeline.png       029E monthly recorded crime, Jan 2024 - Apr 2026
  manager_geo_ranked.png                    all 168 signed Geo coefficients, top 20 by expert region
  researcher_beta_vs_exact_shift.png        signed beta vs single-LSOA W1 shift, by LISA class
  researcher_lisa_map.png                   LISA classes of the Geo coefficient
  governance_seed_consistency.png           top-20 coefficients across five perturbation seeds
plus copies of geo_top20_xgboost.png and stability_four_directions_area_data.png.

    python plot_stakeholder_figures.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from geosmile_hull_twobranch import SEQUENTIAL

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "stakeholder_narratives"
LSOA = "E01033104"
TEXT, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, PINK, DARK = "#2a78d6", "#FF0051", "#8A0033"
LISA_COLOURS = {"HH": "#B5003C", "HL": "#F28CB1", "LH": "#9CC3EE", "LL": "#1F5FA8", "NS": "#E3E5E8"}
LISA_NAMES = {"HH": "High-High", "HL": "High-Low", "LH": "Low-High", "LL": "Low-Low", "NS": "Not significant"}


def save(fig, name: str) -> None:
    fig.savefig(DEST / name, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("Saved", DEST / name)


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, length=0, labelsize=8.5)


def boundaries() -> gpd.GeoDataFrame:
    return gpd.read_file(ROOT / "data" / "hull_lsoa_2021_official.geojson")[["lsoa21cd", "lsoa21nm", "geometry"]]


def outline(ax, g: gpd.GeoDataFrame, code: str, label: str) -> None:
    sel = g[g["lsoa21cd"] == code]
    sel.plot(ax=ax, facecolor="none", edgecolor=TEXT, linewidth=1.8, zorder=5)
    pt = sel.geometry.iloc[0].representative_point()
    ax.annotate(label, xy=(pt.x, pt.y), xytext=(pt.x + 2300, pt.y - 2600), fontsize=8.5, color=TEXT,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color=TEXT, linewidth=0.8), zorder=6)


def police_map(g, local) -> None:
    d = g.merge(local[["lsoa21cd", "crime_rolling_mean_6"]], on="lsoa21cd")
    d["v"] = d["crime_rolling_mean_6"].abs()
    cap = float(np.percentile(d["v"], 98))
    fig, ax = plt.subplots(figsize=(7.5, 7.2), facecolor=SURFACE)
    d.plot(ax=ax, column="v", cmap=SEQUENTIAL, norm=Normalize(0, cap), edgecolor="#d9d9d9", linewidth=0.35)
    v = float(d.loc[d["lsoa21cd"] == LSOA, "v"].iloc[0])
    outline(ax, d, LSOA, f"Kingston upon Hull 029E\ncity centre / Old Town: {v:.1f}, rank 1 of 168")
    ax.set_axis_off()
    ax.set_title("Local reliance on the six-month crime average across Hull\n"
                 "Feature branch, |local coefficient| per LSOA", loc="left", color=TEXT, fontsize=11, pad=8)
    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=Normalize(0, cap))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, extend="max", shrink=0.7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=MUTED, length=0)
    cb.set_label(f"|Local coefficient|: darker = model relies on the six-month average more (capped at {cap:.1f})",
                 fontsize=8.5, color=MUTED)
    save(fig, f"police_{LSOA}_rm6_reliance_map.png")


def police_timeline(panel) -> None:
    p = panel[panel["lsoa21cd"] == LSOA].sort_values("month")
    med = panel.groupby("month")["crime_count"].median()
    fig, ax = plt.subplots(figsize=(8.5, 4.4), facecolor=SURFACE)
    style(ax)
    ax.axvspan(pd.Timestamp("2025-12-16"), p["month"].max() + pd.Timedelta(days=15), color="#eef2f7", zorder=0)
    ax.plot(p["month"], p["crime_count"], color=DARK, marker="o", ms=3.5, lw=1.8, label="029E recorded crime")
    ax.plot(p["month"], p["crime_rolling_mean_6"], color=BLUE, lw=1.6, ls="--", label="029E six-month average (model input)")
    ax.plot(med.index, med.values, color="#9aa0a6", lw=1.4, label="Median Hull LSOA")
    ax.set_ylim(0, p["crime_count"].max() * 1.15)
    ax.set_ylabel("Recorded crimes per month", color=MUTED, fontsize=9)
    ax.text(pd.Timestamp("2026-01-05"), p["crime_count"].max() * 1.08, "test period", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="center left")
    ax.set_title("Kingston upon Hull 029E (city centre / Old Town): recorded crime, Jan 2024 to Apr 2026",
                 loc="left", color=TEXT, fontsize=11, pad=8)
    save(fig, f"police_{LSOA}_crime_timeline.png")


def manager_ranked(geo, regions) -> None:
    r = geo.sort_values("geo_coefficient", ascending=False).reset_index(drop=True)
    r["rank"] = np.arange(1, len(r) + 1)
    inside = r["lsoa21cd"].isin(regions)
    colour = np.where(r["rank"] > 20, "#c9ccd1", np.where(inside, PINK, "#f2a541"))
    fig, ax = plt.subplots(figsize=(8.5, 4.4), facecolor=SURFACE)
    style(ax)
    ax.bar(r["rank"], r["geo_coefficient"], color=colour, width=0.85)
    ax.axhline(0, color="#c9c8c3", lw=0.8)
    ax.axvline(20.5, color=MUTED, lw=0.8, ls=":")
    ax.annotate(f"029E city centre: {r.loc[0, 'geo_coefficient']:.1f}\n(about 3x the next area, {r.loc[1, 'geo_coefficient']:.1f})",
                xy=(1, r.loc[0, "geo_coefficient"]), xytext=(25, 17), fontsize=8.5, color=TEXT,
                arrowprops=dict(arrowstyle="-", color=TEXT, lw=0.8))
    ax.text(21.5, 9, "top 20", fontsize=8, color=MUTED)
    ax.set_xlim(0, len(r) + 1)
    ax.set_xlabel("LSOA rank (1 = model most sensitive to the area's data)", color=MUTED, fontsize=9)
    ax.set_ylabel("Geo-SMILE geographic\nsensitivity coefficient", color=MUTED, fontsize=9)
    n_in = int(inside[:20].sum())
    ax.legend(handles=[Patch(color=PINK, label=f"Top 20, inside expert-highlighted regions ({n_in})"),
                       Patch(color="#f2a541", label=f"Top 20, outside ({20 - n_in})"),
                       Patch(color="#c9ccd1", label="Ranks 21-168")], frameon=False, fontsize=8.5, loc="center right")
    ax.set_title("All 168 Hull LSOAs ranked by geographic sensitivity", loc="left", color=TEXT, fontsize=11, pad=8)
    save(fig, "manager_geo_ranked.png")


def researcher_scatter(geo, lisa) -> None:
    d = geo.merge(lisa[["lsoa21cd", "lisa"]], on="lsoa21cd")
    rho = spearmanr(d["geo_coefficient"], d["exact_w1"]).statistic
    fig, ax = plt.subplots(figsize=(7.2, 5.2), facecolor=SURFACE)
    style(ax)
    for cls in ("NS", "LL", "LH", "HL", "HH"):
        s = d[d["lisa"] == cls]
        ax.scatter(s["geo_coefficient"], s["exact_w1"], s=26, color=LISA_COLOURS[cls], edgecolor="white", lw=0.5,
                   label=f"{LISA_NAMES[cls]} ({len(s)})", zorder=3 if cls != "NS" else 2)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    top = d[d["lsoa21cd"] == LSOA].iloc[0]
    ax.annotate(f"029E: beta {top['geo_coefficient']:.2f}, W1 {top['exact_w1']:.1f}", xy=(top["geo_coefficient"], top["exact_w1"]),
                xytext=(1.2, 120), fontsize=8.5, color=TEXT, arrowprops=dict(arrowstyle="-", color=TEXT, lw=0.8))
    ax.set_xlabel("Signed Geo-SMILE coefficient (symmetric log scale)", color=MUTED, fontsize=9)
    ax.set_ylabel("Single-LSOA Wasserstein prediction shift\n(crimes per month, log scale)", color=MUTED, fontsize=9)
    ax.legend(title="LISA class", title_fontsize=8.5, frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"Coalition surrogate vs exact single-LSOA masking ($\\rho_s$ = {rho:.3f})",
                 loc="left", color=TEXT, fontsize=11, pad=8)
    save(fig, "researcher_beta_vs_exact_shift.png")


def researcher_lisa_map(g, lisa, moran) -> None:
    d = g.merge(lisa[["lsoa21cd", "lisa"]], on="lsoa21cd")
    fig, ax = plt.subplots(figsize=(7.5, 7.2), facecolor=SURFACE)
    d.plot(ax=ax, color=d["lisa"].map(LISA_COLOURS), edgecolor="white", linewidth=0.35)
    outline(ax, d, LSOA, "Kingston upon Hull 029E\nlocal I = 12.08, p = 0.003")
    ax.set_axis_off()
    counts = d["lisa"].value_counts()
    ax.legend(handles=[Patch(color=LISA_COLOURS[c], label=f"{LISA_NAMES[c]} ({int(counts.get(c, 0))})")
                       for c in ("HH", "HL", "LH", "LL", "NS")], frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title(f"Local Moran's I of the Geo-SMILE coefficient (queen contiguity, 999 permutations)\n"
                 f"Global Moran's I = {moran['Moran_I']:.2f}, p = {moran['p_value']:.3f}",
                 loc="left", color=TEXT, fontsize=10.5, pad=8)
    save(fig, "researcher_lisa_map.png")


def governance_seeds(cons, names) -> None:
    seeds = [c for c in cons.columns if c.startswith("seed_")]
    top = cons.sort_values("mean_coefficient", ascending=False).head(20).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    style(ax)
    x = np.arange(len(top))
    for j, s in enumerate(seeds):
        ax.scatter(x + (j - 2) * 0.09, top[s], s=14, color=BLUE, alpha=0.75, zorder=3,
                   label="Five independently seeded runs" if j == 0 else None)
    ax.scatter(x, top["mean_coefficient"], marker="_", s=260, color=TEXT, zorder=4, label="Mean across runs")
    ax.set_yscale("symlog", linthresh=2)
    ax.set_xticks(x, [names.get(c, c).replace("Kingston upon Hull ", "") for c in top["lsoa21cd"]], rotation=60,
                  fontsize=8)
    t = top.iloc[0]
    ax.annotate(f"029E: mean {t['mean_coefficient']:.2f}, SD {t['sd']:.2f} ({t['sd_pct_of_abs_mean']:.1f}%)",
                xy=(0, t["mean_coefficient"]), xytext=(3, 14), fontsize=8.5, color=TEXT,
                arrowprops=dict(arrowstyle="-", color=TEXT, lw=0.8))
    ax.set_ylabel("Geo-SMILE coefficient\n(symmetric log scale)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Reproducibility of the top 20 across perturbation seeds (N = 4,000 each)",
                 loc="left", color=TEXT, fontsize=11, pad=8)
    save(fig, "governance_seed_consistency.png")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    g = boundaries()
    local = pd.read_csv(OUT / "feature_local_coefficients.csv")
    geo = pd.read_csv(OUT / "geo_area_data_eval" / "geo_coefficients_N4000.csv")
    sp = OUT / "geo_area_data_eval" / "spatial_patterns"
    lisa = pd.read_csv(sp / "lisa_lsoa.csv")
    moran = json.loads((sp / "llm_spatial_pattern.json").read_text(encoding="utf-8"))["GLOBAL_SPATIAL_PATTERN"]
    regions = set(pd.read_csv(OUT / "geo_area_data_models" / "top50_maps" / "highlighted_regions_lsoas.csv")["lsoa21cd"])
    cons = pd.read_csv(OUT / "geo_area_data_eval" / "n_4000" / "consistency_per_lsoa.csv")
    panel = pd.read_csv(OUT / "hull_lsoa_month_panel_with_covariates.csv",
                        usecols=["lsoa21cd", "month", "crime_count", "crime_rolling_mean_6"], parse_dates=["month"])
    names = dict(zip(geo["lsoa21cd"], geo["lsoa21nm"]))

    police_map(g, local)
    police_timeline(panel)
    manager_ranked(geo, regions)
    researcher_scatter(geo, lisa)
    researcher_lisa_map(g, lisa, moran)
    governance_seeds(cons, names)
    for src, name in ((OUT / "geo_area_data_models" / "top50_maps" / "geo_top20_xgboost.png", "manager_geo_top20.png"),
                      (OUT / "geo_area_data_eval" / "stability_four_directions_area_data.png",
                       "governance_stability_four_directions.png")):
        shutil.copyfile(src, DEST / name)
        print("Copied", DEST / name)


if __name__ == "__main__":
    main()
