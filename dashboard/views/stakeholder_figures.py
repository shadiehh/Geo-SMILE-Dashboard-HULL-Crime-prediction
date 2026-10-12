"""Stakeholder evidence figures (Chapter 6 design) for ANY selected LSOA.

Each builder takes the StakeholderData bundle and an LSOA code and returns
(matplotlib Figure | Path to a static PNG, caption title, caption text). The caption text is
computed from the same data as the figure, so a new LSOA gets correct numbers.

    STAKEHOLDER_PANELS[stakeholder](data, lsoa, contrast) -> list of panels
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from dashboard.data.schema import human_label
from dashboard.data.stakeholder_data import TEST_START, TOP_K, StakeholderData
from dashboard.llm import card_mode as cm

TEXT, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, PINK, DARK, ORANGE, GREY = "#2a78d6", "#FF0051", "#8A0033", "#f2a541", "#c9ccd1"
SEQUENTIAL = LinearSegmentedColormap.from_list("geo_seq", ["#FFFFFF", "#FFD1DC", PINK, DARK])
LISA_COLOURS = {"HH": "#B5003C", "HL": "#F28CB1", "LH": "#9CC3EE", "LL": "#1F5FA8", "NS": "#E3E5E8"}
LISA_NAMES = {"HH": "High-High", "HL": "High-Low", "LH": "Low-High", "LL": "Low-Low", "NS": "Not significant"}
FAMILIES = {
    "Recent crime history": ("#2a78d6", ("crime_",)),
    "Built environment (OSM)": ("#1baf7a", ("lsoa_area", "nightlife", "retail", "transit", "road_",
                                             "intersection", "green_space")),
    "Season": ("#eda100", ("month_",)),
}
CENSUS = ("Census (people & housing)", "#eb6834")
PLAIN = cm.PLAIN_LABELS

Panel = tuple  # (figure_or_path, title, caption)


# ---------------------------------------------------------------- shared drawing
def _family(feature: str) -> tuple[str, str]:
    for name, (colour, prefixes) in FAMILIES.items():
        if feature.startswith(prefixes):
            return name, colour
    return CENSUS


def _label(feature: str) -> str:
    return PLAIN.get(feature, human_label(feature))


def _fig(w=6.4, h=5.6):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def _style(ax) -> None:
    ax.grid(color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, length=0, labelsize=8)


def _outline(ax, d: StakeholderData, lsoa: str, label: str, offset=(2300, -2600), colour=TEXT) -> None:
    sel = d.geometry[d.geometry["lsoa21cd"] == lsoa]
    sel.plot(ax=ax, facecolor="none", edgecolor=colour, linewidth=1.8, zorder=5)
    pt = sel.geometry.iloc[0].representative_point()
    ax.annotate(label, xy=(pt.x, pt.y), xytext=(pt.x + offset[0], pt.y + offset[1]), fontsize=8, color=TEXT,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color=colour, linewidth=0.8), zorder=6)


def _offset_away(d: StakeholderData, lsoa: str, dist=2600) -> tuple[float, float]:
    pt = d.geometry.loc[d.geometry["lsoa21cd"] == lsoa].geometry.iloc[0].representative_point()
    c = d.geometry.union_all().centroid
    dx, dy = pt.x - c.x, pt.y - c.y
    n = max((dx ** 2 + dy ** 2) ** 0.5, 1.0)
    return dist * dx / n if n > 1 else dist, dist * dy / n if n > 1 else -dist


def _colourbar(fig, ax, cmap, norm, label: str, extend="neither") -> None:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.75, extend=extend)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, colors=MUTED, length=0)
    cb.set_label(label, fontsize=8, color=MUTED)


def _name(d: StakeholderData, lsoa: str) -> str:
    place = d.place(lsoa)
    return f"{d.short(lsoa)}" + (f" ({place})" if place else "")


def _ordinal(k: int) -> str:
    return f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}"


# ---------------------------------------------------------------- figure builders
def geo_map(d: StakeholderData, lsoa: str, others: tuple[str, ...] = (), plain: bool = False) -> Panel:
    """GeoSensitivity map; plain=True drops model terms and coefficients (Resident / Public view)."""
    g = d.geometry
    cap = float(np.percentile(np.abs(g["geo_coefficient"]), 98))
    norm = Normalize(float(g["geo_coefficient"].min()), cap)
    fig, ax = _fig()
    g.plot(ax=ax, column="geo_coefficient", cmap=SEQUENTIAL, norm=norm, edgecolor="#d9d9d9", linewidth=0.3)
    for code in (lsoa, *others):
        r = d.row(code)
        label = (f"{_name(d, code)}: {_ordinal(int(r['geo_rank']))} of {d.n}" if plain else
                 f"{d.short(code)}: {r['geo_coefficient']:.2f}, rank {int(r['geo_rank'])}/{d.n}")
        _outline(ax, d, code, label, offset=_offset_away(d, code))
    ax.set_axis_off()
    if plain:
        ax.set_title("Where predictions depend most on an area's own data", loc="left", color=TEXT, fontsize=10)
        _colourbar(fig, ax, SEQUENTIAL, norm, "Lighter = less, darker = more dependence", extend="max")
        cb_ax = fig.axes[-1]
        cb_ax.set_xticks([])
    else:
        ax.set_title("Geo-SMILE geographic sensitivity across Hull", loc="left", color=TEXT, fontsize=10)
        _colourbar(fig, ax, SEQUENTIAL, norm, f"Darker = more sensitive to the area's data (cap {cap:.1f})",
                   extend="max")
    r = d.row(lsoa)
    rank = int(r["geo_rank"])
    if plain:
        return fig, "Where it matters.", (
            f"Predictions for {_name(d, lsoa)} depend on its own data about as much as a typical area "
            f"({_ordinal(rank)} of {d.n})." if TOP_K < rank < d.n - TOP_K else
            f"{_name(d, lsoa)} is {_ordinal(rank)} of {d.n} for how much predictions depend on its own data.")
    where = "has the highest GeoSensitivity in Hull" if rank == 1 else (
        f"is among the {TOP_K} most sensitive areas" if rank <= TOP_K else
        "has below-average GeoSensitivity" if r["geo_coefficient"] < 0 else "has above-average GeoSensitivity")
    return fig, "Spatial view.", f"{d.short(lsoa)} {where} (rank {rank} of {d.n}; {r['geo_coefficient']:.2f})."


def feature_bars(d: StakeholderData, lsoa: str, top: int = 10, plain: bool = False) -> Panel:
    """Top local features; plain=True shows each as a share of what the model weighs (no coefficients)."""
    if plain:
        return _feature_share_bars(d, lsoa, top)
    vals = d.abs_local(lsoa).head(top)[::-1]
    fig, ax = _fig(6.4, 4.4)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.barh(range(len(vals)), vals.values, color=[_family(f)[1] for f in vals.index], height=0.7)
    ax.set_yticks(range(len(vals)), [_label(f) for f in vals.index], fontsize=8, color=TEXT)
    digits = 1 if vals.max() >= 5 else 2
    for i, v in enumerate(vals.values):
        ax.text(v + vals.max() * 0.01, i, f"{v:.{digits}f}", va="center", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, vals.max() * 1.12)
    ax.set_xlabel("Local importance |coefficient|", color=MUTED, fontsize=8)
    present = dict.fromkeys(_family(f) for f in vals.index[::-1])
    ax.legend(handles=[Patch(color=c, label=n) for n, c in present], frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title(f"Top {top} local features: {_name(d, lsoa)}", loc="left", color=TEXT, fontsize=10)
    a = d.abs_local(lsoa)
    first, second = a.index[0], a.index[1]
    text = (f"{_label(first)} ({a.iloc[0]:.{digits}f}) and {_label(second).lower()} ({a.iloc[1]:.{digits}f}) lead; "
            f"crime history is {100 * d.crime_share(lsoa):.0f}% of the local explanation.")
    return fig, "Local feature view.", text


def _feature_share_bars(d: StakeholderData, lsoa: str, top: int) -> Panel:
    a = d.abs_local(lsoa)
    share = (a / a.sum()).head(top)[::-1]
    fig, ax = _fig(6.4, 4.4)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.barh(range(len(share)), share.values, color=[_family(f)[1] for f in share.index], height=0.7)
    ax.set_yticks(range(len(share)), [_label(f) for f in share.index], fontsize=8, color=TEXT)
    for i, v in enumerate(share.values):
        ax.text(v + share.max() * 0.01, i, f"{100 * v:.0f}%", va="center", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, share.max() * 1.12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel("Share of what the model weighs here", color=MUTED, fontsize=8)
    present = dict.fromkeys(_family(f) for f in share.index[::-1])
    ax.legend(handles=[Patch(color=c, label=n) for n, c in present], frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title(f"What the model weighs most in {_name(d, lsoa)}", loc="left", color=TEXT, fontsize=10)
    return fig, "What matters here.", (
        f"{_label(share.index[-1])} is {100 * share.iloc[-1]:.0f}% of what the model weighs here; past crime "
        f"together is {100 * d.crime_share(lsoa):.0f}%.")


def planner_map(d: StakeholderData, lsoa: str, contrast: str) -> Panel:
    g = d.geometry
    cap = float(np.percentile(np.abs(g["geo_coefficient"]), 98))
    norm = Normalize(float(g["geo_coefficient"].min()), cap)
    fig, ax = _fig()
    g.plot(ax=ax, column="geo_coefficient", cmap=SEQUENTIAL, norm=norm, edgecolor="#d9d9d9", linewidth=0.3)
    for kind, colour in (("HH", DARK), ("LL", BLUE)):
        members = g[g["lisa"] == kind]
        if not members.empty:
            members.dissolve().boundary.plot(ax=ax, color=colour, linewidth=1.1, linestyle="--", zorder=4)
    for code in (lsoa, contrast):
        r = d.row(code)
        _outline(ax, d, code, f"{d.short(code)}: {r['geo_coefficient']:.2f}, rank {int(r['geo_rank'])}",
                 offset=_offset_away(d, code))
    ax.set_axis_off()
    ax.legend(handles=[plt.Line2D([], [], color=DARK, ls="--", label="High-High cluster"),
                       plt.Line2D([], [], color=BLUE, ls="--", label="Low-Low cluster")],
              frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("High-low contrast in geographic sensitivity", loc="left", color=TEXT, fontsize=10)
    _colourbar(fig, ax, SEQUENTIAL, norm, "Geo-SMILE geographic sensitivity coefficient", extend="max")
    a, b = d.row(lsoa), d.row(contrast)
    return fig, "Spatial contrast.", (
        f"{d.short(lsoa)} is {a['geo_coefficient']:.2f} (rank {int(a['geo_rank'])}/{d.n}); "
        f"{d.short(contrast)} is {b['geo_coefficient']:.2f} (rank {int(b['geo_rank'])}/{d.n}).")


def leading_feature_map(d: StakeholderData, lsoa: str | None) -> Panel:
    """|Local coefficient| of the area's leading feature; lsoa=None maps Hull's top global feature."""
    feature = d.abs_local(lsoa).index[0] if lsoa is not None else d.local.abs().sum().idxmax()
    g = d.geometry.merge(d.local[[feature]].abs().rename(columns={feature: "v"}), left_on="lsoa21cd",
                         right_index=True)
    cap = float(np.percentile(g["v"], 98))
    norm = Normalize(0, cap)
    fig, ax = _fig()
    g.plot(ax=ax, column="v", cmap=SEQUENTIAL, norm=norm, edgecolor="#d9d9d9", linewidth=0.3)
    leading = int((d.local.abs().idxmax(axis=1) == feature).sum())
    if lsoa is None:
        ax.set_axis_off()
        ax.set_title(f"Local effect of the top feature: {_label(feature).lower()}", loc="left", color=TEXT,
                     fontsize=10)
        _colourbar(fig, ax, SEQUENTIAL, norm, f"|Local coefficient| (capped at {cap:.1f})", extend="max")
        return fig, "Top feature, local effect.", (
            f"'{_label(feature)}' leads the local explanation in {leading} of {d.n} LSOAs; its local magnitude "
            f"ranges from {g['v'].min():.1f} to {g['v'].max():.1f} (median {g['v'].median():.1f}).")
    v = float(g.loc[g["lsoa21cd"] == lsoa, "v"].iloc[0])
    rank = int(g["v"].rank(ascending=False).loc[g["lsoa21cd"] == lsoa].iloc[0])
    _outline(ax, d, lsoa, f"{d.short(lsoa)}: {v:.1f}, rank {rank}/{d.n}", offset=_offset_away(d, lsoa))
    ax.set_axis_off()
    ax.set_title(f"Local reliance on: {_label(feature).lower()}", loc="left", color=TEXT, fontsize=10)
    _colourbar(fig, ax, SEQUENTIAL, norm, f"|Local coefficient| (capped at {cap:.1f})", extend="max")
    return fig, "Local feature view.", (
        f"Reliance on '{_label(feature)}' here is {v:.1f} (rank {rank} of {d.n}); it is the "
        f"leading local signal in {leading} of {d.n} LSOAs.")


def crime_timeline(d: StakeholderData, lsoa: str) -> Panel:
    p = d.crime[d.crime["lsoa21cd"] == lsoa].sort_values("month")
    med = d.crime.groupby("month")["crime_count"].median()
    fig, ax = _fig(6.4, 3.6)
    _style(ax)
    ax.axvspan(TEST_START - np.timedelta64(15, "D"), p["month"].max() + np.timedelta64(15, "D"), color="#eef2f7")
    ax.plot(p["month"], p["crime_count"], color=DARK, marker="o", ms=2.8, lw=1.6, label="Recorded crime")
    ax.plot(p["month"], p["crime_rolling_mean_6"], color=BLUE, lw=1.4, ls="--", label="Six-month average (model input)")
    ax.plot(med.index, med.values, color="#9aa0a6", lw=1.2, label="Median Hull LSOA")
    ax.set_ylim(0, max(p["crime_count"].max(), med.max()) * 1.18)
    ax.set_ylabel("Crimes per month", color=MUTED, fontsize=8)
    # One-column legend in the empty band above the Hull median; half-yearly date ticks, so the
    # panel stays readable when exported at print size (export_dashboard_panels.py).
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 0.1), ncol=1)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title(f"Recorded crime: {d.short(lsoa)} (shaded: test period)", loc="left", color=TEXT, fontsize=10)
    test = p[p["month"] >= TEST_START]
    lo, hi = int(test["crime_count"].min()), int(test["crime_count"].max())
    before = p[(p["month"] >= TEST_START - np.timedelta64(365, "D")) & (p["month"] < TEST_START)]["crime_count"].mean()
    trend = ("broadly stable" if abs(test["crime_count"].mean() - before) <= 0.15 * max(before, 1)
             else "higher than" if test["crime_count"].mean() > before else "lower than")
    trend_text = "broadly stable against the previous year" if trend == "broadly stable" else f"{trend} the previous year"
    return fig, "Persistence view.", (
        f"Between {lo} and {hi} crimes a month in the test period ({trend_text}), against a Hull median of "
        f"about {med[med.index >= TEST_START].mean():.0f}.")


def ranked_bars(d: StakeholderData, lsoa: str) -> Panel:
    r = d.geo.sort_values("geo_rank")
    colour = np.where(r["geo_rank"] > TOP_K, GREY, np.where(r["expert_region"], PINK, ORANGE))
    fig, ax = _fig(6.4, 3.8)
    _style(ax)
    ax.bar(r["geo_rank"], r["geo_coefficient"], color=colour, width=0.85)
    ax.axhline(0, color="#c9c8c3", lw=0.8)
    ax.axvline(TOP_K + 0.5, color=MUTED, lw=0.8, ls=":")
    sel = d.row(lsoa)
    ax.bar([sel["geo_rank"]], [sel["geo_coefficient"]], color=TEXT, width=1.6)
    ax.annotate(f"{d.short(lsoa)}: {sel['geo_coefficient']:.2f}, rank {int(sel['geo_rank'])}",
                xy=(sel["geo_rank"], sel["geo_coefficient"]),
                xytext=(min(sel["geo_rank"] + 15, d.n * 0.6), r["geo_coefficient"].max() * 0.5), fontsize=8,
                color=TEXT, arrowprops=dict(arrowstyle="-", color=TEXT, lw=0.8))
    n_in = int(r.head(TOP_K)["expert_region"].sum())
    ax.legend(handles=[Patch(color=PINK, label=f"Top {TOP_K}, in expert regions ({n_in})"),
                       Patch(color=ORANGE, label=f"Top {TOP_K}, outside ({TOP_K - n_in})"),
                       Patch(color=GREY, label=f"Ranks {TOP_K + 1}-{d.n}")], frameon=False, fontsize=7.5,
              loc="upper right")
    ax.set_xlabel("LSOA rank (1 = most sensitive)", color=MUTED, fontsize=8)
    ax.set_ylabel("GeoSensitivity coefficient", color=MUTED, fontsize=8)
    ax.set_title(f"All {d.n} LSOAs ranked by geographic sensitivity", loc="left", color=TEXT, fontsize=10)
    first, second = r["geo_coefficient"].iloc[0], r["geo_coefficient"].iloc[1]
    share = r.head(TOP_K)["exact_w1"].sum() / r["exact_w1"].sum()
    return fig, "Concentration view.", (
        f"The top area is {first:.2f}, {first / second:.1f}x the next; the top {TOP_K} carry {100 * share:.0f}% of "
        f"the total single-area shift. {d.short(lsoa)} is {_ordinal(int(sel['geo_rank']))}.")


def top20_map(d: StakeholderData, lsoa: str) -> Panel:
    g = d.geometry
    top = g[g["geo_rank"] <= TOP_K]
    cmap = LinearSegmentedColormap.from_list("rank", [DARK, PINK, "#FFD1DC"])
    fig, ax = _fig()
    g.plot(ax=ax, color="#e8eaed", edgecolor="white", linewidth=0.2)
    top.plot(ax=ax, column="geo_rank", cmap=cmap, norm=Normalize(1, TOP_K), edgecolor="white", linewidth=0.3)
    for _, row in top.iterrows():
        pt = row.geometry.representative_point()
        ax.text(pt.x, pt.y, str(int(row["geo_rank"])), ha="center", va="center", fontsize=6, fontweight="bold")
    if int(d.row(lsoa)["geo_rank"]) > TOP_K:
        _outline(ax, d, lsoa, f"{d.short(lsoa)}: rank {int(d.row(lsoa)['geo_rank'])}", offset=_offset_away(d, lsoa))
    ax.set_axis_off()
    n_in = int(top["expert_region"].sum())
    ax.set_title(f"Top {TOP_K} LSOAs by geographic sensitivity\n({n_in} of {TOP_K} in expert regions)",
                 loc="left", color=TEXT, fontsize=10)
    chance = len(d.regions) / d.n
    return fig, "Expert agreement.", (
        f"{n_in} of the {TOP_K} most sensitive areas lie in the regions the expert highlighted "
        f"({100 * n_in / TOP_K:.0f}% against {100 * chance:.0f}% by chance).")


def beta_vs_shift(d: StakeholderData, lsoa: str | None) -> Panel:
    """Surrogate check; lsoa=None draws the Hull-wide view with no area highlighted."""
    g = d.geo
    rho = spearmanr(g["geo_coefficient"], g["exact_w1"]).statistic
    fig, ax = _fig(6.4, 4.6)
    _style(ax)
    if lsoa is None:  # Hull-wide card view: one colour, no LISA classes
        ax.scatter(g["geo_coefficient"], g["exact_w1"], s=18, color=PINK, edgecolor="white", lw=0.4, zorder=3)
    for cls in ("NS", "LL", "LH", "HL", "HH") if lsoa is not None else ():
        s = g[g["lisa"] == cls]
        ax.scatter(s["geo_coefficient"], s["exact_w1"], s=18, color=LISA_COLOURS[cls], edgecolor="white", lw=0.4,
                   label=f"{LISA_NAMES[cls]} ({len(s)})", zorder=2 if cls == "NS" else 3)
    if lsoa is not None:
        r = d.row(lsoa)
        ax.scatter([r["geo_coefficient"]], [r["exact_w1"]], s=90, facecolor="none", edgecolor=TEXT, lw=1.6, zorder=5)
        ax.annotate(f"{d.short(lsoa)}: beta {r['geo_coefficient']:.2f}, W1 {r['exact_w1']:.1f}",
                    xy=(r["geo_coefficient"], r["exact_w1"]), xytext=(-0.9, g["exact_w1"].max() * 0.6), fontsize=8,
                    arrowprops=dict(arrowstyle="-", color=TEXT, lw=0.8))
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.set_xlabel("Signed Geo-SMILE coefficient (symlog)", color=MUTED, fontsize=8)
    ax.set_ylabel("Single-LSOA W1 shift (log)", color=MUTED, fontsize=8)
    if lsoa is not None:
        ax.legend(title="LISA class", title_fontsize=7.5, frameon=False, fontsize=7, loc="lower right")
    ax.set_title(f"Surrogate vs exact single-LSOA masking (rho_s = {rho:.3f})", loc="left", color=TEXT, fontsize=10)
    head = f"Signed beta tracks the exact single-LSOA W1 shift (rho_s = {rho:.3f}); "
    if lsoa is None:
        return fig, "Surrogate check.", head + (
            f"median W1 {g['exact_w1'].median():.1f}, 90th percentile {g['exact_w1'].quantile(0.9):.1f}.")
    return fig, "Surrogate check.", head + (
        f"{d.short(lsoa)} has W1 {r['exact_w1']:.1f} against a median of {g['exact_w1'].median():.1f}.")


def lisa_map(d: StakeholderData, lsoa: str | None) -> Panel:
    """LISA classes; lsoa=None draws the Hull-wide view with no area outlined."""
    g = d.geometry
    fig, ax = _fig()
    g.plot(ax=ax, color=g["lisa"].map(LISA_COLOURS).fillna("#E3E5E8"), edgecolor="white", linewidth=0.3)
    if lsoa is not None:
        r = d.row(lsoa)
        _outline(ax, d, lsoa, f"{d.short(lsoa)}: {LISA_NAMES.get(r['lisa'], r['lisa'])}\nlocal I = {r['local_I']:.2f}, "
                              f"p = {r['p_value']:.3f}", offset=_offset_away(d, lsoa))
    counts = g["lisa"].value_counts()
    ax.legend(handles=[Patch(color=LISA_COLOURS[c], label=f"{LISA_NAMES[c]} ({int(counts.get(c, 0))})")
                       for c in ("HH", "HL", "LH", "LL", "NS")], frameon=False, fontsize=7.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)  # below the map, clear of outlines
    ax.set_axis_off()
    ax.set_title(f"Local Moran's I (global I = {d.moran.get('Moran_I', float('nan')):.2f}, "
                 f"p = {d.moran.get('p_value', float('nan')):.3f})", loc="left", color=TEXT, fontsize=10)
    if lsoa is None:
        return fig, "Spatial structure.", (
            f"{int(counts.get('NS', 0))} of {d.n} LSOAs are not significant; "
            f"{int(counts.get('HH', 0))} are High-High (unadjusted local p-values).")
    cl = d.clusters.get(lsoa)
    member = f", in a contiguous {cl[0]} cluster of {len(cl[2])}" if cl else ""
    return fig, "Spatial structure.", (
        f"{d.short(lsoa)} is {LISA_NAMES.get(r['lisa'], r['lisa'])} (local I = {r['local_I']:.2f}, unadjusted "
        f"p = {r['p_value']:.3f}){member}.")


def seed_consistency(d: StakeholderData, lsoa: str) -> Panel:
    c = d.consistency
    seeds = [s for s in c.columns if s.startswith("seed_")]
    order = c.sort_values("mean_coefficient", ascending=False)
    shown = list(order.head(TOP_K).index)
    if lsoa not in shown:
        shown = shown[:TOP_K - 1] + [lsoa]
    sub = c.loc[shown]
    fig, ax = _fig(6.4, 3.8)
    _style(ax)
    x = np.arange(len(sub))
    for j, s in enumerate(seeds):
        ax.scatter(x + (j - 2) * 0.09, sub[s], s=10, color=BLUE, alpha=0.75, zorder=3,
                   label="Seeded runs" if j == 0 else None)
    ax.scatter(x, sub["mean_coefficient"], marker="_", s=180, color=TEXT, zorder=4, label="Mean")
    ax.set_yscale("symlog", linthresh=2)
    labels = [d.short(code) for code in sub.index]
    ax.set_xticks(x, labels, rotation=60, fontsize=7)
    for tick, code in zip(ax.get_xticklabels(), sub.index):
        if code == lsoa:
            tick.set_color(PINK)
            tick.set_fontweight("bold")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title(f"Coefficient across {len(seeds)} perturbation seeds\n(N = 4,000 each)", loc="left", color=TEXT,
                 fontsize=10)
    r = c.loc[lsoa]
    return fig, "Reproducibility.", (
        f"Across {len(seeds)} seeds {d.short(lsoa)} varies by SD {r['sd']:.2f} ({r['sd_pct_of_abs_mean']:.1f}% of "
        f"its mean); Geo rank Spearman across runs is {d.metrics['consistency']['rank_spearman_mean']:.3f}.")


CATEGORY_STYLE = {  # card_mode category key -> (label, colour), colours as FAMILIES
    "crime": ("Recent crime history", FAMILIES["Recent crime history"][0]),
    "census": CENSUS,
    "built": ("Built environment (OSM)", FAMILIES["Built environment (OSM)"][0]),
    "season": ("Season", FAMILIES["Season"][0]),
}
TEAL = LinearSegmentedColormap.from_list("planner_teal", ["#F4F8F9", "#7FB7C2", "#1F4E5F"])


def _paired_bars(ax, labels: list[str], hull: list[float], local: list[float], colours: list[str], who: str) -> None:
    """Hull-wide share (solid) above the focus LSOA's share (hatched outline) for each row."""
    y = np.arange(len(labels))[::-1]
    ax.barh(y + 0.19, hull, height=0.36, color=colours, label="Hull-wide")
    ax.barh(y - 0.19, local, height=0.36, color="none", edgecolor=colours, linewidth=1.2, hatch="////",
            label=who)
    top = max(max(hull), max(local))
    digits = 1 if top < 0.1 else 0  # small shares (built features) need a decimal to differ
    for yy, h, v in zip(y, hull, local):
        ax.text(h + top * 0.01, yy + 0.19, f"{100 * h:.{digits}f}%", va="center", fontsize=7.5, color=MUTED)
        ax.text(v + top * 0.01, yy - 0.19, f"{100 * v:.{digits}f}%", va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(y, labels, fontsize=8, color=TEXT)
    ax.set_xlim(0, top * 1.15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.{digits}f}%"))
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")


def category_bars(d: StakeholderData, lsoa: str) -> Panel:
    hull, local = cm.category_shares(d), cm.category_shares(d, lsoa)
    keys = list(hull.sort_values(ascending=False).index)
    fig, ax = _fig(6.4, 3.2)
    _style(ax)
    ax.grid(axis="y", visible=False)
    _paired_bars(ax, [CATEGORY_STYLE[k][0] for k in keys], [hull[k] for k in keys], [local[k] for k in keys],
                 [CATEGORY_STYLE[k][1] for k in keys], d.short(lsoa))
    ax.set_xlabel("Share of local importance", color=MUTED, fontsize=8)
    ax.set_title(f"What the model relies on: Hull vs {_name(d, lsoa)}", loc="left", color=TEXT, fontsize=10)
    return fig, "Feature categories.", (
        f"Across Hull, crime history carries {100 * hull['crime']:.0f}% of local importance, census "
        f"characteristics {100 * hull['census']:.0f}% and the built environment {100 * hull['built']:.0f}%; "
        f"in {d.short(lsoa)} crime history is {100 * local['crime']:.0f}% and the built environment "
        f"{100 * local['built']:.0f}%.")


def built_feature_bars(d: StakeholderData, lsoa: str) -> Panel:
    hull, local = cm.built_feature_shares(d), cm.built_feature_shares(d, lsoa)
    keys = list(hull.index)
    colour = CATEGORY_STYLE["built"][1]
    fig, ax = _fig(6.4, 3.6)
    _style(ax)
    ax.grid(axis="y", visible=False)
    _paired_bars(ax, [k.capitalize() for k in keys], [hull[k] for k in keys], [local.get(k, 0.0) for k in keys],
                 [colour] * len(keys), d.short(lsoa))
    ax.set_xlabel("Share of total local importance", color=MUTED, fontsize=8)
    ax.set_title(f"Built-environment features: Hull vs {_name(d, lsoa)}", loc="left", color=TEXT, fontsize=10)
    lead = local.idxmax()
    return fig, "Built-environment features.", (
        f"Across Hull {keys[0]} ({100 * hull.iloc[0]:.1f}%) and {keys[1]} ({100 * hull.iloc[1]:.1f}%) lead the "
        f"built-environment share; in {d.short(lsoa)} {lead} leads ({100 * local[lead]:.1f}%).")


def built_share_map(d: StakeholderData, lsoa: str) -> Panel:
    share = cm.built_share_table(d)
    g = d.geometry.merge(share[["built_share", "rank"]], left_on="lsoa21cd", right_index=True)
    norm = Normalize(0, float(g["built_share"].max()))
    fig, ax = _fig()
    g.plot(ax=ax, column="built_share", cmap=TEAL, norm=norm, edgecolor="#d9d9d9", linewidth=0.3)
    r = share.loc[lsoa]
    _outline(ax, d, lsoa, f"{d.short(lsoa)}: {100 * r['built_share']:.0f}%, {r['dominant_built_feature']}",
             offset=_offset_away(d, lsoa))
    ax.set_axis_off()
    ax.set_title("Built-environment share of local importance", loc="left", color=TEXT, fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=TEAL, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.75,
                      format=plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, colors=MUTED, length=0)
    cb.set_label("Share from 12 built-environment features", fontsize=8, color=MUTED)
    top = share.iloc[0]
    return fig, "Built-environment share.", (
        f"{d.short(lsoa)} gets {100 * r['built_share']:.0f}% of its local importance from built-environment "
        f"features (rank {int(r['rank'])} of {d.n}, led by {r['dominant_built_feature']}); the highest is "
        f"{top['locality']} at {100 * top['built_share']:.0f}% ({top['dominant_built_feature']}).")


CRIME_BLUE = LinearSegmentedColormap.from_list("police_blue", ["#F3F7FC", "#8DB6E8", "#1F4E8C"])
TREND = LinearSegmentedColormap.from_list("police_trend", ["#1F5FA8", "#9CC3EE", "#F4F4F2", "#F28CB1", "#B5003C"])


def recent_share_map(d: StakeholderData, lsoa: str) -> Panel:
    sig = cm.crime_signal_shares(d)
    g = d.geometry.merge(sig[["recent_share"]], left_on="lsoa21cd", right_index=True)
    norm = Normalize(0, float(g["recent_share"].max()))
    fig, ax = _fig()
    g.plot(ax=ax, column="recent_share", cmap=CRIME_BLUE, norm=norm, edgecolor="#d9d9d9", linewidth=0.3)
    r = sig.loc[lsoa]
    _outline(ax, d, lsoa, f"{d.short(lsoa)}: {100 * r['recent_share']:.0f}% recent", offset=_offset_away(d, lsoa))
    ax.set_axis_off()
    ax.set_title("Share of the local explanation from recent months", loc="left", color=TEXT, fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=CRIME_BLUE, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.75,
                      format=plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, colors=MUTED, length=0)
    cb.set_label("Share from crime in the last 1-3 months", fontsize=8, color=MUTED)
    rank = int(list(sig.index).index(lsoa)) + 1
    return fig, "Recent or long-run signal.", (
        f"Recent months carry {100 * r['recent_share']:.0f}% of {d.short(lsoa)}'s local explanation (rank {rank} of "
        f"{d.n}) against a Hull median of {100 * sig['recent_share'].median():.0f}%; the 6-month average carries "
        f"{100 * r['long_run_share']:.0f}%.")


def trend_map(d: StakeholderData, lsoa: str) -> Panel:
    trend, recent, previous = cm.recorded_trend(d)
    g = d.geometry.merge(trend, left_on="lsoa21cd", right_index=True)
    norm = Normalize(-1, 1)
    fig, ax = _fig()
    g[~g["assessed"]].plot(ax=ax, color="#E3E5E8", edgecolor="#d9d9d9", linewidth=0.3, hatch="///")
    g[g["assessed"]].plot(ax=ax, column="change", cmap=TREND, norm=norm, edgecolor="#d9d9d9", linewidth=0.3)
    r = trend.loc[lsoa]
    change = "not assessed" if pd.isna(r["change"]) else f"{100 * r['change']:+.0f}%"
    _outline(ax, d, lsoa, f"{d.short(lsoa)}: {change}", offset=_offset_away(d, lsoa))
    ax.set_axis_off()
    ax.set_title("Recorded change: last 3 vs previous 6 months", loc="left", color=TEXT, fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=TREND, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.75, extend="both",
                      format=plt.FuncFormatter(lambda v, _: f"{100 * v:+.0f}%"))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, colors=MUTED, length=0)
    cb.set_label(f"Change in monthly crime (grey: under {cm.TREND_MIN_MEAN}/month)", fontsize=8, color=MUTED)
    a = trend[trend["assessed"]]
    return fig, "Recorded change.", (
        f"{d.short(lsoa)}: {r['previous_mean']:.1f} to {r['recent_mean']:.1f} crimes a month ({change}). Of "
        f"{len(a)} assessed areas, {int((a['change'] > cm.TREND_THRESHOLD).sum())} rose and "
        f"{int((a['change'] < -cm.TREND_THRESHOLD).sum())} fell by more than a quarter.")


def stability_image(d: StakeholderData, lsoa: str) -> Panel:
    stab = d.metrics["stability"]
    ranks = ", ".join(f"{k} {v['added_rank']}" for k, v in stab["directions"].items())
    return d.stability_png, "Stability.", (
        f"The top {TOP_K} is unchanged under all four neighbouring-LSOA additions (Jaccard "
        f"{stab['jaccard_mean']:.1f}; added units rank {ranks}). City-wide test, the same for every LSOA.")


NEIGHBOUR_GEOJSON = Path(__file__).resolve().parents[2] / "data" / "cardinal_neighbor_lsoas.geojson"
ORANGE_EDGE = "#E07B00"


def stability_map(d: StakeholderData, lsoa: str | None = None) -> Panel:
    """One-map stability view: Hull's top 20 by rank, the four added neighbouring LSOAs outlined."""
    import geopandas as gpd

    stab = d.metrics["stability"]
    g = d.geometry
    top = g[g["geo_rank"] <= TOP_K]
    added = gpd.read_file(NEIGHBOUR_GEOJSON).to_crs(g.crs)
    cmap = LinearSegmentedColormap.from_list("rank", [DARK, PINK, "#FFD1DC"])
    fig, ax = _fig()
    g.plot(ax=ax, color="#e8eaed", edgecolor="white", linewidth=0.2)
    top.plot(ax=ax, column="geo_rank", cmap=cmap, norm=Normalize(1, TOP_K), edgecolor="white", linewidth=0.3)
    added.plot(ax=ax, color="#FFE08A", edgecolor=ORANGE_EDGE, linewidth=1.2)
    # Label each added LSOA just outside its own outline, on the side facing away from Hull.
    place = {"north": ("top", "center", "bottom", (0, 3)), "south": ("bottom", "center", "top", (0, -3)),
             "east": ("right", "left", "center", (3, 0)), "west": ("left", "right", "center", (-3, 0))}
    for _, row in added.iterrows():
        side, ha, va, off = place[row["direction"]]
        x0, y0, x1, y1 = row.geometry.bounds
        xy = {"top": ((x0 + x1) / 2, y1), "bottom": ((x0 + x1) / 2, y0),
              "right": (x1, (y0 + y1) / 2), "left": (x0, (y0 + y1) / 2)}[side]
        ax.annotate(f"{row['direction'].title()}: rank {stab['directions'][row['direction']]['added_rank']}",
                    xy=xy, xytext=off, textcoords="offset points", fontsize=9, color=ORANGE_EDGE,
                    fontweight="bold", ha=ha, va=va)
    ax.set_axis_off()
    s = d.metrics["smile_diagnostics"]
    kept = min(v["top20_retained"] for v in stab["directions"].values())
    ax.set_title(f"Top {TOP_K} kept {kept}/{TOP_K} under each neighbouring-LSOA addition\n(XGBoost, "
                 f"area-data design, N = {int(d.metrics['n']):,}, seed 42)", loc="left", color=TEXT, fontsize=10)
    ax.legend(handles=[Patch(color=PINK, label=f"Top {TOP_K} (darker = higher rank)"),
                       Patch(facecolor="#FFE08A", edgecolor=ORANGE_EDGE, label="Added neighbouring LSOA")],
              frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2)
    return fig, "Stability.", (
        f"The top {TOP_K} is unchanged under all four neighbouring-LSOA additions (Jaccard "
        f"{stab['jaccard_mean']:.1f}); each perturbation masks {s['min_lsoas_per_perturbation']}-"
        f"{s['max_lsoas_per_perturbation']} LSOAs. City-wide test, the same for every LSOA.")


# ---------------------------------------------------------------- stakeholder layouts
STAKEHOLDER_PANELS: dict[str, Callable[[StakeholderData, str, str], list[Callable[[], Panel]]]] = {
    "public": lambda d, a, b: [lambda: geo_map(d, a, plain=True), lambda: feature_bars(d, a, plain=True)],
    # Short public card: a typical area (Stoneferry), not an extreme one; plain-language panels.
    "public_short": lambda d, a, b: [lambda: geo_map(d, cm.PUBLIC_FOCUS, plain=True),
                                     lambda: feature_bars(d, cm.PUBLIC_FOCUS, 8, plain=True)],
    # Comprehensive planner view: contrast and clustering, feature charts, both local explanations.
    "planner": lambda d, a, b: [lambda: planner_map(d, a, b), lambda: lisa_map(d, a),
                                lambda: category_bars(d, a), lambda: built_feature_bars(d, a),
                                lambda: built_share_map(d, a), lambda: feature_bars(d, a, 8),
                                lambda: feature_bars(d, b, 8)],
    # Short planner card: the two Hull-wide card maps, rank-1 LSOA outlined.
    "planner_short": lambda d, a, b: [lambda: geo_map(d, cm.CENTRE), lambda: built_share_map(d, cm.CENTRE)],
    # Comprehensive police view: leading signal, recent vs long-run, recorded change, timeline, features.
    "police": lambda d, a, b: [lambda: geo_map(d, a), lambda: leading_feature_map(d, a), lambda: recent_share_map(d, a),
                               lambda: trend_map(d, a), lambda: crime_timeline(d, a),
                               lambda: category_bars(d, a), lambda: feature_bars(d, a, 8)],
    # Short police card: the three card panels, Old Town outlined. GeoSensitivity stays in the
    # text and tiles; its map is the dashboard's main view and the planner card's map 01.
    "police_short": lambda d, a, b: [lambda: recent_share_map(d, cm.CENTRE), lambda: trend_map(d, cm.CENTRE)],
    "manager": lambda d, a, b: [lambda: ranked_bars(d, a), lambda: top20_map(d, a), lambda: category_bars(d, a)],
    # Short manager card: where (top-20 map) and what (feature categories, Hull against Newland,
    # the most sensitive area not led by crime history).
    "manager_short": lambda d, a, b: [lambda: top20_map(d, cm.CENTRE),
                                      lambda: category_bars(d, cm.MANAGER_CONTRAST)],
    "researcher": lambda d, a, b: [lambda: beta_vs_shift(d, a), lambda: lisa_map(d, a), lambda: seed_consistency(d, a)],
    # Short researcher card: Hull-wide results of both branches, no single area highlighted.
    "researcher_short": lambda d, a, b: [lambda: beta_vs_shift(d, None), lambda: leading_feature_map(d, None)],
    "governance": lambda d, a, b: [lambda: seed_consistency(d, a), lambda: stability_image(d, a)],
    # Short governance card: Hull-wide assurance; the most seed-variable area is marked.
    "governance_short": lambda d, a, b: [lambda: seed_consistency(d, cm.CENTRE), lambda: stability_map(d)],
}

STAKEHOLDER_EMPHASIS = {
    "public": "City-wide spatial pattern plus the local feature explanation; technical statistics omitted.",
    "planner": "High-low spatial comparison: where sensitivity concentrates and how model reliance differs.",
    "police": "Dominant local crime-history signals and persistence; not a deployment recommendation.",
    "manager": "Finding, exception, confidence and an evidence-review action.",
    "researcher": "Exact values, spatial statistics and explanation-quality diagnostics.",
    "governance": "Provenance, reproducibility and limitations before the evidence is relied upon.",
}


def is_path(obj) -> bool:
    return isinstance(obj, Path)
