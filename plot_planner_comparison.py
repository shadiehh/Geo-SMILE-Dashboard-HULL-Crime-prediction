"""City Planner case-study figures: a high- and a low-GeoSensitivity LSOA side by side.

1. <hi>_vs_<lo>_planner_map.png  signed area-data GeoSensitivity across Hull with
   the High-High and Low-Low LISA clusters that contain the two LSOAs outlined
   and both LSOAs labelled.
2. <hi>_vs_<lo>_feature_profiles.png  each LSOA's top local features as a share
   of that LSOA's total local importance, on one common 0-100% axis, with the
   total magnitude stated (the two totals differ by ~30x, so raw bars on one
   axis would hide the low-sensitivity case).

Uses the same sources as plot_lsoa_feature_importance.py and the official ONS
LSOA 2021 boundaries.

    python plot_planner_comparison.py [HIGH_LSOA LOW_LSOA]
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from dashboard.data.schema import human_label  # noqa: E402
from geosmile_hull_twobranch import SEQUENTIAL  # noqa: E402
from plot_lsoa_feature_importance import (  # noqa: E402
    FAMILIES, MUTED, OUT_DIR, PLACE_OVERRIDES, PLAIN_LABELS, SURFACE, TEXT, family,
)

GEO_CSV = ROOT / "outputs" / "geo_area_data_eval" / "n_4000" / "geo_coefficients.csv"
LISA_JSON = ROOT / "outputs" / "geo_area_data_eval" / "spatial_patterns" / "llm_spatial_pattern.json"
BOUNDARIES = ROOT / "data" / "hull_lsoa_2021_official.geojson"
HH_EDGE, LL_EDGE = "#0b0b0b", "#1c5cab"  # dark fill needs a black HH outline
TOP = 8


def _clusters() -> dict[str, tuple[str, list[str]]]:
    import json

    local = json.loads(LISA_JSON.read_text(encoding="utf-8"))["LOCAL_SPATIAL_PATTERN"]
    out = {}
    for key, kind in (("contiguous_HH_clusters", "HH"), ("contiguous_LL_clusters", "LL")):
        for c in local[key]:
            for code in c["lsoa21cd"]:
                out[code] = (f"{kind}-{c['cluster_id']}", c["lsoa21cd"])
    return out


def planner_map(hi: str, lo: str) -> Path:
    geo = pd.read_csv(GEO_CSV)
    geo["rank"] = geo["geo_coefficient"].rank(ascending=False).astype(int)
    g = gpd.read_file(BOUNDARIES).merge(geo, on="lsoa21cd", how="left")
    clusters = _clusters()
    cap = float(np.percentile(np.abs(g["geo_coefficient"]), 98))
    norm = Normalize(vmin=float(g["geo_coefficient"].min()), vmax=cap)

    fig, ax = plt.subplots(figsize=(8.5, 7.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    g.plot(ax=ax, column="geo_coefficient", cmap=SEQUENTIAL, norm=norm,
           edgecolor="#d9d9d9", linewidth=0.35)

    handles = []
    for code, edge in ((hi, HH_EDGE), (lo, LL_EDGE)):
        if code not in clusters:
            continue
        cid, members = clusters[code]
        outline = g[g["lsoa21cd"].isin(members)].dissolve()
        outline.boundary.plot(ax=ax, color=edge, linewidth=1.8, linestyle="--")
        kind = "High–High" if cid.startswith("HH") else "Low–Low"
        handles.append(Line2D([0], [0], color=edge, linestyle="--", linewidth=1.8,
                              label=f"{kind} cluster ({len(members)} LSOAs)"))

    centre = g.union_all().centroid
    for code in (hi, lo):
        sel = g[g["lsoa21cd"] == code]
        sel.boundary.plot(ax=ax, color=TEXT, linewidth=2.0)
        r = sel.iloc[0]
        pt = r.geometry.representative_point()
        dx, dy = pt.x - centre.x, pt.y - centre.y
        n = max((dx ** 2 + dy ** 2) ** 0.5, 1.0)
        place = PLACE_OVERRIDES.get(code, "")
        ax.annotate(
            f"{r['lsoa21nm'].replace('Kingston upon Hull ', '')} — {place}\n"
            f"GeoSensitivity {r['geo_coefficient']:.2f}, rank {int(r['rank'])} of {len(geo)}",
            xy=(pt.x, pt.y), xytext=(pt.x + 2800 * dx / n, pt.y + 2800 * dy / n),
            fontsize=8.5, color=TEXT, ha="left" if dx >= 0 else "right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor="#c9c8c3"),
            arrowprops=dict(arrowstyle="-", color=TEXT, linewidth=0.8),
        )
    handles.append(Line2D([0], [0], color=TEXT, linewidth=2.0, label="Case-study LSOA"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.06, 0.06), frameon=False, fontsize=8.5)
    ax.set_axis_off()
    ax.set_title("Where the model is most and least sensitive to an area's own data\n"
                 "Signed GeoSensitivity (area-data design), Hull LSOAs",
                 loc="left", color=TEXT, fontsize=11.5, pad=8)
    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02,
                        extend="max", shrink=0.65)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=MUTED, length=0)
    cbar.set_label(f"Signed GeoSensitivity: darker = more sensitive (capped at {cap:.1f})",
                   fontsize=8.5, color=MUTED)

    out = OUT_DIR / f"{hi}_vs_{lo}_planner_map.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def feature_profiles(hi: str, lo: str) -> Path:
    local = pd.read_csv(ROOT / "outputs" / "feature_local_coefficients.csv").set_index("lsoa21cd")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE, sharex=True)
    for ax, code in zip(axes, (hi, lo)):
        row = local.loc[code]
        vals = row.drop("lsoa21nm").astype(float).abs()
        total = float(vals.sum())
        share = (vals / total * 100).sort_values(ascending=False).head(TOP)[::-1]
        ax.set_facecolor(SURFACE)
        ax.barh(range(len(share)), share.values,
                color=[FAMILIES[family(f)][0] for f in share.index],
                height=0.7, edgecolor=SURFACE, linewidth=2)
        ax.set_yticks(range(len(share)),
                      [PLAIN_LABELS.get(f, human_label(f)) for f in share.index],
                      color=TEXT, fontsize=9.5)
        for i, v in enumerate(share.values):
            ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=8.5, color=MUTED)
        ax.set_xlim(0, 45)
        ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c3")
        ax.tick_params(colors=MUTED, length=0)
        ax.set_xlabel("Share of this area's total local importance (%)", color=MUTED, fontsize=9)
        ax.set_title(f"{row['lsoa21nm']} — {PLACE_OVERRIDES.get(code, '')}\n"
                     f"Total local importance {total:.1f}",
                     loc="left", color=TEXT, fontsize=10.5, pad=8)
    present = [n for n in FAMILIES]
    fig.legend(handles=[Patch(color=FAMILIES[n][0], label=n) for n in present],
               loc="lower center", ncol=len(present), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"What the model relies on locally: top {TOP} features in each case-study area",
                 x=0.01, ha="left", color=TEXT, fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / f"{hi}_vs_{lo}_feature_profiles.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    hi, lo = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("E01033104", "E01012877")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(planner_map(hi, lo))
    print(feature_profiles(hi, lo))
