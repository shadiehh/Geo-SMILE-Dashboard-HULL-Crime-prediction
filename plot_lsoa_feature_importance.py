"""Local GeoSMILE feature-importance figure for one LSOA (thesis narrative figure).

Bars show |local coefficient| from outputs/feature_local_coefficients.csv (the
published Feature branch), coloured by feature family. A separate figure maps the
signed area-data GeoSensitivity across Hull with the LSOA outlined.

    python plot_lsoa_feature_importance.py [LSOA_CODE] [--top N]
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from dashboard.data.schema import human_label  # noqa: E402

OUT_DIR = ROOT / "outputs" / "stakeholder_narratives"

FAMILIES = {
    "Recent crime history": ("#2a78d6", ("crime_",)),
    "Census (people & housing)": ("#eb6834", (
        "population_density", "young_adult", "older_adult", "social_rented",
        "private_rented", "owner_occupied", "unemployed", "economically_inactive",
        "long_term_sick", "no_qualifications", "level4_plus",
    )),
    "Built environment (OSM)": ("#1baf7a", (
        "lsoa_area", "nightlife", "retail", "transit", "road_", "intersection", "green_space",
    )),
    "Season": ("#eda100", ("month_",)),
}
PLAIN_LABELS = {
    "crime_rolling_mean_6": "Crime, 6-month average",
    "crime_rolling_mean_3": "Crime, 3-month average",
    "crime_lag_1": "Crime, last month",
    "crime_lag_2": "Crime, 2 months ago",
    "crime_lag_3": "Crime, 3 months ago",
    "month_sin": "Time of year",
    "month_cos": "Time of year (phase)",
}
# Hand-checked labels for the thesis case studies (others use the OSM lookup).
PLACE_OVERRIDES = {
    "E01033104": "Hull city centre (Old Town)",
    "E01012877": "Orchard Park (around Danepark Road)",
}
TEXT, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
# Diverging blue <-> red with a neutral grey midpoint at 0.
# One sequential scale for signed GeoSensitivity (higher = more sensitive; the
# sign is not a direction, so no diverging midpoint). Same ramp as the thesis maps.
from geosmile_hull_twobranch import SEQUENTIAL  # noqa: E402


def family(feature: str) -> str:
    for name, (_, prefixes) in FAMILIES.items():
        if feature.startswith(prefixes):
            return name
    return "Census (people & housing)"


def _style_axis(ax) -> None:
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, length=0)


def plot_features(lsoa: str, top: int, place: str | None) -> Path:
    local = pd.read_csv(ROOT / "outputs" / "feature_local_coefficients.csv").set_index("lsoa21cd")
    row = local.loc[lsoa]
    name = row["lsoa21nm"]
    vals = row.drop("lsoa21nm").astype(float).abs().sort_values(ascending=False)
    shown = vals.head(top)[::-1]

    fig, ax = plt.subplots(figsize=(8.5, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    colours = [FAMILIES[family(f)][0] for f in shown.index]
    ax.barh(range(len(shown)), shown.values, color=colours, height=0.7,
            edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(range(len(shown)), [PLAIN_LABELS.get(f, human_label(f)) for f in shown.index],
                  color=TEXT, fontsize=9.5)
    for i, v in enumerate(shown.values):
        ax.text(v + shown.max() * 0.01, i, f"{v:.{1 if shown.max() >= 5 else 2}f}",
                va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, shown.max() * 1.1)
    ax.set_xlabel("Local importance |coefficient| (larger = model relies on it more here)",
                  color=MUTED, fontsize=9)
    _style_axis(ax)
    present = [n for n in FAMILIES if n in {family(f) for f in shown.index}]
    ax.legend(handles=[Patch(color=FAMILIES[n][0], label=n) for n in present],
              loc="lower right", frameon=False, fontsize=8.5)
    where = f"\n{place}" if place else ""
    ax.set_title(f"Top {top} features for {name} ({lsoa}){where}",
                 loc="left", color=TEXT, fontsize=11.5, pad=10)

    out = OUT_DIR / f"{lsoa}_feature_importance.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_geo_map(lsoa: str, place: str | None) -> Path:
    # Area-data Geo design, XGBoost, N = 4,000; identical to the coefficients in
    # outputs/geo_area_data_models/xgboost.json used by Hull_Geo_Area_Data_Evaluation.
    geo = pd.read_csv(ROOT / "outputs" / "geo_area_data_eval" / "n_4000" / "geo_coefficients.csv")
    geo["rank"] = geo["geo_coefficient"].rank(ascending=False).astype(int)
    g = gpd.read_file(ROOT / "data" / "hull_lsoa_2021_official.geojson").merge(
        geo[["lsoa21cd", "geo_coefficient", "rank"]], on="lsoa21cd", how="left"
    )
    sel = g[g["lsoa21cd"] == lsoa]
    name = sel["lsoa21nm"].iloc[0]
    value = float(sel["geo_coefficient"].iloc[0])
    rank = int(sel["rank"].iloc[0])

    # Cap the colour range at the 98th percentile of |value| so one extreme
    # LSOA does not wash out the rest; the colour bar marks the cap.
    cap = float(np.percentile(np.abs(g["geo_coefficient"].to_numpy(dtype=float)), 98))
    norm = Normalize(vmin=float(g["geo_coefficient"].min()), vmax=cap)

    fig, ax = plt.subplots(figsize=(7.5, 7.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    g.plot(ax=ax, column="geo_coefficient", cmap=SEQUENTIAL, norm=norm,
           edgecolor="#d9d9d9", linewidth=0.35)
    sel.plot(ax=ax, facecolor="none", edgecolor=TEXT, linewidth=2.0)
    pt = sel.geometry.iloc[0].representative_point()
    label = f"{name}\n{place}" if place else name
    # Push the label away from the city centre so the leader line points outward.
    centre = g.union_all().centroid
    dx, dy = pt.x - centre.x, pt.y - centre.y
    norm_len = max((dx ** 2 + dy ** 2) ** 0.5, 1.0)
    ax.annotate(label, xy=(pt.x, pt.y),
                xytext=(pt.x + 2600 * dx / norm_len, pt.y + 2600 * dy / norm_len),
                fontsize=8.5, color=TEXT, ha="left" if dx >= 0 else "right",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color=TEXT, linewidth=0.8))
    ax.set_axis_off()
    ax.set_title(f"Signed GeoSensitivity (area-data design) across Hull\n"
                 f"{name} ({lsoa}): {value:.1f}, rank {rank} of {len(geo)}",
                 loc="left", color=TEXT, fontsize=11, pad=8)
    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02,
                        extend="max", shrink=0.7)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=MUTED, length=0)
    cbar.set_label(f"Signed GeoSensitivity: darker = model more sensitive to the area's data (capped at {cap:.1f})",
                   fontsize=8.5, color=MUTED)

    out = OUT_DIR / f"{lsoa}_geo_sensitivity_map.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 12
    if "--top" in sys.argv:
        args.remove(str(top))
    lsoa = args[0] if args else "E01033104"
    places = pd.read_csv(ROOT / "data" / "hull_lsoa_place_names.csv").set_index("lsoa21cd")
    place = None
    if lsoa in places.index and isinstance(places.loc[lsoa, "locality_label"], str):
        place = f"around {places.loc[lsoa, 'locality_label']}"
    place = PLACE_OVERRIDES.get(lsoa, place)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(plot_features(lsoa, top, place))
    print(plot_geo_map(lsoa, place))
