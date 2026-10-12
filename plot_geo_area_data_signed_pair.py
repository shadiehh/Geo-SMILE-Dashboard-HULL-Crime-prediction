"""Two-panel Hull GeoSensitivity map: the SAME signed beta drawn two ways, annotated.

(a) diverging scale centred at beta = 0 (blue below zero, pink above);
(b) single-hue sequential red scale (light = lower beta, dark = higher beta).

Both panels plot the raw signed Ridge coefficient `geo_coefficient` of the area-data
geo branch (N=4,000, seed 42; mask 1 = masked, response = W1). No absolute value is
taken: ranking, colours and labels all use signed beta. The sign is not a direction of
prediction change. Both scales stop at the 98th percentile of beta (rank 1 at +22.3 would
otherwise wash out the map); the colourbar's arrow marks the capped units.

Reads outputs/geo_area_data_eval/geo_coefficients_N4000.csv.
Writes outputs/geo_area_data_eval/geo_signed_pair_N4000{,_plain}.{pdf,png}. Not causal.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402

from geosmile_hull_twobranch import DIVERGING, INK, SEQUENTIAL  # noqa: E402
from hull_boundaries import hull_outline, load_boundaries  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "outputs" / "geo_area_data_eval"
CAP_PCT = 98
N_LSOA = 168

# Callouts: name -> (LSOA, label position in axes fraction). The LSOAs are the ones the
# hand-drawn callouts pointed at; rank and beta in the label are read from the data.
CALLOUTS = {
    "extreme core": ("E01033104", (0.62, 0.04)),
    "strong ridge": ("E01012817", (0.70, 0.93)),
    "low-ranked corridor": ("E01012877", (0.01, 0.93)),
}
LABEL = r"Signed GeoSensitivity coefficient $\beta$"


def load() -> gpd.GeoDataFrame:
    coef = pd.read_csv(DEST / "geo_coefficients_N4000.csv", dtype={"lsoa21cd": str})
    coef = coef[["lsoa21cd", "geo_coefficient"]]  # signed beta only; geo_abs_coefficient never read
    coef["rank"] = coef["geo_coefficient"].rank(ascending=False, method="first").astype(int)
    assert len(coef) == N_LSOA and coef["lsoa21cd"].is_unique, f"{len(coef)} LSOAs"
    assert (coef["geo_coefficient"] < 0).any(), "no negative beta -- values look like |beta|"
    boundaries, _ = load_boundaries(sorted(coef["lsoa21cd"]))
    gdf = boundaries[["lsoa21cd", "geometry"]].merge(coef, on="lsoa21cd").to_crs("EPSG:27700")
    assert len(gdf) == N_LSOA, "an LSOA has no boundary"
    return gdf


def draw(gdf: gpd.GeoDataFrame, annotate: bool) -> None:
    beta = gdf["geo_coefficient"].to_numpy()
    lo, cap = float(beta.min()), float(np.percentile(beta, CAP_PCT))
    outline = hull_outline(gdf)
    panels = [
        ("(a) Signed GeoSensitivity, diverging scale", DIVERGING, TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=cap)),
        ("(b) Signed GeoSensitivity, sequential red scale", SEQUENTIAL, Normalize(vmin=lo, vmax=cap)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    for ax, (title, cmap, norm) in zip(axes, panels):
        gdf.plot(column="geo_coefficient", cmap=cmap, norm=norm, edgecolor="#d9d9d9", linewidth=0.12, ax=ax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.01, extend="max").set_label(LABEL, fontsize=9)
        if outline is not None:
            gpd.GeoSeries([outline], crs=gdf.crs).plot(ax=ax, facecolor="none", edgecolor=INK, linewidth=0.9)
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK)
        ax.set_axis_off()
        if annotate:
            for name, (code, xytext) in CALLOUTS.items():
                row = gdf.loc[gdf["lsoa21cd"] == code].iloc[0]
                point = row.geometry.representative_point()
                ax.plot(point.x, point.y, "o", ms=3.5, color=INK, zorder=5)
                ax.annotate(f"{name}\nrank {row['rank']} of {N_LSOA}, β = {row['geo_coefficient']:+.2f}",
                            xy=(point.x, point.y), xycoords="data", xytext=xytext, textcoords="axes fraction",
                            fontsize=8.5, color=INK, zorder=6,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cfd6dc", alpha=0.92),
                            arrowprops=dict(arrowstyle="-", color=INK, linewidth=1.0, shrinkB=3))
    fig.text(
        0.5, 0.02,
        "Both panels show the same raw signed geo-branch Ridge coefficient β of the W1 response (area-data design: masking "
        "an LSOA replaces all of its inputs with the training median; N=4,000, seed 42); only the colour scheme differs. "
        "Higher β = higher relative contribution to the coalition W1 response; the sign does NOT indicate whether predicted "
        f"crime rises or falls. No absolute values are taken. Scales run from the minimum β ({lo:+.2f}) to its "
        f"{CAP_PCT}th percentile ({cap:+.2f}); higher units are capped (arrow). Model sensitivity, not causal effects.",
        ha="center", fontsize=7.5, wrap=True,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    stem = DEST / f"geo_signed_pair_N4000{'' if annotate else '_plain'}"
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=250, facecolor="white")
    plt.close(fig)
    print(f"Saved {stem}.pdf / .png")


def main() -> None:
    gdf = load()
    print(f"{len(gdf)} LSOAs, beta {gdf['geo_coefficient'].min():+.3f} .. {gdf['geo_coefficient'].max():+.3f}, "
          f"{int((gdf['geo_coefficient'] < 0).sum())} negative")
    for code, _ in CALLOUTS.values():
        assert code in set(gdf["lsoa21cd"]), code
    draw(gdf, annotate=True)
    draw(gdf, annotate=False)


if __name__ == "__main__":
    main()
