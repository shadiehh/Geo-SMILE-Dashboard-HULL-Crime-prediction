"""Hull: location as a joint Feature player vs the area-data Geo branch (Chapter 4).

Hull counterpart of Chicago/Seattle chi_/sea_geo_sens_vs_location_effect.

Feature-B: an XGBoost with the published settings is fitted on the 30 non-spatial
predictors plus the LSOA centroid coordinates (BNG easting/northing). Feature SMILE is
run with 31 players: the 30 predictors and one JOINT location player whose mask
replaces both coordinates with their training medians. Local coefficients are fitted
per observation to signed prediction differences (original - perturbed), as for the
published local Feature maps, so the location coefficient is directional.

Geo branch: the published area-data Geo-SMILE coefficient (XGBoost, N=4,000, seed 42),
signed, drawn on one sequential blue-to-red scale (lowest to highest) because its sign is not a direction.

Writes outputs/location_player_vs_geo/: hull_geo_sens_vs_location_effect{,_plain}.{png,pdf},
location_player_local.csv, summary.json.

    python plot_location_player_vs_geo.py [--force]
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from geosmile_hull_twobranch import (
    DIVERGING,
    INK,
    RANDOM_SEED,
    SPATIAL_COLUMNS,
    apply_kernel,
    choose_feature_players,
    local_feature_coefficients,
    make_model,
    mask_pool,
    metrics,
    temporal_split,
)
from hull_boundaries import load_boundaries
from map_geo_importance_ranks import _draw_outline, _prepare_plot_gdf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "location_player_vs_geo"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"
GEO_COEF = OUT / "geo_area_data_eval" / "n_4000" / "geo_coefficients.csv"
PLACES = ROOT / "data" / "hull_lsoa_place_names.csv"
Y_COL = "target_next_month_crime"
N_MASKS = 4000
SEED = RANDOM_SEED
LOCATION = "location"
INK_MUTED = "#57646D"
GEO_LABEL = "Geo-SMILE geographic sensitivity coefficient"
# blue (lowest) -> red (highest), ordered from min to p98: sequential, zero is not a neutral midpoint
GEO_CMAP = plt.cm.colors.LinearSegmentedColormap.from_list(
    "geo_blue_red", ["#1E5AA8", "#7FA7D9", "#E9C4D6", "#FF0051", "#8A0033"])


def feature_b_local(force: bool) -> tuple[pd.DataFrame, dict]:
    path, meta_path = DEST / "location_player_local.csv", DEST / "summary.json"
    if path.exists() and meta_path.exists() and not force:
        return pd.read_csv(path, dtype={"lsoa21cd": str}), json.loads(meta_path.read_text(encoding="utf-8"))
    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    base_players = choose_feature_players(panel)
    inputs = base_players + SPATIAL_COLUMNS
    players = base_players + [LOCATION]
    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, inputs].median()
    model = make_model(SEED)
    model.fit(panel.loc[train, inputs], panel.loc[train, Y_COL],
              eval_set=[(panel.loc[validation, inputs], panel.loc[validation, Y_COL])], verbose=False)
    X = panel.loc[test].reset_index(drop=True)
    original = np.asarray(model.predict(X[inputs]), float)
    holdout = metrics(X[Y_COL].to_numpy(float), original)

    rng = np.random.default_rng(SEED)
    masks = mask_pool(N_MASKS, len(players), rng)
    responses = np.zeros(len(masks))
    all_pert = np.zeros((len(masks), len(X)), dtype=np.float32)
    for i, mask in enumerate(masks):
        pert = X[inputs].copy()
        for j in np.flatnonzero(mask == 0):
            cols = SPATIAL_COLUMNS if players[j] == LOCATION else [players[j]]
            for c in cols:  # the joint player masks both coordinates together
                pert[c] = baseline[c]
        pred = np.asarray(model.predict(pert), float)
        all_pert[i] = pred
        responses[i] = wasserstein_distance(original, pred)

    # kernel choice and global fidelity exactly as smile_global_feature
    distances = 1.0 - masks.mean(axis=1)
    order = rng.permutation(len(masks))
    a, b = int(0.6 * len(order)), int(0.8 * len(order))
    tr, va, te = order[:a], order[a:b], order[b:]
    sigma = max(float(np.median(distances[tr])), 1 / len(players))
    scores, fitted = {}, {}
    for kernel in ("rbf", "laplace", "inverse"):
        s = Ridge(alpha=1.0).fit(masks[tr], responses[tr], sample_weight=apply_kernel(distances[tr], kernel, sigma))
        scores[kernel], fitted[kernel] = r2_score(responses[va], s.predict(masks[va])), s
    best = max(scores, key=scores.get)
    glob = pd.Series(fitted[best].coef_, index=players)
    rank = int((glob.abs() > abs(glob[LOCATION])).sum() + 1)

    local = local_feature_coefficients(masks, all_pert, original, players, X["lsoa21cd"].astype(str).to_numpy(),
                                       best, SEED)
    local = local[["lsoa21cd", LOCATION]].rename(columns={LOCATION: "location_effect"})
    meta = {
        "design": "Feature-B: 30 predictors + joint location player (BNG centroid easting/northing masked together)",
        "n_masks": N_MASKS, "seed": SEED, "kernel": best,
        "holdout": holdout,
        "global_test_r2": float(r2_score(responses[te], fitted[best].predict(masks[te]))),
        "location_global_coefficient": float(glob[LOCATION]),
        "location_global_rank_of_31": rank,
    }
    DEST.mkdir(parents=True, exist_ok=True)
    local.to_csv(path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return local, meta


def place(code: str, places: pd.DataFrame) -> str:
    if code in places.index:
        r = places.loc[code]
        for col in ("locality_label",):
            if col in places.columns and isinstance(r.get(col), str) and r.get(col):
                return r[col]
    return code


def figure(local: pd.DataFrame, meta: dict, annotate: bool) -> dict:
    geo = pd.read_csv(GEO_COEF, dtype={"lsoa21cd": str})
    geo = geo.sort_values(["geo_coefficient", "lsoa21cd"], ascending=[False, True]).reset_index(drop=True)
    geo["geo_rank"] = np.arange(1, len(geo) + 1)
    d = local.merge(geo[["lsoa21cd", "geo_coefficient", "geo_rank"]], on="lsoa21cd", validate="one_to_one")
    rho = float(spearmanr(d["location_effect"], d["geo_coefficient"]).statistic)
    rho_abs = float(spearmanr(d["location_effect"].abs(), d["geo_coefficient"]).statistic)

    boundaries, _ = load_boundaries(sorted(d["lsoa21cd"]))
    gdf = _prepare_plot_gdf(boundaries[["lsoa21cd", "geometry"]].drop_duplicates("lsoa21cd")).merge(d, on="lsoa21cd")
    places = pd.read_csv(PLACES, dtype={"lsoa21cd": str}).set_index("lsoa21cd") if PLACES.exists() else pd.DataFrame()
    lo, hi = d.loc[d["location_effect"].idxmin()], d.loc[d["location_effect"].idxmax()]
    top = d.loc[d["geo_rank"] == 1].iloc[0]

    lim = float(np.abs(d["location_effect"]).max())
    loc_norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)  # full range, nothing clipped
    geo_norm = Normalize(float(d["geo_coefficient"].min()), float(np.percentile(d["geo_coefficient"], 98)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    for ax, col, cmap, norm, title, label in (
            (axes[0], "location_effect", DIVERGING, loc_norm,
             "Feature-B: joint location player\n(signed local attribution)", "Signed local Feature-SMILE coefficient"),
            (axes[1], "geo_coefficient", GEO_CMAP, geo_norm,
             "Geo branch, area-data design\n(per-LSOA geographic sensitivity)", GEO_LABEL)):
        gdf.plot(column=col, cmap=cmap, norm=norm, edgecolor="#d9d9d9", linewidth=0.12, ax=ax)
        _draw_outline(ax, gdf)
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK)
        ax.set_axis_off()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.01)
        cb.set_label(label, fontsize=9, color=INK_MUTED)
        if col == "geo_coefficient":
            cb.ax.text(0.5, 1.02, f"max {d['geo_coefficient'].max():.1f}", transform=cb.ax.transAxes,
                       ha="center", va="bottom", fontsize=7.5, color=INK_MUTED)

    if annotate:
        box = dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#C9CED3", alpha=0.93)
        arrow = dict(arrowstyle="-|>", color=INK, linewidth=1.0, connectionstyle="arc3,rad=0.15", shrinkB=4)
        pts = gdf.set_index("lsoa21cd").geometry.representative_point()
        for ax, row, text, xy in (
                (axes[0], lo, f"{place(lo['lsoa21cd'], places)} ({lo['lsoa21cd']})\nlocation effect "
                              f"{lo['location_effect']:+.2f}, the lowest", (0.02, 0.93)),
                (axes[0], hi, f"{place(hi['lsoa21cd'], places)} ({hi['lsoa21cd']})\nlocation effect "
                              f"{hi['location_effect']:+.2f}, the highest", (0.55, 0.06)),
                (axes[1], top, f"{place(top['lsoa21cd'], places)} ({top['lsoa21cd']})\nGeo rank 1 of 168 "
                               f"({top['geo_coefficient']:+.2f}); location\neffect {top['location_effect']:+.2f}",
                 (0.02, 0.93))):
            p = pts[row["lsoa21cd"]]
            ax.annotate(text, xy=(p.x, p.y), xycoords="data", xytext=xy, textcoords="axes fraction",
                        va="top" if xy[1] > 0.5 else "bottom", fontsize=8.5, color=INK, bbox=box, arrowprops=arrow)

    fig.suptitle("Hull: location as a joint feature player vs the Geo branch", fontsize=14, fontweight="bold",
                 color=INK)
    fig.text(0.5, 0.01, textwrap.fill(
        f"168 LSOAs, XGBoost, test period. Left: Feature-B SMILE with 31 players (30 predictors + one joint "
        f"location player masking both centroid coordinates; {meta['n_masks']:,} masks, seed {meta['seed']}); local "
        "coefficients are fitted to signed prediction differences, so red/blue are opposite directions. Right: "
        "area-data Geo-SMILE coefficient (N=4,000), one sequential blue-to-red scale from lowest to highest (zero is not a midpoint) because its sign is not a direction. "
        f"Spearman {rho:+.2f} signed; {rho_abs:+.2f} between |location effect| and the Geo coefficient. "
        "Not causal.", 150), ha="center", va="bottom", fontsize=8, color=INK_MUTED)
    fig.subplots_adjust(left=0.01, right=0.97, top=0.86, bottom=0.1, wspace=0.08)
    suffix = "" if annotate else "_plain"
    for ext in ("png", "pdf"):
        fig.savefig(DEST / f"hull_geo_sens_vs_location_effect{suffix}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)
    return {"spearman_signed": rho, "spearman_abs_location": rho_abs,
            "lowest_location": [lo["lsoa21cd"], float(lo["location_effect"]), int(lo["geo_rank"])],
            "highest_location": [hi["lsoa21cd"], float(hi["location_effect"]), int(hi["geo_rank"])],
            "geo_rank1_location_effect": float(top["location_effect"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    local, meta = feature_b_local(args.force)
    stats = figure(local, meta, annotate=True)
    figure(local, meta, annotate=False)
    meta.update(stats)
    (DEST / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
