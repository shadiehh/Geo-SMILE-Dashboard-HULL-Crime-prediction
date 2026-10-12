"""Area-data sensitivity: how sensitive is the model to the data it takes from each LSOA?

Alternative to the published GeoSensitivity (which masks only easting/northing
in a coordinates-only model). Here the model is the published 30-feature
Feature model, and for the selected LSOAs *all* of their input rows are
replaced with the training-median baseline. The response is the same as the
Geo branch: Wasserstein-1 between original and masked predictions on the
affected LSOA-months, attributed per LSOA with a kernel-weighted Ridge
surrogate. LSOA mask draws (seed, 5-10 LSOAs per perturbation) match
geo_smile_coefficients exactly.

Because every Feature input describes the LSOA itself (no neighbour
features), masking an LSOA only moves its own predictions. An exact per-LSOA
measure is therefore also reported (each LSOA masked alone), as a check on
the SMILE surrogate.

Reading: sensitivity = how atypical the area's data is relative to the Hull
median, weighted by how much the model relies on that data. Not causal.

Outputs go to outputs/area_data_sensitivity/.

    python compute_area_data_sensitivity.py [--perturb 4000] [--noise-seeds 43 44]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Patch
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from compute_geo_spatial_patterns import LISA_COLOURS, attach_boundaries, queen_weights, run_moran_lisa
from geosmile_hull_twobranch import (
    DIVERGING,
    ID_COLUMNS,
    INK,
    RANDOM_SEED,
    SEQUENTIAL,
    SPATIAL_COLUMNS,
    apply_kernel,
    choose_feature_players,
    make_model,
    metrics,
    shared_limit,
    temporal_split,
)
from hull_boundaries import hull_outline

ROOT = Path(__file__).resolve().parent
PANEL_PATH = ROOT / "outputs" / "hull_lsoa_month_panel_with_covariates.csv"
PUBLISHED_GEO = ROOT / "outputs" / "geo_coefficients.csv"
DEST = ROOT / "outputs" / "area_data_sensitivity"
Y_COL = "target_next_month_crime"
TOP_K = 20


def area_data_smile(
    model,
    panel: pd.DataFrame,
    test_mask: pd.Series,
    baselines: pd.Series,
    columns: list[str],
    n_perturb: int,
    seed: int,
    min_sel: int = 5,
    max_sel: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """Per-LSOA SMILE where masking an LSOA replaces ALL its input columns with baseline.

    Same draw order as geo_smile_coefficients, so mask patterns match it for the same seed.
    """
    frame = panel.loc[test_mask, ID_COLUMNS + columns].copy().reset_index(drop=True)
    x = frame[columns].astype(float)
    original = np.asarray(model.predict(x), float)
    lsoa_ids = frame["lsoa21cd"].astype(str).to_numpy()
    unique_lsoas = np.array(sorted(set(lsoa_ids.tolist())), dtype=object)
    n_lsoas = len(unique_lsoas)
    index_by_lsoa = {code: np.flatnonzero(lsoa_ids == code) for code in unique_lsoas}
    rng = np.random.default_rng(seed)
    M = np.zeros((n_perturb, n_lsoas), dtype=np.int8)
    responses = np.zeros(n_perturb, dtype=float)
    baseline_row = baselines[columns].to_numpy(dtype=float)

    for j in range(n_perturb):
        n_sel = int(rng.integers(min_sel, max_sel + 1))
        sel = rng.choice(n_lsoas, size=n_sel, replace=False)
        M[j, sel] = 1
        perturbed = x.copy()
        affected = np.concatenate([index_by_lsoa[unique_lsoas[idx]] for idx in sel])
        perturbed.iloc[affected, :] = baseline_row
        pert_pred = np.asarray(model.predict(perturbed), float)
        responses[j] = wasserstein_distance(original[affected], pert_pred[affected])

    distances = M.sum(axis=1).astype(float) / n_lsoas
    order = rng.permutation(n_perturb)
    a, b = int(0.6 * len(order)), int(0.8 * len(order))
    tr, va, te = order[:a], order[a:b], order[b:]
    sigma = max(float(np.median(distances[tr])), 1 / n_lsoas)
    scores, fitted = {}, {}
    for kernel in ("rbf", "laplace", "inverse"):
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(M[tr], responses[tr], sample_weight=apply_kernel(distances[tr], kernel, sigma))
        scores[kernel] = r2_score(responses[va], surrogate.predict(M[va])) if len(va) > 1 else np.nan
        fitted[kernel] = surrogate
    best = max({k: v for k, v in scores.items() if np.isfinite(v)}, key=scores.get, default="rbf")
    trva = np.concatenate([tr, va])
    final = Ridge(alpha=1.0)
    final.fit(M[trva], responses[trva], sample_weight=apply_kernel(distances[trva], best, sigma))
    test_r2 = r2_score(responses[te], final.predict(M[te])) if len(te) > 1 else np.nan
    coef = pd.DataFrame({"lsoa21cd": unique_lsoas.astype(str), "smile_coefficient": final.coef_})
    diag = {
        "best_kernel": best,
        "validation_r2_by_kernel": scores,
        "test_r2": float(test_r2),
        "n_perturbations": int(n_perturb),
        "response": "wasserstein_1_on_affected_lsoa_month_predictions",
        "perturbation": "replace ALL input columns of selected LSOAs with training median",
    }
    return coef.set_index("lsoa21cd"), diag


def exact_area_shift(model, panel: pd.DataFrame, test_mask: pd.Series, baselines: pd.Series, columns: list[str]) -> pd.DataFrame:
    """Mask each LSOA alone: W1 and signed mean shift on its own test rows."""
    frame = panel.loc[test_mask, ["lsoa21cd"] + columns].reset_index(drop=True)
    original = np.asarray(model.predict(frame[columns]), float)
    masked = frame[columns].astype(float)
    masked.loc[:, :] = baselines[columns].to_numpy(dtype=float)
    # Rows only depend on their own inputs, so one all-masked pass gives every LSOA's masked prediction.
    pert = np.asarray(model.predict(masked), float)
    rows = []
    for code, idx in frame.groupby("lsoa21cd").indices.items():
        rows.append(
            {
                "lsoa21cd": str(code),
                "exact_w1": wasserstein_distance(original[idx], pert[idx]),
                "exact_signed_shift": float(np.mean(original[idx] - pert[idx])),
                "mean_prediction": float(np.mean(original[idx])),
            }
        )
    return pd.DataFrame(rows).set_index("lsoa21cd")


def fit(panel: pd.DataFrame, columns: list[str], seed: int, split):
    train, validation, test = split
    model = make_model(seed)
    model.fit(
        panel.loc[train, columns],
        panel.loc[train, Y_COL],
        eval_set=[(panel.loc[validation, columns], panel.loc[validation, Y_COL])],
        verbose=False,
    )
    return model, metrics(panel.loc[test, Y_COL].to_numpy(float), model.predict(panel.loc[test, columns]))


def lisa_for(values: pd.Series, names: pd.Series) -> dict:
    geo = pd.DataFrame(
        {
            "lsoa21cd": values.index,
            "lsoa21nm": names.reindex(values.index).to_numpy(),
            "geo_coefficient": values.to_numpy(),
            "geo_abs_coefficient": values.abs().to_numpy(),
        }
    )
    gdf = attach_boundaries(geo)
    w, _ = queen_weights(gdf)
    order, _, global_m, _, labels = run_moran_lisa(gdf, w)
    lisa = pd.Series(labels, index=order)
    return {
        "moran_i": float(global_m.I),
        "p_value": float(global_m.p_sim),
        "lisa_counts": {k: int((lisa == k).sum()) for k in ("HH", "LL", "HL", "LH", "NS")},
        "lisa": lisa,
        "gdf": gdf,
    }


def plot_maps(gdf: gpd.GeoDataFrame, path: Path) -> None:
    gdf = gdf.to_crs("EPSG:27700")
    outline = hull_outline(gdf)
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    panels = [
        (
            "exact_w1",
            "Area-data sensitivity (crimes/month)\n30-feature model; each LSOA's data replaced by Hull median",
            False,
        ),
        (
            "published_geo_coefficient",
            "Published GeoSensitivity (signed coefficient)\ncoordinates-only model; higher = more sensitive",
            True,
        ),
        (
            "exact_signed_shift",
            "Direction: area prediction minus typical-area prediction\n(red = area data raises prediction)",
            True,
        ),
    ]
    for ax, (col, title, signed) in zip(axes.flat[:3], panels):
        vmax = shared_limit(gdf[col].to_numpy())
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax) if signed else Normalize(0, vmax)
        cmap = DIVERGING if signed else SEQUENTIAL
        gdf.plot(column=col, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.12, ax=ax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.01)
        ax.set_title(title, fontsize=11, color=INK)
    ax = axes.flat[3]
    for lab, colour in LISA_COLOURS.items():
        sub = gdf[gdf["lisa_area"] == lab]
        if not sub.empty:
            sub.plot(ax=ax, color=colour, edgecolor="white", linewidth=0.25, zorder=1 if lab == "NS" else 3)
    ax.legend(
        handles=[Patch(facecolor=LISA_COLOURS[k], label=k) for k in ("HH", "LL", "HL", "LH", "NS")],
        loc="lower left",
        fontsize=8,
        title="LISA",
    )
    ax.set_title("LISA of area-data sensitivity\n(queen contiguity, 999 permutations, p<0.05)", fontsize=11, color=INK)
    for ax in axes.flat:
        if outline is not None:
            gpd.GeoSeries([outline], crs=gdf.crs).plot(ax=ax, facecolor="none", edgecolor=INK, linewidth=0.9, zorder=5)
        ax.set_axis_off()
    fig.suptitle(
        "Hull: sensitivity of the model to each LSOA's data vs published GeoSensitivity. "
        "Colour scales capped at the 98th percentile. Not causal.",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_single(gdf: gpd.GeoDataFrame, column: str, title: str, cbar_label: str, path: Path, *, signed: bool) -> None:
    """Standalone choropleth in the style of the published maps_twobranch/ maps."""
    gdf = gdf.to_crs("EPSG:27700")
    vmax = shared_limit(gdf[column].to_numpy())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax) if signed else Normalize(0.0, vmax)
    cmap = DIVERGING if signed else SEQUENTIAL
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    gdf.plot(column=column, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.12, ax=ax)
    outline = hull_outline(gdf)
    if outline is not None:
        gpd.GeoSeries([outline], crs=gdf.crs).plot(ax=ax, color="none", edgecolor=INK, linewidth=1.0)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.68, pad=0.015)
    cbar.set_label(cbar_label, fontsize=9)
    fig.text(
        0.5,
        0.012,
        "30-feature model; each LSOA's inputs replaced by the Hull training median. "
        "Colour scale capped at the 98th percentile. Not causal effects.",
        ha="center",
        fontsize=7.0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_single_maps(frame: pd.DataFrame) -> None:
    boundaries = attach_boundaries(
        frame.reset_index()[["lsoa21cd", "lsoa21nm"]].assign(geo_coefficient=0.0, geo_abs_coefficient=0.0)
    )[["lsoa21cd", "geometry"]]
    gdf = boundaries.merge(frame, left_on="lsoa21cd", right_index=True)
    plot_single(
        gdf,
        "exact_signed_shift",
        "Hull — Area-data sensitivity (signed)",
        "Area prediction minus typical-area prediction (crimes/month)",
        DEST / "signed" / "area_data_sensitivity_signed.png",
        signed=True,
    )
    plot_single(
        gdf,
        "exact_w1",
        "Hull — Area-data sensitivity (absolute)",
        "Prediction shift when area data is masked (crimes/month)",
        DEST / "absolute" / "area_data_sensitivity_abs.png",
        signed=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perturb", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--noise-seeds", type=int, nargs="*", default=[43, 44])
    parser.add_argument("--maps-only", action="store_true", help="Redraw signed/absolute maps from the saved CSV.")
    args = parser.parse_args()

    if args.maps_only:
        saved = pd.read_csv(DEST / "area_data_sensitivity.csv", dtype={"lsoa21cd": str}).set_index("lsoa21cd")
        write_single_maps(saved)
        return

    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    split = temporal_split(panel)
    train, _, test = split
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
    names = panel[["lsoa21cd", "lsoa21nm"]].drop_duplicates("lsoa21cd").set_index("lsoa21cd")["lsoa21nm"]

    print(f"Feature model ({len(players)} inputs), seed {args.seed} ...", flush=True)
    model, holdout = fit(panel, players, args.seed, split)
    smile, diag = area_data_smile(model, panel, test, baseline, players, args.perturb, args.seed)
    exact = exact_area_shift(model, panel, test, baseline, players)
    noise_smile, noise_exact = [], []
    for s in args.noise_seeds:
        print(f"Noise floor: seed {s} (model retrained, masks redrawn) ...", flush=True)
        m_s, _ = fit(panel, players, s, split)
        noise_smile.append(area_data_smile(m_s, panel, test, baseline, players, args.perturb, s)[0]["smile_coefficient"])
        noise_exact.append(exact_area_shift(m_s, panel, test, baseline, players)["exact_w1"])

    # Every masked row gets identical inputs, so the masked prediction is one constant and the
    # signed SMILE coefficient is a linear function of exact_w1 (its sign only marks above/below
    # average). Rank and map by exact_w1 / signed coefficient, never by |coefficient|.
    published = pd.read_csv(PUBLISHED_GEO, dtype={"lsoa21cd": str}).set_index("lsoa21cd")["geo_coefficient"]
    frame = pd.DataFrame(index=smile.index)
    frame["lsoa21nm"] = names.reindex(frame.index)
    frame = frame.join(exact)
    frame["smile_coefficient"] = smile["smile_coefficient"]
    frame["published_geo_coefficient"] = published.reindex(frame.index)
    frame["rank_area_data"] = frame["exact_w1"].rank(ascending=False, method="first").astype(int)
    frame["rank_published_geo"] = frame["published_geo_coefficient"].rank(ascending=False, method="first").astype(int)
    frame["rank_shift_vs_published"] = frame["rank_area_data"] - frame["rank_published_geo"]
    runs = np.column_stack([frame["exact_w1"]] + [n.reindex(frame.index) for n in noise_exact])
    frame["noise_range_exact_w1"] = runs.max(axis=1) - runs.min(axis=1)

    moran = lisa_for(frame["exact_w1"], names)
    frame["lisa_area_data"] = moran.pop("lisa").reindex(frame.index)
    gdf = moran.pop("gdf").merge(
        frame[["exact_w1", "exact_signed_shift", "published_geo_coefficient", "lisa_area_data"]].rename(
            columns={"lisa_area_data": "lisa_area"}
        ),
        left_on="lsoa21cd",
        right_index=True,
    )

    DEST.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values("rank_area_data")
    frame.reset_index().to_csv(DEST / "area_data_sensitivity.csv", index=False)
    plot_maps(gdf, DEST / "area_data_sensitivity_maps.png")
    write_single_maps(frame)

    top_new = set(frame.nsmallest(TOP_K, "rank_area_data").index)
    top_pub = set(frame.nsmallest(TOP_K, "rank_published_geo").index)
    rho = lambda a, b: float(spearmanr(a, b).statistic)  # noqa: E731
    typical = float(np.median(frame["mean_prediction"] - frame["exact_signed_shift"]))
    summary = {
        "design": "30-feature Feature model; per-LSOA masking of ALL input columns to training median; "
        "W1 on affected LSOA-months. Exact per-LSOA shift (primary) plus SMILE surrogate "
        "(mask draws identical to the Geo branch).",
        "perturbations": args.perturb,
        "seed": args.seed,
        "holdout": holdout,
        "surrogate": diag,
        "typical_area_prediction": typical,
        "exact_w1_distribution": frame["exact_w1"].describe(percentiles=[0.5, 0.9]).to_dict(),
        "smile_signed_vs_exact_rank_spearman": rho(frame["smile_coefficient"], frame["exact_w1"]),
        "noise_floor_rank_spearman_exact": [rho(frame["exact_w1"], n.reindex(frame.index)) for n in noise_exact],
        "noise_floor_rank_spearman_smile": [rho(frame["smile_coefficient"], n.reindex(frame.index)) for n in noise_smile],
        "vs_published_geo_signed_rank_spearman": rho(frame["exact_w1"], frame["published_geo_coefficient"]),
        f"vs_published_top{TOP_K}_overlap": len(top_new & top_pub),
        "moran": moran,
        "note": "Not causal. Sensitivity = how far the area's predictions sit from the typical-area "
        "prediction when all its data is replaced by the Hull median.",
    }
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    cols = ["lsoa21nm", "exact_w1", "exact_signed_shift", "mean_prediction", "rank_published_geo", "lisa_area_data"]
    print(frame.head(10)[cols].round(2).to_string())
    print(frame.tail(5)[cols].round(2).to_string())


if __name__ == "__main__":
    main()
