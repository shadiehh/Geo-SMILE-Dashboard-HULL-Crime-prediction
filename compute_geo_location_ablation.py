"""GeoSensitivity ablation: coordinates alone vs coordinates alongside the 30 Feature inputs.

GeoSensitivity needs location, so the ablation is on the *other* inputs:

  - geo_only: the published Geo model (easting/northing only);
  - combined: one model trained on the 30 Feature players + easting/northing,
    with only the coordinates masked per LSOA (feature values held at truth).

Both use the same panel, split, XGBoost settings, seed, number of
perturbations, and baselines. Because n_lsoas, min/max selection and seed are
identical, both runs draw the *same* LSOA masking pattern, so differences come
from the model, not from the perturbation sample. Noise floors rerun
both configurations with extra seeds (model and perturbations reseeded).

Outputs go to outputs/location_ablation/geo/. Not causal.

    python compute_geo_location_ablation.py [--perturb 4000] [--noise-seeds 43 44]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from combined_model_fidelity import combined_geo_smile
from compute_geo_spatial_patterns import attach_boundaries, queen_weights, run_moran_lisa
from geosmile_hull_twobranch import (
    RANDOM_SEED,
    SPATIAL_COLUMNS,
    choose_feature_players,
    geo_smile_coefficients,
    make_model,
    metrics,
    temporal_split,
)

ROOT = Path(__file__).resolve().parent
PANEL_PATH = ROOT / "outputs" / "hull_lsoa_month_panel_with_covariates.csv"
DEST = ROOT / "outputs" / "location_ablation" / "geo"
Y_COL = "target_next_month_crime"
TOP_K = 20


def fit(panel: pd.DataFrame, columns: list[str], seed: int, split):
    train, validation, test = split
    model = make_model(seed)
    model.fit(
        panel.loc[train, columns],
        panel.loc[train, Y_COL],
        eval_set=[(panel.loc[validation, columns], panel.loc[validation, Y_COL])],
        verbose=False,
    )
    holdout = metrics(panel.loc[test, Y_COL].to_numpy(float), model.predict(panel.loc[test, columns]))
    return model, holdout


def ranks(values: pd.Series) -> pd.Series:
    return values.rank(ascending=False, method="first").astype(int)


def moran(coef: pd.DataFrame) -> dict:
    gdf = attach_boundaries(coef.rename(columns={"geo_abs_coefficient": "geo_abs_coefficient"}))
    w, _ = queen_weights(gdf)
    order, _, global_m, _, labels = run_moran_lisa(gdf, w)
    lisa = pd.Series(labels, index=order)
    return {
        "moran_i": float(global_m.I),
        "p_value": float(global_m.p_sim),
        "lisa_counts": {k: int((lisa == k).sum()) for k in ("HH", "LL", "HL", "LH", "NS")},
        "lisa": lisa,
    }


def plot(frame: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4))
    x, y = frame["abs_geo_only"], frame["abs_combined"]
    ax1.scatter(x, y, s=16, color="#256abf", alpha=0.8)
    lim = [min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.2]
    ax1.plot(lim, lim, color="#57646D", lw=0.8, ls="--")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("|GeoSensitivity|, coordinates only")
    ax1.set_ylabel("|GeoSensitivity|, coordinates + 30 features")
    ax1.set_title("Per-LSOA GeoSensitivity (log-log; dashed = no change)", fontsize=10)
    ax2.scatter(frame["rank_geo_only"], frame["rank_combined"], s=16, color="#FF0051", alpha=0.8)
    ax2.plot([1, len(frame)], [1, len(frame)], color="#57646D", lw=0.8, ls="--")
    ax2.set_xlabel("Rank, coordinates only")
    ax2.set_ylabel("Rank, coordinates + 30 features")
    ax2.set_title("Rank movement (1 = most sensitive)", fontsize=10)
    fig.suptitle("GeoSensitivity with vs without the other Feature inputs. Not causal.", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perturb", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--noise-seeds", type=int, nargs="*", default=[43, 44])
    args = parser.parse_args()

    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    split = temporal_split(panel)
    train, _, test = split
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()

    def geo_only_run(seed: int):
        model, holdout = fit(panel, SPATIAL_COLUMNS, seed, split)
        coef, diag = geo_smile_coefficients(model, panel, test, baseline, args.perturb, seed)
        return coef.set_index("lsoa21cd"), diag, holdout

    print(f"Geo only (coordinates), seed {args.seed} ...", flush=True)
    geo_only, geo_diag, geo_holdout = geo_only_run(args.seed)
    print(f"Combined ({len(players)} features + coordinates), seed {args.seed} ...", flush=True)
    model_c, comb_holdout = fit(panel, players + SPATIAL_COLUMNS, args.seed, split)
    combined, comb_diag, *_ = combined_geo_smile(
        model_c, panel, test, baseline, players, args.perturb, args.seed
    )
    combined = combined.set_index("lsoa21cd")
    noise, noise_comb = [], []
    for s in args.noise_seeds:
        print(f"Noise floor: geo only, seed {s} ...", flush=True)
        noise.append(geo_only_run(s)[0])
        print(f"Noise floor: combined, seed {s} ...", flush=True)
        model_s, _ = fit(panel, players + SPATIAL_COLUMNS, s, split)
        noise_comb.append(
            combined_geo_smile(model_s, panel, test, baseline, players, args.perturb, s)[0].set_index("lsoa21cd")
        )

    names = panel[["lsoa21cd", "lsoa21nm"]].drop_duplicates("lsoa21cd").set_index("lsoa21cd")["lsoa21nm"]
    idx = geo_only.index
    frame = pd.DataFrame(
        {
            "lsoa21nm": names.reindex(idx).to_numpy(),
            "geo_only": geo_only["geo_coefficient"],
            "combined": combined.loc[idx, "geo_coefficient"],
        },
        index=idx,
    )
    frame["abs_geo_only"] = frame["geo_only"].abs()
    frame["abs_combined"] = frame["combined"].abs()
    frame["rank_geo_only"] = ranks(frame["abs_geo_only"])
    frame["rank_combined"] = ranks(frame["abs_combined"])
    frame["rank_shift"] = frame["rank_combined"] - frame["rank_geo_only"]
    frame["delta_abs"] = frame["abs_combined"] - frame["abs_geo_only"]
    frame["ratio_abs"] = frame["abs_combined"] / frame["abs_geo_only"]
    frame["sign_flipped"] = np.sign(frame["geo_only"]) != np.sign(frame["combined"])
    runs = np.column_stack([frame["abs_geo_only"]] + [n.loc[idx, "geo_abs_coefficient"] for n in noise])
    frame["noise_abs_range"] = runs.max(axis=1) - runs.min(axis=1)
    frame["change_exceeds_noise"] = frame["delta_abs"].abs() > frame["noise_abs_range"]
    runs_c = np.column_stack([frame["abs_combined"]] + [n.loc[idx, "geo_abs_coefficient"] for n in noise_comb])
    frame["noise_abs_range_combined"] = runs_c.max(axis=1) - runs_c.min(axis=1)

    m_geo = moran(frame[["abs_geo_only"]].rename(columns={"abs_geo_only": "geo_abs_coefficient"}).assign(
        geo_coefficient=frame["geo_only"], lsoa21nm=frame["lsoa21nm"]).reset_index())
    m_comb = moran(frame[["abs_combined"]].rename(columns={"abs_combined": "geo_abs_coefficient"}).assign(
        geo_coefficient=frame["combined"], lsoa21nm=frame["lsoa21nm"]).reset_index())
    frame["lisa_geo_only"] = m_geo.pop("lisa").reindex(idx)
    frame["lisa_combined"] = m_comb.pop("lisa").reindex(idx)

    DEST.mkdir(parents=True, exist_ok=True)
    frame = frame.sort_values("rank_geo_only")
    frame.reset_index().to_csv(DEST / "geo_sensitivity_comparison.csv", index=False)
    plot(frame, DEST / "geo_sensitivity_ablation.png")

    top_a = set(frame.nsmallest(TOP_K, "rank_geo_only").index)
    top_b = set(frame.nsmallest(TOP_K, "rank_combined").index)

    def dist(col: str) -> dict:
        v = frame[col]
        return {"mean": v.mean(), "median": v.median(), "p90": v.quantile(0.9), "max": v.max(), "sum": v.sum()}

    summary = {
        "design": "Same panel, split, XGBoost settings, seed, perturbations, baselines and LSOA mask draws; "
        "only difference is whether the model also sees the 30 Feature inputs.",
        "perturbations": args.perturb,
        "seed": args.seed,
        "noise_seeds": args.noise_seeds,
        "holdout": {"geo_only": geo_holdout, "combined": comb_holdout},
        "surrogate_test_r2": {"geo_only": geo_diag["test_r2"], "combined": comb_diag["test_r2"]},
        "abs_distribution": {"geo_only": dist("abs_geo_only"), "combined": dist("abs_combined")},
        "median_ratio_combined_to_geo_only": float(frame["ratio_abs"].median()),
        "rank_spearman": float(spearmanr(frame["abs_geo_only"], frame["abs_combined"]).statistic),
        "signed_pearson": float(np.corrcoef(frame["geo_only"], frame["combined"])[0, 1]),
        # Noise floors reseed both the model and the perturbations.
        "noise_floor_rank_spearman_geo_only": [
            float(spearmanr(frame["abs_geo_only"], n.loc[frame.index, "geo_abs_coefficient"]).statistic)
            for n in noise
        ],
        "noise_floor_rank_spearman_combined": [
            float(spearmanr(frame["abs_combined"], n.loc[frame.index, "geo_abs_coefficient"]).statistic)
            for n in noise_comb
        ],
        "noise_top1_combined": [n["geo_abs_coefficient"].idxmax() for n in noise_comb],
        "noise_max_abs_combined": [float(n["geo_abs_coefficient"].max()) for n in noise_comb],
        f"top{TOP_K}_overlap": len(top_a & top_b),
        f"top{TOP_K}_jaccard": len(top_a & top_b) / len(top_a | top_b),
        "top1": {"geo_only": frame.index[0], "combined": frame["abs_combined"].idxmax()},
        "n_sign_flips": int(frame["sign_flipped"].sum()),
        "n_change_exceeds_noise": int(frame["change_exceeds_noise"].sum()),
        "moran": {"geo_only": m_geo, "combined": m_comb},
        "lisa_class_changed": int((frame["lisa_geo_only"] != frame["lisa_combined"]).sum()),
        "note": "Not causal. Combined-model GeoSensitivity is conditional on true feature values.",
    }
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
