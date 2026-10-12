"""Location ablation: how non-location Feature importances change when location is added.

Runs two Feature-branch configurations under an identical design (same panel,
temporal split, XGBoost settings, seed, mask count, medians as baselines):

  - without location: the published 30 non-spatial Feature players;
  - with location:    the same 30 players plus centroid longitude and latitude.

The already-saved artefacts are not compared directly, because they differ in
more than location (published Feature uses 4000 masks on the current panel;
outputs/feature_with_location/ used 1024 masks on an older panel).

A noise floor is estimated by rerunning the without-location configuration
with extra seeds (model + masks reseeded). A change is flagged only when it
exceeds that seed-to-seed spread.

Outputs go to outputs/location_ablation/. Not causal.

    python compute_location_ablation.py [--masks 4000] [--noise-seeds 43 44]
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

from geosmile_hull_feature_with_location import LOCATION_PLAYERS, add_lonlat
from geosmile_hull_twobranch import (
    FEATURE_LABELS,
    RANDOM_SEED,
    choose_feature_players,
    local_feature_coefficients,
    make_model,
    metrics,
    smile_global_feature,
    temporal_split,
)

ROOT = Path(__file__).resolve().parent
PANEL_PATH = ROOT / "outputs" / "hull_lsoa_month_panel_with_covariates.csv"
DEST = ROOT / "outputs" / "location_ablation"
Y_COL = "target_next_month_crime"
TOP_K = 10


def _label(name: str) -> str:
    return {"centroid_longitude": "Longitude", "centroid_latitude": "Latitude"}.get(
        name, FEATURE_LABELS.get(name, name.replace("_", " "))
    )


def run_branch(panel: pd.DataFrame, players: list[str], n_masks: int, seed: int, local: bool) -> dict:
    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, players].median()
    model = make_model(seed)
    model.fit(
        panel.loc[train, players],
        panel.loc[train, Y_COL],
        eval_set=[(panel.loc[validation, players], panel.loc[validation, Y_COL])],
        verbose=False,
    )
    X_test = panel.loc[test].reset_index(drop=True)
    holdout = metrics(X_test[Y_COL].to_numpy(float), model.predict(X_test[players]))
    importance, diag, masks, all_pert, original = smile_global_feature(
        model, X_test[players], players, baseline, n_masks, seed
    )
    out = {"importance": importance.set_index("feature"), "diag": diag, "holdout": holdout}
    if local:
        out["local"] = local_feature_coefficients(
            masks, all_pert, original, players, X_test["lsoa21cd"].to_numpy(), diag["best_kernel"], seed
        ).set_index("lsoa21cd")
    return out


def ranks(abs_values: pd.Series) -> pd.Series:
    order = abs_values.sort_values(ascending=False, kind="mergesort")
    return pd.Series(np.arange(1, len(order) + 1), index=order.index)


def global_comparison(no_loc: dict, with_loc: dict, noise: list[dict], base: list[str]) -> pd.DataFrame:
    a = no_loc["importance"].loc[base]
    b = with_loc["importance"]
    b_base = b.loc[base]
    frame = pd.DataFrame(
        {
            "feature": base,
            "feature_label": [_label(f) for f in base],
            "coef_without_location": a["coefficient"].to_numpy(),
            "coef_with_location": b_base["coefficient"].to_numpy(),
            "abs_without_location": a["abs_coefficient"].to_numpy(),
            "abs_with_location": b_base["abs_coefficient"].to_numpy(),
        }
    ).set_index("feature")
    frame["rank_without_location"] = ranks(frame["abs_without_location"])
    # Rank among all 32 players, and among the 30 shared players only.
    frame["rank_with_location_all32"] = ranks(b["abs_coefficient"]).loc[base]
    frame["rank_with_location_among30"] = ranks(frame["abs_with_location"])
    frame["rank_shift_among30"] = frame["rank_with_location_among30"] - frame["rank_without_location"]
    frame["delta_abs"] = frame["abs_with_location"] - frame["abs_without_location"]
    frame["delta_abs_pct"] = 100 * frame["delta_abs"] / frame["abs_without_location"]
    frame["sign_flipped"] = np.sign(frame["coef_with_location"]) != np.sign(frame["coef_without_location"])
    frame["share_without_location"] = frame["abs_without_location"] / frame["abs_without_location"].sum()
    frame["share_with_location_among30"] = frame["abs_with_location"] / frame["abs_with_location"].sum()
    frame["delta_share_pp"] = 100 * (frame["share_with_location_among30"] - frame["share_without_location"])
    if noise:
        runs = np.column_stack(
            [frame["abs_without_location"].to_numpy()]
            + [n["importance"].loc[base, "abs_coefficient"].to_numpy() for n in noise]
        )
        frame["noise_abs_range"] = runs.max(axis=1) - runs.min(axis=1)
        frame["change_exceeds_noise"] = frame["delta_abs"].abs() > frame["noise_abs_range"]
    return frame.sort_values("rank_without_location").reset_index()


def local_comparison(no_loc: pd.DataFrame, with_loc: pd.DataFrame, base: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    lsoas = no_loc.index.intersection(with_loc.index)
    a = no_loc.loc[lsoas, base]
    b = with_loc.loc[lsoas, base]
    per_lsoa = pd.DataFrame(index=lsoas)
    per_lsoa["rank_spearman_abs"] = [
        spearmanr(a.loc[i].abs(), b.loc[i].abs()).statistic for i in lsoas
    ]
    per_lsoa["dominant_without_location"] = a.abs().idxmax(axis=1)
    per_lsoa["dominant_with_location_among30"] = b.abs().idxmax(axis=1)
    per_lsoa["dominant_changed"] = per_lsoa["dominant_without_location"] != per_lsoa["dominant_with_location_among30"]
    loc_abs = with_loc.loc[lsoas, LOCATION_PLAYERS].abs().sum(axis=1)
    per_lsoa["location_share_of_total_abs"] = loc_abs / (loc_abs + b.abs().sum(axis=1))
    per_lsoa["mean_abs_delta"] = (b.abs() - a.abs()).abs().mean(axis=1)
    per_feature = pd.DataFrame(
        {
            "feature": base,
            "feature_label": [_label(f) for f in base],
            "mean_local_abs_without_location": a.abs().mean().to_numpy(),
            "mean_local_abs_with_location": b.abs().mean().to_numpy(),
            "mean_abs_local_delta": (b - a).abs().mean().to_numpy(),
            "cross_lsoa_pearson": [np.corrcoef(a[f], b[f])[0, 1] for f in base],
        }
    )
    return per_lsoa.reset_index(), per_feature.sort_values("mean_local_abs_without_location", ascending=False)


def plot(frame: pd.DataFrame, path: Path) -> None:
    top = frame.head(15).iloc[::-1]
    y = np.arange(len(top))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
    ax1.barh(y + 0.2, top["abs_without_location"], height=0.4, color="#B7C0C8", label="Without location")
    ax1.barh(y - 0.2, top["abs_with_location"], height=0.4, color="#FF0051", label="With location")
    ax1.set_xscale("log")
    ax1.set_yticks(y)
    ax1.set_yticklabels(top["feature_label"], fontsize=9)
    ax1.set_xlabel("|global SMILE coefficient| (log scale)")
    ax1.legend(frameon=False, fontsize=9)
    colors = ["#FF0051" if v < 0 else "#256abf" for v in top["delta_abs_pct"]]
    ax2.barh(y, top["delta_abs_pct"], color=colors)
    if "noise_abs_range" in top:
        noise_pct = 100 * top["noise_abs_range"] / top["abs_without_location"]
        ax2.errorbar(np.zeros(len(top)), y, xerr=noise_pct, fmt="none", ecolor="#57646D", capsize=3, lw=1)
    ax2.axvline(0, color="#12171C", lw=0.8)
    ax2.set_xlabel("% change in |coefficient| when location is added\n(grey bar: seed-to-seed noise range)")
    fig.suptitle("Top 15 non-location features: importance with vs without location players. Not causal.", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--noise-seeds", type=int, nargs="*", default=[43, 44])
    args = parser.parse_args()

    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    panel = add_lonlat(panel)
    base = choose_feature_players(panel.drop(columns=LOCATION_PLAYERS))
    with_players = base + LOCATION_PLAYERS

    print(f"Without location ({len(base)} players), seed {args.seed} ...", flush=True)
    no_loc = run_branch(panel, base, args.masks, args.seed, local=True)
    print(f"With location ({len(with_players)} players), seed {args.seed} ...", flush=True)
    with_loc = run_branch(panel, with_players, args.masks, args.seed, local=True)
    noise = []
    for s in args.noise_seeds:
        print(f"Noise floor: without location, seed {s} ...", flush=True)
        noise.append(run_branch(panel, base, args.masks, s, local=False))

    DEST.mkdir(parents=True, exist_ok=True)
    frame = global_comparison(no_loc, with_loc, noise, base)
    frame.to_csv(DEST / "global_importance_comparison.csv", index=False)
    per_lsoa, per_feature = local_comparison(no_loc["local"], with_loc["local"], base)
    per_lsoa.to_csv(DEST / "local_per_lsoa_comparison.csv", index=False)
    per_feature.to_csv(DEST / "local_per_feature_comparison.csv", index=False)
    plot(frame, DEST / "location_ablation_top15.png")

    loc_imp = with_loc["importance"].loc[LOCATION_PLAYERS]
    all_abs = with_loc["importance"]["abs_coefficient"]
    noise_rhos = [
        spearmanr(no_loc["importance"].loc[base, "abs_coefficient"], n["importance"].loc[base, "abs_coefficient"]).statistic
        for n in noise
    ]
    top_a = set(frame.nsmallest(TOP_K, "rank_without_location")["feature"])
    top_b = set(frame.nsmallest(TOP_K, "rank_with_location_among30")["feature"])
    summary = {
        "design": "Same panel, split, XGBoost settings, seed, masks; only difference is the two location players.",
        "masks": args.masks,
        "seed": args.seed,
        "noise_seeds": args.noise_seeds,
        "holdout": {"without_location": no_loc["holdout"], "with_location": with_loc["holdout"]},
        "surrogate_test_r2": {
            "without_location": no_loc["diag"]["test_r2"],
            "with_location": with_loc["diag"]["test_r2"],
        },
        "location_players": {
            f: {
                "coefficient": float(loc_imp.loc[f, "coefficient"]),
                "rank_of_32": int(ranks(all_abs).loc[f]),
            }
            for f in LOCATION_PLAYERS
        },
        "location_share_of_total_abs": float(loc_imp["abs_coefficient"].sum() / all_abs.sum()),
        "global_rank_spearman_30_shared": float(
            spearmanr(frame["abs_without_location"], frame["abs_with_location"]).statistic
        ),
        "noise_floor_rank_spearman": [float(r) for r in noise_rhos],
        f"top{TOP_K}_jaccard": len(top_a & top_b) / len(top_a | top_b),
        "n_features_change_exceeds_noise": int(frame.get("change_exceeds_noise", pd.Series(dtype=bool)).sum()),
        "n_sign_flips": int(frame["sign_flipped"].sum()),
        "local_rank_spearman_median": float(per_lsoa["rank_spearman_abs"].median()),
        "local_dominant_feature_changed_lsoas": int(per_lsoa["dominant_changed"].sum()),
        "local_location_share_median": float(per_lsoa["location_share_of_total_abs"].median()),
        "note": "Not causal. Changes include SMILE sampling noise; compare against noise_abs_range.",
    }
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
