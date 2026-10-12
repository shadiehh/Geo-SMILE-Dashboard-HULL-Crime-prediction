"""Visualize GP's already-computed ATT stability (4 cardinal-neighbour LSOA
directions) and ATT consistency (10 SMILE reseeds, R=10) metrics --
NO recomputation. Both were already produced by verify_metrics_all_models.py
and live in outputs/verify_metrics_all_models/gp/results.json; this only
reads that file and plots it, matching the project's existing chart style
(compute_att_stability.py's palette/savefig conventions).

Complements outputs/top_maps/figures/{geo_top20_gp,feature_top10_gp}.png
(already produced by generate_top_maps.py) -- together these are the full
set of GP visuals: coefficient maps/rankings plus stability/consistency.

Writes outputs/top_maps/figures/gp_stability_directions.png and
outputs/top_maps/figures/gp_consistency_r10.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, PINK = "#1E88E5", "#FF0051"
INK, INK_MUTED = "#12171C", "#57646D"
MUTED_BAR = "#B7C0C8"

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "outputs" / "verify_metrics_all_models" / "gp" / "results.json"
FIG_DIR = ROOT / "outputs" / "top_maps" / "figures"

DIRECTIONS = ("north", "east", "south", "west")


def plot_stability(neighbors: dict, path: Path) -> None:
    labels = [neighbors[d]["lsoa21nm"] for d in DIRECTIONS]
    feat_jaccard = [neighbors[d]["feature_jaccard"] for d in DIRECTIONS]
    feat_spearman = [neighbors[d]["feature_spearman"] for d in DIRECTIONS]
    geo_jaccard = [neighbors[d]["geo_jaccard"] for d in DIRECTIONS]
    geo_spearman = [neighbors[d]["geo_spearman"] for d in DIRECTIONS]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(DIRECTIONS))
    width = 0.32

    for ax, jaccard, spearman, title in (
        (axes[0], feat_jaccard, feat_spearman, "Feature branch"),
        (axes[1], geo_jaccard, geo_spearman, "Geo branch"),
    ):
        ax.bar(x - width / 2, jaccard, width, label="Top-$k$ Jaccard", color=BLUE)
        ax.bar(x + width / 2, spearman, width, label="|coef| Spearman $\\rho$", color=PINK)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d.capitalize()}\n{lbl}" for d, lbl in zip(DIRECTIONS, labels)], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=11, fontweight="bold", color=INK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Metric value", fontsize=10, color=INK_MUTED)
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("GP: cardinal-neighbour ATT stability (add one real LSOA per direction)",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.01,
              "GeoSMILE stability check, seed=42, Feature N=1,024 / Geo N=4,000. "
              "Each direction adds one real neighbouring LSOA to the reference set; "
              "metrics compare the resulting attribution to the unperturbed baseline.",
              ha="center", fontsize=7.0, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)


def plot_consistency(consistency: dict, path: Path) -> None:
    branches = ("feature", "geo")
    branch_labels = ("Feature", "Geo")
    metric_keys = ("Rank Spearman", "Top-20% Jaccard")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(metric_keys))
    width = 0.32

    for i, (branch, label, color) in enumerate(zip(branches, branch_labels, (BLUE, PINK))):
        means = [consistency[branch][f"{k} Mean"] for k in metric_keys]
        stds = [consistency[branch][f"{k} Std"] for k in metric_keys]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value (mean $\\pm$ SD across reseeds)", fontsize=10, color=INK_MUTED)
    ax.set_title("GP: ATT consistency across $R=10$ SMILE reseeds", fontsize=12.5, fontweight="bold", color=INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    fig.text(0.5, 0.01,
              "10 independent SMILE reseeds, seed=42..51, Feature N=1,024 / Geo N=4,000. "
              "Pairwise rank-Spearman / top-20% Jaccard across all reseed pairs, mean ± SD shown.",
              ha="center", fontsize=7.0, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"Missing {RESULTS_PATH}")
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    stability_path = FIG_DIR / "gp_stability_directions.png"
    plot_stability(results["metric3_att_stability"]["neighbors"], stability_path)
    print(f"Saved: {stability_path}")

    consistency_path = FIG_DIR / "gp_consistency_r10.png"
    plot_consistency(results["metric4_att_consistency"], consistency_path)
    print(f"Saved: {consistency_path}")


if __name__ == "__main__":
    main()
