"""Chapter-4 five-diagnostic N-sweep figure with the area-data Geo branch.

Feature series: the published N=4,000-design sweep values (Chapter 4, Table
"Five evaluation diagnostics across independent perturbation budgets"); the
Feature branch is unchanged by the Geo redesign. Geo series: read from
outputs/geo_area_data_eval/summary_by_n.csv.
Writes outputs/geo_area_data_eval/hull_metrics_n_sweep_area_data.{png,pdf}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "outputs" / "geo_area_data_eval"
FEATURE = pd.DataFrame(
    {
        "N": [1000, 2000, 3000, 4000, 5000, 7500, 10000],
        "fidelity_r2": [0.9764, 0.9774, 0.9768, 0.9770, 0.9766, 0.9771, 0.9753],
        "accuracy": [6 / 10, 7 / 10, 7 / 10, 6 / 10, 7 / 10, 6 / 10, 6 / 10],
        "faithfulness": [4.229, 4.280, 4.262, 4.246, 4.279, 4.389, 4.389],
        "stability": [1.0] * 7,
        "consistency_sd": [0.03033, 0.02183, 0.01588, 0.01459, 0.01305, 0.01082, 0.00914],
        "consistency_var": [0.000963, 0.000510, 0.000264, 0.000227, 0.000180, 0.000124, 0.000089],
    }
)


REGIONS = ROOT / "outputs" / "geo_area_data_models" / "top50_maps" / "highlighted_regions_lsoas.csv"


def region_accuracy(n: int, k: int = 20) -> float:
    """Geo ATT accuracy as in Chapter 4: share of the top-k inside the 58 expert-region LSOAs."""
    ref = set(pd.read_csv(REGIONS, dtype={"lsoa21cd": str})["lsoa21cd"])
    c = pd.read_csv(DEST / f"n_{n}" / "geo_coefficients.csv", dtype={"lsoa21cd": str})
    top = c.sort_values(["geo_coefficient", "lsoa21cd"], ascending=[False, True]).head(k)["lsoa21cd"]
    return len(set(top) & ref) / k


def main() -> None:
    # XGBoost only. Panel 4 is the intervention test (top-k minus random-set W1), not ATT
    # Faithfulness: ATT Faithfulness is a cross-model correlation and cannot be swept for one model.
    g = pd.read_csv(DEST / "summary_by_n.csv")
    geo = pd.DataFrame(
        {
            "N": g["N"],
            "fidelity_r2": g["fidelity_r2"],
            "accuracy": [region_accuracy(int(n)) for n in g["N"]],
            "faithfulness": g["faithfulness_top_minus_random"],
            "stability": g["stability_jaccard"],
            "consistency_sd": g["consistency_mean_sd"],
            "consistency_var": g["consistency_mean_variance"],
        }
    )
    panels = [
        ("fidelity_r2", "Fidelity $R^2$"),
        ("accuracy", "ATT accuracy\n(Feature: top-10 overlap; Geo: top-20 in expert regions)"),
        ("faithfulness", "Intervention impact\n" + r"$\Delta W_1$ (top $k$ $-$ random sets)"),
        ("stability", "Stability Jaccard"),
        ("consistency_var", "Consistency mean variance (log scale)"),
        ("consistency_sd", "Consistency mean SD"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, panels):
        # Geo first (larger hollow markers) and Feature on top (dashed), so identical values
        # such as Stability Jaccard = 1.000 in both branches stay visible.
        for frame, label, colour, z, ms, mfc, ls, ss in (
            (geo, "Geo (area data)", "#FF0051", 2, 9, "white", "-", 320),
            (FEATURE, "Feature", "#256abf", 3, 5, "#256abf", "--", 160),
        ):
            ax.plot(frame["N"], frame[col], marker="o", ms=ms, mfc=mfc, mew=1.6, ls=ls, lw=1.8,
                    color=colour, label=label, zorder=z)
            star = frame[frame["N"] == 4000]
            ax.scatter(star["N"], star[col], marker="*", s=ss, color=colour, zorder=z + 3, edgecolor="black")
        if col == "stability":
            ax.set_ylim(0.95, 1.02)
            ax.text(0.5, 0.35, "Feature and Geo both 1.000\nat every N (lines overlap)", transform=ax.transAxes,
                    ha="center", fontsize=9, color="#444444")
        if col == "consistency_var":
            ax.set_yscale("log")  # Feature variances are ~100x smaller than Geo
        ax.set_xscale("log")
        ax.set_xticks([1000, 2000, 4000, 10000])
        ax.set_xticklabels(["1k", "2k", "4k", "10k"])
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelbottom=True)
        ax.grid(alpha=0.3)
    for ax in axes[-2:]:
        ax.set_xlabel("Perturbations N")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Hull evaluation diagnostics across perturbation budgets\nXGBoost (stars: N=4,000 reporting design)",
                 fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(DEST / f"hull_metrics_n_sweep_area_data.{ext}", dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Saved {DEST / 'hull_metrics_n_sweep_area_data.png'}")


if __name__ == "__main__":
    main()
