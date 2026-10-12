"""Plot GeoSMILE surrogate R2 vs perturbation count.

This is explainer fidelity (Ridge vs Wasserstein responses of the ML model),
not XGBoost predictive holdout R2. Writes only under outputs/eval_by_n_perturb/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "eval_by_n_perturb" / "plots"
BLUE, PINK, TEAL, INK, MUTED = "#1E88E5", "#FF0051", "#147D74", "#12171C", "#57646D"

PUBLISHED = {
    "Feature": {"n": 4000, "r2": 0.9769549209642179, "wr2": 0.9770971025312047},
    "Geo": {"n": 4000, "r2": 0.9478707156885263, "wr2": 0.9454139717459202},
}
PUBLISHED_METRICS = {
    "Feature": {
        "n": 4000,
        "fidelity_r2": 0.9769549209642179,
        "accuracy_intersection": 6,
        "faithfulness_top_minus_random": 4.246146297796495,
        "att_stability_jaccard": 1.0,
        "att_consistency_rank_spearman": 0.8310567296996663,
        "att_consistency_top20pct_jaccard": 1.0,
    },
    "Geo": {
        "n": 4000,
        "fidelity_r2": 0.9478707156885263,
        "accuracy_intersection": 15,
        "faithfulness_top_minus_random": 8.02810170066357,
        "att_stability_jaccard": 1.0,
        "att_consistency_rank_spearman": 0.9472694509543891,
        "att_consistency_top20pct_jaccard": 0.8100615190088876,
    },
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_sweep() -> pd.DataFrame:
    return pd.read_csv(OUT / "eval_by_n_perturb" / "summary.csv")


def load_convergence() -> pd.DataFrame:
    conv = pd.read_csv(OUT / "geosmile_hull_convergence.csv")
    conv = conv.drop_duplicates(subset=["Branch", "#Perturb"], keep="last")
    return conv.sort_values(["Branch", "#Perturb"])


def plot_independent_runs(sweep: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=False)
    for ax, branch, color in zip(axes, ["Feature", "Geo"], [BLUE, TEAL]):
        sub = sweep[sweep["branch"] == branch].sort_values("n_perturb")
        ax.plot(
            sub["n_perturb"],
            sub["fidelity_r2"],
            marker="o",
            color=color,
            lw=2,
            ms=8,
            label="Held-out surrogate R²",
        )
        ax.plot(
            sub["n_perturb"],
            sub["fidelity_weighted_r2"],
            marker="s",
            color=PINK,
            lw=1.5,
            ms=7,
            ls="--",
            label="Weighted R²",
        )
        pub = PUBLISHED[branch]
        ax.scatter(
            [pub["n"]],
            [pub["r2"]],
            marker="*",
            s=180,
            color=INK,
            zorder=5,
            label=f"Published ({pub['n']})",
        )
        ax.set_title(branch)
        ax.set_xlabel("Perturbation count N")
        ax.set_ylabel("GeoSMILE surrogate R²")
        ax.set_xticks(list(sub["n_perturb"]))
        ax.tick_params(axis="x", labelrotation=30)
        ymin = min(sub["fidelity_r2"].min(), pub["r2"]) - 0.015
        ymax = max(sub["fidelity_r2"].max(), pub["r2"]) + 0.012
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.28)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "GeoSMILE fidelity: how well the Ridge explainer reconstructs\n"
        "the ML model's perturbation responses (independent runs at each N)",
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_with_convergence(sweep: pd.DataFrame, conv: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for ax, branch, color in zip(axes, ["Feature", "Geo"], [BLUE, TEAL]):
        csub = conv[conv["Branch"] == branch]
        ssub = sweep[sweep["branch"] == branch].sort_values("n_perturb")
        ax.plot(
            csub["#Perturb"],
            csub["R2"],
            marker="o",
            color=color,
            lw=1.8,
            ms=6,
            label="Nested convergence (original pool)",
        )
        ax.plot(
            ssub["n_perturb"],
            ssub["fidelity_r2"],
            marker="D",
            color=PINK,
            lw=1.8,
            ms=7,
            label="Independent full run at N",
        )
        pub = PUBLISHED[branch]
        ax.axhline(pub["r2"], color=MUTED, ls=":", lw=1.2, label=f"Published {pub['n']}: {pub['r2']:.3f}")
        ax.scatter([pub["n"]], [pub["r2"]], marker="*", s=160, color=INK, zorder=5)
        ax.set_title(branch)
        ax.set_xlabel("Number of perturbations")
        ax.set_ylabel("GeoSMILE surrogate R²")
        ax.grid(alpha=0.28)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.set_ylim(0.72 if branch == "Geo" else 0.95, 1.0 if branch == "Feature" else 0.98)
    fig.suptitle(
        "Surrogate R² vs perturbation budget\n"
        "Nested curve: growing subset of the original 1024/2000 pool. "
        "Diamonds: new SMILE runs at 1000–5000.",
        fontweight="bold",
        y=1.05,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(sweep: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for branch, color in [("Feature", BLUE), ("Geo", TEAL)]:
        sub = sweep[sweep["branch"] == branch].sort_values("n_perturb")
        ax.plot(
            sub["n_perturb"],
            sub["fidelity_r2"],
            marker="o",
            color=color,
            lw=2,
            ms=8,
            label=branch,
        )
        pub = PUBLISHED[branch]
        ax.scatter([pub["n"]], [pub["r2"]], marker="*", s=170, color=color, zorder=5)
    ax.set_xlabel("Perturbation count N")
    ax.set_ylabel("GeoSMILE surrogate R²")
    ax.set_title("Feature vs Geo explainer fidelity")
    ax.set_xticks([1000, 2000, 3000, 4000, 5000, 7500, 10000])
    ax.set_ylim(0.90, 1.0)
    ax.grid(alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_five_metrics(sweep: pd.DataFrame, path: Path) -> None:
    """Independent-run sweep of the five Hull evaluation metrics, both branches."""
    panels = [
        ("fidelity_r2", "Fidelity (surrogate R²)", None),
        ("accuracy_intersection", "Expert overlap (Feature k=10, Geo k=20)", None),
        ("faithfulness_top_minus_random", "Faithfulness (top minus random WD)", None),
        ("att_stability_jaccard", "ATT stability (top-k Jaccard)", (0.85, 1.02)),
        ("att_consistency_rank_spearman", "ATT consistency (rank Spearman)", (0.55, 1.02)),
        ("att_consistency_top20pct_jaccard", "ATT consistency (top-20% Jaccard)", (0.50, 1.05)),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11.4, 10.6), sharex=True)
    xticks = [1000, 2000, 3000, 4000, 5000, 7500, 10000]
    for ax, (col, title, ylim) in zip(axes.ravel(), panels):
        for branch, color in [("Feature", BLUE), ("Geo", TEAL)]:
            sub = sweep[sweep["branch"] == branch].sort_values("n_perturb")
            ax.plot(
                sub["n_perturb"],
                sub[col],
                marker="o",
                color=color,
                lw=2,
                ms=7,
                label=branch,
            )
            pub = PUBLISHED_METRICS[branch]
            ax.scatter(
                [pub["n"]],
                [pub[col]],
                marker="*",
                s=160,
                color=color,
                zorder=5,
                edgecolors=INK,
                linewidths=0.4,
            )
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(alpha=0.28)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[2, 0].set_xlabel("Perturbation count N")
    axes[2, 1].set_xlabel("Perturbation count N")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Five Hull GeoSMILE evaluation metrics versus perturbation budget\n"
        "Circles: independent runs. Stars: published Feature 4000 / Geo 4000 (seed 42).",
        fontweight="bold",
        y=1.06,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_table(sweep: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, r in sweep.sort_values(["branch", "n_perturb"]).iterrows():
        rows.append(
            {
                "source": "independent_run",
                "branch": r["branch"],
                "n_perturb": int(r["n_perturb"]),
                "surrogate_r2": float(r["fidelity_r2"]),
                "surrogate_weighted_r2": float(r["fidelity_weighted_r2"]),
            }
        )
    for branch, pub in PUBLISHED.items():
        rows.append(
            {
                "source": "published_4000_4000",
                "branch": branch,
                "n_perturb": pub["n"],
                "surrogate_r2": pub["r2"],
                "surrogate_weighted_r2": pub["wr2"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    _style()
    sweep = load_sweep()
    conv = load_convergence()
    plot_independent_runs(sweep, DEST / "geosmile_fidelity_by_n.png")
    plot_with_convergence(sweep, conv, DEST / "geosmile_fidelity_convergence_and_sweep.png")
    plot_overlay(sweep, DEST / "geosmile_fidelity_feature_vs_geo.png")
    plot_five_metrics(sweep, DEST / "geosmile_eval_metrics_by_n.png")
    write_table(sweep, DEST / "geosmile_fidelity_by_n.csv")
    print(f"Wrote plots under {DEST}")
    print(sweep[["n_perturb", "branch", "fidelity_r2", "fidelity_weighted_r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
