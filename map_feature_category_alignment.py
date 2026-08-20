"""Feature + category alignment using the saved expert ranking v2.

Primary expert source:
  outputs/feature_expert_ranking_v2.csv

Compares GeoSMILE |coefficient| ranks against this fixed expert feature ranking,
and category importance via mean expert feature rank within each category.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
IMP_PATH = OUT / "feature_global_importance.csv"
EXPERT_RANK_PATH = OUT / "feature_expert_ranking_v2.csv"
CAT_DIR = OUT / "feature_category_alignment"

INK, INK_MUTED = "#12171C", "#57646D"
BLUE, PINK = "#1E88E5", "#FF0051"


def load_merged() -> pd.DataFrame:
    imp = pd.read_csv(IMP_PATH).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    imp["geosmile_rank"] = np.arange(1, len(imp) + 1)
    expert = pd.read_csv(EXPERT_RANK_PATH)
    need = {"feature", "expert_rank", "expert_category"}
    missing = need - set(expert.columns)
    if missing:
        raise SystemExit(f"{EXPERT_RANK_PATH} missing {sorted(missing)}")
    if expert["feature"].duplicated().any():
        raise SystemExit("Duplicate features in expert ranking file")
    if set(expert["expert_rank"]) != set(range(1, 31)):
        raise SystemExit("expert_rank must be a permutation of 1..30")
    if len(expert) != 30:
        raise SystemExit(f"Expected 30 expert rows, got {len(expert)}")

    merged = imp.merge(expert, on="feature", how="inner", validate="one_to_one")
    if len(merged) != 30:
        raise SystemExit(
            f"GeoSMILE/expert feature mismatch: matched {len(merged)}; "
            f"missing from expert={sorted(set(imp['feature']) - set(expert['feature']))}; "
            f"extra in expert={sorted(set(expert['feature']) - set(imp['feature']))}"
        )
    merged = merged.rename(columns={"expert_category": "thesis_category"})
    return merged.sort_values("geosmile_rank").reset_index(drop=True)


def category_summary(merged: pd.DataFrame) -> pd.DataFrame:
    total_mass = float(merged["abs_coefficient"].sum())
    rows = []
    for cat, g in merged.groupby("thesis_category", sort=False):
        mass = float(g["abs_coefficient"].sum())
        rows.append(
            {
                "category": cat,
                "n_features": int(len(g)),
                "geosmile_abs_mass": mass,
                "geosmile_mass_share": mass / total_mass if total_mass else np.nan,
                "geosmile_mean_abs": float(g["abs_coefficient"].mean()),
                "expert_mean_feature_rank": float(g["expert_rank"].mean()),
                "expert_median_feature_rank": float(g["expert_rank"].median()),
                "n_expert_top10": int((g["expert_rank"] <= 10).sum()),
                "n_geosmile_top10": int((g["geosmile_rank"] <= 10).sum()),
                "features": ", ".join(g.sort_values("expert_rank")["feature"].tolist()),
            }
        )
    out = pd.DataFrame(rows)
    # Lower mean expert feature rank = higher importance.
    out = out.sort_values(
        ["expert_mean_feature_rank", "category"], ascending=[True, True], kind="mergesort"
    ).reset_index(drop=True)
    out["expert_category_rank"] = np.arange(1, len(out) + 1)
    out = out.sort_values(
        ["geosmile_abs_mass", "category"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    out["geosmile_category_rank"] = np.arange(1, len(out) + 1)
    out["expert_rank_source"] = "mean_feature_rank_from_feature_expert_ranking_v2"
    return out.sort_values("geosmile_category_rank").reset_index(drop=True)


def topk_metrics(merged: pd.DataFrame, k: int = 6) -> dict:
    g = set(merged.loc[merged["geosmile_rank"] <= k, "feature"])
    e = set(merged.loc[merged["expert_rank"] <= k, "feature"])
    both = g & e
    return {
        "k": k,
        "overlap_count": int(len(both)),
        "precision_at_k": float(len(both) / k),
        "recall_at_k": float(len(both) / k),
        "jaccard": float(len(both) / len(g | e)) if (g | e) else np.nan,
        "geosmile_topk": sorted(g),
        "expert_topk": sorted(e),
        "both": sorted(both),
        "geosmile_only": sorted(g - e),
        "expert_only": sorted(e - g),
    }


def agreement_metrics(merged: pd.DataFrame, summary: pd.DataFrame) -> dict:
    rho_f, p_f = spearmanr(merged["geosmile_rank"], merged["expert_rank"])
    tau_f, p_tf = kendalltau(merged["geosmile_rank"], merged["expert_rank"])
    rho_c, p_c = spearmanr(summary["geosmile_category_rank"], summary["expert_category_rank"])
    tau_c, p_tc = kendalltau(summary["geosmile_category_rank"], summary["expert_category_rank"])
    # Also correlate mass share with inverse mean expert rank (higher = more important)
    inv = summary["expert_mean_feature_rank"].max() + summary["expert_mean_feature_rank"].min()
    # simpler: lower mean rank is better -> use negative mean rank
    r_mass, p_mass = spearmanr(summary["geosmile_mass_share"], -summary["expert_mean_feature_rank"])
    top3_g = set(summary.nsmallest(3, "geosmile_category_rank")["category"])
    top3_e = set(summary.nsmallest(3, "expert_category_rank")["category"])
    return {
        "feature_spearman_rho": float(rho_f),
        "feature_spearman_p": float(p_f),
        "feature_kendall_tau": float(tau_f),
        "feature_kendall_p": float(p_tf),
        "category_spearman_rho": float(rho_c),
        "category_spearman_p": float(p_c),
        "category_kendall_tau": float(tau_c),
        "category_kendall_p": float(p_tc),
        "category_mass_vs_neg_mean_expert_rank_spearman": float(r_mass),
        "category_mass_vs_neg_mean_expert_rank_p": float(p_mass),
        "top3_category_overlap": int(len(top3_g & top3_e)),
        "top3_geosmile": sorted(top3_g),
        "top3_expert": sorted(top3_e),
        "top3_both": sorted(top3_g & top3_e),
    }


def plot_feature_top10_comparison(top10: dict, path: Path) -> None:
    """Side-by-side Feature top-10 comparison (expert vs GeoSMILE)."""
    both = list(top10["both"])
    geo_only = list(top10["geosmile_only"])
    exp_only = list(top10["expert_only"])
    groups = [
        ("Mutual\n(expert ∩ model)", both, "#0D7377"),
        ("GeoSMILE only", geo_only, PINK),
        ("Expert only", exp_only, BLUE),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.2), sharey=False)
    for ax, (title, items, color) in zip(axes, groups):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 11)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontweight="bold", color=INK, fontsize=11)
        ax.set_facecolor("#F7F8FA")
        for spine in ax.spines.values():
            spine.set_color("#D0D5DB")
        if not items:
            ax.text(0.5, 5.5, "(none)", ha="center", va="center", color=INK_MUTED)
            continue
        for i, name in enumerate(items, start=1):
            ax.add_patch(
                plt.Rectangle((0.08, i - 0.35), 0.84, 0.7, facecolor=color, alpha=0.18, edgecolor=color)
            )
            ax.text(0.12, i, f"{i}. {name}", va="center", ha="left", fontsize=8.5, color=INK)

    fig.suptitle(
        f"Feature top-10 comparison (10/30)\nOverlap = {top10['overlap_count']}/10",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.5,
        0.02,
        "Expert feature ranking vs GeoSMILE |global coefficient|.",
        ha="center",
        fontsize=8,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_category_comparison(summary: pd.DataFrame, path: Path) -> None:
    s = summary.sort_values("expert_category_rank", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    axes[0].barh(s["category"], s["geosmile_mass_share"], color=PINK, edgecolor="white")
    axes[0].set_xlabel("Share of total |GeoSMILE coefficient|")
    axes[0].set_title("GeoSMILE attribution mass", fontweight="bold", color=INK)

    # Invert mean expert rank for display: higher bar = more important
    display = s["expert_mean_feature_rank"].max() + s["expert_mean_feature_rank"].min()
    # Use (max+1 - mean_rank) so rank 1 -> high bar
    priority = (s["expert_mean_feature_rank"].max() + 1) - s["expert_mean_feature_rank"]
    axes[1].barh(s["category"], priority, color=BLUE, edgecolor="white")
    axes[1].set_xlabel("Expert priority (from mean feature rank; higher = more important)")
    axes[1].set_title("Expert category importance", fontweight="bold", color=INK)

    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(colors=INK)
    fig.suptitle(
        "Feature categories: GeoSMILE mass vs expert ranking v2",
        fontweight="bold",
        color=INK,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Expert ranks from feature_expert_ranking_v2.csv. "
        "Category importance = mean feature expert rank.",
        ha="center",
        fontsize=7.5,
        color=INK_MUTED,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_rank_scatter(merged: pd.DataFrame, summary: pd.DataFrame, metrics: dict, path_prefix: Path) -> None:
    # Feature scatter
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(merged["expert_rank"], merged["geosmile_rank"], s=45, c=BLUE, edgecolors=INK, zorder=3)
    lim = [0.5, 30.5]
    ax.plot(lim, lim, "--", color=INK_MUTED, linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("Expert feature rank (1 = most important)")
    ax.set_ylabel("GeoSMILE feature rank (1 = highest |coef|)")
    ax.set_title(
        f"Feature rank agreement\nSpearman rho = {metrics['feature_spearman_rho']:.2f}",
        fontweight="bold",
        color=INK,
    )
    ax.grid(alpha=0.25)
    fig.savefig(path_prefix.parent / "feature_rank_agreement.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Category scatter
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(
        summary["expert_category_rank"],
        summary["geosmile_category_rank"],
        s=70,
        c=BLUE,
        edgecolors=INK,
        zorder=3,
    )
    for _, row in summary.iterrows():
        ax.annotate(
            row["category"],
            (row["expert_category_rank"], row["geosmile_category_rank"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color=INK,
        )
    n = len(summary)
    lim = [0.5, n + 0.5]
    ax.plot(lim, lim, "--", color=INK_MUTED, linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("Expert category rank (1 = highest; from mean feature rank)")
    ax.set_ylabel("GeoSMILE category rank (1 = highest mass)")
    ax.set_title(
        f"Category rank agreement\nSpearman rho = {metrics['category_spearman_rho']:.2f}",
        fontweight="bold",
        color=INK,
    )
    ax.grid(alpha=0.25)
    fig.savefig(path_prefix, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path_prefix.parent / 'feature_rank_agreement.png'}")
    print(f"Saved: {path_prefix}")


def main() -> None:
    CAT_DIR.mkdir(parents=True, exist_ok=True)
    if not EXPERT_RANK_PATH.exists():
        raise SystemExit(f"Missing expert ranking file: {EXPERT_RANK_PATH}")

    merged = load_merged()
    summary = category_summary(merged)
    top10 = topk_metrics(merged, k=10)
    metrics = agreement_metrics(merged, summary)

    merged.sort_values("geosmile_rank").to_csv(CAT_DIR / "feature_with_categories.csv", index=False)
    summary.to_csv(CAT_DIR / "category_alignment_thesis.csv", index=False)
    pd.DataFrame(
        [
            {
                "branch": "Feature (location excluded)",
                "level": "feature",
                "metric": "Top-10 overlap",
                "value": top10["overlap_count"],
                "notes": f"both={top10['both']}",
            },
            {
                "branch": "Feature (location excluded)",
                "level": "feature",
                "metric": "Top-10 overlap ratio",
                "value": top10["overlap_count"] / 10.0,
                "notes": "overlap/10",
            },
            {
                "branch": "Feature (location excluded)",
                "level": "feature",
                "metric": "Spearman rho (ranks)",
                "value": metrics["feature_spearman_rho"],
                "notes": f"p={metrics['feature_spearman_p']:.4g}",
            },
            {
                "branch": "Feature (location excluded)",
                "level": "feature",
                "metric": "Kendall tau-b",
                "value": metrics["feature_kendall_tau"],
                "notes": f"p={metrics['feature_kendall_p']:.4g}",
            },
            {
                "branch": "Feature (location excluded)",
                "level": "category",
                "metric": "Spearman rho (ranks)",
                "value": metrics["category_spearman_rho"],
                "notes": f"p={metrics['category_spearman_p']:.4g}",
            },
            {
                "branch": "Feature (location excluded)",
                "level": "category",
                "metric": "Top-3 category overlap",
                "value": metrics["top3_category_overlap"],
                "notes": f"both={metrics['top3_both']}",
            },
        ]
    ).to_csv(CAT_DIR / "feature_expert_alignment_eval_v2.csv", index=False)

    payload = {
        "expert_ranking_file": str(EXPERT_RANK_PATH.name),
        "annotation_status": "expert_reference",
        "note": (
            "Expert feature ranks over the exact 30 Hull Feature-branch predictors. "
            "Primary feature cut is top-10 of 30."
        ),
        "top10": top10,
        "metrics": metrics,
        "category_table": summary.to_dict(orient="records"),
    }
    (CAT_DIR / "category_alignment_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    plot_category_comparison(summary, CAT_DIR / "category_mass_vs_expert_rating.png")
    plot_rank_scatter(merged, summary, metrics, CAT_DIR / "category_rank_agreement.png")
    plot_feature_top10_comparison(top10, CAT_DIR / "feature_top10_comparison.png")

    print("\n=== Feature top-10 ===")
    print(f"Mutual {top10['overlap_count']}/10: {top10['both']}")
    print(f"GeoSMILE only: {top10['geosmile_only']}")
    print(f"Expert only:   {top10['expert_only']}")
    print(
        f"Feature Spearman rho = {metrics['feature_spearman_rho']:.3f} "
        f"(p={metrics['feature_spearman_p']:.3g})"
    )
    print("\n=== Category ranks ===")
    print(
        summary[
            [
                "geosmile_category_rank",
                "category",
                "geosmile_mass_share",
                "expert_mean_feature_rank",
                "expert_category_rank",
                "n_geosmile_top10",
                "n_expert_top10",
            ]
        ].to_string(index=False)
    )
    print(
        f"\nCategory Spearman rho = {metrics['category_spearman_rho']:.3f} "
        f"(p={metrics['category_spearman_p']:.3g})"
    )
    print(
        f"Top-3 category overlap = {metrics['top3_category_overlap']}/3 "
        f"both={metrics['top3_both']}"
    )
    print(f"\nSaved expert ranking: {EXPERT_RANK_PATH.resolve()}")
    print(f"Outputs: {CAT_DIR.resolve()}")


if __name__ == "__main__":
    main()
