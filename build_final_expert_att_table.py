"""Final expert-reference attribution table: Feature 10/30 + Geo 20/168.

Writes:
  outputs/geosmile_hull_expert_att_final.csv
  outputs/geosmile_hull_expert_att_final_wide.csv
  outputs/geosmile_hull_expert_att_final.tex
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from hull_expert_alignment import graded_gains, topk_indices

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
RANK_DIR = OUT / "maps_twobranch" / "ranking"


def ndcg_full_from_ranks(magnitude: np.ndarray, expert_rank: np.ndarray, ids: np.ndarray) -> float:
    """nDCG over the full list using gain = 1/expert_rank (rank-1 = highest gain)."""
    gains = 1.0 / np.asarray(expert_rank, dtype=float)
    n = len(gains)
    order = topk_indices(magnitude, ids, n)
    ideal = np.sort(gains)[::-1]
    dcg = float(np.sum(gains[order] / np.log2(np.arange(2, n + 2))))
    idcg = float(np.sum(ideal / np.log2(np.arange(2, n + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def ndcg_full_from_ratings(magnitude: np.ndarray, ratings: np.ndarray, ids: np.ndarray) -> float:
    """nDCG using 1-5 style ratings via graded_gains."""
    gains = graded_gains(ratings)
    n = len(gains)
    order = topk_indices(magnitude, ids, n)
    ideal = np.sort(gains)[::-1]
    dcg = float(np.sum(gains[order] / np.log2(np.arange(2, n + 2))))
    idcg = float(np.sum(ideal / np.log2(np.arange(2, n + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def _topk_suite(
    *,
    ids: np.ndarray,
    expert_rank: np.ndarray,
    model_rank: np.ndarray,
    scores: np.ndarray,
    ratings: np.ndarray | None,
    k: int,
) -> dict[str, float]:
    n = len(ids)
    y_true = (expert_rank <= k).astype(int)
    y_pred = (model_rank <= k).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    rho, p_rho = spearmanr(expert_rank, model_rank)
    tau, p_tau = kendalltau(expert_rank, model_rank)
    if ratings is None:
        ndcg = ndcg_full_from_ranks(scores, expert_rank, ids)
    else:
        ndcg = ndcg_full_from_ratings(scores, ratings, ids)

    return {
        "n": float(n),
        "k": float(k),
        "intersection": float(tp),
        "overlap_ratio": float(tp / k),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision_at_k": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_k": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_k": float(f1_score(y_true, y_pred, zero_division=0)),
        "jaccard_at_k": float(tp / (tp + fp + fn)) if (tp + fp + fn) else np.nan,
        "overlap_coefficient": float(tp / min(k, k)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "lift": float((tp / k) / (k / n)),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "kendall_tau_b": float(tau),
        "kendall_p": float(p_tau),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "ndcg_full": float(ndcg),
    }


def feature_metrics() -> dict[str, float | str]:
    merged = pd.read_csv(OUT / "feature_category_alignment" / "feature_with_categories.csv")
    cat = pd.read_csv(OUT / "feature_category_alignment" / "category_alignment_thesis.csv")
    suite = _topk_suite(
        ids=merged["feature"].astype(str).to_numpy(),
        expert_rank=merged["expert_rank"].to_numpy(dtype=int),
        model_rank=merged["geosmile_rank"].to_numpy(dtype=int),
        scores=pd.to_numeric(merged["abs_coefficient"], errors="coerce").to_numpy(dtype=float),
        ratings=None,  # use 1/expert_rank gains
        k=10,
    )
    rho_c, _ = spearmanr(cat["expert_category_rank"], cat["geosmile_category_rank"])
    top3_e = set(cat.nsmallest(3, "expert_category_rank")["category"])
    top3_g = set(cat.nsmallest(3, "geosmile_category_rank")["category"])
    suite["factor_group_spearman_rho"] = float(rho_c)
    suite["factor_group_top3_overlap"] = float(len(top3_e & top3_g))
    suite["factor_group_top3_of"] = 3.0
    return suite


def geo_metrics() -> dict[str, float | str]:
    g = pd.read_csv(RANK_DIR / "geo_expert_importance_ranks.csv")
    suite = _topk_suite(
        ids=g["lsoa21cd"].astype(str).to_numpy(),
        expert_rank=g["expert_importance_rank"].to_numpy(dtype=int),
        model_rank=g["geo_importance_rank"].to_numpy(dtype=int),
        scores=pd.to_numeric(g["geo_abs_coefficient"], errors="coerce").to_numpy(dtype=float),
        ratings=pd.to_numeric(g["reference_rating"], errors="coerce").to_numpy(dtype=float),
        k=20,
    )
    suite["factor_group_spearman_rho"] = np.nan
    suite["factor_group_top3_overlap"] = np.nan
    suite["factor_group_top3_of"] = np.nan
    return suite


ROWS = [
    ("Number of evaluated factors", "n", "{:.0f}"),
    ("Reference set size $k$", "k", "{:.0f}"),
    ("Reference/model intersection", "intersection", "{:.0f}"),
    ("Overlap ratio", "overlap_ratio", "{:.4f}"),
    ("Precision at $k$", "precision_at_k", "{:.4f}"),
    ("Recall at $k$", "recall_at_k", "{:.4f}"),
    ("F1 at $k$", "f1_at_k", "{:.4f}"),
    ("Jaccard at fixed $k$", "jaccard_at_k", "{:.4f}"),
    ("Overlap coefficient", "overlap_coefficient", "{:.4f}"),
    ("Spearman $\\rho$", "spearman_rho", "{:.4f}"),
    ("Spearman $p$-value", "spearman_p", "{:.4g}"),
    ("Kendall $\\tau_b$", "kendall_tau_b", "{:.4f}"),
    ("Kendall $p$-value", "kendall_p", "{:.4g}"),
    ("Accuracy", "accuracy", "{:.4f}"),
    ("Balanced accuracy", "balanced_accuracy", "{:.4f}"),
    ("Matthews correlation coefficient", "mcc", "{:.4f}"),
    ("Lift over prevalence", "lift", "{:.4f}"),
    ("AUROC", "auroc", "{:.4f}"),
    ("AUPRC", "auprc", "{:.4f}"),
    ("nDCG over full ranking", "ndcg_full", "{:.4f}"),
    ("Factor-group Spearman $\\rho$", "factor_group_spearman_rho", "{:.4f}"),
    ("Factor-group top-3 overlap", "factor_group_top3_overlap", "{:.0f}"),
]


def _fmt(val, fmt: str, *, na_label: str) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na_label
    if fmt.endswith("f") or fmt.endswith("g") or ".0f" in fmt:
        return fmt.format(float(val))
    return fmt.format(val)


def main() -> None:
    feat = feature_metrics()
    geo = geo_metrics()

    long_rows = []
    wide_rows = []
    for label, key, fmt in ROWS:
        fv = feat.get(key, np.nan)
        gv = geo.get(key, np.nan)
        if key == "factor_group_top3_overlap" and not (isinstance(fv, float) and np.isnan(fv)):
            f_cell = f"{int(fv)} of {int(feat['factor_group_top3_of'])}"
            g_cell = "Not applicable"
        elif key.startswith("factor_group") and (isinstance(gv, float) and np.isnan(gv)):
            f_cell = _fmt(fv, fmt, na_label="Not applicable")
            g_cell = "Not applicable"
        else:
            f_cell = _fmt(fv, fmt, na_label="—")
            g_cell = _fmt(gv, fmt, na_label="—")
        long_rows.extend(
            [
                {
                    "branch": "Feature (location excluded)",
                    "metric": label.replace("$", "").replace("\\rho", "rho").replace("\\tau_b", "tau_b"),
                    "value": fv if not (isinstance(fv, float) and np.isnan(fv)) else np.nan,
                    "display": f_cell,
                    "k": 10,
                    "n": 30,
                    "reference": "feature_expert_ranking_v2",
                },
                {
                    "branch": "Geo",
                    "metric": label.replace("$", "").replace("\\rho", "rho").replace("\\tau_b", "tau_b"),
                    "value": gv if not (isinstance(gv, float) and np.isnan(gv)) else np.nan,
                    "display": g_cell,
                    "k": 20,
                    "n": 168,
                    "reference": "expert_top20",
                },
            ]
        )
        wide_rows.append(
            {
                "metric": label.replace("$", "").replace("\\rho", "rho").replace("\\tau_b", "tau_b"),
                "feature_top10": f_cell,
                "geo_top20": g_cell,
                "feature_value": fv,
                "geo_value": gv,
            }
        )

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    long_path = OUT / "geosmile_hull_expert_att_final.csv"
    wide_path = OUT / "geosmile_hull_expert_att_final_wide.csv"
    tex_path = OUT / "geosmile_hull_expert_att_final.tex"
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    lines = [
        r"\begin{longtable}{p{0.38\textwidth} r r}",
        r"\caption{Expert-reference attribution-accuracy and ranking metrics. "
        r"Feature uses the top-10 reference over 30 predictors "
        r"(\texttt{feature\_expert\_ranking\_v2}). Geo uses the top-20 "
        r"reference over 168 LSOAs (\texttt{expert\_top20}); "
        r"set-based Geo quantities correspond to $\mathrm{TP}=16$, $\mathrm{FP}=\mathrm{FN}=4$, "
        r"$\mathrm{TN}=144$. Rank and score diagnostics for Geo are computed against this "
        r"expert view over all 168 LSOAs.}",
        r"\label{tab:hull_expert_att_final}\\",
        r"\toprule",
        r"Metric & Feature top 10 & Geo top 20 \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Metric & Feature top 10 & Geo top 20 \\",
        r"\midrule",
        r"\endhead",
    ]
    for label, key, fmt in ROWS:
        fv = feat.get(key, np.nan)
        gv = geo.get(key, np.nan)
        if key == "factor_group_top3_overlap":
            f_cell = f"{int(fv)} of {int(feat['factor_group_top3_of'])}"
            g_cell = r"Not applicable"
        elif key.startswith("factor_group") and (isinstance(gv, float) and np.isnan(gv)):
            f_cell = _fmt(fv, fmt, na_label=r"Not applicable")
            g_cell = r"Not applicable"
        else:
            f_cell = _fmt(fv, fmt, na_label="---")
            g_cell = _fmt(gv, fmt, na_label="---")
        lines.append(f"{label} & {f_cell} & {g_cell} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}", ""]
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    print("=== EXPERT ATT (Feature 10/30 | Geo 20/168) ===")
    print(wide_df[["metric", "feature_top10", "geo_top20"]].to_string(index=False))
    print(f"\nWrote {long_path}")
    print(f"Wrote {wide_path}")
    print(f"Wrote {tex_path}")
    print(
        f"Geo contingency: TP={int(geo['tp'])}, FP={int(geo['fp'])}, "
        f"FN={int(geo['fn'])}, TN={int(geo['tn'])}"
    )


if __name__ == "__main__":
    main()
