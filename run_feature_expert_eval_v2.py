"""Re-run Feature expert-reference evaluation metrics using ranking v2."""

from __future__ import annotations

import json
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

import map_feature_category_alignment as mfa

OUT = Path("outputs")
CAT = OUT / "feature_category_alignment"


def topk_metrics(merged: pd.DataFrame, k: int) -> dict:
    g = set(merged.loc[merged["geosmile_rank"] <= k, "feature"])
    e = set(merged.loc[merged["expert_rank"] <= k, "feature"])
    both = g & e
    y_true = (merged["expert_rank"] <= k).astype(int).to_numpy()
    y_pred = (merged["geosmile_rank"] <= k).astype(int).to_numpy()
    y_score = merged["abs_coefficient"].to_numpy()
    out = {
        "k": k,
        "overlap_count": int(len(both)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "jaccard": float(len(both) / len(g | e)) if (g | e) else np.nan,
        "overlap_coefficient": float(len(both) / min(len(g), len(e))) if min(len(g), len(e)) else np.nan,
        "lift": float((len(both) / k) / (k / len(merged))),
        "both": sorted(both),
        "geosmile_only": sorted(g - e),
        "expert_only": sorted(e - g),
    }
    if y_true.min() != y_true.max():
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    else:
        out["auroc"] = out["auprc"] = out["balanced_accuracy"] = out["mcc"] = None
    return out


def main() -> None:
    CAT.mkdir(parents=True, exist_ok=True)
    merged = mfa.load_merged()
    summary = mfa.category_summary(merged)
    t10 = topk_metrics(merged, 10)

    rho, p_rho = spearmanr(merged["geosmile_rank"], merged["expert_rank"])
    tau, p_tau = kendalltau(merged["geosmile_rank"], merged["expert_rank"])
    rho_c, p_c = spearmanr(summary["geosmile_category_rank"], summary["expert_category_rank"])
    tau_c, p_tc = kendalltau(summary["geosmile_category_rank"], summary["expert_category_rank"])
    top3_g = set(summary.nsmallest(3, "geosmile_category_rank")["category"])
    top3_e = set(summary.nsmallest(3, "expert_category_rank")["category"])
    top3_both = sorted(top3_g & top3_e)

    rows = []
    for metric in [
        "overlap_count",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "jaccard",
        "overlap_coefficient",
        "lift",
        "auroc",
        "auprc",
        "balanced_accuracy",
        "mcc",
    ]:
        rows.append(
            {
                "branch": "Feature (location excluded)",
                "scope": "top10",
                "category": "Expert-reference attribution accuracy",
                "metric": metric,
                "value": t10[metric],
            }
        )
    rows.extend(
        [
            {
                "branch": "Feature (location excluded)",
                "scope": "full_ranking",
                "category": "Expert-reference attribution accuracy",
                "metric": "spearman_rho",
                "value": float(rho),
            },
            {
                "branch": "Feature (location excluded)",
                "scope": "full_ranking",
                "category": "Expert-reference attribution accuracy",
                "metric": "spearman_p",
                "value": float(p_rho),
            },
            {
                "branch": "Feature (location excluded)",
                "scope": "full_ranking",
                "category": "Expert-reference attribution accuracy",
                "metric": "kendall_tau",
                "value": float(tau),
            },
            {
                "branch": "Feature (location excluded)",
                "scope": "full_ranking",
                "category": "Expert-reference attribution accuracy",
                "metric": "kendall_p",
                "value": float(p_tau),
            },
            {
                "branch": "Feature (location excluded)",
                "scope": "category",
                "category": "Expert-reference attribution accuracy",
                "metric": "spearman_rho",
                "value": float(rho_c),
            },
            {
                "branch": "Feature (location excluded)",
                "scope": "category",
                "category": "Expert-reference attribution accuracy",
                "metric": "spearman_p",
                "value": float(p_c),
            },
            {
                "branch": "Feature (location excluded)",
                "scope": "category",
                "category": "Expert-reference attribution accuracy",
                "metric": "kendall_tau",
                "value": float(tau_c),
            },
            {
                "branch": "Feature (location excluded)",
                "scope": "category",
                "category": "Expert-reference attribution accuracy",
                "metric": "kendall_p",
                "value": float(p_tc),
            },
            {
                "branch": "Feature (location excluded)",
                "scope": "category",
                "category": "Expert-reference attribution accuracy",
                "metric": "top3_overlap",
                "value": len(top3_both),
            },
        ]
    )
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(CAT / "feature_expert_alignment_eval_v2.csv", index=False)
    merged.sort_values("geosmile_rank").to_csv(CAT / "feature_with_categories.csv", index=False)
    summary.to_csv(CAT / "category_alignment_thesis.csv", index=False)

    payload = {
        "expert_ranking_file": "feature_expert_ranking_v2.csv",
        "annotation_status": "expert_reference",
        "note": (
            "Expert ranking over exact 30 Feature predictors; "
            "primary Feature cut is top-10 of 30."
        ),
        "top10": t10,
        "full_ranking": {
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
            "kendall_tau": float(tau),
            "kendall_p": float(p_tau),
        },
        "category": {
            "spearman_rho": float(rho_c),
            "spearman_p": float(p_c),
            "kendall_tau": float(tau_c),
            "kendall_p": float(p_tc),
            "top3_overlap": len(top3_both),
            "top3_both": top3_both,
        },
        "categories": summary.to_dict(orient="records"),
    }
    (CAT / "category_alignment_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Refresh figures
    mfa.main()

    print("\n========== FEATURE EVALUATION (expert ranking v2; top-10) ==========")
    print(f"Full-rank Spearman rho={rho:.3f} (p={p_rho:.3g})  Kendall tau={tau:.3f}")
    print("\n--- Top-10 ---")
    print(
        f"Overlap {t10['overlap_count']}/10  P={t10['precision']:.3f} R={t10['recall']:.3f} "
        f"F1={t10['f1']:.3f} Jaccard={t10['jaccard']:.3f}"
    )
    print(
        f"AUROC={t10['auroc']:.3f} AUPRC={t10['auprc']:.3f} "
        f"MCC={t10['mcc']:.3f} bal_acc={t10['balanced_accuracy']:.3f}"
    )
    print("both:", t10["both"])
    print("GeoSMILE only:", t10["geosmile_only"])
    print("Expert only:  ", t10["expert_only"])
    print("\n--- Categories ---")
    print(
        summary[
            [
                "geosmile_category_rank",
                "category",
                "geosmile_mass_share",
                "expert_mean_feature_rank",
                "expert_category_rank",
            ]
        ].to_string(index=False)
    )
    print(
        f"Category Spearman rho={rho_c:.3f} (p={p_c:.3g})  "
        f"top3 overlap={len(top3_both)}/3 both={top3_both}"
    )
    print(f"\nWrote {CAT / 'feature_expert_alignment_eval_v2.csv'}")


if __name__ == "__main__":
    main()
