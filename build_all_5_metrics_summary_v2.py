"""All five evaluation metrics per (model, branch), with ATT Faithfulness under the
KG-SMILE (2025, Sec. 5.3) definition.

Supersedes the faithfulness columns of build_all_5_metrics_summary.py. That table's
faithfulness was the per-model intervention test (top-k minus random response); here
ATT Faithfulness is the across-model Pearson r between predictive R2 and attribution
quality, from faithfulness_accuracy_r2.py. Being a property of the whole model set, it
is one value per branch, repeated on every row of that branch.

Columns taken unchanged from outputs/all_5_metrics_summary.csv (build_all_5_metrics_summary.py):
predictive R2/MAE/RMSE, ATT fidelity, ATT stability, ATT consistency, and ATT accuracy /
F1 / AUROC at the selected design. Dropped: faithfulness_top_minus_random,
faithfulness_empirical_p.

Added from outputs/faithfulness_accuracy_r2/ (N=4,000; Feature uses its N=4,096 run):
  att_overlap_hits, att_overlap_k, att_overlap         expert top-k overlap per model
  att_auc_n4000                                        ATT-AUC per model
  att_faithfulness_r, att_faithfulness_p               Pearson r(R2, overlap), two-sided p
  att_faithfulness_r_auc, att_faithfulness_p_auc       Pearson r(R2, ATT-AUC), two-sided p

Run faithfulness_accuracy_r2.py first. Writes outputs/all_5_metrics_summary_v2.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FAITH = OUT / "faithfulness_accuracy_r2"


def main() -> None:
    base = pd.read_csv(OUT / "all_5_metrics_summary.csv")
    base = base.drop(columns=["faithfulness_top_minus_random", "faithfulness_empirical_p"])

    models = pd.read_csv(FAITH / "model_table.csv")[
        ["model", "branch", "overlap_hits", "k", "att_accuracy", "att_auc", "score_n"]
    ].rename(columns={"overlap_hits": "att_overlap_hits", "k": "att_overlap_k", "att_accuracy": "att_overlap",
                      "att_auc": "att_auc_n4000", "score_n": "att_score_n"})

    faith = pd.read_csv(FAITH / "faithfulness.csv")
    wide = faith.pivot(index="branch", columns="measure", values=["pearson_r", "pearson_p"])
    branch = pd.DataFrame({
        "att_faithfulness_r": wide[("pearson_r", "att_accuracy")],
        "att_faithfulness_p": wide[("pearson_p", "att_accuracy")],
        "att_faithfulness_r_auc": wide[("pearson_r", "att_auc")],
        "att_faithfulness_p_auc": wide[("pearson_p", "att_auc")],
    }).reset_index()

    out = (base.merge(models, on=["model", "branch"], how="outer")
               .merge(branch, on="branch", how="left")
               .sort_values(["branch", "model"]).reset_index(drop=True))
    path = OUT / "all_5_metrics_summary_v2.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} rows to {path}")
    pd.set_option("display.width", 220)
    print(out[["model", "branch", "predictive_r2", "att_overlap", "att_auc_n4000",
               "att_faithfulness_r", "att_faithfulness_r_auc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
