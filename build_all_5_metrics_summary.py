"""Consolidate all 5 Dehghani-style evaluation metrics -- ATT accuracy,
faithfulness, ATT stability, ATT consistency, ATT fidelity -- for all 10
candidate models, both branches, into one CSV. Read-only over already
computed results; no recomputation.

Sources, at the selected design (Feature N=1,024 / Geo N=4,000, seed 42):
  - ATT accuracy + faithfulness: outputs/accuracy_faithfulness_nsweep/
    summary.csv (accuracy_faithfulness_nsweep.py), filtered to the selected
    N per branch. ATT accuracy is verify_metrics_all_models.att_metrics()'s
    (TP+TN)/n_total formula against the current canonical expert reference
    -- genuine accuracy, not an overlap/intersection metric.
  - ATT stability + ATT consistency + ATT fidelity + predictive R2:
    outputs/verify_metrics_all_models/all_models_summary.csv
    (verify_metrics_all_models.py), for the 9 models it has been run for.
  - CNN is the one exception: its stability/consistency were deliberately
    deprioritized (CNN's Conv1D has no genuine spatial/temporal adjacency
    on either branch -- see model_factory.py -- so this was judged not
    worth the remaining compute; explicit user decision, not an oversight).
    CNN's predictive R2 comes from outputs/eval_by_model/
    predictive_performance.csv and its fidelity from outputs/
    response_metric_comparison/response_metric_comparison_summary.csv
    (wasserstein, Ridge surrogate, same N) instead.

Missing stability/consistency cells are left as NaN, not imputed or
back-filled -- the gap is real and should read as real in the output.

Writes outputs/all_5_metrics_summary.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

FEAT_N_SELECTED = 1024
GEO_N_SELECTED = 4000
R2_GATE = 0.7


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    af = pd.read_csv(OUT / "accuracy_faithfulness_nsweep" / "summary.csv")
    af_sel = af[((af["branch"] == "Feature") & (af["n"] == FEAT_N_SELECTED)) |
                ((af["branch"] == "Geo") & (af["n"] == GEO_N_SELECTED))].copy()

    verify = pd.read_csv(OUT / "verify_metrics_all_models" / "all_models_summary.csv")

    predictive = pd.read_csv(OUT / "eval_by_model" / "predictive_performance.csv")
    ns = pd.read_csv(OUT / "response_metric_comparison" / "response_metric_comparison_summary.csv")
    ns_ridge = ns[ns["response_metric"] == "wasserstein"]

    all_models = sorted(set(predictive["model"]) | set(verify["model"]) | set(af["model"]))
    all_branches = ["Feature", "Geo"]

    rows = []
    for model in all_models:
        for branch in all_branches:
            row = {"model": model, "branch": branch}

            pred_row = predictive[(predictive["model"] == model) & (predictive["branch"] == branch)]
            r2 = float(pred_row.iloc[0]["r2"]) if not pred_row.empty else np.nan
            row["predictive_r2"] = r2
            row["predictive_mae"] = float(pred_row.iloc[0]["mae"]) if not pred_row.empty else np.nan
            row["predictive_rmse"] = float(pred_row.iloc[0]["rmse"]) if not pred_row.empty else np.nan
            passes_gate = np.isfinite(r2) and r2 > R2_GATE

            v_row = verify[(verify["model"] == model) & (verify["branch"] == branch)]
            if not passes_gate:
                # R2<=0.7: GeoSMILE was never validly fit for this (model, branch) per the
                # project-wide gate. Some underlying files (verify_metrics_all_models'
                # all_models_summary.csv, specifically mlp/Geo) predate that gate being
                # introduced and still carry stale numbers -- exclude them here rather
                # than propagate them into this consolidated table.
                row["fidelity_r2"] = np.nan
                row["fidelity_weighted_r2"] = np.nan
                row["stability_top_k_jaccard"] = np.nan
                row["stability_abs_coef_spearman"] = np.nan
                row["consistency_rank_spearman_r10"] = np.nan
                row["consistency_top20pct_jaccard_r10"] = np.nan
            elif not v_row.empty:
                v = v_row.iloc[0]
                row["fidelity_r2"] = v["fidelity_r2"]
                row["fidelity_weighted_r2"] = v["fidelity_weighted_r2"]
                row["stability_top_k_jaccard"] = v["stability_top_k_jaccard"]
                row["stability_abs_coef_spearman"] = v["stability_abs_coef_spearman"]
                row["consistency_rank_spearman_r10"] = v["consistency_rank_spearman_r10"]
                row["consistency_top20pct_jaccard_r10"] = v["consistency_top20pct_jaccard_r10"]
            else:
                n_sel = FEAT_N_SELECTED if branch == "Feature" else GEO_N_SELECTED
                fid_row = ns_ridge[(ns_ridge["model"] == model) & (ns_ridge["branch"] == branch) & (ns_ridge["n"] == n_sel)]
                row["fidelity_r2"] = float(fid_row.iloc[0]["r2"]) if not fid_row.empty else np.nan
                row["fidelity_weighted_r2"] = float(fid_row.iloc[0]["weighted_r2"]) if not fid_row.empty else np.nan
                row["stability_top_k_jaccard"] = np.nan
                row["stability_abs_coef_spearman"] = np.nan
                row["consistency_rank_spearman_r10"] = np.nan
                row["consistency_top20pct_jaccard_r10"] = np.nan

            a_row = af_sel[(af_sel["model"] == model) & (af_sel["branch"] == branch)]
            if not a_row.empty:
                a = a_row.iloc[0]
                row["att_accuracy"] = a["acc_ATT_accuracy"]
                row["att_f1"] = a["acc_ATT_F1"]
                row["att_auroc"] = a["acc_ATT_AUROC_continuous"]
                row["faithfulness_top_minus_random"] = a["faith_top_minus_random"]
                row["faithfulness_empirical_p"] = a["faith_empirical_p_top_vs_random"]
            else:
                for k in ("att_accuracy", "att_f1", "att_auroc", "faithfulness_top_minus_random", "faithfulness_empirical_p"):
                    row[k] = np.nan

            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["predictive_r2"], how="all").reset_index(drop=True)
    # Drop rows where literally nothing is known for this (model, branch) -- e.g. mlp/Geo, svr/Geo, cnn/Geo, all gated out everywhere.
    metric_cols = [c for c in df.columns if c not in ("model", "branch")]
    df = df[df[metric_cols].notna().any(axis=1)].reset_index(drop=True)

    out_path = OUT / "all_5_metrics_summary.csv"
    df.to_csv(out_path, index=False)
    _log(f"Wrote {len(df)} rows to {out_path}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    _log(df.to_string(index=False))


if __name__ == "__main__":
    main()
