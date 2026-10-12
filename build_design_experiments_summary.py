"""Consolidate every SMILE design-robustness experiment run in this project
(perturbation-count sweep, response-metric sweep, full surrogate menu, and
the Feature-branch surrogate x N interaction) into one long-format CSV, plus
copy the combined-model-architecture surrogate check as a labeled secondary
table. Read-only over existing result CSVs -- no recomputation.

Unified schema: experiment, model, category, branch, n, response_metric,
role, surrogate, r2, weighted_r2, mae, wmae, rmse, wrmse, best_kernel.
"missing" fields (e.g. weighted metrics for tree surrogates, which were
never fit with sample_weight in some rows) are left blank (NaN), not
imputed or dropped.

Writes outputs/design_experiments/{design_experiments_summary,
combined_model_surrogate_comparison}.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "design_experiments"

COLUMNS = ["experiment", "model", "category", "branch", "n", "response_metric",
           "role", "surrogate", "r2", "weighted_r2", "mae", "wmae", "rmse", "wrmse", "best_kernel"]

CATEGORY = {
    "xgboost": "boosting", "lightgbm": "boosting", "catboost": "boosting",
    "svr": "kernel", "gp": "kernel",
    "mlp": "neural", "dl": "neural",
}

# Fixed N used to generate the pools surrogate_comparison_extended.py reused
# (from surrogate_comparison.py's FEAT_N_MASKS / GEO_N_PERTURB constants).
FIXED_N = {"Feature": 1024, "Geo": 4000}


def load_n_perturbation_sweep() -> pd.DataFrame:
    df = pd.read_csv(OUT / "response_metric_comparison" / "response_metric_comparison_summary.csv")
    df["experiment"] = "n_perturbation_sweep"
    df["category"] = df["model"].map(CATEGORY)
    df["role"] = "candidate"
    df["surrogate"] = "ridge_kernel_weighted"
    return df.reindex(columns=COLUMNS)


def load_surrogate_menu_fixed_n() -> pd.DataFrame:
    df = pd.read_csv(OUT / "surrogate_comparison_extended" / "surrogate_comparison_extended_summary.csv")
    df["experiment"] = "surrogate_menu_fixed_n"
    df["n"] = df["branch"].map(FIXED_N)
    df["response_metric"] = "wasserstein"
    return df.reindex(columns=COLUMNS)


def load_surrogate_by_n_feature() -> pd.DataFrame:
    df = pd.read_csv(OUT / "surrogate_by_n" / "surrogate_by_n_summary.csv")
    df["experiment"] = "surrogate_by_n_feature"
    df["response_metric"] = "wasserstein"
    return df.reindex(columns=COLUMNS)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    parts = [load_n_perturbation_sweep(), load_surrogate_menu_fixed_n(), load_surrogate_by_n_feature()]
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["experiment", "model", "branch", "n", "role", "surrogate"]).reset_index(drop=True)
    out_path = DEST / "design_experiments_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.groupby(["experiment", "branch"]).size())

    combined = pd.read_csv(OUT / "surrogate_comparison_combined" / "surrogate_comparison_summary.csv")
    combined_out = DEST / "combined_model_surrogate_comparison.csv"
    combined.to_csv(combined_out, index=False)
    print(f"\nWrote {len(combined)} rows to {combined_out} (combined-model architecture, separate reference table)")


if __name__ == "__main__":
    main()
