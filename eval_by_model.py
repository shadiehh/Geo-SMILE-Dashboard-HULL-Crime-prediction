"""Compare GeoSMILE's candidate black-box models: predictive fit and full attribution.

For each model in model_factory.MODEL_NAMES this:
  1. Fits Feature and Geo branch models and scores them on the temporal-holdout test
     split (MAE/RMSE/R2) -- independent of GeoSMILE.
  2. Runs the full GeoSMILE attribution pipeline against that model by invoking
     eval_by_n_perturb.py as a subprocess with --model <name> and a model-specific
     --output-dir, so no model's cached rankings are ever mistaken for another's.
  3. Computes expert-reference top-k overlap directly from each model's own raw SMILE
     rankings, not accuracy.json's accuracy/intersection/precision/recall/f1/
     overlap_ratio/jaccard_at_k fields -- those are known-unreliable at every N tested
     (see the module docstring of eval_by_n_perturb.py / hull_expert_topk.py). auroc/
     auprc from accuracy.json are reused as-is (confirmed reliable).

Writes outputs/eval_by_model/<model>/... (subprocess output, mirrors the n_<N> layout
eval_by_n_perturb.py already uses), outputs/eval_by_model/predictive_performance.csv,
and outputs/eval_by_model/model_comparison_summary.csv.

Runtime: training is cheap for every candidate, but GaussianProcessRegressor's predict
cost scales with training-set size and the SMILE loops call .predict() up to N times
per branch per invocation (1 main pass + 4 stability directions + 5 consistency
reseeds by default) -- expect roughly 1-1.5 hours for gp across both branches at the
default N=2000 with the full suite; the other five models combined add only a few
minutes. Use --skip-stability/--skip-consistency for a faster first pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from eval_by_n_perturb import Y_COL, compact_row
from geosmile_hull_twobranch import (
    FORBIDDEN_FEATURE_TOKENS,
    SPATIAL_COLUMNS,
    choose_feature_players,
    metrics,
    temporal_split,
)
from hull_expert_topk import (
    FEATURE_TOP_K,
    GEO_TOP_K,
    load_feature_expert_reference,
    load_geo_expert_reference,
)
from model_factory import MODEL_NAMES, fit_model, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "eval_by_model"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"

SEED = 42


def _log(msg: str) -> None:
    print(msg, flush=True)


def raw_topk_overlap(
    importance: pd.DataFrame, expert_ids: set[str], *, id_col: str, mag_col: str, k: int
) -> dict:
    """Expert-reference overlap computed directly from a model's own raw ranking.

    Bypasses hull_expert_topk.assign_geosmile_ranks/compute_att_suite -- their
    accuracy/intersection/precision/recall/f1/overlap_ratio/jaccard_at_k fields are
    confirmed unreliable at every N tested (the automated scorer under-counts matches
    against the expert reference).
    """
    ranked = importance.assign(_abs=importance[mag_col].astype(float).abs())
    top_ids = (
        ranked.sort_values(["_abs", id_col], ascending=[False, True], kind="mergesort")
        .head(k)[id_col]
        .astype(str)
    )
    top_set = set(top_ids)
    intersection = len(top_set & expert_ids)
    union = len(top_set | expert_ids)
    return {
        "expert_overlap_k": int(k),
        "expert_overlap_intersection": intersection,
        "expert_overlap_recall_at_k": intersection / len(expert_ids) if expert_ids else None,
        "expert_overlap_jaccard": intersection / union if union else None,
    }


def predictive_performance(
    panel: pd.DataFrame, players: list[str], train, validation, test, model_name: str
) -> list[dict]:
    rows = []
    for branch, feature_cols in (("Feature", players), ("Geo", SPATIAL_COLUMNS)):
        model = make_model(model_name, SEED)
        fit_model(
            model_name,
            model,
            panel.loc[train, feature_cols],
            panel.loc[train, Y_COL],
            panel.loc[validation, feature_cols],
            panel.loc[validation, Y_COL],
        )
        pred = model.predict(panel.loc[test, feature_cols])
        row = {"model": model_name, "branch": branch}
        row.update(metrics(panel.loc[test, Y_COL], pred))
        rows.append(row)
    return rows


def run_geosmile_pipeline(model_name: str, n: int, extra_args: list[str]) -> Path:
    model_dir = DEST / model_name
    cmd = [
        sys.executable,
        str(ROOT / "eval_by_n_perturb.py"),
        "--model",
        model_name,
        "--output-dir",
        str(model_dir),
        "--n-grid",
        str(n),
        *extra_args,
    ]
    _log(f"\n=== [{model_name}] running GeoSMILE pipeline (n={n}) ===")
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=ROOT)
    _log(f"=== [{model_name}] done in {(time.time() - t0) / 60:.1f} min ===")
    return model_dir / f"n_{n}" / "metrics.json"


def build_summary_rows(
    model_name: str,
    n: int,
    metrics_payload: dict,
    feat_expert: set[str],
    geo_expert: set[str],
    model_dir: Path,
) -> list[dict]:
    rows = []
    for branch, branch_key, importance_csv, id_col, mag_col, expert_ids, top_k in (
        ("Feature", "feature", "feature_importance.csv", "feature", "abs_coefficient", feat_expert, FEATURE_TOP_K),
        ("Geo", "geo", "geo_coefficients.csv", "lsoa21cd", "geo_abs_coefficient", geo_expert, GEO_TOP_K),
    ):
        row = compact_row(n, branch, metrics_payload)
        row.pop("accuracy", None)
        row.pop("accuracy_intersection", None)
        row["fidelity_r2_evaluation"] = row.pop("fidelity_r2")
        row["fidelity_r2_kernel_diagnostic"] = row.pop("smile_diagnostics_test_r2")

        acc = metrics_payload.get("accuracy", {}).get(branch_key, {}) or {}
        cons = metrics_payload.get("att_consistency", {}).get(branch_key, {}) or {}
        row["model"] = model_name
        row["auroc"] = acc.get("auroc")
        row["auprc"] = acc.get("auprc")
        row["fidelity_r2_consistency_mean"] = cons.get("Fidelity R2 Mean")

        importance = pd.read_csv(model_dir / f"n_{n}" / importance_csv)
        importance[id_col] = importance[id_col].astype(str)
        row.update(
            raw_topk_overlap(importance, expert_ids, id_col=id_col, mag_col=mag_col, k=top_k)
        )
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument(
        "--n",
        type=int,
        default=2000,
        help="Shared SMILE mask/perturbation count for both branches (default 2000).",
    )
    p.add_argument("--skip-stability", action="store_true")
    p.add_argument("--skip-consistency", action="store_true")
    p.add_argument("--skip-faithfulness", action="store_true")
    p.add_argument("--n-jobs", type=int, default=2, help="Model threads (default 2). Use -1 for all cores.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not PANEL_PATH.exists():
        raise SystemExit(f"Missing {PANEL_PATH}")
    DEST.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    bad = [p for p in players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise SystemExit(f"Spatial leak in Feature players: {bad}")
    train, validation, test = temporal_split(panel)

    feat_ref = load_feature_expert_reference()
    feat_expert = set(feat_ref.loc[feat_ref["expert_rank"] <= FEATURE_TOP_K, "feature"].astype(str))
    geo_ref = load_geo_expert_reference()
    geo_expert = set(geo_ref["lsoa21cd"].astype(str))

    extra_args = ["--n-jobs", str(args.n_jobs)]
    if args.skip_stability:
        extra_args.append("--skip-stability")
    if args.skip_consistency:
        extra_args.append("--skip-consistency")
    if args.skip_faithfulness:
        extra_args.append("--skip-faithfulness")

    predictive_rows: list[dict] = []
    summary_rows: list[dict] = []
    for model_name in args.models:
        _log(f"\n[{model_name}] predictive performance (Feature + Geo)...")
        predictive_rows.extend(
            predictive_performance(panel, players, train, validation, test, model_name)
        )

        metrics_path = run_geosmile_pipeline(model_name, args.n, extra_args)
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary_rows.extend(
            build_summary_rows(
                model_name, args.n, metrics_payload, feat_expert, geo_expert, DEST / model_name
            )
        )

    pred_new = pd.DataFrame(predictive_rows)
    summary_new = pd.DataFrame(summary_rows)
    pred_lookup = pred_new.set_index(["model", "branch"])[["mae", "rmse", "r2"]]
    summary_new = summary_new.join(pred_lookup, on=["model", "branch"])

    # Merge with any prior run's output rather than clobbering it, so a partial
    # --models run (e.g. adding one new candidate later) doesn't erase the rest.
    pred_path = DEST / "predictive_performance.csv"
    if pred_path.exists():
        pred_old = pd.read_csv(pred_path)
        pred_df = pd.concat(
            [pred_old[~pred_old["model"].isin(args.models)], pred_new], ignore_index=True
        )
    else:
        pred_df = pred_new
    pred_df.to_csv(pred_path, index=False)

    summary_path = DEST / "model_comparison_summary.csv"
    if summary_path.exists():
        summary_old = pd.read_csv(summary_path)
        summary_df = pd.concat(
            [summary_old[~summary_old["model"].isin(args.models)], summary_new], ignore_index=True
        )
    else:
        summary_df = summary_new
    summary_df.to_csv(summary_path, index=False)
    _log(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()
