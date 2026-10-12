"""For each candidate model, train ONE combined model (30 Feature players + 2
coordinates) and report:

  1. Its predictive R2/MAE/RMSE on the temporal holdout -- a single number per
     model, since there is only one model here (not a separate Feature R2 and
     Geo R2).
  2. GeoSMILE's Feature-branch fidelity/faithfulness/consistency/expert-overlap
     explaining THAT model, holding coordinates fixed at each row's true value.
  3. GeoSMILE's Geo-branch fidelity/faithfulness/consistency/expert-overlap
     explaining THAT SAME model, holding the 30 Feature players fixed.

Fidelity is the same "evaluation-split surrogate R2" computed everywhere else
in this pipeline (Dehghani-style). Expert-overlap is computed directly from
each model's own raw ranking (never accuracy.json's flagged fields -- see
eval_by_model.py's module docstring for why). Cardinal-neighbor ATT stability
is intentionally NOT computed here -- see combined_model_fidelity.py's
docstring.

Historical context: a combined predictive model ("Feature-B") was already
tried in this project and dropped -- no significant accuracy gain and worse
spatial-holdout generalization (outputs/hull_results_interpretation.md). This
script exists to make that comparison concrete across all candidate model
families, not to argue for reviving the combined design.

Writes outputs/eval_combined_model/<model>/results.json and
outputs/eval_combined_model/combined_model_comparison_summary.csv. Does not
touch outputs/eval_by_model/ or outputs/eval_by_n_perturb/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from combined_model_fidelity import (
    combined_consistency,
    combined_feature_smile,
    combined_geo_smile,
    combined_model_faithfulness,
)
from eval_by_model import raw_topk_overlap
from eval_by_n_perturb import Y_COL, dehghani_fidelity
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
DEST = OUT / "eval_combined_model"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"

SEED = 42
CONSISTENCY_SEED_BASE = 100
N_REPEATS = 5


def _log(msg: str) -> None:
    print(msg, flush=True)


def evaluate_model(
    model_name: str,
    *,
    panel: pd.DataFrame,
    players: list[str],
    columns: list[str],
    train,
    validation,
    test,
    X_test: pd.DataFrame,
    X_test_with_id: pd.DataFrame,
    baseline: pd.Series,
    feat_expert: set[str],
    geo_expert: set[str],
    args: argparse.Namespace,
) -> dict:
    _log(f"\n=== [{model_name}] combined model (32 inputs) ===")
    t0 = time.time()
    model = make_model(model_name, SEED, n_jobs=args.n_jobs)
    fit_model(
        model_name,
        model,
        panel.loc[train, columns],
        panel.loc[train, Y_COL],
        panel.loc[validation, columns],
        panel.loc[validation, Y_COL],
    )
    pred = model.predict(X_test)
    predictive = metrics(panel.loc[test, Y_COL], pred)
    _log(
        f"[{model_name}] predictive MAE={predictive['mae']:.4f} RMSE={predictive['rmse']:.4f} "
        f"R2={predictive['r2']:.4f}"
    )

    feat_imp, feat_diag, feat_masks, feat_resp, feat_dist = combined_feature_smile(
        model, X_test, players, baseline, args.n_feature, SEED
    )
    feature_fidelity = dehghani_fidelity(feat_masks, feat_resp, feat_dist)
    feature_fidelity["smile_diagnostics_test_r2"] = float(feat_diag["test_r2"])
    _log(f"[{model_name}] Feature fidelity R2={feature_fidelity['R2']:.4f}")

    geo_imp, geo_diag, geo_M, geo_resp, geo_dist = combined_geo_smile(
        model, panel, test, baseline, players, args.n_geo, SEED
    )
    geo_fidelity = dehghani_fidelity(geo_M, geo_resp, geo_dist)
    geo_fidelity["smile_diagnostics_test_r2"] = float(geo_diag["test_r2"])
    _log(f"[{model_name}] Geo fidelity R2={geo_fidelity['R2']:.4f}")

    if args.skip_faithfulness:
        faithfulness = {"feature": {}, "geo": {}}
    else:
        _log(f"[{model_name}] faithfulness (n_random={args.n_random})...")
        faithfulness = combined_model_faithfulness(
            model, X_test_with_id, players, baseline, feat_imp, geo_imp,
            FEATURE_TOP_K, GEO_TOP_K, args.n_random, SEED,
        )
        _log(
            f"[{model_name}] faithfulness top-random Feature={faithfulness['feature']['top_minus_random']:.4f} "
            f"| Geo={faithfulness['geo']['top_minus_random']:.4f}"
        )

    if args.skip_consistency:
        consistency = {"feature": {}, "geo": {}}
    else:
        _log(f"[{model_name}] consistency ({N_REPEATS} seeds from {CONSISTENCY_SEED_BASE})...")
        consistency = combined_consistency(
            model, X_test, panel, test, players, baseline,
            args.n_feature, args.n_geo, N_REPEATS, CONSISTENCY_SEED_BASE,
        )
        _log(
            f"[{model_name}] consistency Rank Spearman Feature={consistency['feature']['Rank Spearman Mean']:.3f} "
            f"| Geo={consistency['geo']['Rank Spearman Mean']:.3f}"
        )

    feat_overlap = raw_topk_overlap(
        feat_imp, feat_expert, id_col="feature", mag_col="abs_coefficient", k=FEATURE_TOP_K
    )
    geo_overlap = raw_topk_overlap(
        geo_imp, geo_expert, id_col="lsoa21cd", mag_col="geo_abs_coefficient", k=GEO_TOP_K
    )

    elapsed = time.time() - t0
    _log(f"[{model_name}] done in {elapsed / 60:.1f} min")

    result = {
        "model": model_name,
        "n_feature": args.n_feature,
        "n_geo": args.n_geo,
        "elapsed_sec": elapsed,
        "predictive_performance": predictive,
        "feature_branch": {
            "fidelity": feature_fidelity,
            "faithfulness": faithfulness["feature"],
            "consistency": consistency["feature"],
            "expert_overlap": feat_overlap,
        },
        "geo_branch": {
            "fidelity": geo_fidelity,
            "faithfulness": faithfulness["geo"],
            "consistency": consistency["geo"],
            "expert_overlap": geo_overlap,
        },
    }
    model_dir = DEST / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    feat_imp.to_csv(model_dir / "feature_importance.csv", index=False)
    geo_imp.to_csv(model_dir / "geo_coefficients.csv", index=False)
    return result


def summary_rows(result: dict) -> list[dict]:
    rows = []
    for branch, key in (("Feature", "feature_branch"), ("Geo", "geo_branch")):
        b = result[key]
        rows.append(
            {
                "model": result["model"],
                "branch": branch,
                "mae": result["predictive_performance"]["mae"],
                "rmse": result["predictive_performance"]["rmse"],
                "r2": result["predictive_performance"]["r2"],
                "fidelity_r2_evaluation": b["fidelity"].get("R2"),
                "fidelity_r2_kernel_diagnostic": b["fidelity"].get("smile_diagnostics_test_r2"),
                "fidelity_r2_consistency_mean": b["consistency"].get("Fidelity R2 Mean"),
                "faithfulness_top_minus_random": b["faithfulness"].get("top_minus_random"),
                "att_consistency_rank_spearman": b["consistency"].get("Rank Spearman Mean"),
                "att_consistency_top20pct_jaccard": b["consistency"].get("Top-20% Jaccard Mean"),
                "expert_overlap_intersection": b["expert_overlap"].get("expert_overlap_intersection"),
                "expert_overlap_recall_at_k": b["expert_overlap"].get("expert_overlap_recall_at_k"),
                "expert_overlap_jaccard": b["expert_overlap"].get("expert_overlap_jaccard"),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--n-feature", type=int, default=2000, help="Feature-branch SMILE mask count (default 2000).")
    p.add_argument("--n-geo", type=int, default=2000, help="Geo-branch SMILE perturbation count (default 2000).")
    p.add_argument("--n-random", type=int, default=1000, help="Faithfulness random-draw count (default 1000).")
    p.add_argument("--skip-faithfulness", action="store_true")
    p.add_argument("--skip-consistency", action="store_true")
    p.add_argument("--n-jobs", type=int, default=2)
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
    columns = players + SPATIAL_COLUMNS
    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, columns].median()
    X_test = panel.loc[test, columns].reset_index(drop=True)
    X_test_with_id = panel.loc[test, ["lsoa21cd"] + columns].reset_index(drop=True)

    feat_ref = load_feature_expert_reference()
    feat_expert = set(feat_ref.loc[feat_ref["expert_rank"] <= FEATURE_TOP_K, "feature"].astype(str))
    geo_ref = load_geo_expert_reference()
    geo_expert = set(geo_ref["lsoa21cd"].astype(str))

    all_rows: list[dict] = []
    for model_name in args.models:
        result = evaluate_model(
            model_name,
            panel=panel,
            players=players,
            columns=columns,
            train=train,
            validation=validation,
            test=test,
            X_test=X_test,
            X_test_with_id=X_test_with_id,
            baseline=baseline,
            feat_expert=feat_expert,
            geo_expert=geo_expert,
            args=args,
        )
        all_rows.extend(summary_rows(result))

    new_df = pd.DataFrame(all_rows)
    summary_path = DEST / "combined_model_comparison_summary.csv"
    if summary_path.exists():
        old_df = pd.read_csv(summary_path)
        df = pd.concat([old_df[~old_df["model"].isin(args.models)], new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(summary_path, index=False)
    _log(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()
