"""Full rigorous 5-metric verification (ATT accuracy/F1/AUROC, faithfulness,
ATT stability, ATT consistency at R=10, ATT fidelity with weighted-adjusted R2)
at the fixed design (Feature = 1,024 masks / 30 players; Geo = 4,000
perturbations / 168 LSOA players; Ridge surrogate; Wasserstein-1 response;
SMILE seed 42) -- extended across all model_factory.MODEL_NAMES, not just the
published XGBoost.

Geo N was raised from the originally-published 2,000 to 4,000 following a
direct finding (a further XGBoost sweep at other N) that N=4,000 achieves
higher Geo fidelity than N=2,000. Since this deliberately departs from the
published design, this script never compares against the archived beta_0/
accuracy numbers (those are N=2,000-specific by definition) -- see
verify_published_metrics.py for that strict N=2,000 reproduction check, left
unchanged at N=2,000.

Reuses verify_published_metrics.py's metric-1/2/5 logic and adds a fresh
ATT-stability run per model (att_stability_asymmetric below, an N_feat/N_geo
-separated adaptation of eval_by_n_perturb.att_stability_for_n -- that
function only supports one shared N for both branches, which the published
asymmetric design doesn't fit).

Only XGBoost has an archived beta_0 to check reproduction against (see
verify_published_metrics.py for that specific check, already run separately).
For every model here -- including XGBoost, for a consistent comparison row --
metrics are computed fresh from that model's own Feature/Geo SMILE ranking at
seed 42, not by loading anything archived.

Predictive R2/MAE/RMSE are NOT recomputed here -- reused as-is from
outputs/eval_by_model/predictive_performance.csv (same temporal-holdout
computation regardless of SMILE settings, already run for all 7 models).

Writes outputs/verify_metrics_all_models/<model>/results.json and
outputs/verify_metrics_all_models/all_models_summary.csv.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compute_att_stability import DIRECTIONS, compare_attributions, hull_row_pred_spearman
from eval_by_n_perturb import Y_COL, dehghani_fidelity, feature_responses, load_cached_neighbors, model_based_faithfulness
from geosmile_hull_twobranch import (
    FORBIDDEN_FEATURE_TOKENS,
    SPATIAL_COLUMNS,
    choose_feature_players,
    geo_smile_coefficients,
    smile_global_feature,
    temporal_split,
)
from hull_boundary_neighbors import expand_panel
from hull_dehghani_eval import consistency_metrics, split_masks
from hull_expert_topk import (
    FEATURE_TOP_K,
    GEO_TOP_K,
    load_feature_expert_reference,
    load_geo_expert_reference,
)
from model_factory import MODEL_NAMES, fit_model, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "verify_metrics_all_models"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"
PREDICTIVE_CSV = OUT / "eval_by_model" / "predictive_performance.csv"

FEAT_N_MASKS = 1024
GEO_N_PERTURB = 4000  # per user: N=4000 achieved higher Geo fidelity for xgboost than the published N=2000
SMILE_SEED = 42
CONSISTENCY_SEEDS = list(range(42, 52))  # R = 10
N_RANDOM_FAITH = 1000


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Metric 1
# --------------------------------------------------------------------------
def att_metrics(importance: pd.DataFrame, expert_set: set[str], *, id_col: str, mag_col: str, k: int, n_total: int) -> dict:
    ranked = importance.assign(_abs=importance[mag_col].astype(float).abs())
    ranked = ranked.sort_values(["_abs", id_col], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    S_k = set(ranked.head(k)[id_col].astype(str))
    G_k = set(expert_set)
    TP, FP, FN = len(S_k & G_k), len(S_k - G_k), len(G_k - S_k)
    TN = n_total - len(S_k | G_k)
    acc = (TP + TN) / n_total
    f1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) else float("nan")
    y_true = ranked[id_col].astype(str).isin(G_k).astype(int).to_numpy()
    y_score = ranked["_abs"].to_numpy()
    auroc = float(roc_auc_score(y_true, y_score))
    return {
        "n": n_total, "k": k, "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "ATT_accuracy": acc, "ATT_F1": f1, "ATT_AUROC_continuous": auroc,
        "precision_at_k": TP / len(S_k) if S_k else float("nan"),
        "recall_at_k": TP / len(G_k) if G_k else float("nan"),
        "jaccard_at_k": TP / len(S_k | G_k) if (S_k | G_k) else float("nan"),
    }


# --------------------------------------------------------------------------
# Metric 3: N_feat/N_geo-separated cardinal-neighbor stability
# (adapted from eval_by_n_perturb.att_stability_for_n, which only supports
#  one shared N for both branches)
# --------------------------------------------------------------------------
def att_stability_asymmetric(
    *, n_feat: int, n_geo: int, model_feat, model_geo, panel, X_test, players, baseline,
    hull_lsoas, base_feat, base_geo, neighbors, extra_panel, seed,
) -> dict:
    extra_by_dir = {
        str(row.direction): extra_panel[extra_panel["lsoa21cd"].astype(str) == str(row.lsoa21cd)].copy()
        for row in neighbors.itertuples()
    }
    base_feat_top = set(
        base_feat.sort_values(["abs_coefficient", "feature"], ascending=[False, True], kind="mergesort")
        .head(FEATURE_TOP_K)["feature"].astype(str)
    )
    base_geo_top = set(
        base_geo.sort_values(["geo_abs_coefficient", "lsoa21cd"], ascending=[False, True], kind="mergesort")
        .head(GEO_TOP_K)["lsoa21cd"].astype(str)
    )
    per_direction = {}
    feat_j, feat_rho, geo_j, geo_rho, geo_rank = [], [], [], [], []
    for direction in DIRECTIONS:
        extra_rows = extra_by_dir[direction]
        code = str(extra_rows["lsoa21cd"].iloc[0])
        nm = str(extra_rows["lsoa21nm"].iloc[0])
        expanded = expand_panel(panel, extra_rows)
        _, _, exp_test = temporal_split(expanded)
        X_exp = expanded.loc[exp_test].reset_index(drop=True)
        feat_imp, _, _, _, _ = smile_global_feature(model_feat, X_exp, players, baseline, n_feat, seed)
        geo_imp, _ = geo_smile_coefficients(
            model_geo, expanded, exp_test, baseline, n_geo, seed, sample_lsoas=hull_lsoas
        )
        metrics = compare_attributions(
            base_feat, feat_imp, base_geo, geo_imp,
            base_feat_top=base_feat_top, base_geo_top=base_geo_top, added_lsoa=code,
        )
        f_pred, _ = hull_row_pred_spearman(model_feat, X_test, X_exp, players)
        g_pred, _ = hull_row_pred_spearman(model_geo, X_test, X_exp, SPATIAL_COLUMNS)
        feat_j.append(metrics["feature_jaccard"]); feat_rho.append(metrics["feature_spearman"])
        geo_j.append(metrics["geo_jaccard"]); geo_rho.append(metrics["geo_spearman"])
        geo_rank.append(metrics["geo_extra_rank"])
        per_direction[direction] = {
            "lsoa21cd": code, "lsoa21nm": nm,
            **{k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v) for k, v in metrics.items()},
            "feature_pred_spearman": f_pred, "geo_pred_spearman": g_pred,
        }
    return {
        "n_feat": int(n_feat), "n_geo": int(n_geo), "smile_seed": int(seed),
        "neighbors": per_direction,
        "feature": {"top_k_jaccard_mean": float(np.mean(feat_j)), "abs_coef_spearman_mean": float(np.mean(feat_rho))},
        "geo": {"top_k_jaccard_mean": float(np.mean(geo_j)), "abs_coef_spearman_mean": float(np.mean(geo_rho)),
                "mean_geo_extra_rank_out_of_169": float(np.mean(geo_rank))},
    }


R2_GATE = 0.7  # branches at/below this (or non-positive) skip GeoSMILE + all other metrics entirely


def _skipped(r2) -> dict:
    return {"skipped": True, "reason": f"predictive R2={r2} <= {R2_GATE} (or non-positive)"}


def evaluate_model(model_name: str, *, panel, players, train, validation, test, baseline, X_test,
                    hull_lsoas, feat_expert, geo_expert, neighbors, extra_panel, n_jobs: int,
                    do_feature: bool, do_geo: bool, feat_r2: float, geo_r2: float) -> dict:
    _log(f"\n=== [{model_name}] === Feature: {'run' if do_feature else f'SKIPPED (R2={feat_r2:.3f})'}, "
         f"Geo: {'run' if do_geo else f'SKIPPED (R2={geo_r2:.3f})'}")
    t0 = time.time()
    model_feat = model_geo = None
    feat_imp = feat_diag = feat_masks_pool = feat_all_pert = feat_original = None
    geo_imp = geo_diag = geo_M = geo_resp2 = geo_dist2 = None

    if do_feature:
        model_feat = make_model(model_name, SMILE_SEED, n_jobs=n_jobs)
        fit_model(model_name, model_feat, panel.loc[train, players], panel.loc[train, Y_COL],
                  panel.loc[validation, players], panel.loc[validation, Y_COL])
        _log(f"[{model_name}] Feature SMILE n={FEAT_N_MASKS}, seed={SMILE_SEED}...")
        feat_imp, feat_diag, feat_masks_pool, feat_all_pert, feat_original = smile_global_feature(
            model_feat, X_test[players], players, baseline, FEAT_N_MASKS, SMILE_SEED
        )
    if do_geo:
        model_geo = make_model(model_name, SMILE_SEED, n_jobs=n_jobs)
        fit_model(model_name, model_geo, panel.loc[train, SPATIAL_COLUMNS], panel.loc[train, Y_COL],
                  panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, Y_COL])
        _log(f"[{model_name}] Geo SMILE n={GEO_N_PERTURB}, seed={SMILE_SEED}...")
        geo_imp, geo_diag, geo_M, geo_resp2, geo_dist2 = geo_smile_coefficients(
            model_geo, panel, test, baseline, GEO_N_PERTURB, SMILE_SEED, return_pool=True
        )
        geo_imp["lsoa21cd"] = geo_imp["lsoa21cd"].astype(str)

    metric1 = {
        "feature": att_metrics(feat_imp, feat_expert, id_col="feature", mag_col="coefficient", k=FEATURE_TOP_K, n_total=30) if do_feature else _skipped(feat_r2),
        "geo": att_metrics(geo_imp, geo_expert, id_col="lsoa21cd", mag_col="geo_coefficient", k=GEO_TOP_K, n_total=168) if do_geo else _skipped(geo_r2),
    }
    if do_feature or do_geo:
        _log(f"[{model_name}] ATT accuracy Feature={'%.4f' % metric1['feature']['ATT_accuracy'] if do_feature else 'skipped'} "
             f"Geo={'%.4f' % metric1['geo']['ATT_accuracy'] if do_geo else 'skipped'}")

    if do_feature and do_geo:
        _log(f"[{model_name}] faithfulness (n_random={N_RANDOM_FAITH})...")
        metric2 = model_based_faithfulness(
            model_feat=model_feat, model_geo=model_geo, X_test=X_test, players=players, baseline=baseline,
            feat_imp=feat_imp, geo_imp=geo_imp, n_random=N_RANDOM_FAITH, seed=SMILE_SEED,
        )
        _log(f"[{model_name}] ATT stability (4 cardinal neighbors)...")
        metric3 = att_stability_asymmetric(
            n_feat=FEAT_N_MASKS, n_geo=GEO_N_PERTURB, model_feat=model_feat, model_geo=model_geo,
            panel=panel, X_test=X_test, players=players, baseline=baseline, hull_lsoas=hull_lsoas,
            base_feat=feat_imp, base_geo=geo_imp, neighbors=neighbors, extra_panel=extra_panel, seed=SMILE_SEED,
        )
    else:
        note = "joint metric requires both Feature and Geo to pass the R2 gate; skipped since at least one did not"
        metric2 = {"feature": _skipped(feat_r2) if not do_feature else {"note": note}, "geo": _skipped(geo_r2) if not do_geo else {"note": note}, "note": note}
        metric3 = {"feature": _skipped(feat_r2) if not do_feature else {"note": note}, "geo": _skipped(geo_r2) if not do_geo else {"note": note}, "note": note}

    metric4 = {"seeds": CONSISTENCY_SEEDS, "feature": _skipped(feat_r2), "geo": _skipped(geo_r2)}
    if do_feature or do_geo:
        _log(f"[{model_name}] ATT consistency R={len(CONSISTENCY_SEEDS)}...")
        feat_runs, geo_runs, feat_r2s, geo_r2s = [], [], [], []
        geo_index = None
        for seed in CONSISTENCY_SEEDS:
            if do_feature:
                f_imp, f_diag, *_ = smile_global_feature(model_feat, X_test[players], players, baseline, FEAT_N_MASKS, seed)
                f_imp = f_imp.set_index("feature").loc[players]
                feat_runs.append(f_imp["coefficient"].to_numpy(dtype=float))
                feat_r2s.append(float(f_diag["test_r2"]))
            if do_geo:
                g_imp, g_diag = geo_smile_coefficients(model_geo, panel, test, baseline, GEO_N_PERTURB, seed)
                g_imp["lsoa21cd"] = g_imp["lsoa21cd"].astype(str)
                g_imp = g_imp.sort_values("lsoa21cd").reset_index(drop=True)
                if geo_index is None:
                    geo_index = g_imp["lsoa21cd"].tolist()
                g_imp = g_imp.set_index("lsoa21cd").loc[geo_index]
                geo_runs.append(g_imp["geo_coefficient"].to_numpy(dtype=float))
                geo_r2s.append(float(g_diag["test_r2"]))
        if do_feature:
            feat_consist = consistency_metrics(np.abs(np.asarray(feat_runs)), top_pct=20)
            feat_consist["Fidelity R2 Mean"] = float(np.mean(feat_r2s))
            metric4["feature"] = feat_consist
        if do_geo:
            geo_consist = consistency_metrics(np.abs(np.asarray(geo_runs)), top_pct=20)
            geo_consist["Fidelity R2 Mean"] = float(np.mean(geo_r2s))
            metric4["geo"] = geo_consist

    metric5 = {"feature": _skipped(feat_r2), "geo": _skipped(geo_r2)}
    if do_feature or do_geo:
        _log(f"[{model_name}] ATT fidelity (weighted-adjusted R2)...")

        def adj_r2(fid, n_p, n_s):
            denom = n_p - n_s - 1
            if denom <= 0:
                return None, denom
            return 1 - (1 - fid["Weighted R2"]) * (n_p - 1) / denom, denom

        if do_feature:
            _, _, te_feat = split_masks(FEAT_N_MASKS, SMILE_SEED)
            feat_resp = feature_responses(feat_masks_pool, feat_all_pert, feat_original)
            feat_dist = 1.0 - feat_masks_pool.mean(axis=1)
            feat_fid = dehghani_fidelity(feat_masks_pool, feat_resp, feat_dist)
            feat_r2adj, feat_denom = adj_r2(feat_fid, len(te_feat), 30)
            metric5["feature"] = {**feat_fid, "N_p": len(te_feat), "N_s": 30, "N_p_minus_N_s_minus_1": feat_denom,
                                   "R2_adj_w": feat_r2adj, "guard_triggered": feat_denom <= 0}
        if do_geo:
            _, _, te_geo = split_masks(GEO_N_PERTURB, SMILE_SEED)
            geo_fid = dehghani_fidelity(geo_M, geo_resp2, geo_dist2)
            geo_r2adj, geo_denom = adj_r2(geo_fid, len(te_geo), 168)
            metric5["geo"] = {**geo_fid, "N_p": len(te_geo), "N_s": 168, "N_p_minus_N_s_minus_1": geo_denom,
                               "R2_adj_w": geo_r2adj, "guard_triggered": geo_denom <= 0}

    elapsed = time.time() - t0
    _log(f"[{model_name}] done in {elapsed / 60:.1f} min")

    result = {
        "model": model_name, "elapsed_sec": elapsed, "do_feature": do_feature, "do_geo": do_geo,
        "feature_kernel": feat_diag["best_kernel"] if do_feature else None,
        "geo_kernel": geo_diag["best_kernel"] if do_geo else None,
        "metric1_att_accuracy": metric1, "metric2_faithfulness": metric2,
        "metric3_att_stability": metric3, "metric4_att_consistency": metric4,
        "metric5_att_fidelity": metric5,
    }
    model_dir = DEST / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if feat_imp is not None:
        feat_imp.to_csv(model_dir / "feature_importance.csv", index=False)
    if geo_imp is not None:
        geo_imp.to_csv(model_dir / "geo_coefficients.csv", index=False)
    return result


def summary_row(result: dict, predictive: pd.DataFrame) -> list[dict]:
    rows = []
    for branch, key, ran in (("Feature", "feature", result["do_feature"]), ("Geo", "geo", result["do_geo"])):
        pred_row = predictive[(predictive["model"] == result["model"]) & (predictive["branch"] == branch)].iloc[0]
        m1, m2, m3, m4, m5 = (result[f"metric{i}_{n}"][key] for i, n in (
            (1, "att_accuracy"), (2, "faithfulness"), (3, "att_stability"), (4, "att_consistency"), (5, "att_fidelity")))
        rows.append({
            "model": result["model"], "branch": branch, "geosmile_fit": ran,
            "predictive_r2": pred_row["r2"], "predictive_mae": pred_row["mae"], "predictive_rmse": pred_row["rmse"],
            "att_accuracy": m1.get("ATT_accuracy"), "att_f1": m1.get("ATT_F1"), "att_auroc": m1.get("ATT_AUROC_continuous"),
            "faithfulness_top_minus_random": m2.get("top_minus_random"), "faithfulness_p": m2.get("empirical_p_top_vs_random"),
            "stability_top_k_jaccard": m3.get("top_k_jaccard_mean"), "stability_abs_coef_spearman": m3.get("abs_coef_spearman_mean"),
            "consistency_rank_spearman_r10": m4.get("Rank Spearman Mean"), "consistency_top20pct_jaccard_r10": m4.get("Top-20% Jaccard Mean"),
            "consistency_fidelity_r2_mean_r10": m4.get("Fidelity R2 Mean"),
            "fidelity_r2": m5.get("R2"), "fidelity_weighted_r2": m5.get("Weighted R2"), "fidelity_r2_adj_w": m5.get("R2_adj_w"),
            "N_p": m5.get("N_p"), "N_s": m5.get("N_s"), "fidelity_guard_triggered": m5.get("guard_triggered"),
            "kernel": result["feature_kernel"] if branch == "Feature" else result["geo_kernel"],
        })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--n-jobs", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not PANEL_PATH.exists():
        raise SystemExit(f"Missing {PANEL_PATH}")
    if not PREDICTIVE_CSV.exists():
        raise SystemExit(f"Missing {PREDICTIVE_CSV} -- run eval_by_model.py first")
    DEST.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    bad = [p for p in players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise SystemExit(f"Spatial leak in Feature players: {bad}")
    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
    X_test = panel.loc[test].reset_index(drop=True)
    hull_lsoas = sorted(panel["lsoa21cd"].astype(str).unique())
    neighbors, extra_panel = load_cached_neighbors()

    feat_ref = load_feature_expert_reference()
    feat_expert = set(feat_ref.loc[feat_ref["expert_rank"] <= FEATURE_TOP_K, "feature"].astype(str))
    geo_ref = load_geo_expert_reference()
    geo_expert = set(geo_ref["lsoa21cd"].astype(str))

    predictive = pd.read_csv(PREDICTIVE_CSV)

    all_rows: list[dict] = []
    for model_name in args.models:
        pred_feat = predictive[(predictive["model"] == model_name) & (predictive["branch"] == "Feature")]
        pred_geo = predictive[(predictive["model"] == model_name) & (predictive["branch"] == "Geo")]
        feat_r2 = float(pred_feat.iloc[0]["r2"]) if not pred_feat.empty else float("nan")
        geo_r2 = float(pred_geo.iloc[0]["r2"]) if not pred_geo.empty else float("nan")
        do_feature = np.isfinite(feat_r2) and feat_r2 > R2_GATE
        do_geo = np.isfinite(geo_r2) and geo_r2 > R2_GATE
        if not do_feature and not do_geo:
            _log(f"\n=== [{model_name}] === SKIPPED ENTIRELY: both branches at/below R2={R2_GATE} "
                 f"(Feature={feat_r2:.3f}, Geo={geo_r2:.3f})")
            continue
        result = evaluate_model(
            model_name, panel=panel, players=players, train=train, validation=validation, test=test,
            baseline=baseline, X_test=X_test, hull_lsoas=hull_lsoas, feat_expert=feat_expert, geo_expert=geo_expert,
            neighbors=neighbors, extra_panel=extra_panel, n_jobs=args.n_jobs,
            do_feature=do_feature, do_geo=do_geo, feat_r2=feat_r2, geo_r2=geo_r2,
        )
        all_rows.extend(summary_row(result, predictive))

    new_df = pd.DataFrame(all_rows)
    summary_path = DEST / "all_models_summary.csv"
    if summary_path.exists():
        old_df = pd.read_csv(summary_path)
        df = pd.concat([old_df[~old_df["model"].isin(args.models)], new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(summary_path, index=False)
    _log(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()
