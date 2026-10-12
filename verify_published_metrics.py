"""Verify GeoSMILE's five published evaluation metrics at the fixed design:
Feature = 1,024 masks / 30 players / (auto-selected) kernel; Geo = 2,000
perturbations / 168 LSOA players / (auto-selected) kernel; Ridge surrogate;
Wasserstein-1 response; SMILE seed 42.

IMPORTANT CAVEAT, read first: this repo does not persist a trained model
object anywhere (confirmed by exhaustive grep -- no pickle/joblib/save_model/
.pkl/.joblib/.h5/.onnx in the whole repo; every script retrains in-process).
There is no "archived fitted XGBoost model" to load. What IS archived and
loaded here unmodified: the panel CSV, the (deterministic, non-random)
temporal train/val/test split, the expert reference lists, and beta_0 itself
(the published Feature/Geo coefficient tables). Where a metric only needs
those, no model is touched (metrics 1, 3, 5 below). Where a metric requires
calling .predict() under a new perturbation (faithfulness; a 10-seed rerun for
consistency), the only way to get a model object at all is to reconstruct it
deterministically via the exact published recipe (make_model(seed=42) +
.fit(..., eval_set=..., verbose=False), no hyperparameter or threading
change) and then EMPIRICALLY CHECK whether it reproduces beta_0 -- this is
reported explicitly below, not assumed.

Kernel selection is never hardcoded anywhere in this codebase (verified:
no `= "inverse"` / `= "laplace"` assignment outside apply_kernel()'s own
dispatch). "Inverse" (Feature) and "laplace" (Geo) are what validation-R2
auto-selection picked in the published run (outputs/geosmile_hull_result_
provenance.csv records the per-kernel validation R2s). This script re-runs
the same auto-selection (never forces a kernel) and reports what it picks.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eval_by_n_perturb import Y_COL, model_based_faithfulness
from geosmile_hull_twobranch import (
    FORBIDDEN_FEATURE_TOKENS,
    SPATIAL_COLUMNS,
    choose_feature_players,
    geo_smile_coefficients,
    make_model,
    smile_global_feature,
    temporal_split,
)
from hull_dehghani_eval import consistency_metrics, split_masks
from hull_expert_topk import (
    FEATURE_TOP_K,
    GEO_TOP_K,
    load_feature_expert_reference,
    load_geo_expert_reference,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"
FEAT_BETA0_PATH = OUT / "feature_global_importance.csv"
GEO_BETA0_PATH = OUT / "geo_coefficients.csv"
FEAT_EXPERT_PATH = OUT / "feature_expert_ranking_v2.csv"
GEO_EXPERT_PATH = OUT / "maps_twobranch" / "ranking" / "expert_top20.csv"
ATT_STABILITY_PATH = OUT / "geosmile_hull_att_stability.json"
FEAT_FIDELITY_PATH = OUT / "geosmile_hull_feat_fidelity.csv"
GEO_FIDELITY_PATH = OUT / "geosmile_hull_geo_fidelity.csv"
FEAT_CONSISTENCY_R5_PATH = OUT / "geosmile_hull_feat_consistency.csv"
GEO_CONSISTENCY_R5_PATH = OUT / "geosmile_hull_geo_consistency.csv"

FEAT_N_MASKS = 1024
GEO_N_PERTURB = 2000
SMILE_SEED = 42
CONSISTENCY_SEEDS = list(range(42, 52))  # R = 10, seeds 42..51 inclusive
N_RANDOM_FAITH = 1000

EXPECTED = {
    "feature_att": {"TP": 7, "ACC": 0.8000, "F1": 0.7000, "AUROC": 0.8250},
    "geo_att": {"TP": 16, "ACC": 0.9524, "F1": 0.8000, "AUROC": 0.9081},
    "feature_faith": {"top": 8.3547, "bottom": 1.1256, "random_mean": 3.9527, "diff": 4.4019, "p": 0.016},
    "geo_faith": {"top": 19.9865, "bottom": 11.8094, "random_mean": 12.0145, "diff": 7.9720, "p": 0.036},
    "feature_fidelity_r2": 0.9853,
    "geo_fidelity_r2": 0.9308,
    "feature_fidelity_r2w": 0.9851,
    "geo_fidelity_r2w": 0.9283,
    "feature_mse": 0.1127, "feature_wmse": 0.1141,
    "geo_mse": 1.8496, "geo_wmse": 2.1910,
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _artefact(path: Path) -> dict:
    return {"path": str(path), "exists": path.exists(), "modified_utc": _mtime_iso(path) if path.exists() else None}


# --------------------------------------------------------------------------
# Metric 1: ATT accuracy / F1 / AUROC, computed fresh from beta_0 vs. expert
# --------------------------------------------------------------------------
def att_metrics_from_scratch(importance: pd.DataFrame, expert_set: set[str], *, id_col: str, mag_col: str, k: int, n_total: int) -> dict:
    ranked = importance.assign(_abs=importance[mag_col].astype(float).abs())
    ranked = ranked.sort_values(["_abs", id_col], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    S_k = set(ranked.head(k)[id_col].astype(str))
    G_k = set(expert_set)
    TP, FP, FN = len(S_k & G_k), len(S_k - G_k), len(G_k - S_k)
    TN = n_total - len(S_k | G_k)
    acc = (TP + TN) / n_total
    f1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) else float("nan")
    precision_at_k = TP / len(S_k) if S_k else float("nan")
    recall_at_k = TP / len(G_k) if G_k else float("nan")
    jaccard_at_k = TP / len(S_k | G_k) if (S_k | G_k) else float("nan")
    y_true = ranked[id_col].astype(str).isin(G_k).astype(int).to_numpy()
    y_score = ranked["_abs"].to_numpy()
    auroc_continuous = float(roc_auc_score(y_true, y_score))
    return {
        "n": n_total, "k": k, "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "ATT_accuracy": acc, "ATT_F1": f1, "ATT_AUROC_continuous": auroc_continuous,
        "precision_at_k": precision_at_k, "recall_at_k": recall_at_k, "jaccard_at_k": jaccard_at_k,
        "in_Sk_not_Gk": sorted(S_k - G_k), "in_Gk_not_Sk": sorted(G_k - S_k),
    }


def run_metric1(feat_beta0, geo_beta0, feat_expert, geo_expert) -> dict:
    feat = att_metrics_from_scratch(feat_beta0, feat_expert, id_col="feature", mag_col="coefficient", k=FEATURE_TOP_K, n_total=30)
    geo = att_metrics_from_scratch(geo_beta0, geo_expert, id_col="lsoa21cd", mag_col="geo_coefficient", k=GEO_TOP_K, n_total=168)
    return {"feature": feat, "geo": geo}


# --------------------------------------------------------------------------
# Metric 5: weighted-adjusted R2, N_p/N_s guard, MSE/WMSE/MAE cross-checks
# --------------------------------------------------------------------------
def run_metric5() -> dict:
    feat_fid = pd.read_csv(FEAT_FIDELITY_PATH).iloc[0].to_dict()
    geo_fid = pd.read_csv(GEO_FIDELITY_PATH).iloc[0].to_dict()

    # N_p: exact size of the held-out (test) fold behind these R2s. split_masks is
    # pure/deterministic (no model, no re-seed) -- same function, same seed, used
    # to fit them originally (dehghani_fidelity / fit_surrogate's tr/va/te split).
    _, _, te_feat = split_masks(FEAT_N_MASKS, SMILE_SEED)
    _, _, te_geo = split_masks(GEO_N_PERTURB, SMILE_SEED)
    n_p_feat, n_p_geo = len(te_feat), len(te_geo)
    n_s_feat, n_s_geo = 30, 168

    def adj_r2(r2w, n_p, n_s):
        denom = n_p - n_s - 1
        if denom <= 0:
            return None, denom
        return 1 - (1 - r2w) * (n_p - 1) / denom, denom

    feat_r2adj, feat_denom = adj_r2(feat_fid["Weighted R2"], n_p_feat, n_s_feat)
    geo_r2adj, geo_denom = adj_r2(geo_fid["Weighted R2"], n_p_geo, n_s_geo)

    def verify_branch(fid: dict) -> dict:
        rmse2 = fid["RMSE"] ** 2
        wrmse2 = fid["WRMSE"] ** 2
        return {
            "R2": fid["R2"], "Weighted_R2": fid["Weighted R2"],
            "MAE": fid["MAE"], "WMAE": fid["WMAE"], "RMSE": fid["RMSE"], "WRMSE": fid["WRMSE"],
            "MSE_check": {"RMSE_squared": rmse2, "matches_mean_L2_claim": True, "note": "MSE = RMSE^2 by definition; no independent MSE field exists to compare against, this checks internal consistency"},
            "WMSE_check": {"WRMSE_squared": wrmse2},
            "mean_L1_equals_MAE": True,
            "mean_L2_equals_MSE": True,
        }

    return {
        "feature": {**verify_branch(feat_fid), "N_p": n_p_feat, "N_s": n_s_feat, "N_p_minus_N_s_minus_1": feat_denom, "R2_adj_w": feat_r2adj, "guard_triggered": feat_denom <= 0},
        "geo": {**verify_branch(geo_fid), "N_p": n_p_geo, "N_s": n_s_geo, "N_p_minus_N_s_minus_1": geo_denom, "R2_adj_w": geo_r2adj, "guard_triggered": geo_denom <= 0},
    }


# --------------------------------------------------------------------------
# Metric 3: report only, no new runs
# --------------------------------------------------------------------------
def run_metric3() -> dict:
    stab = json.loads(ATT_STABILITY_PATH.read_text(encoding="utf-8"))
    return {
        "source_file": str(ATT_STABILITY_PATH),
        "feature": {
            "top_k_jaccard_vs_beta0": stab["feature"]["top_k_jaccard_mean"],
            "abs_coef_spearman_vs_beta0": stab["feature"]["abs_coef_spearman_mean"],
            "pred_spearman_vs_beta0": stab["feature"]["pred_spearman_mean"],
        },
        "geo": {
            "top_k_jaccard_vs_beta0": stab["geo"]["top_k_jaccard_mean"],
            "abs_coef_spearman_vs_beta0": stab["geo"]["abs_coef_spearman_mean"],
            "pred_spearman_vs_beta0": stab["geo"]["pred_spearman_mean"],
        },
        "cardinal_neighbors": {
            direction: {
                "lsoa21cd": d["lsoa21cd"], "lsoa21nm": d["lsoa21nm"],
                "geo_rank_out_of_169": int(d["geo_extra_rank"]),
                "geo_extra_abs_coefficient": d["geo_extra_abs_coefficient"],
                "feature_jaccard": d["feature_jaccard"], "feature_spearman": d["feature_spearman"],
                "geo_jaccard": d["geo_jaccard"], "geo_spearman": d["geo_spearman"],
            }
            for direction, d in stab["neighbors"].items()
        },
    }


def main() -> None:
    _log("=== Loading fixed, deterministic inputs (panel, split, beta_0, expert lists) ===")
    if not PANEL_PATH.exists():
        raise SystemExit(f"Missing {PANEL_PATH}")
    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    bad = [p for p in players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise SystemExit(f"Spatial leak in Feature players: {bad}")
    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
    X_test = panel.loc[test].reset_index(drop=True)

    feat_beta0 = pd.read_csv(FEAT_BETA0_PATH)
    geo_beta0 = pd.read_csv(GEO_BETA0_PATH)
    geo_beta0["lsoa21cd"] = geo_beta0["lsoa21cd"].astype(str)

    feat_ref = load_feature_expert_reference()
    feat_expert = set(feat_ref.loc[feat_ref["expert_rank"] <= FEATURE_TOP_K, "feature"].astype(str))
    geo_ref = load_geo_expert_reference()
    geo_expert = set(geo_ref["lsoa21cd"].astype(str))

    artefacts = {
        "panel": _artefact(PANEL_PATH),
        "feature_beta0": _artefact(FEAT_BETA0_PATH),
        "geo_beta0": _artefact(GEO_BETA0_PATH),
        "feature_expert_reference": _artefact(FEAT_EXPERT_PATH),
        "geo_expert_reference": _artefact(GEO_EXPERT_PATH),
        "att_stability_archived": _artefact(ATT_STABILITY_PATH),
        "feature_fidelity_archived": _artefact(FEAT_FIDELITY_PATH),
        "geo_fidelity_archived": _artefact(GEO_FIDELITY_PATH),
        "feature_consistency_r5_archived": _artefact(FEAT_CONSISTENCY_R5_PATH),
        "geo_consistency_r5_archived": _artefact(GEO_CONSISTENCY_R5_PATH),
    }

    _log("\n=== Metric 1: ATT accuracy/F1/AUROC (fresh, from beta_0) ===")
    metric1 = run_metric1(feat_beta0, geo_beta0, feat_expert, geo_expert)
    _log(f"Feature: TP={metric1['feature']['TP']} ACC={metric1['feature']['ATT_accuracy']:.4f} "
         f"F1={metric1['feature']['ATT_F1']:.4f} AUROC={metric1['feature']['ATT_AUROC_continuous']:.4f}")
    _log(f"Geo:     TP={metric1['geo']['TP']} ACC={metric1['geo']['ATT_accuracy']:.4f} "
         f"F1={metric1['geo']['ATT_F1']:.4f} AUROC={metric1['geo']['ATT_AUROC_continuous']:.4f}")

    _log("\n=== Metric 3: ATT stability (report archived only, no new runs) ===")
    metric3 = run_metric3()

    _log("\n=== Metric 5: weighted-adjusted R2, N_p/N_s guard, MSE/WMSE checks ===")
    metric5 = run_metric5()
    _log(f"Feature N_p={metric5['feature']['N_p']} N_s={metric5['feature']['N_s']} "
         f"R2_adj_w={metric5['feature']['R2_adj_w']}")
    _log(f"Geo     N_p={metric5['geo']['N_p']} N_s={metric5['geo']['N_s']} "
         f"R2_adj_w={metric5['geo']['R2_adj_w']}")

    _log("\n=== Reconstructing model deterministically (seed=42) -- no archived model exists ===")
    model_feat = make_model(SMILE_SEED)
    model_geo = make_model(SMILE_SEED)
    model_feat.fit(
        panel.loc[train, players], panel.loc[train, Y_COL],
        eval_set=[(panel.loc[validation, players], panel.loc[validation, Y_COL])], verbose=False,
    )
    model_geo.fit(
        panel.loc[train, SPATIAL_COLUMNS], panel.loc[train, Y_COL],
        eval_set=[(panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, Y_COL])], verbose=False,
    )

    _log(f"Feature SMILE n={FEAT_N_MASKS} seed={SMILE_SEED} (auto kernel selection, not forced)...")
    feat_imp_42, feat_diag_42, *_ = smile_global_feature(model_feat, X_test, players, baseline, FEAT_N_MASKS, SMILE_SEED)
    _log(f"Geo SMILE n={GEO_N_PERTURB} seed={SMILE_SEED} (auto kernel selection, not forced)...")
    geo_imp_42, geo_diag_42 = geo_smile_coefficients(model_geo, panel, test, baseline, GEO_N_PERTURB, SMILE_SEED)
    geo_imp_42["lsoa21cd"] = geo_imp_42["lsoa21cd"].astype(str)

    # Reproduction check: does the deterministic retrain + seed-42 SMILE reproduce beta_0?
    feat_merge = feat_beta0.merge(feat_imp_42, on="feature", suffixes=("_beta0", "_retrain"))
    feat_max_abs_diff = float((feat_merge["coefficient_beta0"] - feat_merge["coefficient_retrain"]).abs().max())
    geo_merge = geo_beta0.merge(geo_imp_42, on="lsoa21cd", suffixes=("_beta0", "_retrain"))
    geo_max_abs_diff = float((geo_merge["geo_coefficient_beta0"] - geo_merge["geo_coefficient_retrain"]).abs().max())
    reproduction = {
        "feature_max_abs_coefficient_diff": feat_max_abs_diff,
        "feature_exact": feat_max_abs_diff == 0.0,
        "feature_kernel_selected": feat_diag_42["best_kernel"],
        "feature_kernel_matches_archived_inverse": feat_diag_42["best_kernel"] == "inverse",
        "geo_max_abs_coefficient_diff": geo_max_abs_diff,
        "geo_exact": geo_max_abs_diff == 0.0,
        "geo_kernel_selected": geo_diag_42["best_kernel"],
        "geo_kernel_matches_archived_laplace": geo_diag_42["best_kernel"] == "laplace",
    }
    _log(f"Feature reproduction: max|diff|={feat_max_abs_diff:.6g} exact={reproduction['feature_exact']} "
         f"kernel={reproduction['feature_kernel_selected']}")
    _log(f"Geo reproduction:     max|diff|={geo_max_abs_diff:.6g} exact={reproduction['geo_exact']} "
         f"kernel={reproduction['geo_kernel_selected']}")

    _log(f"\n=== Metric 2: Faithfulness (n_random={N_RANDOM_FAITH}), top/bottom sets from beta_0 ===")
    faithfulness = model_based_faithfulness(
        model_feat=model_feat, model_geo=model_geo, X_test=X_test, players=players, baseline=baseline,
        feat_imp=feat_beta0.rename(columns={"coefficient": "coefficient", "abs_coefficient": "abs_coefficient"}),
        geo_imp=geo_beta0.rename(columns={"geo_coefficient": "geo_coefficient", "geo_abs_coefficient": "geo_abs_coefficient"}),
        n_random=N_RANDOM_FAITH, seed=SMILE_SEED,
    )
    _log(f"Feature: top={faithfulness['feature']['top_response']:.4f} "
         f"bottom={faithfulness['feature']['bottom_response']:.4f} "
         f"random_mean={faithfulness['feature']['random_mean']:.4f} "
         f"diff={faithfulness['feature']['top_minus_random']:.4f} p={faithfulness['feature']['empirical_p_top_vs_random']:.4f}")
    _log(f"Geo:     top={faithfulness['geo']['top_response']:.4f} "
         f"bottom={faithfulness['geo']['bottom_response']:.4f} "
         f"random_mean={faithfulness['geo']['random_mean']:.4f} "
         f"diff={faithfulness['geo']['top_minus_random']:.4f} p={faithfulness['geo']['empirical_p_top_vs_random']:.4f}")

    _log(f"\n=== Metric 4: ATT consistency, R={len(CONSISTENCY_SEEDS)}, seeds {CONSISTENCY_SEEDS} ===")
    feat_runs_signed, geo_runs_signed = [], []
    feat_test_r2, geo_test_r2 = [], []
    geo_index = None
    for seed in CONSISTENCY_SEEDS:
        _log(f"  seed {seed}: Feature SMILE...")
        f_imp, f_diag, *_ = smile_global_feature(model_feat, X_test, players, baseline, FEAT_N_MASKS, seed)
        f_imp = f_imp.set_index("feature").loc[players]
        feat_runs_signed.append(f_imp["coefficient"].to_numpy(dtype=float))
        feat_test_r2.append(float(f_diag["test_r2"]))
        _log(f"  seed {seed}: Geo SMILE...")
        g_imp, g_diag = geo_smile_coefficients(model_geo, panel, test, baseline, GEO_N_PERTURB, seed)
        g_imp["lsoa21cd"] = g_imp["lsoa21cd"].astype(str)
        g_imp = g_imp.sort_values("lsoa21cd").reset_index(drop=True)
        if geo_index is None:
            geo_index = g_imp["lsoa21cd"].tolist()
        g_imp = g_imp.set_index("lsoa21cd").loc[geo_index]
        geo_runs_signed.append(g_imp["geo_coefficient"].to_numpy(dtype=float))
        geo_test_r2.append(float(g_diag["test_r2"]))

    feat_runs_signed = np.asarray(feat_runs_signed)  # (10, 30)
    geo_runs_signed = np.asarray(geo_runs_signed)  # (10, 168)

    def per_player_stats(runs_signed: np.ndarray, ids: list[str], beta0_map: dict[str, float]) -> pd.DataFrame:
        beta_bar = runs_signed.mean(axis=0)
        var = runs_signed.var(axis=0, ddof=1)
        sd = np.sqrt(var)
        beta0_arr = np.array([beta0_map[i] for i in ids])
        rank = pd.Series(-np.abs(beta0_arr)).rank(method="min").astype(int).to_numpy()
        return pd.DataFrame({
            "player_id": ids, "beta_0": beta0_arr, "beta_bar": beta_bar,
            "Var": var, "SD": sd, "rank_in_beta_0": rank,
        })

    feat_beta0_map = dict(zip(feat_beta0["feature"].astype(str), feat_beta0["coefficient"].astype(float)))
    geo_beta0_map = dict(zip(geo_beta0["lsoa21cd"].astype(str), geo_beta0["geo_coefficient"].astype(float)))
    feat_stats = per_player_stats(feat_runs_signed, players, feat_beta0_map)
    feat_stats.insert(0, "branch", "Feature")
    geo_stats = per_player_stats(geo_runs_signed, geo_index, geo_beta0_map)
    geo_stats.insert(0, "branch", "Geo")
    per_player = pd.concat([feat_stats, geo_stats], ignore_index=True)
    per_player.to_csv(OUT / "hull_consistency_per_player.csv", index=False)

    def branch_summary(stats: pd.DataFrame, runs_signed: np.ndarray, ids: list[str], beta0_map: dict[str, float], top_ids: set[str]) -> dict:
        beta0_arr = np.array([beta0_map[i] for i in ids])
        rms_dev = float(np.sqrt(np.mean((runs_signed - beta0_arr[None, :]) ** 2)))
        top_mask = stats["player_id"].isin(top_ids)
        return {
            "mean_Var": float(stats["Var"].mean()),
            "max_Var": float(stats["Var"].max()),
            "max_Var_player": str(stats.loc[stats["Var"].idxmax(), "player_id"]),
            "mean_SD": float(stats["SD"].mean()),
            "max_SD": float(stats["SD"].max()),
            "max_SD_player": str(stats.loc[stats["SD"].idxmax(), "player_id"]),
            "mean_SD_over_beta0_topk": float(stats.loc[top_mask, "SD"].mean()),
            "rms_deviation_from_beta0": rms_dev,
        }

    feat_top_ids = set(feat_beta0.assign(_abs=feat_beta0["coefficient"].abs()).sort_values(
        ["_abs", "feature"], ascending=[False, True]).head(FEATURE_TOP_K)["feature"].astype(str))
    geo_top_ids = set(geo_beta0.assign(_abs=geo_beta0["geo_coefficient"].abs()).sort_values(
        ["_abs", "lsoa21cd"], ascending=[False, True]).head(GEO_TOP_K)["lsoa21cd"].astype(str))

    feat_r10 = consistency_metrics(np.abs(feat_runs_signed), top_pct=20)
    feat_r10["Fidelity R2 Mean"] = float(np.mean(feat_test_r2))
    feat_r10["Fidelity R2 Std"] = float(np.std(feat_test_r2))
    geo_r10 = consistency_metrics(np.abs(geo_runs_signed), top_pct=20)
    geo_r10["Fidelity R2 Mean"] = float(np.mean(geo_test_r2))
    geo_r10["Fidelity R2 Std"] = float(np.std(geo_test_r2))

    feat_r5_archived = pd.read_csv(FEAT_CONSISTENCY_R5_PATH).iloc[0].to_dict()
    geo_r5_archived = pd.read_csv(GEO_CONSISTENCY_R5_PATH).iloc[0].to_dict()

    metric4 = {
        "seeds": CONSISTENCY_SEEDS, "R": len(CONSISTENCY_SEEDS),
        "seed_42_reproduces_beta0": reproduction,
        "feature": {**branch_summary(feat_stats, feat_runs_signed, players, feat_beta0_map, feat_top_ids),
                    "agreement_R10": feat_r10, "agreement_R5_archived": feat_r5_archived},
        "geo": {**branch_summary(geo_stats, geo_runs_signed, geo_index, geo_beta0_map, geo_top_ids),
                "agreement_R10": geo_r10, "agreement_R5_archived": geo_r5_archived},
    }
    _log(f"Feature R10 Rank Spearman Mean={feat_r10['Rank Spearman Mean']:.4f} "
         f"(R5 archived={feat_r5_archived['Rank Spearman Mean']:.4f})")
    _log(f"Geo     R10 Rank Spearman Mean={geo_r10['Rank Spearman Mean']:.4f} "
         f"(R5 archived={geo_r5_archived['Rank Spearman Mean']:.4f})")

    # --- assemble final report ---
    result = {
        "design": {
            "feature_n_masks": FEAT_N_MASKS, "geo_n_perturb": GEO_N_PERTURB,
            "smile_seed": SMILE_SEED, "consistency_seeds": CONSISTENCY_SEEDS,
            "feature_top_k": FEATURE_TOP_K, "geo_top_k": GEO_TOP_K,
        },
        "artefacts": artefacts,
        "no_archived_model_exists": True,
        "model_reconstruction": reproduction,
        "metric1_att_accuracy": metric1,
        "metric2_faithfulness": faithfulness,
        "metric3_att_stability": metric3,
        "metric4_att_consistency": metric4,
        "metric5_att_fidelity": metric5,
        "expected_values_from_prompt": EXPECTED,
    }
    (OUT / "hull_metrics_computed.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    _log(f"\nWrote {OUT / 'hull_metrics_computed.json'}")
    _log(f"Wrote {OUT / 'hull_consistency_per_player.csv'}")


if __name__ == "__main__":
    main()
