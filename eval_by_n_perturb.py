"""Sweep Feature/Geo GeoSMILE metrics across perturbation counts.

Writes only under outputs/eval_by_n_perturb/. Does not replace the published
1024-mask / 2000-perturbation tables under outputs/.

Metrics (both branches, each N in 1000, 2000, 3000, 4000, 5000):
  accuracy       expert-reference ATT accuracy (Feature k=10, Geo k=20)
  fidelity       SMILE Ridge held-out surrogate R2 (Dehghani-style, same pool)
  faithfulness   model-based top-vs-random Wasserstein (k=10 / k=20, n_random=1000)
  att_stability  cardinal-neighbor Jaccard of Feature top-10 / Geo top-20
  att_consistency multi-seed Rank Spearman and Top-20% Jaccard (base 100, 5 repeats)

Models are the published two-branch XGBoosts, trained once (seed 42).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from compute_att_stability import (
    DIRECTIONS,
    compare_attributions,
    hull_row_pred_spearman,
)
from geosmile_hull_twobranch import (
    FORBIDDEN_FEATURE_TOKENS,
    SPATIAL_COLUMNS,
    choose_feature_players,
    geo_smile_coefficients,
    make_model,
    smile_global_feature,
    temporal_split,
)
from hull_boundary_neighbors import NEIGHBOR_META, NEIGHBOR_PANEL, expand_panel
from hull_dehghani_eval import (
    consistency_metrics,
    fit_surrogate,
    select_best_kernel,
    split_masks,
)
from hull_expert_topk import (
    FEATURE_TOP_K,
    GEO_TOP_K,
    assign_geosmile_ranks,
    compute_att_suite,
    feature_expert_binary_frame,
    geo_expert_binary_frame,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "eval_by_n_perturb"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"
FEAT_EXPERT = OUT / "geosmile_hull_expert_alignment_merged_feature.csv"
COV_PATH = ROOT / "data" / "hull_static_covariates.csv"

SEED = 42
SPLIT_SEED = 42
CONSISTENCY_SEED_BASE = 100
N_REPEATS = 5
N_GRID_DEFAULT = (1000, 2000, 3000, 4000, 5000)
N_RANDOM_FAITH = 1000
Y_COL = "target_next_month_crime"

PUBLISHED_FORBIDDEN = (
    OUT / "feature_global_importance.csv",
    OUT / "geo_coefficients.csv",
    OUT / "geosmile_hull_feat_consistency.csv",
    OUT / "geosmile_hull_geo_consistency.csv",
    OUT / "geosmile_hull_att_stability.json",
    OUT / "geosmile_hull_feat_fidelity.csv",
    OUT / "geosmile_hull_geo_fidelity.csv",
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        x = float(value)
        if not np.isfinite(x):
            return None
        return x
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def load_cached_neighbors() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not NEIGHBOR_PANEL.exists() or not NEIGHBOR_META.exists():
        raise SystemExit(
            f"Missing cached cardinal-neighbor panel ({NEIGHBOR_PANEL}) or meta "
            f"({NEIGHBOR_META}). Run compute_att_stability.py once to build them."
        )
    extra = pd.read_csv(NEIGHBOR_PANEL)
    extra["month"] = pd.to_datetime(extra["month"])
    meta = json.loads(NEIGHBOR_META.read_text(encoding="utf-8"))
    neighbors = pd.DataFrame(
        {
            "lsoa21cd": [str(c) for c in meta["lsoa21cd"]],
            "lsoa21nm": [str(n) for n in meta["lsoa21nm"]],
            "direction": [str(d) for d in meta["direction"]],
        }
    )
    return neighbors, extra


def feature_responses(masks: np.ndarray, all_pert: np.ndarray, original: np.ndarray) -> np.ndarray:
    return np.asarray(
        [wasserstein_distance(original, all_pert[i]) for i in range(len(masks))],
        dtype=float,
    )


def dehghani_fidelity(masks: np.ndarray, responses: np.ndarray, distances: np.ndarray) -> dict:
    tr, va, te = split_masks(len(masks), SPLIT_SEED)
    best, scores = select_best_kernel(
        masks[tr], masks[va], distances[tr], responses[tr], responses[va]
    )
    _, fid = fit_surrogate(masks, responses, distances, best, tr, va, te)
    fid = dict(fid)
    fid["best_kernel"] = best
    fid["validation_r2_by_kernel"] = scores
    fid["n_masks"] = int(len(masks))
    fid["n_test"] = int(len(te))
    return fid


def att_from_importance(
    importance: pd.DataFrame,
    expert: pd.DataFrame,
    *,
    id_col: str,
    mag_col: str,
    k: int,
) -> dict:
    left = expert[[id_col, "expert_top10_reference"]].copy()
    right = importance[[id_col, mag_col]].copy()
    left[id_col] = left[id_col].astype(str)
    right[id_col] = right[id_col].astype(str)
    frame = left.merge(right, on=id_col, how="inner", validate="one_to_one")
    frame = assign_geosmile_ranks(frame, id_col=id_col, mag_col=mag_col, k=k)
    frame["geosmile_abs_attribution"] = frame[mag_col].astype(float)
    suite = compute_att_suite(frame, id_col=id_col)
    return {
        "accuracy": suite["att"]["ATT accuracy"],
        "balanced_accuracy": suite["att"]["balanced accuracy"],
        "precision": suite["att"]["ATT precision"],
        "recall": suite["att"]["ATT recall"],
        "f1": suite["att"]["ATT F1"],
        "auroc": suite["att"]["ATT AUROC"],
        "auprc": suite["att"]["ATT AUPRC"],
        "intersection": suite["overlap"]["intersection"],
        "overlap_ratio": suite["overlap"]["Recall@k"],
        "jaccard_at_k": suite["overlap"]["Jaccard@k"],
        "k": int(k),
        "n": int(suite["att"]["n"]),
        "model_ids": suite["model_ids"],
    }


def model_based_faithfulness(
    *,
    model_feat,
    model_geo,
    X_test: pd.DataFrame,
    players: list[str],
    baseline: pd.Series,
    feat_imp: pd.DataFrame,
    geo_imp: pd.DataFrame,
    n_random: int,
    seed: int,
) -> dict:
    feat_imp = feat_imp.sort_values(
        ["abs_coefficient", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    geo_imp = geo_imp.sort_values(
        ["geo_abs_coefficient", "lsoa21cd"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    k_f, k_g = FEATURE_TOP_K, GEO_TOP_K
    top_feat = feat_imp.head(k_f)["feature"].tolist()
    bottom_feat = feat_imp.tail(k_f)["feature"].tolist()
    all_feat = feat_imp["feature"].tolist()
    top_geo = geo_imp.head(k_g)["lsoa21cd"].astype(str).tolist()
    bottom_geo = geo_imp.tail(k_g)["lsoa21cd"].astype(str).tolist()
    all_geo = geo_imp["lsoa21cd"].astype(str).tolist()

    pop_area = None
    merged_geo = OUT / "geosmile_hull_expert_alignment_merged_geo.csv"
    if merged_geo.exists() and COV_PATH.exists():
        mg = pd.read_csv(merged_geo)
        cov = pd.read_csv(COV_PATH)
        if "population_mid_2022" in mg.columns and "lsoa_area_km2" in cov.columns:
            pop_area = (
                mg[["lsoa21cd", "population_mid_2022"]]
                .merge(cov[["lsoa21cd", "lsoa_area_km2"]], on="lsoa21cd", how="left")
                .assign(lsoa21cd=lambda d: d["lsoa21cd"].astype(str))
                .set_index("lsoa21cd")
            )

    rng = np.random.default_rng(seed)
    orig_feat = np.asarray(model_feat.predict(X_test[players]), float)
    x_geo = X_test[SPATIAL_COLUMNS].copy()
    orig_geo = np.asarray(model_geo.predict(x_geo), float)
    lsoa_ids = X_test["lsoa21cd"].astype(str).to_numpy()
    baseline_xy = baseline[SPATIAL_COLUMNS].to_numpy(dtype=float)

    def feature_response(mask_features: list[str]) -> float:
        pert = X_test[players].copy()
        for name in mask_features:
            pert[name] = baseline[name]
        return float(wasserstein_distance(orig_feat, model_feat.predict(pert)))

    def geo_response(mask_lsoas: list[str]) -> float:
        perturbed = x_geo.copy()
        affected: list[int] = []
        for code in mask_lsoas:
            rows = np.flatnonzero(lsoa_ids == str(code))
            perturbed.iloc[rows, :] = baseline_xy
            affected.extend(rows.tolist())
        affected_idx = np.asarray(affected, dtype=int)
        if len(affected_idx) == 0:
            return 0.0
        pert_pred = np.asarray(model_geo.predict(perturbed), float)
        return float(wasserstein_distance(orig_geo[affected_idx], pert_pred[affected_idx]))

    def sample_geo_random(k: int) -> list[str]:
        if pop_area is None:
            return list(rng.choice(all_geo, size=k, replace=False))
        present = [c for c in top_geo if c in pop_area.index]
        if len(present) < k:
            return list(rng.choice(all_geo, size=k, replace=False))
        target_pop = float(pop_area.loc[present, "population_mid_2022"].sum())
        target_area = float(pop_area.loc[present, "lsoa_area_km2"].sum())
        best = None
        best_score = np.inf
        for _ in range(40):
            cand = list(rng.choice(all_geo, size=k, replace=False))
            if any(c not in pop_area.index for c in cand):
                continue
            pop = float(pop_area.loc[cand, "population_mid_2022"].sum())
            area = float(pop_area.loc[cand, "lsoa_area_km2"].sum())
            score = abs(pop - target_pop) / (target_pop + 1e-9) + abs(area - target_area) / (
                target_area + 1e-9
            )
            if score < best_score:
                best_score = score
                best = cand
        return best or list(rng.choice(all_geo, size=k, replace=False))

    def pack(branch: str, top_v: float, bot_v: float, rand_arr: np.ndarray, k: int) -> dict:
        lo, hi = np.percentile(rand_arr, [2.5, 97.5]).tolist()
        p_top = float((np.sum(rand_arr >= top_v) + 1) / (len(rand_arr) + 1))
        return {
            "branch": branch,
            "k": int(k),
            "top_response": float(top_v),
            "bottom_response": float(bot_v),
            "random_mean": float(rand_arr.mean()),
            "random_ci95_low": float(lo),
            "random_ci95_high": float(hi),
            "top_minus_random": float(top_v - rand_arr.mean()),
            "top_minus_bottom": float(top_v - bot_v),
            "empirical_p_top_vs_random": p_top,
            "n_random": int(len(rand_arr)),
        }

    top_f = feature_response(top_feat)
    bot_f = feature_response(bottom_feat)
    rand_f = np.array(
        [feature_response(list(rng.choice(all_feat, size=k_f, replace=False))) for _ in range(n_random)],
        dtype=float,
    )
    top_g = geo_response(top_geo)
    bot_g = geo_response(bottom_geo)
    rand_g = np.array([geo_response(sample_geo_random(k_g)) for _ in range(n_random)], dtype=float)
    return {
        "feature": pack("Feature", top_f, bot_f, rand_f, k_f),
        "geo": pack("Geo", top_g, bot_g, rand_g, k_g),
    }


def att_stability_for_n(
    *,
    n: int,
    model_feat,
    model_geo,
    panel: pd.DataFrame,
    X_test: pd.DataFrame,
    players: list[str],
    baseline: pd.Series,
    hull_lsoas: list[str],
    base_feat: pd.DataFrame,
    base_geo: pd.DataFrame,
    neighbors: pd.DataFrame,
    extra_panel: pd.DataFrame,
    seed: int,
) -> dict:
    extra_by_dir = {
        str(row.direction): extra_panel[extra_panel["lsoa21cd"].astype(str) == str(row.lsoa21cd)].copy()
        for row in neighbors.itertuples()
    }
    base_feat_top = set(
        base_feat.sort_values(["abs_coefficient", "feature"], ascending=[False, True], kind="mergesort")
        .head(FEATURE_TOP_K)["feature"]
        .astype(str)
    )
    base_geo_top = set(
        base_geo.sort_values(
            ["geo_abs_coefficient", "lsoa21cd"], ascending=[False, True], kind="mergesort"
        )
        .head(GEO_TOP_K)["lsoa21cd"]
        .astype(str)
    )
    per_direction = {}
    feat_j, feat_rho, geo_j, geo_rho = [], [], [], []
    for direction in DIRECTIONS:
        extra_rows = extra_by_dir[direction]
        code = str(extra_rows["lsoa21cd"].iloc[0])
        nm = str(extra_rows["lsoa21nm"].iloc[0])
        _log(f"  [{_now()}] ATT stability {direction}: adding {code} {nm}")
        expanded = expand_panel(panel, extra_rows)
        _, _, exp_test = temporal_split(expanded)
        X_exp = expanded.loc[exp_test].reset_index(drop=True)
        feat_imp, _, _, _, _ = smile_global_feature(
            model_feat, X_exp, players, baseline, n, seed
        )
        geo_imp, _ = geo_smile_coefficients(
            model_geo,
            expanded,
            exp_test,
            baseline,
            n,
            seed,
            sample_lsoas=hull_lsoas,
        )
        metrics = compare_attributions(
            base_feat,
            feat_imp,
            base_geo,
            geo_imp,
            base_feat_top=base_feat_top,
            base_geo_top=base_geo_top,
            added_lsoa=code,
        )
        f_pred, _ = hull_row_pred_spearman(model_feat, X_test, X_exp, players)
        g_pred, _ = hull_row_pred_spearman(model_geo, X_test, X_exp, SPATIAL_COLUMNS)
        feat_j.append(metrics["feature_jaccard"])
        feat_rho.append(metrics["feature_spearman"])
        geo_j.append(metrics["geo_jaccard"])
        geo_rho.append(metrics["geo_spearman"])
        per_direction[direction] = {
            "lsoa21cd": code,
            "lsoa21nm": nm,
            **{k: float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v for k, v in metrics.items()},
            "feature_pred_spearman": f_pred,
            "geo_pred_spearman": g_pred,
        }
        _log(
            f"    Feature J={metrics['feature_jaccard']:.3f} | "
            f"Geo J={metrics['geo_jaccard']:.3f} overlap="
            f"{int(metrics['geo_top20_overlap_n'])}/{GEO_TOP_K}"
        )
    return {
        "n_perturb": int(n),
        "smile_seed": int(seed),
        "baseline": "hull_only_smile_at_this_n",
        "n_reps": int(len(DIRECTIONS)),
        "feature_k": FEATURE_TOP_K,
        "geo_k": GEO_TOP_K,
        "neighbors": per_direction,
        "feature": {
            "top_k_jaccard_mean": float(np.mean(feat_j)),
            "top_k_jaccard_std": float(np.std(feat_j)),
            "spearman_mean": float(np.mean(feat_rho)),
            "spearman_std": float(np.std(feat_rho)),
        },
        "geo": {
            "top_k_jaccard_mean": float(np.mean(geo_j)),
            "top_k_jaccard_std": float(np.std(geo_j)),
            "spearman_mean": float(np.mean(geo_rho)),
            "spearman_std": float(np.std(geo_rho)),
        },
    }


def consistency_for_n(
    *,
    n: int,
    model_feat,
    model_geo,
    panel: pd.DataFrame,
    X_test: pd.DataFrame,
    test_mask: pd.Series,
    players: list[str],
    baseline: pd.Series,
    n_repeats: int,
    seed_base: int,
) -> dict:
    feat_runs = []
    feat_fid = []
    geo_runs = []
    geo_fid = []
    geo_index = None
    for seed in range(seed_base, seed_base + n_repeats):
        _log(f"  [{_now()}] consistency seed {seed} Feature n={n}")
        feat_imp, feat_diag, _, _, _ = smile_global_feature(
            model_feat, X_test, players, baseline, n, seed
        )
        aligned_f = feat_imp.set_index("feature").loc[players]["coefficient"].to_numpy(dtype=float)
        feat_runs.append(np.abs(aligned_f))
        feat_fid.append(float(feat_diag["test_r2"]))
        _log(f"  [{_now()}] consistency seed {seed} Geo n={n}")
        geo_imp, geo_diag = geo_smile_coefficients(
            model_geo, panel, test_mask, baseline, n, seed
        )
        geo_imp = geo_imp.sort_values("lsoa21cd").reset_index(drop=True)
        if geo_index is None:
            geo_index = geo_imp["lsoa21cd"].astype(str).tolist()
        aligned_g = (
            geo_imp.set_index("lsoa21cd")
            .loc[geo_index]["geo_coefficient"]
            .to_numpy(dtype=float)
        )
        geo_runs.append(np.abs(aligned_g))
        geo_fid.append(float(geo_diag["test_r2"]))
    feat_consist = consistency_metrics(np.asarray(feat_runs), top_pct=20)
    feat_consist["Fidelity R2 Mean"] = float(np.mean(feat_fid))
    feat_consist["Fidelity R2 Std"] = float(np.std(feat_fid))
    geo_consist = consistency_metrics(np.asarray(geo_runs), top_pct=20)
    geo_consist["Fidelity R2 Mean"] = float(np.mean(geo_fid))
    geo_consist["Fidelity R2 Std"] = float(np.std(geo_fid))
    return {
        "n_perturb": int(n),
        "n_repeats": int(n_repeats),
        "seed_base": int(seed_base),
        "seeds": list(range(seed_base, seed_base + n_repeats)),
        "feature": feat_consist,
        "geo": geo_consist,
    }


def _branch_key(branch: str) -> str:
    return "geo" if branch.lower() == "geo" else "feature"


def compact_row(n: int, branch: str, metrics: dict) -> dict:
    key = _branch_key(branch)
    faith = metrics.get("faithfulness", {}).get(key, {}) or {}
    acc = metrics.get("accuracy", {}).get(key, {}) or {}
    fid = metrics.get("fidelity", {}).get(key, {}) or {}
    stab = metrics.get("att_stability", {}).get(key, {}) or {}
    cons = metrics.get("att_consistency", {}).get(key, {}) or {}
    return {
        "n_perturb": n,
        "branch": branch,
        "accuracy": acc.get("accuracy"),
        "accuracy_intersection": acc.get("intersection"),
        "fidelity_r2": fid.get("R2"),
        "fidelity_weighted_r2": fid.get("Weighted R2"),
        "faithfulness_top_minus_random": faith.get("top_minus_random"),
        "faithfulness_top_response": faith.get("top_response"),
        "faithfulness_random_mean": faith.get("random_mean"),
        "att_stability_jaccard": stab.get("top_k_jaccard_mean"),
        "att_stability_spearman": stab.get("spearman_mean"),
        "att_consistency_rank_spearman": cons.get("Rank Spearman Mean"),
        "att_consistency_top20pct_jaccard": cons.get("Top-20% Jaccard Mean"),
        "smile_diagnostics_test_r2": fid.get("smile_diagnostics_test_r2"),
        "n_unique_masks_or_perturbations": fid.get("n_masks"),
    }


def evaluate_n(
    n: int,
    *,
    model_feat,
    model_geo,
    panel: pd.DataFrame,
    X_test: pd.DataFrame,
    test_mask: pd.Series,
    players: list[str],
    baseline: pd.Series,
    hull_lsoas: list[str],
    feat_expert: pd.DataFrame,
    geo_expert: pd.DataFrame,
    neighbors: pd.DataFrame,
    extra_panel: pd.DataFrame,
    n_random: int,
    n_repeats: int,
    seed_base: int,
    skip_stability: bool,
    skip_consistency: bool,
    skip_faithfulness: bool,
) -> dict:
    n_dir = DEST / f"n_{n}"
    n_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _log(f"\n=== [{_now()}] N={n} Feature masks / Geo perturbations (this N only) ===")

    feat_csv = n_dir / "feature_importance.csv"
    geo_csv = n_dir / "geo_coefficients.csv"
    acc_path = n_dir / "accuracy.json"
    fid_path = n_dir / "fidelity.json"
    diag_path = n_dir / "smile_diagnostics.json"
    have_rankings = feat_csv.exists() and geo_csv.exists()

    if have_rankings:
        _log(f"[{_now()}] Reusing saved Feature/Geo coefficients for N={n}")
        feat_imp = pd.read_csv(feat_csv)
        geo_imp = pd.read_csv(geo_csv)
        geo_imp["lsoa21cd"] = geo_imp["lsoa21cd"].astype(str)
        acc_feat = att_from_importance(
            feat_imp, feat_expert, id_col="feature", mag_col="abs_coefficient", k=FEATURE_TOP_K
        )
        acc_geo = att_from_importance(
            geo_imp, geo_expert, id_col="lsoa21cd", mag_col="geo_abs_coefficient", k=GEO_TOP_K
        )
        _write_json(acc_path, {"feature": acc_feat, "geo": acc_geo})
        if fid_path.exists() and diag_path.exists():
            fid_saved = json.loads(fid_path.read_text(encoding="utf-8"))
            feat_fid, geo_fid = fid_saved["feature"], fid_saved["geo"]
            diag_saved = json.loads(diag_path.read_text(encoding="utf-8"))
            feat_diag, geo_diag = diag_saved.get("feature", {}), diag_saved.get("geo", {})
        else:
            _log(f"[{_now()}] Recomputing fidelity pool without replacing saved rankings")
            _, feat_diag, feat_masks, feat_pert, feat_orig = smile_global_feature(
                model_feat, X_test, players, baseline, n, SEED
            )
            feat_fid = dehghani_fidelity(
                feat_masks, feature_responses(feat_masks, feat_pert, feat_orig), 1.0 - feat_masks.mean(axis=1)
            )
            feat_fid["smile_diagnostics_test_r2"] = float(feat_diag["test_r2"])
            feat_fid["smile_best_kernel"] = feat_diag["best_kernel"]
            _, geo_diag, geo_m, geo_y, geo_d = geo_smile_coefficients(
                model_geo, panel, test_mask, baseline, n, SEED, return_pool=True
            )
            geo_fid = dehghani_fidelity(geo_m, geo_y, geo_d)
            geo_fid["smile_diagnostics_test_r2"] = float(geo_diag["test_r2"])
            geo_fid["smile_best_kernel"] = geo_diag["best_kernel"]
            _write_json(fid_path, {"feature": feat_fid, "geo": geo_fid})
            _write_json(diag_path, {"feature": feat_diag, "geo": geo_diag})
    else:
        _log(f"[{_now()}] Feature SMILE n={n} seed={SEED}")
        feat_imp, feat_diag, feat_masks, feat_pert, feat_orig = smile_global_feature(
            model_feat, X_test, players, baseline, n, SEED
        )
        feat_y = feature_responses(feat_masks, feat_pert, feat_orig)
        feat_d = 1.0 - feat_masks.mean(axis=1)
        feat_fid = dehghani_fidelity(feat_masks, feat_y, feat_d)
        feat_fid["smile_diagnostics_test_r2"] = float(feat_diag["test_r2"])
        feat_fid["smile_best_kernel"] = feat_diag["best_kernel"]
        feat_imp.to_csv(feat_csv, index=False)

        _log(f"[{_now()}] Geo SMILE n={n} seed={SEED}")
        geo_imp, geo_diag, geo_m, geo_y, geo_d = geo_smile_coefficients(
            model_geo, panel, test_mask, baseline, n, SEED, return_pool=True
        )
        geo_fid = dehghani_fidelity(geo_m, geo_y, geo_d)
        geo_fid["smile_diagnostics_test_r2"] = float(geo_diag["test_r2"])
        geo_fid["smile_best_kernel"] = geo_diag["best_kernel"]
        geo_imp.to_csv(geo_csv, index=False)

        acc_feat = att_from_importance(
            feat_imp, feat_expert, id_col="feature", mag_col="abs_coefficient", k=FEATURE_TOP_K
        )
        acc_geo = att_from_importance(
            geo_imp, geo_expert, id_col="lsoa21cd", mag_col="geo_abs_coefficient", k=GEO_TOP_K
        )
        _write_json(acc_path, {"feature": acc_feat, "geo": acc_geo})
        _write_json(fid_path, {"feature": feat_fid, "geo": geo_fid})
        _write_json(diag_path, {"feature": feat_diag, "geo": geo_diag})

    _log(
        f"[{_now()}] Accuracy Feature={acc_feat['accuracy']:.4f} "
        f"({acc_feat['intersection']}/{FEATURE_TOP_K}) | "
        f"Geo={acc_geo['accuracy']:.4f} ({acc_geo['intersection']}/{GEO_TOP_K})"
    )
    _log(
        f"[{_now()}] Fidelity Feature R2={feat_fid['R2']:.4f} | Geo R2={geo_fid['R2']:.4f}"
    )

    faith_path = n_dir / "faithfulness.json"
    if skip_faithfulness:
        faith = {"feature": {}, "geo": {}, "skipped": True}
    elif faith_path.exists():
        _log(f"[{_now()}] Reusing saved faithfulness")
        faith = json.loads(faith_path.read_text(encoding="utf-8"))
    else:
        _log(f"[{_now()}] Faithfulness n_random={n_random}")
        faith = model_based_faithfulness(
            model_feat=model_feat,
            model_geo=model_geo,
            X_test=X_test,
            players=players,
            baseline=baseline,
            feat_imp=feat_imp,
            geo_imp=geo_imp,
            n_random=n_random,
            seed=SEED,
        )
        _log(
            f"[{_now()}] Faithfulness top-random Feature={faith['feature']['top_minus_random']:.4f} | "
            f"Geo={faith['geo']['top_minus_random']:.4f}"
        )
        _write_json(faith_path, faith)

    stab_path = n_dir / "att_stability.json"
    if skip_stability:
        stability = {
            "feature": {"top_k_jaccard_mean": None, "spearman_mean": None},
            "geo": {"top_k_jaccard_mean": None, "spearman_mean": None},
            "skipped": True,
        }
    elif stab_path.exists():
        _log(f"[{_now()}] Reusing saved ATT stability")
        stability = json.loads(stab_path.read_text(encoding="utf-8"))
    else:
        _log(f"[{_now()}] ATT stability (4 cardinal neighbors), baseline = Hull-only at N={n}")
        stability = att_stability_for_n(
            n=n,
            model_feat=model_feat,
            model_geo=model_geo,
            panel=panel,
            X_test=X_test,
            players=players,
            baseline=baseline,
            hull_lsoas=hull_lsoas,
            base_feat=feat_imp,
            base_geo=geo_imp,
            neighbors=neighbors,
            extra_panel=extra_panel,
            seed=SEED,
        )
        _write_json(stab_path, stability)
        _log(
            f"[{_now()}] ATT stability Jaccard Feature={stability['feature']['top_k_jaccard_mean']:.3f} | "
            f"Geo={stability['geo']['top_k_jaccard_mean']:.3f}"
        )

    cons_path = n_dir / "att_consistency.json"
    if skip_consistency:
        consist = {
            "feature": {
                "Rank Spearman Mean": None,
                "Top-20% Jaccard Mean": None,
            },
            "geo": {
                "Rank Spearman Mean": None,
                "Top-20% Jaccard Mean": None,
            },
            "skipped": True,
        }
    elif cons_path.exists():
        _log(f"[{_now()}] Reusing saved ATT consistency")
        consist = json.loads(cons_path.read_text(encoding="utf-8"))
    else:
        _log(f"[{_now()}] ATT consistency {n_repeats} seeds from {seed_base} at N={n}")
        consist = consistency_for_n(
            n=n,
            model_feat=model_feat,
            model_geo=model_geo,
            panel=panel,
            X_test=X_test,
            test_mask=test_mask,
            players=players,
            baseline=baseline,
            n_repeats=n_repeats,
            seed_base=seed_base,
        )
        pd.DataFrame([consist["feature"]]).assign(branch="Feature").to_csv(
            n_dir / "consistency_feature.csv", index=False
        )
        pd.DataFrame([consist["geo"]]).assign(branch="Geo").to_csv(
            n_dir / "consistency_geo.csv", index=False
        )
        _write_json(cons_path, consist)
        _log(
            f"[{_now()}] Consistency Rank Spearman Feature="
            f"{consist['feature']['Rank Spearman Mean']:.3f} | "
            f"Geo={consist['geo']['Rank Spearman Mean']:.3f}"
        )

    payload = {
        "n_perturb": int(n),
        "seed": SEED,
        "elapsed_sec": float(time.time() - t0),
        "accuracy": {"feature": acc_feat, "geo": acc_geo},
        "fidelity": {"feature": feat_fid, "geo": geo_fid},
        "faithfulness": faith,
        "att_stability": stability,
        "att_consistency": consist,
        "smile_diagnostics": {"feature": feat_diag, "geo": geo_diag},
    }
    _write_json(n_dir / "metrics.json", payload)
    _log(f"[{_now()}] N={n} done in {payload['elapsed_sec'] / 60:.1f} min -> {n_dir}")
    return payload


def refresh_saved_accuracy(feat_expert: pd.DataFrame, geo_expert: pd.DataFrame) -> None:
    """Recompute expert ATT for every saved N without retraining or replacing rankings."""
    n_dirs = sorted(DEST.glob("n_*"))
    if not n_dirs:
        raise SystemExit(f"No saved N folders under {DEST}")
    for n_dir in n_dirs:
        feat_csv = n_dir / "feature_importance.csv"
        geo_csv = n_dir / "geo_coefficients.csv"
        if not (feat_csv.exists() and geo_csv.exists()):
            _log(f"Skip {n_dir.name}: missing coefficient CSVs")
            continue
        feat_imp = pd.read_csv(feat_csv)
        geo_imp = pd.read_csv(geo_csv)
        geo_imp["lsoa21cd"] = geo_imp["lsoa21cd"].astype(str)
        acc_feat = att_from_importance(
            feat_imp, feat_expert, id_col="feature", mag_col="abs_coefficient", k=FEATURE_TOP_K
        )
        acc_geo = att_from_importance(
            geo_imp, geo_expert, id_col="lsoa21cd", mag_col="geo_abs_coefficient", k=GEO_TOP_K
        )
        acc = {"feature": acc_feat, "geo": acc_geo}
        _write_json(n_dir / "accuracy.json", acc)
        metrics_path = n_dir / "metrics.json"
        if metrics_path.exists():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            payload["accuracy"] = acc
            _write_json(metrics_path, payload)
        _log(
            f"{n_dir.name}: Feature {acc_feat['intersection']}/{FEATURE_TOP_K} "
            f"Geo {acc_geo['intersection']}/{GEO_TOP_K}"
        )
    write_summaries()


def write_summaries(results: list[dict] | None = None) -> None:
    payloads = list(results or [])
    seen = {int(p["n_perturb"]) for p in payloads if "n_perturb" in p}
    if DEST.exists():
        for path in sorted(DEST.glob("n_*/metrics.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            n = int(payload.get("n_perturb", -1))
            if n not in seen:
                payloads.append(payload)
                seen.add(n)
    payloads.sort(key=lambda p: int(p.get("n_perturb", 0)))
    rows = []
    for payload in payloads:
        n = int(payload["n_perturb"])
        rows.append(compact_row(n, "Feature", payload))
        rows.append(compact_row(n, "Geo", payload))
    if not rows:
        return
    summary = pd.DataFrame(rows)
    summary.to_csv(DEST / "summary.csv", index=False)
    long_rows = []
    metric_cols = [
        "accuracy",
        "fidelity_r2",
        "faithfulness_top_minus_random",
        "att_stability_jaccard",
        "att_consistency_rank_spearman",
        "att_consistency_top20pct_jaccard",
    ]
    for row in rows:
        for metric in metric_cols:
            long_rows.append(
                {
                    "n_perturb": row["n_perturb"],
                    "branch": row["branch"],
                    "metric": metric,
                    "value": row[metric],
                }
            )
    pd.DataFrame(long_rows).to_csv(DEST / "summary_long.csv", index=False)
    _log(f"\nSaved {DEST / 'summary.csv'}")
    _log(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-grid", type=int, nargs="+", default=list(N_GRID_DEFAULT))
    p.add_argument("--n-random", type=int, default=N_RANDOM_FAITH)
    p.add_argument("--n-repeats", type=int, default=N_REPEATS)
    p.add_argument("--consistency-seed-base", type=int, default=CONSISTENCY_SEED_BASE)
    p.add_argument(
        "--refresh-accuracy-only",
        action="store_true",
        help=(
            "Recompute expert-reference accuracy from the canonical Geo list "
            "for saved N folders. Does not retrain or replace coefficient CSVs."
        ),
    )
    p.add_argument("--force", action="store_true", help="Recompute even if metrics.json exists")
    p.add_argument("--skip-stability", action="store_true")
    p.add_argument("--skip-consistency", action="store_true")
    p.add_argument("--skip-faithfulness", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEST,
        help="Destination folder (default: outputs/eval_by_n_perturb). Never writes published tables.",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help="XGBoost threads (default 2 so the machine stays usable). Use -1 for all cores.",
    )
    return p.parse_args()


def main() -> None:
    global DEST
    args = parse_args()
    DEST = Path(args.output_dir)
    if not PANEL_PATH.exists():
        raise SystemExit(f"Missing {PANEL_PATH}")
    DEST.mkdir(parents=True, exist_ok=True)
    existing = [p for p in PUBLISHED_FORBIDDEN if p.exists()]
    _log(
        f"Writing sweep to {DEST} only. "
        f"Will not replace {len(existing)} published tables "
        "(1024/2000 Feature/Geo artefacts left untouched)."
    )

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
    feat_imp_pub = pd.read_csv(OUT / "feature_global_importance.csv")
    feat_expert = feature_expert_binary_frame(feat_imp_pub["feature"])
    if int(feat_expert["expert_top10_reference"].sum()) != FEATURE_TOP_K:
        raise SystemExit("Canonical Feature expert list did not mark 10 of 30 features")
    geo_expert = geo_expert_binary_frame(hull_lsoas)
    if int(geo_expert["expert_top10_reference"].sum()) != GEO_TOP_K:
        raise SystemExit("Canonical Geo expert list did not mark 20 Hull LSOAs")

    if args.refresh_accuracy_only:
        refresh_saved_accuracy(feat_expert, geo_expert)
        return

    if args.n_jobs > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.n_jobs)
        os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    _log(
        f"[{_now()}] Training Feature and Geo XGBoost (seed {SEED}) once "
        f"with n_jobs={args.n_jobs}..."
    )
    model_feat = make_model(SEED)
    model_geo = make_model(SEED)
    model_feat.set_params(n_jobs=args.n_jobs)
    model_geo.set_params(n_jobs=args.n_jobs)
    model_feat.fit(
        panel.loc[train, players],
        panel.loc[train, Y_COL],
        eval_set=[(panel.loc[validation, players], panel.loc[validation, Y_COL])],
        verbose=False,
    )
    model_geo.fit(
        panel.loc[train, SPATIAL_COLUMNS],
        panel.loc[train, Y_COL],
        eval_set=[(panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, Y_COL])],
        verbose=False,
    )

    neighbors, extra_panel = load_cached_neighbors()
    missing_players = [p for p in players if p not in extra_panel.columns]
    if missing_players:
        raise SystemExit(f"Neighbor panel missing Feature players: {missing_players}")

    results = []
    for n in args.n_grid:
        metrics_path = DEST / f"n_{n}" / "metrics.json"
        if metrics_path.exists() and not args.force:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            complete = all(
                key in payload
                for key in (
                    "accuracy",
                    "fidelity",
                    "faithfulness",
                    "att_stability",
                    "att_consistency",
                )
            )
            skipped_ok = True
            if not args.skip_consistency and payload.get("att_consistency", {}).get("skipped"):
                skipped_ok = False
            if not args.skip_stability and payload.get("att_stability", {}).get("skipped"):
                skipped_ok = False
            if not args.skip_faithfulness and payload.get("faithfulness", {}).get("skipped"):
                skipped_ok = False
            if complete and skipped_ok:
                _log(f"Skipping N={n}; found {metrics_path}. Use --force to recompute.")
                results.append(payload)
                continue
        payload = evaluate_n(
            n,
            model_feat=model_feat,
            model_geo=model_geo,
            panel=panel,
            X_test=X_test,
            test_mask=test,
            players=players,
            baseline=baseline,
            hull_lsoas=hull_lsoas,
            feat_expert=feat_expert,
            geo_expert=geo_expert,
            neighbors=neighbors,
            extra_panel=extra_panel,
            n_random=args.n_random,
            n_repeats=args.n_repeats,
            seed_base=args.consistency_seed_base,
            skip_stability=args.skip_stability,
            skip_consistency=args.skip_consistency,
            skip_faithfulness=args.skip_faithfulness,
        )
        results.append(payload)
        write_summaries(results)

    write_summaries(results)
    _log(f"\nDone. Published 1024/2000 tables were not modified. Results: {DEST}")


if __name__ == "__main__":
    main()
