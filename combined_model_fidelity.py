"""SMILE attribution for a single combined (Feature+Geo) predictive model.

Everywhere else in this pipeline, Feature and Geo are two SEPARATELY-trained
models (see geosmile_hull_twobranch.py) -- a deliberate choice, not an oversight:
the legacy three-branch script (geosmile_hull.py) tried a combined "Feature-B"
model (features + coordinates for prediction) and found no significant accuracy
gain on the temporal holdout and measurably worse generalization on a spatial
holdout (coordinates let the model use location as an identifier). See
outputs/hull_results_interpretation.md.

This module answers a narrower, explicit question instead: given ONE model
trained on all 32 inputs (30 Feature players + 2 coordinates), how well does
GeoSMILE explain it along each axis separately? Coordinates are never masked
during the Feature-branch pass (they condition prediction but are never
attribution players, exactly matching the legacy Feature-B convention); the 30
Feature players are never masked during the Geo-branch pass. Both passes reuse
the model's own predictions -- there is only ever one underlying model here.

Each function below is a faithful adaptation of its geosmile_hull_twobranch.py /
eval_by_n_perturb.py counterpart (smile_global_feature, geo_smile_coefficients,
model_based_faithfulness, consistency_for_n respectively), changed only to hold
the other branch's columns fixed at their true value instead of omitting them.
Cardinal-neighbor ATT stability is intentionally not adapted here (it requires
the neighbor-panel "extra LSOA" machinery in geo_smile_coefficients, which is
Geo-specific plumbing not worth duplicating for this one-off investigation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from geosmile_hull_twobranch import ID_COLUMNS, SPATIAL_COLUMNS, apply_kernel, mask_pool
from hull_dehghani_eval import consistency_metrics


def combined_feature_smile(
    model,
    reference_full: pd.DataFrame,
    players: list[str],
    baselines: pd.Series,
    n_masks: int,
    seed: int,
) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray, np.ndarray]:
    """Feature-branch SMILE for a combined model. `reference_full` must contain
    both `players` and SPATIAL_COLUMNS, in the model's trained column order.
    Coordinates stay at each row's true value throughout.
    """
    rng = np.random.default_rng(seed)
    masks = mask_pool(n_masks, len(players), rng)
    original = np.asarray(model.predict(reference_full), float)
    responses = np.zeros(len(masks), dtype=float)
    for i, mask in enumerate(masks):
        perturbed = reference_full.copy()
        for j in np.flatnonzero(mask == 0):
            perturbed[players[j]] = baselines[players[j]]
        pred = np.asarray(model.predict(perturbed), float)
        responses[i] = wasserstein_distance(original, pred)
    distances = 1.0 - masks.mean(axis=1)
    order = rng.permutation(len(masks))
    a, b = int(0.6 * len(order)), int(0.8 * len(order))
    tr, va, te = order[:a], order[a:b], order[b:]
    sigma = max(float(np.median(distances[tr])), 1 / len(players))
    scores, fitted = {}, {}
    for kernel in ("rbf", "laplace", "inverse"):
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(masks[tr], responses[tr], sample_weight=apply_kernel(distances[tr], kernel, sigma))
        scores[kernel] = r2_score(responses[va], surrogate.predict(masks[va])) if len(va) > 1 else np.nan
        fitted[kernel] = surrogate
    best = max({k: v for k, v in scores.items() if np.isfinite(v)}, key=scores.get, default="rbf")
    surrogate = fitted[best]
    test_r2 = r2_score(responses[te], surrogate.predict(masks[te])) if len(te) > 1 else np.nan
    importance = pd.DataFrame(
        {"feature": players, "coefficient": surrogate.coef_, "abs_coefficient": np.abs(surrogate.coef_)}
    ).sort_values("abs_coefficient", ascending=False)
    diagnostics = {
        "best_kernel": best,
        "validation_r2_by_kernel": scores,
        "test_r2": float(test_r2),
        "n_unique_masks": int(len(masks)),
    }
    return importance, diagnostics, masks, responses, distances


def combined_geo_smile(
    model,
    panel: pd.DataFrame,
    test_mask: pd.Series,
    baselines: pd.Series,
    players: list[str],
    n_perturb: int,
    seed: int,
    min_sel: int = 5,
    max_sel: int = 10,
) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray, np.ndarray]:
    """Geo-branch SMILE for a combined model, per-LSOA masking (matches
    geo_smile_coefficients' methodology). Feature players stay at each row's
    true value throughout -- only SPATIAL_COLUMNS are masked to baseline, per
    selected LSOA.
    """
    columns = players + SPATIAL_COLUMNS
    frame = panel.loc[test_mask, ID_COLUMNS + columns].copy().reset_index(drop=True)
    x = frame[columns].copy()
    original = np.asarray(model.predict(x), float)
    lsoa_ids = frame["lsoa21cd"].astype(str).to_numpy()
    unique_lsoas = np.array(sorted(set(lsoa_ids.tolist())), dtype=object)
    n_lsoas = len(unique_lsoas)
    index_by_lsoa = {code: np.flatnonzero(lsoa_ids == code) for code in unique_lsoas}
    rng = np.random.default_rng(seed)
    M = np.zeros((n_perturb, n_lsoas), dtype=np.int8)
    responses = np.zeros(n_perturb, dtype=float)
    baseline_xy = baselines[SPATIAL_COLUMNS].to_numpy(dtype=float)

    for j in range(n_perturb):
        n_sel = int(rng.integers(min_sel, max_sel + 1))
        sel = rng.choice(n_lsoas, size=n_sel, replace=False)
        M[j, sel] = 1
        perturbed = x.copy()
        affected = []
        for idx in sel:
            rows = index_by_lsoa[unique_lsoas[idx]]
            perturbed.loc[rows, SPATIAL_COLUMNS] = baseline_xy
            affected.extend(rows.tolist())
        affected = np.asarray(affected, dtype=int)
        pert_pred = np.asarray(model.predict(perturbed), float)
        responses[j] = wasserstein_distance(original[affected], pert_pred[affected])

    distances = M.sum(axis=1).astype(float) / n_lsoas
    order = rng.permutation(n_perturb)
    a, b = int(0.6 * len(order)), int(0.8 * len(order))
    tr, va, te = order[:a], order[a:b], order[b:]
    sigma = max(float(np.median(distances[tr])), 1 / n_lsoas)
    scores, fitted = {}, {}
    for kernel in ("rbf", "laplace", "inverse"):
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(M[tr], responses[tr], sample_weight=apply_kernel(distances[tr], kernel, sigma))
        scores[kernel] = r2_score(responses[va], surrogate.predict(M[va])) if len(va) > 1 else np.nan
        fitted[kernel] = surrogate
    best = max({k: v for k, v in scores.items() if np.isfinite(v)}, key=scores.get, default="rbf")
    trva = np.concatenate([tr, va])
    final = Ridge(alpha=1.0)
    final.fit(M[trva], responses[trva], sample_weight=apply_kernel(distances[trva], best, sigma))
    test_r2 = r2_score(responses[te], final.predict(M[te])) if len(te) > 1 else np.nan
    coef = pd.DataFrame(
        {
            "lsoa21cd": unique_lsoas.astype(str),
            "geo_coefficient": final.coef_,
            "geo_abs_coefficient": np.abs(final.coef_),
        }
    )
    diagnostics = {
        "best_kernel": best,
        "validation_r2_by_kernel": scores,
        "test_r2": float(test_r2),
        "n_perturbations": int(n_perturb),
        "n_spatial_units": int(n_lsoas),
    }
    return coef, diagnostics, M, responses, distances


def combined_model_faithfulness(
    model,
    X_test: pd.DataFrame,
    players: list[str],
    baseline: pd.Series,
    feat_imp: pd.DataFrame,
    geo_imp: pd.DataFrame,
    feature_top_k: int,
    geo_top_k: int,
    n_random: int,
    seed: int,
) -> dict:
    """Top-vs-random Wasserstein faithfulness for a combined model, both
    branches explained by the SAME model. Adapted from
    eval_by_n_perturb.model_based_faithfulness (drops its population/area
    stratified Geo-random sampling -- plain uniform random LSOA sampling here).
    """
    columns = players + SPATIAL_COLUMNS
    feat_imp = feat_imp.sort_values(
        ["abs_coefficient", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    geo_imp = geo_imp.sort_values(
        ["geo_abs_coefficient", "lsoa21cd"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    top_feat = feat_imp.head(feature_top_k)["feature"].tolist()
    bottom_feat = feat_imp.tail(feature_top_k)["feature"].tolist()
    all_feat = feat_imp["feature"].tolist()
    top_geo = geo_imp.head(geo_top_k)["lsoa21cd"].astype(str).tolist()
    bottom_geo = geo_imp.tail(geo_top_k)["lsoa21cd"].astype(str).tolist()
    all_geo = geo_imp["lsoa21cd"].astype(str).tolist()

    rng = np.random.default_rng(seed)
    x_full = X_test[columns].copy()
    original = np.asarray(model.predict(x_full), float)
    lsoa_ids = X_test["lsoa21cd"].astype(str).to_numpy()
    baseline_xy = baseline[SPATIAL_COLUMNS].to_numpy(dtype=float)

    def feature_response(mask_features: list[str]) -> float:
        pert = x_full.copy()
        for name in mask_features:
            pert[name] = baseline[name]
        return float(wasserstein_distance(original, model.predict(pert)))

    def geo_response(mask_lsoas: list[str]) -> float:
        pert = x_full.copy()
        affected: list[int] = []
        for code in mask_lsoas:
            rows = np.flatnonzero(lsoa_ids == str(code))
            pert.loc[rows, SPATIAL_COLUMNS] = baseline_xy
            affected.extend(rows.tolist())
        affected_idx = np.asarray(affected, dtype=int)
        if len(affected_idx) == 0:
            return 0.0
        pert_pred = np.asarray(model.predict(pert), float)
        return float(wasserstein_distance(original[affected_idx], pert_pred[affected_idx]))

    def pack(branch: str, top_v: float, bot_v: float, rand_arr: np.ndarray, k: int) -> dict:
        p_top = float((np.sum(rand_arr >= top_v) + 1) / (len(rand_arr) + 1))
        return {
            "branch": branch,
            "k": int(k),
            "top_response": float(top_v),
            "bottom_response": float(bot_v),
            "random_mean": float(rand_arr.mean()),
            "top_minus_random": float(top_v - rand_arr.mean()),
            "top_minus_bottom": float(top_v - bot_v),
            "empirical_p_top_vs_random": p_top,
            "n_random": int(len(rand_arr)),
        }

    top_f = feature_response(top_feat)
    bot_f = feature_response(bottom_feat)
    rand_f = np.array(
        [feature_response(list(rng.choice(all_feat, size=feature_top_k, replace=False))) for _ in range(n_random)],
        dtype=float,
    )
    top_g = geo_response(top_geo)
    bot_g = geo_response(bottom_geo)
    rand_g = np.array(
        [geo_response(list(rng.choice(all_geo, size=geo_top_k, replace=False))) for _ in range(n_random)],
        dtype=float,
    )
    return {
        "feature": pack("Feature", top_f, bot_f, rand_f, feature_top_k),
        "geo": pack("Geo", top_g, bot_g, rand_g, geo_top_k),
    }


def combined_consistency(
    model,
    X_test: pd.DataFrame,
    panel: pd.DataFrame,
    test_mask: pd.Series,
    players: list[str],
    baseline: pd.Series,
    n_feature: int,
    n_geo: int,
    n_repeats: int,
    seed_base: int,
) -> dict:
    """Multi-seed rank-consistency for a combined model, both branches.
    Adapted from eval_by_n_perturb.consistency_for_n.
    """
    feat_runs, feat_fid_r2 = [], []
    geo_runs, geo_fid_r2 = [], []
    geo_index = None
    for seed in range(seed_base, seed_base + n_repeats):
        feat_imp, feat_diag, *_ = combined_feature_smile(model, X_test, players, baseline, n_feature, seed)
        aligned_f = feat_imp.set_index("feature").loc[players]["coefficient"].to_numpy(dtype=float)
        feat_runs.append(np.abs(aligned_f))
        feat_fid_r2.append(float(feat_diag["test_r2"]))

        geo_imp, geo_diag, *_ = combined_geo_smile(model, panel, test_mask, baseline, players, n_geo, seed)
        geo_imp = geo_imp.sort_values("lsoa21cd").reset_index(drop=True)
        if geo_index is None:
            geo_index = geo_imp["lsoa21cd"].astype(str).tolist()
        aligned_g = geo_imp.set_index("lsoa21cd").loc[geo_index]["geo_coefficient"].to_numpy(dtype=float)
        geo_runs.append(np.abs(aligned_g))
        geo_fid_r2.append(float(geo_diag["test_r2"]))

    feat_consist = consistency_metrics(np.asarray(feat_runs), top_pct=20)
    feat_consist["Fidelity R2 Mean"] = float(np.mean(feat_fid_r2))
    geo_consist = consistency_metrics(np.asarray(geo_runs), top_pct=20)
    geo_consist["Fidelity R2 Mean"] = float(np.mean(geo_fid_r2))
    return {"feature": feat_consist, "geo": geo_consist}
