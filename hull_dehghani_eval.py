"""Dehghani / Seattle-style GeoSMILE evaluation metrics for Hull (two-branch).

Matches the Seattle GeoSMILE evaluation design used to align with Dehghani et al.:

* Fidelity — held-out surrogate R², weighted R², MAE/WMAE, RMSE/WRMSE
* Perturbation response monotonicity — Spearman/Pearson of mask cardinality vs Wasserstein response
  (formerly labelled "Faithfulness"; not expert-reference accuracy and not model faithfulness)
* Sparsity/Entropy — Hoyer, Gini, entropy of absolute coefficients
* Consistency — multi-seed rank/score agreement of attribution coefficients
* Stability — top-set Jaccard across multi-seed reruns (repeated-run; not controlled-input stability)
* Convergence — fidelity vs number of training perturbations

Branches evaluated:
* Feature — non-spatial players only (no coordinates)
* Geo — per-LSOA location masking
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from geosmile_hull_twobranch import (
    DERIVED_FEATURES,
    FORBIDDEN_FEATURE_TOKENS,
    ID_COLUMNS,
    SPATIAL_COLUMNS,
    choose_feature_players,
    make_model,
    temporal_split,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"
BLUE, PINK, TEAL = "#1E88E5", "#FF0051", "#147D74"
INK_MUTED = "#57646D"

KERNELS = ["rbf", "laplace", "inverse"]


def apply_kernel(distances, kernel_type, sigma=None):
    d = np.asarray(distances, dtype=float)
    if sigma is None:
        sigma = np.median(d) + 1e-12
    if kernel_type == "rbf":
        return np.exp(-(d**2) / (sigma**2))
    if kernel_type == "laplace":
        return np.exp(-d / sigma)
    if kernel_type == "inverse":
        return 1.0 / (1.0 + d / sigma)
    raise ValueError(kernel_type)


def select_best_kernel(M_tr, M_va, dist_tr, resp_tr, resp_va):
    scores = {}
    for kernel in KERNELS:
        w = np.clip(apply_kernel(dist_tr, kernel), 1e-8, None)
        ridge = Ridge(alpha=1.0)
        ridge.fit(M_tr, resp_tr, sample_weight=w)
        scores[kernel] = r2_score(resp_va, ridge.predict(M_va))
    best = max(scores, key=scores.get)
    return best, scores


def weighted_r2_score(y_true, y_pred, sample_weight):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    sample_weight = np.asarray(sample_weight, float)
    if np.sum(sample_weight) <= 0:
        return np.nan
    weighted_mean = np.average(y_true, weights=sample_weight)
    ss_res = np.sum(sample_weight * (y_true - y_pred) ** 2)
    ss_tot = np.sum(sample_weight * (y_true - weighted_mean) ** 2)
    if ss_tot <= 1e-12:
        return np.nan
    return float(1 - (ss_res / ss_tot))


def weighted_mae(y_true, y_pred, sample_weight):
    return float(
        np.average(np.abs(np.asarray(y_true) - np.asarray(y_pred)), weights=sample_weight)
    )


def weighted_rmse(y_true, y_pred, sample_weight):
    return float(
        np.sqrt(
            np.average((np.asarray(y_true) - np.asarray(y_pred)) ** 2, weights=sample_weight)
        )
    )


def fidelity_metrics(y_true, y_pred, weights) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "Weighted R2": weighted_r2_score(y_true, y_pred, weights),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "WMAE": weighted_mae(y_true, y_pred, weights),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "WRMSE": weighted_rmse(y_true, y_pred, weights),
    }


def sparsity_entropy_metrics(scores_raw) -> dict[str, float]:
    s = np.abs(np.asarray(scores_raw, dtype=float))
    s_sum = s.sum() + 1e-12
    p = s / s_sum
    hoyer = (np.sqrt(len(s)) - np.linalg.norm(s, 1) / (np.linalg.norm(s, 2) + 1e-12)) / (
        np.sqrt(len(s)) - 1 + 1e-12
    )
    n_nz = int((s > 1e-10).sum())
    gini = 2 * np.sum((np.arange(1, len(s) + 1)) * np.sort(s)) / (len(s) * s_sum) - (
        len(s) + 1
    ) / len(s)
    ent = float(-np.sum(p * np.log(p + 1e-12)))
    return {
        "hoyer": float(hoyer),
        "gini": float(gini),
        "entropy": ent,
        "nonzero": n_nz,
        "n": int(len(s)),
    }


def faithfulness_stats(cardinality, wd_response) -> dict[str, float]:
    card = np.asarray(cardinality, float)
    wd = np.asarray(wd_response, float)
    rho, p_rho = spearmanr(card, wd)
    r, p_r = pearsonr(card, wd)
    return {
        "spearman": float(rho),
        "p_spearman": float(p_rho),
        "pearson": float(r),
        "p_pearson": float(p_r),
    }


def split_masks(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    return order[:a], order[a:b], order[b:]


def build_feature_pool(
    model: XGBRegressor,
    X: pd.DataFrame,
    players: list[str],
    baselines: pd.Series,
    n_masks: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    masks = rng.integers(0, 2, size=(n_masks, len(players)), dtype=np.int8)
    masks[0] = 1
    masks[1] = 0
    original = np.asarray(model.predict(X[players]), float)
    responses = np.zeros(n_masks, dtype=float)
    for i, mask in enumerate(masks):
        pert = X[players].copy()
        for j in np.flatnonzero(mask == 0):
            pert[players[j]] = baselines[players[j]]
        responses[i] = wasserstein_distance(original, model.predict(pert))
    distances = 1.0 - masks.mean(axis=1)
    return masks, responses, distances


def build_geo_pool(
    model: XGBRegressor,
    frame: pd.DataFrame,
    baselines: pd.Series,
    n_perturb: int,
    seed: int,
    min_sel: int = 5,
    max_sel: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = frame[SPATIAL_COLUMNS].copy().reset_index(drop=True)
    original = np.asarray(model.predict(x), float)
    lsoa_ids = frame["lsoa21cd"].to_numpy()
    unique_lsoas = np.array(sorted(frame["lsoa21cd"].unique()))
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
            perturbed.iloc[rows, :] = baseline_xy
            affected.extend(rows.tolist())
        affected = np.asarray(affected, dtype=int)
        pert_pred = np.asarray(model.predict(perturbed), float)
        responses[j] = wasserstein_distance(original[affected], pert_pred[affected])
    distances = M.sum(axis=1).astype(float) / n_lsoas
    return M, responses, distances, unique_lsoas


def fit_surrogate(masks, responses, distances, best_kernel, tr, va, te):
    sigma = max(float(np.median(distances[tr])), 1e-12)
    trva = np.concatenate([tr, va])
    ridge = Ridge(alpha=1.0)
    ridge.fit(
        masks[trva],
        responses[trva],
        sample_weight=apply_kernel(distances[trva], best_kernel, sigma),
    )
    w_te = np.clip(apply_kernel(distances[te], best_kernel, sigma), 1e-8, None)
    pred_te = ridge.predict(masks[te])
    return ridge, fidelity_metrics(responses[te], pred_te, w_te)


def consistency_metrics(coef_runs: np.ndarray, top_pct: float = 20.0) -> dict[str, float]:
    """coef_runs shape: (n_runs, n_units)."""
    rank_vals, score_vals, set_vals = [], [], []
    for i, j in combinations(range(len(coef_runs)), 2):
        a = np.abs(coef_runs[i])
        b = np.abs(coef_runs[j])
        rank_vals.append(spearmanr(a, b)[0])
        score_vals.append(pearsonr(a, b)[0])
        ta = np.percentile(a, 100 - top_pct)
        tb = np.percentile(b, 100 - top_pct)
        A = set(np.flatnonzero(a >= ta))
        B = set(np.flatnonzero(b >= tb))
        set_vals.append(len(A & B) / len(A | B) if (A | B) else 1.0)
    return {
        "Rank Spearman Mean": float(np.nanmean(rank_vals)),
        "Rank Spearman Std": float(np.nanstd(rank_vals)),
        "Score Pearson Mean": float(np.nanmean(score_vals)),
        "Score Pearson Std": float(np.nanstd(score_vals)),
        f"Top-{int(top_pct)}% Jaccard Mean": float(np.nanmean(set_vals)),
        f"Top-{int(top_pct)}% Jaccard Std": float(np.nanstd(set_vals)),
    }


def convergence_table(masks, responses, distances, best_kernel, tr, va, sizes):
    rows = []
    w_va = np.clip(apply_kernel(distances[va], best_kernel), 1e-8, None)
    for n in sizes:
        n_use = min(n, len(tr))
        idx = tr[:n_use]
        ridge = Ridge(alpha=1.0)
        ridge.fit(
            masks[idx],
            responses[idx],
            sample_weight=apply_kernel(distances[idx], best_kernel),
        )
        pred = ridge.predict(masks[va])
        resid = responses[va] - pred
        rows.append(
            {
                "#Perturb": int(n_use),
                "Kernel": best_kernel,
                "R2": float(r2_score(responses[va], pred)),
                "Weighted R2": weighted_r2_score(responses[va], pred, w_va),
                "WMAE": float(np.average(np.abs(resid), weights=w_va)),
                "WRMSE": float(np.sqrt(np.average(resid**2, weights=w_va))),
                "MAE": float(mean_absolute_error(responses[va], pred)),
                "RMSE": float(np.sqrt(mean_squared_error(responses[va], pred))),
            }
        )
    return pd.DataFrame(rows)


def _rows(metrics: dict, branch: str, category: str) -> list[dict]:
    return [
        {"Branch": branch, "Category": category, "Metric": k, "Value": v}
        for k, v in metrics.items()
    ]


def save_fidelity_chart(summary: pd.DataFrame, path: Path) -> None:
    fid = summary[
        (summary["Category"] == "Fidelity") & (summary["Metric"].isin(["R2", "Weighted R2"]))
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    branches = ["Feature", "Geo"]
    x = np.arange(len(branches))
    width = 0.35
    for i, metric in enumerate(["R2", "Weighted R2"]):
        vals = [
            float(fid[(fid["Branch"] == b) & (fid["Metric"] == metric)]["Value"].iloc[0])
            for b in branches
        ]
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=metric, color=[BLUE, PINK][i])
    ax.set_xticks(x)
    ax.set_xticklabels(branches)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Surrogate fidelity")
    ax.set_title("Dehghani-style surrogate fidelity (Hull two-branch)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.text(
        0.5,
        -0.04,
        "Held-out W1 Ridge surrogate fidelity. Matches Seattle/Dehghani fidelity suite.",
        ha="center",
        fontsize=8,
        color=INK_MUTED,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    if not PANEL_PATH.exists():
        raise SystemExit(
            f"Missing {PANEL_PATH}. Run geosmile_hull_twobranch.py first."
        )
    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])

    # Guard: Feature players must remain non-spatial.
    players = choose_feature_players(panel)
    bad = [p for p in players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise ValueError(f"Spatial feature leaked into Feature branch: {bad}")

    train, validation, test = temporal_split(panel)
    y_col = "target_next_month_crime"
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()

    print("Fitting black-box models...")
    model_feat = make_model(args.seed)
    model_geo = make_model(args.seed)
    model_feat.fit(
        panel.loc[train, players],
        panel.loc[train, y_col],
        eval_set=[(panel.loc[validation, players], panel.loc[validation, y_col])],
        verbose=False,
    )
    model_geo.fit(
        panel.loc[train, SPATIAL_COLUMNS],
        panel.loc[train, y_col],
        eval_set=[(panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, y_col])],
        verbose=False,
    )

    X_test = panel.loc[test].reset_index(drop=True)
    print(f"Building Feature perturbation pool ({args.feat_masks})...")
    M_feat, y_feat, d_feat = build_feature_pool(
        model_feat, X_test, players, baseline, args.feat_masks, args.seed
    )
    tr_f, va_f, te_f = split_masks(len(M_feat), args.seed)
    best_feat, feat_kernel_scores = select_best_kernel(
        M_feat[tr_f], M_feat[va_f], d_feat[tr_f], y_feat[tr_f], y_feat[va_f]
    )
    ridge_feat, feat_fidelity = fit_surrogate(
        M_feat, y_feat, d_feat, best_feat, tr_f, va_f, te_f
    )
    print(f"Feature best kernel={best_feat}; fidelity R2={feat_fidelity['R2']:.4f}")

    print(f"Building Geo perturbation pool ({args.geo_perturbations})...")
    M_geo, y_geo, d_geo, unique_lsoas = build_geo_pool(
        model_geo, X_test, baseline, args.geo_perturbations, args.seed
    )
    tr_g, va_g, te_g = split_masks(len(M_geo), args.seed + 1)
    best_geo, geo_kernel_scores = select_best_kernel(
        M_geo[tr_g], M_geo[va_g], d_geo[tr_g], y_geo[tr_g], y_geo[va_g]
    )
    ridge_geo, geo_fidelity = fit_surrogate(
        M_geo, y_geo, d_geo, best_geo, tr_g, va_g, te_g
    )
    print(f"Geo best kernel={best_geo}; fidelity R2={geo_fidelity['R2']:.4f}")

    # Perturbation response monotonicity (cardinality vs WD).
    # Not expert-reference accuracy and not model faithfulness.
    faith_feat = faithfulness_stats(d_feat, y_feat)
    faith_geo = faithfulness_stats(d_geo, y_geo)
    perturbation_monotonicity = {
        "Feature Spearman: perturbation cardinality vs WD": faith_feat["spearman"],
        "Feature Pearson: perturbation cardinality vs WD": faith_feat["pearson"],
        "Geo Spearman: perturbation cardinality vs WD": faith_geo["spearman"],
        "Geo Pearson: perturbation cardinality vs WD": faith_geo["pearson"],
    }

    # Sparsity on absolute coefficients
    feat_sparsity = sparsity_entropy_metrics(ridge_feat.coef_)
    geo_sparsity = sparsity_entropy_metrics(ridge_geo.coef_)

    # Convergence
    conv_sizes = [100, 200, 400, 600, 800, len(tr_f)]
    feat_conv = convergence_table(M_feat, y_feat, d_feat, best_feat, tr_f, va_f, conv_sizes)
    feat_conv.insert(0, "Branch", "Feature")
    geo_conv_sizes = [200, 400, 800, 1200, 1600, len(tr_g)]
    geo_conv = convergence_table(M_geo, y_geo, d_geo, best_geo, tr_g, va_g, geo_conv_sizes)
    geo_conv.insert(0, "Branch", "Geo")
    convergence = pd.concat([feat_conv, geo_conv], ignore_index=True)

    # Multi-run consistency / stability (Seattle Steps 25–26 style)
    base = args.consistency_seed_base
    print(f"Multi-run consistency ({args.n_repeats} seeds from {base})...")
    feat_runs = []
    feat_fid_runs = []
    for seed in range(base, base + args.n_repeats):
        M_r, y_r, d_r = build_feature_pool(
            model_feat, X_test, players, baseline, args.repeat_masks, seed
        )
        tr, va, te = split_masks(len(M_r), args.split_seed)
        best, _ = select_best_kernel(M_r[tr], M_r[va], d_r[tr], y_r[tr], y_r[va])
        ridge, fid = fit_surrogate(M_r, y_r, d_r, best, tr, va, te)
        feat_runs.append(np.abs(ridge.coef_))
        feat_fid_runs.append(fid["R2"])
    feat_consist = consistency_metrics(np.asarray(feat_runs), top_pct=20)
    feat_consist["Fidelity R2 Mean"] = float(np.mean(feat_fid_runs))
    feat_consist["Fidelity R2 Std"] = float(np.std(feat_fid_runs))

    geo_runs = []
    geo_fid_runs = []
    for seed in range(base, base + args.n_repeats):
        M_r, y_r, d_r, _ = build_geo_pool(
            model_geo, X_test, baseline, args.repeat_geo, seed
        )
        tr, va, te = split_masks(len(M_r), args.split_seed)
        best, _ = select_best_kernel(M_r[tr], M_r[va], d_r[tr], y_r[tr], y_r[va])
        ridge, fid = fit_surrogate(M_r, y_r, d_r, best, tr, va, te)
        geo_runs.append(np.abs(ridge.coef_))
        geo_fid_runs.append(fid["R2"])
    geo_consist = consistency_metrics(np.asarray(geo_runs), top_pct=20)
    geo_consist["Fidelity R2 Mean"] = float(np.mean(geo_fid_runs))
    geo_consist["Fidelity R2 Std"] = float(np.std(geo_fid_runs))

    # Combined evaluation table (Seattle Step 28 style)
    summary_rows = (
        _rows(feat_fidelity, "Feature", "Fidelity")
        + _rows(geo_fidelity, "Geo", "Fidelity")
        + _rows(perturbation_monotonicity, "Both", "Perturbation response monotonicity")
        + _rows(feat_consist, "Feature", "Consistency (multi-run)")
        + _rows(geo_consist, "Geo", "Consistency (multi-run)")
        + _rows(feat_sparsity, "Feature", "Sparsity/Entropy")
        + _rows(geo_sparsity, "Geo", "Sparsity/Entropy")
        + [
            {"Branch": "Feature", "Category": "Setup", "Metric": "Best kernel", "Value": best_feat},
            {"Branch": "Geo", "Category": "Setup", "Metric": "Best kernel", "Value": best_geo},
            {
                "Branch": "Feature",
                "Category": "Setup",
                "Metric": "Players",
                "Value": len(players),
            },
            {
                "Branch": "Geo",
                "Category": "Setup",
                "Metric": "Spatial units",
                "Value": len(unique_lsoas),
            },
        ]
    )
    summary = pd.DataFrame(summary_rows)

    # Paper-style compact table (Chicago three_branch_summary analogue)
    compact = pd.DataFrame(
        [
            {"branch": "Feature", "category": "Fidelity", "metric": "Test R2", "value": feat_fidelity["R2"]},
            {
                "branch": "Feature",
                "category": "Fidelity",
                "metric": "Test Weighted R2",
                "value": feat_fidelity["Weighted R2"],
            },
            {
                "branch": "Feature",
                "category": "Fidelity",
                "metric": "Test WMAE",
                "value": feat_fidelity["WMAE"],
            },
            {"branch": "Geo", "category": "Fidelity", "metric": "Test R2", "value": geo_fidelity["R2"]},
            {
                "branch": "Geo",
                "category": "Fidelity",
                "metric": "Test Weighted R2",
                "value": geo_fidelity["Weighted R2"],
            },
            {"branch": "Geo", "category": "Fidelity", "metric": "Test WMAE", "value": geo_fidelity["WMAE"]},
            {
                "branch": "Feature",
                "category": "Perturbation response monotonicity",
                "metric": "Spearman: perturbation cardinality vs WD",
                "value": faith_feat["spearman"],
            },
            {
                "branch": "Feature",
                "category": "Perturbation response monotonicity",
                "metric": "Pearson: perturbation cardinality vs WD",
                "value": faith_feat["pearson"],
            },
            {
                "branch": "Geo",
                "category": "Perturbation response monotonicity",
                "metric": "Spearman: perturbation cardinality vs WD",
                "value": faith_geo["spearman"],
            },
            {
                "branch": "Geo",
                "category": "Perturbation response monotonicity",
                "metric": "Pearson: perturbation cardinality vs WD",
                "value": faith_geo["pearson"],
            },
            {
                "branch": "Feature",
                "category": "Stability",
                "metric": "Top-20% Jaccard",
                "value": feat_consist["Top-20% Jaccard Mean"],
            },
            {
                "branch": "Geo",
                "category": "Stability",
                "metric": "Top-20% Jaccard",
                "value": geo_consist["Top-20% Jaccard Mean"],
            },
            {
                "branch": "Feature",
                "category": "Consistency",
                "metric": "Mean Rank Spearman",
                "value": feat_consist["Rank Spearman Mean"],
            },
            {
                "branch": "Geo",
                "category": "Consistency",
                "metric": "Mean Rank Spearman",
                "value": geo_consist["Rank Spearman Mean"],
            },
            {
                "branch": "Feature",
                "category": "Sparsity",
                "metric": "Hoyer",
                "value": feat_sparsity["hoyer"],
            },
            {"branch": "Geo", "category": "Sparsity", "metric": "Hoyer", "value": geo_sparsity["hoyer"]},
            {"branch": "Feature", "category": "Setup", "metric": "Best kernel", "value": best_feat},
            {"branch": "Geo", "category": "Setup", "metric": "Best kernel", "value": best_geo},
        ]
    )

    # Save
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "geosmile_hull_evaluation_full.csv", index=False)
    compact.to_csv(OUT / "geosmile_hull_two_branch_summary.csv", index=False)
    convergence.to_csv(OUT / "geosmile_hull_convergence.csv", index=False)
    pd.DataFrame([feat_fidelity]).assign(branch="Feature").to_csv(
        OUT / "geosmile_hull_feat_fidelity.csv", index=False
    )
    pd.DataFrame([geo_fidelity]).assign(branch="Geo").to_csv(
        OUT / "geosmile_hull_geo_fidelity.csv", index=False
    )
    pd.DataFrame(
        [
            {"branch": "Feature", **faith_feat},
            {"branch": "Geo", **faith_geo},
        ]
    ).to_csv(OUT / "geosmile_hull_faithfulness.csv", index=False)
    pd.DataFrame(
        [
            {"branch": "Feature", **feat_sparsity},
            {"branch": "Geo", **geo_sparsity},
        ]
    ).to_csv(OUT / "geosmile_hull_sparsity.csv", index=False)
    pd.DataFrame([feat_consist]).assign(branch="Feature").to_csv(
        OUT / "geosmile_hull_feat_consistency.csv", index=False
    )
    pd.DataFrame([geo_consist]).assign(branch="Geo").to_csv(
        OUT / "geosmile_hull_geo_consistency.csv", index=False
    )
    pd.DataFrame(
        [{"kernel": k, "validation_r2": v} for k, v in feat_kernel_scores.items()]
    ).to_csv(OUT / "geosmile_hull_kernel_selection_feat.csv", index=False)
    pd.DataFrame(
        [{"kernel": k, "validation_r2": v} for k, v in geo_kernel_scores.items()]
    ).to_csv(OUT / "geosmile_hull_kernel_selection_geo.csv", index=False)

    save_fidelity_chart(summary, OUT / "figures" / "dehghani_fidelity_comparison.png")

    # Convergence plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, branch, color in zip(axes, ["Feature", "Geo"], [BLUE, TEAL]):
        sub = convergence[convergence["Branch"] == branch]
        ax.plot(sub["#Perturb"], sub["R2"], marker="o", color=color, label="R2")
        ax.plot(
            sub["#Perturb"],
            sub["Weighted R2"],
            marker="s",
            color=PINK,
            label="Weighted R2",
        )
        ax.set_title(f"{branch}: fidelity vs #perturbations")
        ax.set_xlabel("# training perturbations")
        ax.set_ylabel("Validation R²")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Dehghani-style convergence (Hull)", fontweight="bold")
    fig.savefig(
        OUT / "figures" / "dehghani_convergence.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    meta = {
        "paper_alignment": "Dehghani et al. SMILE evaluation suite via Seattle GeoSMILE implementation",
        "branches": ["Feature", "Geo"],
        "feature_players": players,
        "feature_excludes_coordinates": True,
        "n_lsoas": int(len(unique_lsoas)),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "consistency_seed_base": args.consistency_seed_base,
        "feat_masks": args.feat_masks,
        "geo_perturbations": args.geo_perturbations,
        "n_consistency_repeats": args.n_repeats,
        "feat_fidelity": feat_fidelity,
        "geo_fidelity": geo_fidelity,
        "best_kernel_feature": best_feat,
        "best_kernel_geo": best_geo,
    }
    (OUT / "geosmile_hull_dehghani_eval_summary.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("\n=== Dehghani-style two-branch summary ===")
    print(compact.to_string(index=False))
    print(f"\nSaved evaluation tables under {OUT.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42, help="Model + primary perturbation seed")
    p.add_argument("--split-seed", type=int, default=42, help="Train/val/test split of mask pool")
    p.add_argument(
        "--consistency-seed-base",
        type=int,
        default=10,
        help="First seed for multi-run consistency/stability (uses base..base+n_repeats-1)",
    )
    p.add_argument("--feat-masks", type=int, default=1024)
    p.add_argument("--geo-perturbations", type=int, default=2000)
    p.add_argument("--n-repeats", type=int, default=5, help="Consistency seeds (Seattle uses 10)")
    p.add_argument("--repeat-masks", type=int, default=600)
    p.add_argument("--repeat-geo", type=int, default=800)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
