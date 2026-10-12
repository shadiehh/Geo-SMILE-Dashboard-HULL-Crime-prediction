"""Geo branch under the area-data design.

Published Geo branch: a coordinates-only model; masking an LSOA replaces its
easting/northing with the training median. Area-data design (this module):
the Geo branch explains the same 30-feature model as the Feature branch, and
masking an LSOA replaces ALL of its input columns with the training median.
Each Geo coefficient therefore measures how sensitive the model is to the data
it takes from that location.

Because every masked row receives identical inputs, the masked prediction is a
single "typical-area" constant, so the per-LSOA response is monotone in how far
the LSOA's predictions sit from that constant. The signed SMILE coefficient
preserves that ordering (the sign only marks above/below the average
contribution); |coefficient| folds it. Rankings in this design therefore use
the SIGNED coefficient (`geo_score`), never |coefficient|.

`area_geo_coefficients` mirrors `geosmile_hull_twobranch.geo_smile_coefficients`
exactly (same RNG draw order, extra-LSOA handling, kernel selection, final fit
on train+validation), changing only which columns are masked. `surrogate_menu`
scores the full surrogate menu reported in Chapter 4 on one mask/response pool.
Not causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    LinearRegression,
    PoissonRegressor,
    Ridge,
    TweedieRegressor,
)
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

from geosmile_hull_twobranch import ID_COLUMNS, apply_kernel
from hull_dehghani_eval import split_masks

SPLIT_SEED = 42
N_NULL_PERMUTATIONS = 20
GEO_SCORE = "geo_score"


def area_geo_coefficients(
    model,
    panel: pd.DataFrame,
    mask: pd.Series,
    baselines: pd.Series,
    columns: list[str],
    n_perturb: int,
    seed: int,
    min_sel: int = 5,
    max_sel: int = 10,
    sample_lsoas: list[str] | None = None,
    return_pool: bool = False,
    kernel: str | None = None,
) -> tuple:
    """Per-LSOA Geo SMILE masking ALL `columns` of the selected LSOAs' rows.

    `kernel` fixes the surrogate kernel instead of selecting it on validation
    (used for the fixed-kernel stability comparison).
    """
    frame = panel.loc[mask, ID_COLUMNS + columns].copy().reset_index(drop=True)
    x = frame[columns].astype(float)
    original = np.asarray(model.predict(x), float)
    lsoa_ids = frame["lsoa21cd"].astype(str).to_numpy()
    present = sorted(set(lsoa_ids.tolist()))
    if sample_lsoas is None:
        unique_lsoas = np.array(present, dtype=object)
        n_sample = len(unique_lsoas)
        extras: list[str] = []
    else:
        core = [str(c) for c in sample_lsoas]
        missing = [c for c in core if c not in set(present)]
        if missing:
            raise ValueError(f"sample_lsoas missing from panel: {missing[:8]}")
        extras = [c for c in present if c not in set(core)]
        unique_lsoas = np.array(core + extras, dtype=object)
        n_sample = len(core)
    n_lsoas = len(unique_lsoas)
    index_by_lsoa = {code: np.flatnonzero(lsoa_ids == code) for code in unique_lsoas}
    rng = np.random.default_rng(seed)
    extra_rng = np.random.default_rng(seed + 1_000_003)
    M = np.zeros((n_perturb, n_lsoas), dtype=np.int8)
    responses = np.zeros(n_perturb, dtype=float)
    baseline_row = baselines[columns].to_numpy(dtype=float)
    x_values = x.to_numpy()

    for j in range(n_perturb):
        n_sel = int(rng.integers(min_sel, max_sel + 1))
        sel = rng.choice(n_sample, size=n_sel, replace=False)
        M[j, sel] = 1
        chosen = list(sel)
        if extras:
            p_extra = n_sel / n_sample
            for e, _code in enumerate(extras):
                if extra_rng.random() < p_extra:
                    extra_col = n_sample + e
                    M[j, extra_col] = 1
                    chosen.append(extra_col)
        affected = np.concatenate([index_by_lsoa[unique_lsoas[idx]] for idx in chosen])
        perturbed = x_values.copy()
        perturbed[affected, :] = baseline_row
        pert_pred = np.asarray(model.predict(pd.DataFrame(perturbed, columns=columns)), float)
        responses[j] = wasserstein_distance(original[affected], pert_pred[affected])

    distances = M[:, :n_sample].sum(axis=1).astype(float) / n_sample
    order = rng.permutation(n_perturb)
    a, b = int(0.6 * len(order)), int(0.8 * len(order))
    tr, va, te = order[:a], order[a:b], order[b:]
    sigma = max(float(np.median(distances[tr])), 1 / n_sample)
    scores, fitted = {}, {}
    for k in ("rbf", "laplace", "inverse"):
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(M[tr], responses[tr], sample_weight=apply_kernel(distances[tr], k, sigma))
        scores[k] = r2_score(responses[va], surrogate.predict(M[va])) if len(va) > 1 else np.nan
        fitted[k] = surrogate
    best = kernel or max({k: v for k, v in scores.items() if np.isfinite(v)}, key=scores.get, default="rbf")
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
    coef[GEO_SCORE] = coef["geo_coefficient"]
    diagnostics = {
        "method": "per_lsoa_area_data_masking",
        "n_spatial_units": int(n_lsoas),
        "n_perturbations": int(n_perturb),
        "n_sampled_lsoas": int(n_sample),
        "n_extra_lsoas": int(len(extras)),
        "min_lsoas_per_perturbation": min_sel,
        "max_lsoas_per_perturbation": max_sel,
        "best_kernel": best,
        "kernel_fixed": kernel is not None,
        "validation_r2_by_kernel": scores,
        "test_r2": float(test_r2),
        "intercept": float(final.intercept_),
        "response": "wasserstein_1_on_affected_lsoa_month_predictions",
        "perturbation": "replace ALL model inputs of selected LSOAs with training median",
        "ranking": "signed coefficient (geo_score); |coefficient| folds the ordering",
    }
    if return_pool:
        return coef, diagnostics, M, responses, distances
    return coef, diagnostics


def exact_area_shift(model, panel: pd.DataFrame, mask: pd.Series, baselines: pd.Series, columns: list[str]) -> pd.DataFrame:
    """Each LSOA masked alone: W1 on its own rows (crimes/month), and the signed mean shift."""
    frame = panel.loc[mask, ["lsoa21cd"] + columns].reset_index(drop=True)
    original = np.asarray(model.predict(frame[columns].astype(float)), float)
    masked = pd.DataFrame(
        np.tile(baselines[columns].to_numpy(dtype=float), (len(frame), 1)), columns=columns
    )
    pert = np.asarray(model.predict(masked), float)
    rows = []
    for code, idx in frame.groupby("lsoa21cd").indices.items():
        rows.append(
            {
                "lsoa21cd": str(code),
                "exact_w1": float(wasserstein_distance(original[idx], pert[idx])),
                "exact_signed_shift": float(np.mean(original[idx] - pert[idx])),
            }
        )
    return pd.DataFrame(rows)


# --- surrogate menu -----------------------------------------------------------------


def _score(y, pred, w) -> dict:
    resid = y - pred
    ybar_w = np.average(y, weights=w)
    return {
        "r2": float(r2_score(y, pred)),
        "weighted_r2": float(1 - np.sum(w * resid**2) / np.sum(w * (y - ybar_w) ** 2)),
        "mae": float(np.mean(np.abs(resid))),
        "wmae": float(np.average(np.abs(resid), weights=w)),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "wrmse": float(np.sqrt(np.average(resid**2, weights=w))),
    }


def surrogate_menu(masks: np.ndarray, responses: np.ndarray, distances: np.ndarray, seed: int = SPLIT_SEED) -> list[dict]:
    """Fit the Chapter-4 surrogate menu on one pool; score on the shared held-out split.

    Split and kernel selection follow `hull_dehghani_eval` (split_masks, seed 42),
    so the weighted-Ridge row equals `dehghani_fidelity` on the same pool.
    """
    from eval_by_n_perturb import dehghani_fidelity

    tr, va, te = split_masks(len(masks), seed)
    trva = np.concatenate([tr, va])
    Xtr, ytr, Xte, yte = masks[trva].astype(float), responses[trva], masks[te].astype(float), responses[te]
    ridge_fid = dehghani_fidelity(masks, responses, distances)
    kernel = ridge_fid["best_kernel"]
    sigma = max(float(np.median(distances[tr])), 1e-12)
    w_tr = np.clip(apply_kernel(distances[trva], kernel, sigma), 1e-8, None)
    w_te = np.clip(apply_kernel(distances[te], kernel, sigma), 1e-8, None)
    ypos = np.clip(ytr, 1e-9, None)  # GLMs need a strictly positive response

    def fit(reg, weighted: bool, y=None):
        y = ytr if y is None else y
        if weighted:
            reg.fit(Xtr, y, sample_weight=w_tr)
        else:
            reg.fit(Xtr, y)
        return _score(yte, reg.predict(Xte), w_te)

    rows = {
        "Weighted OLS": fit(LinearRegression(), True),
        "Weighted Ridge": {
            "r2": ridge_fid["R2"],
            "weighted_r2": ridge_fid["Weighted R2"],
            "mae": ridge_fid["MAE"],
            "wmae": ridge_fid["WMAE"],
            "rmse": ridge_fid["RMSE"],
            "wrmse": ridge_fid["WRMSE"],
        },
        "Bayesian ridge": fit(BayesianRidge(), True),
        "Linear regression": fit(LinearRegression(), False),
        "Ridge (unweighted)": fit(Ridge(alpha=1.0), False),
        # Settings calibrated to reproduce the published coordinates-only XGBoost Geo
        # row (N=4,000) to within ~0.01 R2 before being applied to the new design.
        "Elastic net (weak)": fit(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000), True),
        "Random forest": fit(RandomForestRegressor(n_estimators=300, max_depth=8, random_state=seed, n_jobs=-1), False),
        "Hist. grad. boosting": fit(HistGradientBoostingRegressor(max_depth=4, random_state=seed), False),
        "Kernel ridge (RBF)": fit(KernelRidge(kernel="rbf", alpha=1.0), True),
        "Tweedie GLM": fit(TweedieRegressor(power=1.5, alpha=1.0, link="log", max_iter=1000), True, ypos),
        "Poisson GLM": fit(PoissonRegressor(alpha=1.0, max_iter=1000), True, ypos),
        "Unweighted OLS": fit(LinearRegression(), False),
        "Ridge $\\alpha=0$": fit(LinearRegression(), True),
        "Shallow tree": fit(DecisionTreeRegressor(max_depth=3, random_state=seed), False),
    }
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(N_NULL_PERMUTATIONS):
        perm = rng.permutation(len(responses))
        yp = responses[perm]
        reg = Ridge(alpha=1.0).fit(Xtr, yp[trva], sample_weight=w_tr)
        null.append(r2_score(yp[te], reg.predict(Xte)))
    rows["Null, SD"] = {"r2": float(np.std(null))}
    return [{"surrogate": name, "best_kernel": kernel, **m} for name, m in rows.items()]
