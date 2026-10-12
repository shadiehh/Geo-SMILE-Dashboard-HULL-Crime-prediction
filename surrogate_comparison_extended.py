"""Extended SMILE surrogate comparison: a category-tailored candidate menu plus
a shared set of diagnostic controls, fit on the SAME mask->response pools
already saved by surrogate_comparison.py (outputs/surrogate_comparison/
<model>_{feature,geo}_pool.npz) -- no black-box model retraining needed.

Candidate menu (varies by which category the black-box model belongs to):
  Core (every model, both branches): weighted OLS, Ridge (kernel-weighted,
    the incumbent SMILE surrogate -- reuses dehghani_fidelity unmodified),
    Bayesian Ridge.
  + Gradient boosting (xgboost, lightgbm, catboost): Huber, Elastic Net.
  + Kernel/instance-based (svr, gp): ARD (a negative control on the GP
    kernel-selection instability noted elsewhere in this project -- ARD has
    no sample_weight support in this sklearn version, so it is fit unweighted).
  + Neural network (mlp, dl): narrow-bandwidth weighted OLS (sigma/4 of the
    incumbent Ridge's auto-selected bandwidth), Huber.

Controls (run for every model/branch, reported separately -- NOT eligible for
"best surrogate" selection, purely diagnostic):
  unweighted OLS; Ridge alpha->0 (identical to weighted OLS, reported under
  both labels per confirmed scope); Ridge alpha->large (1e6); a deliberately
  aggressive Lasso (alpha=1.0, no CV tuning); a shallow decision tree
  (max_depth=3); and a permuted-response null (20 repeats: responses randomly
  permuted before fitting the SAME kernel-weighted Ridge, held-out R2/MAE/RMSE
  averaged) -- establishes the chance-level fit this method would achieve if
  the mask/response relationship were pure noise.

Writes outputs/surrogate_comparison_extended/surrogate_comparison_extended_summary.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from eval_by_n_perturb import dehghani_fidelity
from hull_dehghani_eval import apply_kernel, split_masks, weighted_mae, weighted_r2_score, weighted_rmse

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
POOL_DIR = OUT / "surrogate_comparison"
DEST = OUT / "surrogate_comparison_extended"

SPLIT_SEED = 42
SMILE_SEED = 42
N_PERMUTATIONS = 20

CATEGORY = {
    "xgboost": "boosting", "lightgbm": "boosting", "catboost": "boosting",
    "svr": "kernel", "gp": "kernel",
    "mlp": "neural", "dl": "neural", "ann": "neural", "cnn": "neural", "transformer": "neural",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def score(y_true, y_pred, test_weight=None) -> dict:
    out = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    if test_weight is not None:
        out["weighted_r2"] = weighted_r2_score(y_true, y_pred, test_weight)
        out["wmae"] = weighted_mae(y_true, y_pred, test_weight)
        out["wrmse"] = weighted_rmse(y_true, y_pred, test_weight)
    return out


def fit_score(reg, X_tr, y_tr, X_te, y_te, sample_weight=None, test_weight=None) -> dict:
    if sample_weight is not None:
        try:
            reg.fit(X_tr, y_tr, sample_weight=sample_weight)
        except TypeError:
            reg.fit(X_tr, y_tr)
    else:
        reg.fit(X_tr, y_tr)
    return score(y_te, reg.predict(X_te), test_weight)


def run_pool(masks: np.ndarray, responses: np.ndarray, distances: np.ndarray, category: str) -> list[dict]:
    tr, va, te = split_masks(len(masks), SPLIT_SEED)
    trva = np.concatenate([tr, va])
    X_trva, y_trva = masks[trva], responses[trva]
    X_te, y_te = masks[te], responses[te]

    ridge_fid = dehghani_fidelity(masks, responses, distances)
    best_kernel = ridge_fid["best_kernel"]
    sigma = max(float(np.median(distances[tr])), 1e-12)
    w_trva = np.clip(apply_kernel(distances[trva], best_kernel, sigma), 1e-8, None)
    w_trva_narrow = np.clip(apply_kernel(distances[trva], best_kernel, sigma / 4.0), 1e-8, None)
    # Test-set weights use the SAME (standard, not narrow) kernel/sigma for every surrogate,
    # so the weighted metrics stay comparable across all of them -- including the
    # narrow-bandwidth candidate, whose narrow weights only affect its own training fit.
    w_te = np.clip(apply_kernel(distances[te], best_kernel, sigma), 1e-8, None)

    rows: list[dict] = []

    def add(role: str, name: str, m: dict) -> None:
        rows.append({"role": role, "surrogate": name, **m})

    # --- core candidates (every model, every branch) ---
    weighted_ols_score = fit_score(LinearRegression(), X_trva, y_trva, X_te, y_te, w_trva, w_te)
    add("candidate", "weighted_ols", weighted_ols_score)
    add("candidate", "ridge_kernel_weighted", {
        "r2": ridge_fid["R2"], "mae": ridge_fid["MAE"], "rmse": ridge_fid["RMSE"],
        "weighted_r2": ridge_fid["Weighted R2"], "wmae": ridge_fid["WMAE"], "wrmse": ridge_fid["WRMSE"],
        "best_kernel": best_kernel,
    })
    add("candidate", "bayesian_ridge", fit_score(BayesianRidge(), X_trva, y_trva, X_te, y_te, w_trva, w_te))

    # --- category-specific candidates ---
    if category == "boosting":
        add("candidate", "huber", fit_score(HuberRegressor(max_iter=1000), X_trva, y_trva, X_te, y_te, w_trva, w_te))
        add("candidate", "elastic_net", fit_score(ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000), X_trva, y_trva, X_te, y_te, w_trva, w_te))
    elif category == "kernel":
        add("candidate", "ard", fit_score(ARDRegression(), X_trva, y_trva, X_te, y_te, sample_weight=None, test_weight=w_te))
    elif category == "neural":
        add("candidate", "narrow_bandwidth_weighted_ols", fit_score(LinearRegression(), X_trva, y_trva, X_te, y_te, w_trva_narrow, w_te))
        add("candidate", "huber", fit_score(HuberRegressor(max_iter=1000), X_trva, y_trva, X_te, y_te, w_trva, w_te))

    # --- controls (every model, every branch) ---
    add("control", "unweighted_ols", fit_score(LinearRegression(), X_trva, y_trva, X_te, y_te, sample_weight=None, test_weight=w_te))
    add("control", "ridge_alpha_zero", weighted_ols_score)  # identical to weighted OLS, per confirmed scope
    add("control", "ridge_alpha_large", fit_score(Ridge(alpha=1e6), X_trva, y_trva, X_te, y_te, w_trva, w_te))
    add("control", "strong_lasso", fit_score(Lasso(alpha=1.0, max_iter=5000), X_trva, y_trva, X_te, y_te, w_trva, w_te))
    add("control", "shallow_tree", fit_score(DecisionTreeRegressor(max_depth=3, random_state=SMILE_SEED), X_trva, y_trva, X_te, y_te, sample_weight=None, test_weight=w_te))

    rng = np.random.default_rng(SMILE_SEED)
    null_scores = []
    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(len(responses))
        resp_perm = responses[perm]
        ridge = Ridge(alpha=1.0)
        ridge.fit(masks[trva], resp_perm[trva], sample_weight=w_trva)
        null_scores.append(score(resp_perm[te], ridge.predict(masks[te]), w_te))
    metric_keys = ("r2", "mae", "rmse", "weighted_r2", "wmae", "wrmse")
    add("control", "permuted_null_mean", {k: float(np.mean([s[k] for s in null_scores])) for k in metric_keys})
    add("control", "permuted_null_sd", {k: float(np.std([s[k] for s in null_scores])) for k in metric_keys})

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool-dir", type=Path, default=POOL_DIR,
                    help="Directory containing <model>_<branch>_pool.npz files (default: separate-models architecture).")
    p.add_argument("--dest", type=Path, default=DEST, help="Output directory for the summary CSV.")
    p.add_argument("--out-name", default="surrogate_comparison_extended_summary.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    pool_files = sorted(args.pool_dir.glob("*_pool.npz"))
    if not pool_files:
        raise SystemExit(f"No saved pools found in {args.pool_dir}.")

    for pool_path in pool_files:
        stem = pool_path.stem.removesuffix("_pool")
        model_name, branch = stem.rsplit("_", 1)
        category = CATEGORY.get(model_name)
        if category is None:
            _log(f"Skipping {pool_path.name}: unknown model {model_name!r}")
            continue
        data = np.load(pool_path)
        masks, responses, distances = data["masks"], data["responses"], data["distances"]
        _log(f"[{model_name}/{branch}] category={category}, n={len(masks)}...")
        for row in run_pool(masks, responses, distances, category):
            all_rows.append({"model": model_name, "branch": branch.capitalize(), "category": category, **row})
        _log(f"[{model_name}/{branch}] done ({sum(1 for r in all_rows if r['model'] == model_name and r['branch'] == branch.capitalize())} rows)")

    df = pd.DataFrame(all_rows)
    out_path = args.dest / args.out_name
    df.to_csv(out_path, index=False)
    _log(f"\nDone. {len(df)} rows across {df['model'].nunique()} models. Summary: {out_path}")


if __name__ == "__main__":
    main()
