"""Cross the two design axes that have so far only been tested separately:
surrogate choice (surrogate_comparison_extended.py, tested at a single fixed
N per model) and perturbation budget (response_metric_comparison.py, tested
with the surrogate fixed at the incumbent kernel-weighted Ridge). This script
fits the FULL category-tailored surrogate menu -- core (weighted OLS, Ridge
kernel-weighted, Bayesian Ridge) plus category-specific candidates (boosting:
Huber/Elastic Net; kernel: ARD; neural: narrow-bandwidth weighted OLS/Huber)
plus the 6 diagnostic controls -- at EVERY N in the Feature perturbation grid,
for every model.

Motivation: MLP and DL's Feature-branch fidelity collapses (R2 ~0.04-0.23)
regardless of N when the surrogate is held fixed at Ridge, and is flat near
ceiling for every other model regardless of N. The open question this answers
is whether a different surrogate rescues MLP/DL at SOME N -- i.e. whether the
two axes interact -- rather than assuming (from testing them separately) that
neither axis matters on its own.

Scope: Feature branch only. Geo branch already plateaus at N>=4000 with Ridge
alone for every model (including MLP/DL, whose Geo fidelity is already fine,
~0.94-0.95) -- there is no open question there to justify repeating this
model's expensive predict-loop a second time. Response metric fixed at
Wasserstein (the incumbent, canonical metric used everywhere else in this
project) to keep the combined sweep tractable; L1/L2 were already shown not to
help MLP/DL's Feature fidelity under the Ridge-only sweep.

Reuses feature_pool_multi_response() from response_metric_comparison.py for
pool generation (unmodified) and run_pool()/CATEGORY from
surrogate_comparison_extended.py for surrogate fitting (unmodified) -- no
black-box model retraining logic duplicated.

Writes outputs/surrogate_by_n/surrogate_by_n_summary.csv.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from geosmile_hull_twobranch import FORBIDDEN_FEATURE_TOKENS, SPATIAL_COLUMNS, choose_feature_players, temporal_split
from hull_dehghani_eval import apply_kernel, split_masks
from model_factory import MODEL_NAMES, fit_model, make_model
from response_metric_comparison import FEAT_N_GRID, SMILE_SEED, feature_pool_multi_response
from eval_by_n_perturb import Y_COL, dehghani_fidelity
from surrogate_comparison_extended import CATEGORY, SPLIT_SEED, run_pool

# Tree-ensemble candidates, added only for the "neural" category (mlp, dl) --
# their Feature-branch fidelity collapses under every linear/robust surrogate
# in run_pool()'s menu, so this checks whether a genuinely non-linear
# surrogate can fit the mask->response relationship better than any of those.
# Both an unweighted fit (matching surrogate_comparison.py's original
# convention: no established analogue for kernel weighting on tree ensembles)
# and a kernel-weighted fit (same best_kernel/sigma as the linear candidates,
# since RandomForestRegressor/HistGradientBoostingRegressor both DO support
# sample_weight, unlike ARDRegression) are tried, to see whether weighting
# stabilizes the occasional crash the unweighted fit shows at some N.


def tree_candidates(masks: np.ndarray, responses: np.ndarray, distances: np.ndarray, split_seed: int = SPLIT_SEED) -> list[dict]:
    tr, va, te = split_masks(len(masks), split_seed)
    trva = np.concatenate([tr, va])
    X_trva, y_trva = masks[trva], responses[trva]
    X_te, y_te = masks[te], responses[te]

    ridge_fid = dehghani_fidelity(masks, responses, distances)
    sigma = max(float(np.median(distances[tr])), 1e-12)
    w_trva = np.clip(apply_kernel(distances[trva], ridge_fid["best_kernel"], sigma), 1e-8, None)

    def score(y_true, y_pred) -> dict:
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }

    rows = []
    for label, weight in (("", None), ("_weighted", w_trva)):
        rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=SMILE_SEED, n_jobs=-1)
        rf.fit(X_trva, y_trva, sample_weight=weight)
        rows.append({"role": "candidate", "surrogate": f"random_forest{label}", **score(y_te, rf.predict(X_te))})

        hgb = HistGradientBoostingRegressor(max_depth=4, random_state=SMILE_SEED)
        hgb.fit(X_trva, y_trva, sample_weight=weight)
        rows.append({"role": "candidate", "surrogate": f"hist_gradient_boosting{label}", **score(y_te, hgb.predict(X_te))})

    return rows

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "surrogate_by_n"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"
PREDICTIVE_CSV = OUT / "eval_by_model" / "predictive_performance.csv"

R2_GATE = 0.7


def _log(msg: str) -> None:
    print(msg, flush=True)


def evaluate_model(model_name: str, *, panel, players, train, validation, test, baseline, X_test, n_jobs: int,
                    n_grid=FEAT_N_GRID) -> list[dict]:
    t0 = time.time()
    category = CATEGORY[model_name]
    model_feat = make_model(model_name, SMILE_SEED, n_jobs=n_jobs)
    fit_model(model_name, model_feat, panel.loc[train, players], panel.loc[train, Y_COL],
              panel.loc[validation, players], panel.loc[validation, Y_COL])
    rows: list[dict] = []
    for n in n_grid:
        _log(f"[{model_name}] Feature pool (n={n})...")
        masks, distances, resp_by_metric = feature_pool_multi_response(model_feat, X_test, players, baseline, n, SMILE_SEED)
        responses = resp_by_metric["wasserstein"]
        menu_rows = run_pool(masks, responses, distances, category)
        if category == "neural":
            menu_rows += tree_candidates(masks, responses, distances)
        for row in menu_rows:
            rows.append({"model": model_name, "branch": "Feature", "n": n, "category": category, **row})
            _log(f"[{model_name}] n={n} {row['role']}/{row['surrogate']}: R2={row['r2']:.4f}")
    _log(f"[{model_name}] done in {(time.time() - t0) / 60:.1f} min")
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--n-grid", nargs="+", type=int, default=None,
                    help="Override the Feature perturbation-count grid (default: %s)." % (FEAT_N_GRID,))
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

    predictive = pd.read_csv(PREDICTIVE_CSV)

    all_rows: list[dict] = []
    for model_name in args.models:
        pred_feat = predictive[(predictive["model"] == model_name) & (predictive["branch"] == "Feature")]
        feat_r2 = float(pred_feat.iloc[0]["r2"]) if not pred_feat.empty else float("nan")
        if not (np.isfinite(feat_r2) and feat_r2 > R2_GATE):
            _log(f"\n=== [{model_name}] === SKIPPED (Feature R2={feat_r2:.4f} <= {R2_GATE})")
            continue
        _log(f"\n=== [{model_name}] === category={CATEGORY[model_name]}")
        all_rows.extend(evaluate_model(
            model_name, panel=panel, players=players, train=train, validation=validation, test=test,
            baseline=baseline, X_test=X_test, n_jobs=args.n_jobs,
            n_grid=args.n_grid if args.n_grid is not None else FEAT_N_GRID,
        ))

    new_df = pd.DataFrame(all_rows)
    out_path = DEST / "surrogate_by_n_summary.csv"
    if out_path.exists() and not new_df.empty:
        old_df = pd.read_csv(out_path)
        recomputed = set(new_df["model"])
        keep_mask = ~old_df["model"].isin(recomputed)
        df = pd.concat([old_df[keep_mask], new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(out_path, index=False)
    _log(f"\nDone. {len(df)} rows. Summary: {out_path}")


if __name__ == "__main__":
    main()
