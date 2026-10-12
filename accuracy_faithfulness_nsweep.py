"""ATT accuracy and model-based faithfulness across the established
perturbation-budget grid (Feature N in {512,1024,2048,4096}, Geo N in
{2000,4000,6000,8000}), for all 10 candidate models, both branches.

ATT accuracy reuses verify_metrics_all_models.att_metrics() UNMODIFIED --
the correct, already-validated (TP+TN)/n_total formula this project reports
everywhere as "ATT accuracy" / "ATT ACC" (Table 3/5 of the XGBoost chapter),
NOT hull_expert_topk.compute_att_suite's accuracy field (confirmed
under-counting matches at every N tested) and NOT a plain intersection/
overlap-ratio metric -- this is genuine classification accuracy over all n
Feature players / Geo LSOAs, scored against the CURRENT canonical expert
reference already in this project (hull_expert_topk.load_feature_expert_
reference/load_geo_expert_reference -- the same "ranking-v2" Feature list
and "overlay" Geo list used throughout, nothing new supplied here).

Faithfulness is eval_by_n_perturb.model_based_faithfulness's logic, split
into independent per-branch functions (feature_faithfulness/geo_faithfulness
below) so sweeping one branch's N grid doesn't force a redundant, unrelated
recomputation of the other branch at some arbitrary paired N -- the two
branches were never coupled in the underlying maths (feature_response only
touches model_feat/players; geo_response only touches model_geo/
SPATIAL_COLUMNS), only bundled for convenience in the original function.
Same population/area-stratified Geo random-set sampling preserved.

Both metrics are derived from the SAME SMILE pool used to fit each N's
ranking (smile_global_feature/geo_smile_coefficients), reusing masks/
all_pert/original already generated -- no separate fidelity-only pass.

Writes outputs/accuracy_faithfulness_nsweep/summary.csv (appended
incrementally per (model, branch, n) so a partial/killed run keeps
whatever finished).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from eval_by_n_perturb import Y_COL
from geosmile_hull_twobranch import FORBIDDEN_FEATURE_TOKENS, SPATIAL_COLUMNS, choose_feature_players, geo_smile_coefficients, smile_global_feature, temporal_split
from hull_expert_topk import FEATURE_TOP_K, GEO_TOP_K, load_feature_expert_reference, load_geo_expert_reference
from model_factory import MODEL_NAMES, fit_model, make_model
from response_metric_comparison import FEAT_N_GRID, GEO_N_GRID
from verify_metrics_all_models import att_metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "accuracy_faithfulness_nsweep"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"
PREDICTIVE_CSV = OUT / "eval_by_model" / "predictive_performance.csv"
COV_PATH = OUT / "hull_lsoa_boundary_features.csv"
MERGED_GEO_PATH = OUT / "geosmile_hull_expert_alignment_merged_geo.csv"

SMILE_SEED = 42
N_RANDOM_FAITH = 1000
R2_GATE = 0.7


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_pop_area() -> pd.DataFrame | None:
    if MERGED_GEO_PATH.exists() and COV_PATH.exists():
        mg = pd.read_csv(MERGED_GEO_PATH)
        cov = pd.read_csv(COV_PATH)
        if "population_mid_2022" in mg.columns and "lsoa_area_km2" in cov.columns:
            return (
                mg[["lsoa21cd", "population_mid_2022"]]
                .merge(cov[["lsoa21cd", "lsoa_area_km2"]], on="lsoa21cd", how="left")
                .assign(lsoa21cd=lambda d: d["lsoa21cd"].astype(str))
                .set_index("lsoa21cd")
            )
    return None


def feature_faithfulness(model, players, baseline, X_test, feat_imp, k, n_random, seed) -> dict:
    ranked = feat_imp.sort_values(["abs_coefficient", "feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    top, bottom, all_feat = ranked.head(k)["feature"].tolist(), ranked.tail(k)["feature"].tolist(), ranked["feature"].tolist()
    orig = np.asarray(model.predict(X_test[players]), float)

    def response(mask_features: list[str]) -> float:
        pert = X_test[players].copy()
        for name in mask_features:
            pert[name] = baseline[name]
        return float(wasserstein_distance(orig, model.predict(pert)))

    rng = np.random.default_rng(seed)
    top_v, bot_v = response(top), response(bottom)
    rand_arr = np.array([response(list(rng.choice(all_feat, size=k, replace=False))) for _ in range(n_random)])
    lo, hi = np.percentile(rand_arr, [2.5, 97.5]).tolist()
    p_top = float((np.sum(rand_arr >= top_v) + 1) / (len(rand_arr) + 1))
    return {
        "top_response": top_v, "bottom_response": bot_v, "random_mean": float(rand_arr.mean()),
        "random_ci95_low": lo, "random_ci95_high": hi, "top_minus_random": top_v - float(rand_arr.mean()),
        "top_minus_bottom": top_v - bot_v, "empirical_p_top_vs_random": p_top, "n_random": n_random,
    }


def geo_faithfulness(model, panel, test_mask, baseline, geo_imp, k, n_random, seed, pop_area) -> dict:
    ranked = geo_imp.sort_values(["geo_abs_coefficient", "lsoa21cd"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    top = ranked.head(k)["lsoa21cd"].astype(str).tolist()
    bottom = ranked.tail(k)["lsoa21cd"].astype(str).tolist()
    all_geo = ranked["lsoa21cd"].astype(str).tolist()
    frame = panel.loc[test_mask, ["lsoa21cd"] + SPATIAL_COLUMNS].reset_index(drop=True)
    x = frame[SPATIAL_COLUMNS].copy()
    orig = np.asarray(model.predict(x), float)
    lsoa_ids = frame["lsoa21cd"].astype(str).to_numpy()
    baseline_xy = baseline[SPATIAL_COLUMNS].to_numpy(dtype=float)

    def response(mask_lsoas: list[str]) -> float:
        pert = x.copy()
        affected: list[int] = []
        for code in mask_lsoas:
            rows = np.flatnonzero(lsoa_ids == str(code))
            pert.iloc[rows, :] = baseline_xy
            affected.extend(rows.tolist())
        affected_idx = np.asarray(affected, dtype=int)
        if len(affected_idx) == 0:
            return 0.0
        return float(wasserstein_distance(orig[affected_idx], model.predict(pert)[affected_idx]))

    rng = np.random.default_rng(seed)

    def sample_random(k: int) -> list[str]:
        if pop_area is None:
            return list(rng.choice(all_geo, size=k, replace=False))
        present = [c for c in top if c in pop_area.index]
        if len(present) < k:
            return list(rng.choice(all_geo, size=k, replace=False))
        target_pop = float(pop_area.loc[present, "population_mid_2022"].sum())
        target_area = float(pop_area.loc[present, "lsoa_area_km2"].sum())
        best, best_score = None, np.inf
        for _ in range(40):
            cand = list(rng.choice(all_geo, size=k, replace=False))
            if any(c not in pop_area.index for c in cand):
                continue
            pop = float(pop_area.loc[cand, "population_mid_2022"].sum())
            area = float(pop_area.loc[cand, "lsoa_area_km2"].sum())
            score = abs(pop - target_pop) / (target_pop + 1e-9) + abs(area - target_area) / (target_area + 1e-9)
            if score < best_score:
                best_score, best = score, cand
        return best or list(rng.choice(all_geo, size=k, replace=False))

    top_v, bot_v = response(top), response(bottom)
    rand_arr = np.array([response(sample_random(k)) for _ in range(n_random)])
    lo, hi = np.percentile(rand_arr, [2.5, 97.5]).tolist()
    p_top = float((np.sum(rand_arr >= top_v) + 1) / (len(rand_arr) + 1))
    return {
        "top_response": top_v, "bottom_response": bot_v, "random_mean": float(rand_arr.mean()),
        "random_ci95_low": lo, "random_ci95_high": hi, "top_minus_random": top_v - float(rand_arr.mean()),
        "top_minus_bottom": top_v - bot_v, "empirical_p_top_vs_random": p_top, "n_random": n_random,
    }


def append_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if path.exists():
        old_df = pd.read_csv(path)
        keys = set(zip(new_df["model"], new_df["branch"], new_df["n"]))
        keep_mask = ~old_df.apply(lambda r: (r["model"], r["branch"], r["n"]) in keys, axis=1)
        df = pd.concat([old_df[keep_mask], new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--branches", nargs="+", choices=["Feature", "Geo"], default=["Feature", "Geo"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    out_path = DEST / "summary.csv"

    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    bad = [p for p in players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise SystemExit(f"Spatial leak: {bad}")
    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
    X_test = panel.loc[test].reset_index(drop=True)
    pop_area = load_pop_area()

    feat_ref = load_feature_expert_reference()
    feat_expert = set(feat_ref.loc[feat_ref["expert_rank"] <= FEATURE_TOP_K, "feature"].astype(str))
    geo_expert = set(load_geo_expert_reference()["lsoa21cd"].astype(str))

    predictive = pd.read_csv(PREDICTIVE_CSV)

    def r2_of(model_name: str, branch: str) -> float:
        row = predictive[(predictive["model"] == model_name) & (predictive["branch"] == branch)]
        return float(row.iloc[0]["r2"]) if not row.empty else float("nan")

    for model_name in args.models:
        if "Feature" in args.branches:
            r2 = r2_of(model_name, "Feature")
            if np.isfinite(r2) and r2 > R2_GATE:
                _log(f"\n=== [{model_name}/Feature] === R2={r2:.3f}")
                model = make_model(model_name, SMILE_SEED, n_jobs=2)
                fit_model(model_name, model, panel.loc[train, players], panel.loc[train, Y_COL],
                          panel.loc[validation, players], panel.loc[validation, Y_COL])
                for n in FEAT_N_GRID:
                    _log(f"[{model_name}/Feature] n={n}: SMILE ranking...")
                    imp, _diag, _masks, _all_pert, _orig = smile_global_feature(model, X_test[players], players, baseline, n, SMILE_SEED)
                    acc = att_metrics(imp, feat_expert, id_col="feature", mag_col="coefficient", k=FEATURE_TOP_K, n_total=len(players))
                    _log(f"[{model_name}/Feature] n={n}: faithfulness ({N_RANDOM_FAITH} random sets)...")
                    faith = feature_faithfulness(model, players, baseline, X_test, imp, FEATURE_TOP_K, N_RANDOM_FAITH, SMILE_SEED)
                    row = {"model": model_name, "branch": "Feature", "n": n}
                    row.update({f"acc_{k}": v for k, v in acc.items() if k != "model_ids"})
                    row.update({f"faith_{k}": v for k, v in faith.items()})
                    append_csv([row], out_path)
                    _log(f"[{model_name}/Feature] n={n}: ATT_accuracy={acc['ATT_accuracy']:.4f} "
                         f"top_minus_random={faith['top_minus_random']:.4f} p={faith['empirical_p_top_vs_random']:.4f}")
            else:
                _log(f"\n=== [{model_name}/Feature] === SKIPPED (R2={r2:.3f} <= {R2_GATE})")

        if "Geo" in args.branches:
            r2 = r2_of(model_name, "Geo")
            if np.isfinite(r2) and r2 > R2_GATE:
                _log(f"\n=== [{model_name}/Geo] === R2={r2:.3f}")
                model = make_model(model_name, SMILE_SEED, n_jobs=2)
                fit_model(model_name, model, panel.loc[train, SPATIAL_COLUMNS], panel.loc[train, Y_COL],
                          panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, Y_COL])
                for n in GEO_N_GRID:
                    _log(f"[{model_name}/Geo] n={n}: SMILE ranking...")
                    imp, _diag = geo_smile_coefficients(model, panel, test, baseline, n, SMILE_SEED)
                    imp["lsoa21cd"] = imp["lsoa21cd"].astype(str)
                    acc = att_metrics(imp, geo_expert, id_col="lsoa21cd", mag_col="geo_coefficient", k=GEO_TOP_K, n_total=len(imp))
                    _log(f"[{model_name}/Geo] n={n}: faithfulness ({N_RANDOM_FAITH} random sets)...")
                    faith = geo_faithfulness(model, panel, test, baseline, imp, GEO_TOP_K, N_RANDOM_FAITH, SMILE_SEED, pop_area)
                    row = {"model": model_name, "branch": "Geo", "n": n}
                    row.update({f"acc_{k}": v for k, v in acc.items() if k != "model_ids"})
                    row.update({f"faith_{k}": v for k, v in faith.items()})
                    append_csv([row], out_path)
                    _log(f"[{model_name}/Geo] n={n}: ATT_accuracy={acc['ATT_accuracy']:.4f} "
                         f"top_minus_random={faith['top_minus_random']:.4f} p={faith['empirical_p_top_vs_random']:.4f}")
            else:
                _log(f"\n=== [{model_name}/Geo] === SKIPPED (R2={r2:.3f} <= {R2_GATE})")

    _log(f"\nDone. {out_path}")


if __name__ == "__main__":
    main()
