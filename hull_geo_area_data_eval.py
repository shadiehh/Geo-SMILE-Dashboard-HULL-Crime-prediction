"""Chapter-4 Geo-branch evaluation under the area-data design (XGBoost).

Recomputes every XGBoost Geo number in Chapter 4 with the Geo branch redefined
by geo_area_data.py: the Geo branch explains the 30-feature XGBoost model, and
masking an LSOA replaces ALL of its inputs with the training median. LSOAs are
ranked by the SIGNED Geo-SMILE coefficient (higher = more sensitive).

For every N in the grid (default 1,000 ... 10,000):
  fidelity      weighted-Ridge held-out R2 / weighted R2 / MAE / WMAE / RMSE / WRMSE
  accuracy      top-20 overlap with the fixed expert Geo reference, ATT ACC/F1/AUROC,
                precision, recall, Jaccard
  faithfulness  top-20 vs bottom-20 vs 1,000 random 20-sets (population/area matched,
                as in eval_by_n_perturb.model_based_faithfulness), W1 on affected rows
  stability     one real neighbouring LSOA added in each cardinal direction
  consistency   5 reseeded perturbation pools (seeds 100-104), same fitted model:
                per-LSOA coefficient variance/SD (ddof=1) plus rank Spearman,
                score Pearson and top-20% Jaccard on signed coefficients
At N=4,000 additionally: fixed-kernel west comparison, concentration (Hoyer/Gini/
entropy of the exact per-LSOA shift), exact-vs-SMILE agreement, and the maps.

The Feature branch is unchanged by this redesign and is not recomputed.
Writes outputs/geo_area_data_eval/. Not causal.

    python hull_geo_area_data_eval.py [--n-grid 1000 2000 ...] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics import roc_auc_score

from compute_att_stability import DIRECTIONS
from eval_by_n_perturb import COV_PATH, dehghani_fidelity, load_cached_neighbors
from geo_area_data import GEO_SCORE, area_geo_coefficients, exact_area_shift
from geosmile_hull_twobranch import (
    FORBIDDEN_FEATURE_TOKENS,
    RANDOM_SEED,
    SPATIAL_COLUMNS,
    choose_feature_players,
    temporal_split,
)
from hull_boundary_neighbors import expand_panel
from hull_dehghani_eval import sparsity_entropy_metrics
from hull_expert_topk import GEO_TOP_K, geo_expert_binary_frame
from model_factory import fit_model, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "geo_area_data_eval"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"
MERGED_GEO = OUT / "geosmile_hull_expert_alignment_merged_geo.csv"
Y_COL = "target_next_month_crime"
SEED = RANDOM_SEED
CONSISTENCY_SEEDS = tuple(range(100, 105))
N_GRID = (1000, 2000, 3000, 4000, 5000, 7500, 10000)
N_RANDOM = 1000
FOCUS_N = 4000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_json(obj):
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def ranked(coef: pd.DataFrame) -> pd.DataFrame:
    out = coef.sort_values([GEO_SCORE, "lsoa21cd"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    out["geo_rank"] = np.arange(1, len(out) + 1)
    return out


def top_k(coef: pd.DataFrame, k: int = GEO_TOP_K) -> list[str]:
    return ranked(coef).head(k)["lsoa21cd"].astype(str).tolist()


def accuracy(coef: pd.DataFrame, expert: pd.DataFrame) -> dict:
    frame = expert.merge(coef[["lsoa21cd", GEO_SCORE]], on="lsoa21cd", validate="one_to_one")
    ref = set(frame.loc[frame["expert_top10_reference"] == 1, "lsoa21cd"])
    top = set(top_k(coef))
    tp = len(ref & top)
    n, k = len(frame), GEO_TOP_K
    tn = n - len(ref | top)
    return {
        "n": n,
        "k": k,
        "intersection": tp,
        "accuracy": tp / k,
        "att_acc": (tp + tn) / n,
        "att_f1": 2 * tp / (len(ref) + len(top)),
        "att_auroc": float(roc_auc_score(frame["expert_top10_reference"], frame[GEO_SCORE])),
        "precision": tp / len(top),
        "recall": tp / len(ref),
        "jaccard": tp / len(ref | top),
        "shared": sorted(ref & top),
        "reference_only": sorted(ref - top),
        "model_only": sorted(top - ref),
    }


class GeoResponder:
    """W1 response on affected rows when the given LSOAs' inputs are masked."""

    def __init__(self, model, X_test: pd.DataFrame, players: list[str], baseline: pd.Series):
        self.model = model
        self.players = players
        self.x = X_test[players].astype(float).to_numpy()
        self.orig = np.asarray(model.predict(pd.DataFrame(self.x, columns=players)), float)
        self.ids = X_test["lsoa21cd"].astype(str).to_numpy()
        self.base = baseline[players].to_numpy(dtype=float)

    def __call__(self, lsoas: list[str]) -> float:
        rows = np.flatnonzero(np.isin(self.ids, list(lsoas)))
        if len(rows) == 0:
            return 0.0
        pert = self.x.copy()
        pert[rows, :] = self.base
        pred = np.asarray(self.model.predict(pd.DataFrame(pert, columns=self.players)), float)
        return float(wasserstein_distance(self.orig[rows], pred[rows]))


def faithfulness(responder: GeoResponder, coef: pd.DataFrame, seed: int, n_random: int = N_RANDOM) -> dict:
    """Mirror of eval_by_n_perturb.model_based_faithfulness (Geo part), signed ranking."""
    order = ranked(coef)["lsoa21cd"].astype(str).tolist()
    k = GEO_TOP_K
    top, bottom = order[:k], order[-k:]
    pop_area = None
    if MERGED_GEO.exists() and COV_PATH.exists():
        mg = pd.read_csv(MERGED_GEO)
        cov = pd.read_csv(COV_PATH)
        pop_area = (
            mg[["lsoa21cd", "population_mid_2022"]]
            .merge(cov[["lsoa21cd", "lsoa_area_km2"]], on="lsoa21cd", how="left")
            .assign(lsoa21cd=lambda d: d["lsoa21cd"].astype(str))
            .set_index("lsoa21cd")
        )
    rng = np.random.default_rng(seed)

    def sample() -> list[str]:
        if pop_area is None or any(c not in pop_area.index for c in top):
            return list(rng.choice(order, size=k, replace=False))
        t_pop = float(pop_area.loc[top, "population_mid_2022"].sum())
        t_area = float(pop_area.loc[top, "lsoa_area_km2"].sum())
        best, best_score = None, np.inf
        for _ in range(40):
            cand = list(rng.choice(order, size=k, replace=False))
            if any(c not in pop_area.index for c in cand):
                continue
            score = abs(float(pop_area.loc[cand, "population_mid_2022"].sum()) - t_pop) / (t_pop + 1e-9) + abs(
                float(pop_area.loc[cand, "lsoa_area_km2"].sum()) - t_area
            ) / (t_area + 1e-9)
            if score < best_score:
                best, best_score = cand, score
        return best or list(rng.choice(order, size=k, replace=False))

    top_v, bot_v = responder(top), responder(bottom)
    rand = np.array([responder(sample()) for _ in range(n_random)], dtype=float)
    lo, hi = np.percentile(rand, [2.5, 97.5])
    return {
        "k": k,
        "top_response": top_v,
        "bottom_response": bot_v,
        "random_mean": float(rand.mean()),
        "random_ci95_low": float(lo),
        "random_ci95_high": float(hi),
        "top_minus_random": float(top_v - rand.mean()),
        "top_minus_bottom": float(top_v - bot_v),
        "empirical_p_top_vs_random": float((np.sum(rand >= top_v) + 1) / (len(rand) + 1)),
        "n_random": int(len(rand)),
    }


def stability(model, panel, test, baseline, players, hull_lsoas, base_coef, neighbors, extra_panel, n, seed, kernel=None) -> dict:
    base_hull = base_coef[base_coef["lsoa21cd"].isin(hull_lsoas)]
    base_top = set(top_k(base_hull))
    per_dir, jac, rho = {}, [], []
    for direction in DIRECTIONS:
        code = str(neighbors.loc[neighbors["direction"] == direction, "lsoa21cd"].iloc[0])
        extra_rows = extra_panel[extra_panel["lsoa21cd"].astype(str) == code]
        expanded = expand_panel(panel, extra_rows)
        _, _, exp_test = temporal_split(expanded)
        coef, _ = area_geo_coefficients(
            model, expanded, exp_test, baseline, players, n, seed, sample_lsoas=hull_lsoas, kernel=kernel
        )
        r = ranked(coef)
        hull_part = coef[coef["lsoa21cd"].isin(hull_lsoas)]
        top = set(top_k(r))
        merged = base_hull[["lsoa21cd", GEO_SCORE]].merge(hull_part[["lsoa21cd", GEO_SCORE]], on="lsoa21cd")
        j = len(base_top & top) / len(base_top | top)
        s = float(spearmanr(merged[f"{GEO_SCORE}_x"], merged[f"{GEO_SCORE}_y"]).statistic)
        hit = r[r["lsoa21cd"] == code].iloc[0]
        per_dir[direction] = {
            "lsoa21cd": code,
            "jaccard": j,
            "spearman": s,
            "top20_retained": len(base_top & top),
            "added_rank": int(hit["geo_rank"]),
            "added_coefficient": float(hit["geo_coefficient"]),
            "added_in_top20": bool(hit["geo_rank"] <= GEO_TOP_K),
        }
        jac.append(j)
        rho.append(s)
        log(f"    stability {direction} ({code}): J={j:.3f} rho={s:.4f} added rank {int(hit['geo_rank'])}")
    return {
        "kernel_fixed": kernel,
        "directions": per_dir,
        "jaccard_mean": float(np.mean(jac)),
        "spearman_mean": float(np.mean(rho)),
    }


def consistency(model, panel, test, baseline, players, n, names) -> dict:
    runs = []
    for s in CONSISTENCY_SEEDS:
        coef, _ = area_geo_coefficients(model, panel, test, baseline, players, n, s)
        runs.append(coef.set_index("lsoa21cd")[GEO_SCORE])
    mat = pd.concat(runs, axis=1)
    var = mat.var(axis=1, ddof=1)
    sd = np.sqrt(var)
    rank_r, score_r, top_j = [], [], []
    for a, b in combinations(range(mat.shape[1]), 2):
        x, y = mat.iloc[:, a], mat.iloc[:, b]
        rank_r.append(spearmanr(x, y).statistic)
        score_r.append(pearsonr(x, y)[0])
        ta, tb = set(x[x >= x.quantile(0.8)].index), set(y[y >= y.quantile(0.8)].index)
        top_j.append(len(ta & tb) / len(ta | tb))
    worst = sd.idxmax()
    return {
        "seeds": list(CONSISTENCY_SEEDS),
        "mean_variance": float(var.mean()),
        "median_variance": float(var.median()),
        "mean_sd": float(sd.mean()),
        "median_sd": float(sd.median()),
        "max_sd": float(sd.max()),
        "max_sd_lsoa": str(worst),
        "max_sd_lsoa_name": str(names.get(worst, worst)),
        "max_sd_lsoa_mean_coefficient": float(mat.loc[worst].mean()),
        "max_sd_pct_of_mean": float(100 * sd.max() / abs(mat.loc[worst].mean())),
        "rank_spearman_mean": float(np.mean(rank_r)),
        "score_pearson_mean": float(np.mean(score_r)),
        "top20pct_jaccard_mean": float(np.mean(top_j)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-grid", type=int, nargs="+", default=list(N_GRID))
    p.add_argument("--force", action="store_true")
    p.add_argument("--n-jobs", type=int, default=4)
    args = p.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.n_jobs)

    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    assert not [c for c in players if any(t in c.lower() for t in FORBIDDEN_FEATURE_TOKENS)]
    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
    X_test = panel.loc[test].reset_index(drop=True)
    hull_lsoas = sorted(panel["lsoa21cd"].astype(str).unique())
    names = panel.drop_duplicates("lsoa21cd").set_index("lsoa21cd")["lsoa21nm"].astype(str).to_dict()
    expert = geo_expert_binary_frame(hull_lsoas)[["lsoa21cd", "expert_top10_reference"]]
    neighbors, extra_panel = load_cached_neighbors()

    log("Training the 30-feature XGBoost (seed 42) -- explained by both branches under this design")
    model = make_model("xgboost", SEED, n_jobs=args.n_jobs)
    fit_model("xgboost", model, panel.loc[train, players], panel.loc[train, Y_COL],
              panel.loc[validation, players], panel.loc[validation, Y_COL])
    responder = GeoResponder(model, X_test, players, baseline)
    DEST.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in args.n_grid:
        n_dir = DEST / f"n_{n}"
        path = n_dir / "metrics.json"
        if path.exists() and not args.force:
            log(f"N={n}: reusing {path}")
            rows.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        t0 = time.time()
        log(f"=== N={n} ===")
        coef, diag, M, y, d = area_geo_coefficients(model, panel, test, baseline, players, n, SEED, return_pool=True)
        fid = dehghani_fidelity(M, y, d)
        acc = accuracy(coef, expert)
        log(f"  fidelity R2={fid['R2']:.4f}  accuracy {acc['intersection']}/20  AUROC={acc['att_auroc']:.4f}")
        faith = faithfulness(responder, coef, SEED)
        log(f"  faithfulness top-random={faith['top_minus_random']:.3f} p={faith['empirical_p_top_vs_random']:.3f}")
        stab = stability(model, panel, test, baseline, players, hull_lsoas, coef, neighbors, extra_panel, n, SEED)
        cons = consistency(model, panel, test, baseline, players, n, names)
        log(f"  consistency mean var={cons['mean_variance']:.5f} mean SD={cons['mean_sd']:.5f} rank rho={cons['rank_spearman_mean']:.3f}")
        payload = {
            "n": n,
            "smile_diagnostics": diag,
            "fidelity": fid,
            "accuracy": acc,
            "faithfulness": faith,
            "stability": stab,
            "consistency": cons,
        }
        if n == FOCUS_N:
            base_fixed, _ = area_geo_coefficients(model, panel, test, baseline, players, n, SEED, kernel="inverse")
            west = stability(model, panel, test, baseline, players, hull_lsoas, base_fixed, neighbors,
                             extra_panel, n, SEED, kernel="inverse")["directions"]["west"]
            exact = exact_area_shift(model, panel, test, baseline, players)
            both = coef.merge(exact, on="lsoa21cd")
            payload["fixed_kernel_west"] = west
            payload["concentration_exact_shift"] = sparsity_entropy_metrics(both["exact_w1"])
            payload["concentration_abs_coefficient"] = sparsity_entropy_metrics(both["geo_coefficient"])
            payload["exact_vs_smile_spearman"] = float(spearmanr(both[GEO_SCORE], both["exact_w1"]).statistic)
            r = ranked(coef).merge(exact, on="lsoa21cd")
            r["lsoa21nm"] = r["lsoa21cd"].map(names)
            r["expert_reference"] = r["lsoa21cd"].isin(expert.loc[expert["expert_top10_reference"] == 1, "lsoa21cd"])
            r.to_csv(DEST / "geo_coefficients_N4000.csv", index=False)
        payload["elapsed_min"] = (time.time() - t0) / 60
        n_dir.mkdir(parents=True, exist_ok=True)
        coef.to_csv(n_dir / "geo_coefficients.csv", index=False)
        path.write_text(json.dumps(to_json(payload), indent=2), encoding="utf-8")
        log(f"N={n} done in {payload['elapsed_min']:.1f} min")
        rows.append(to_json(payload))

    summary = pd.DataFrame(
        [
            {
                "N": r["n"],
                "fidelity_r2": r["fidelity"]["R2"],
                "fidelity_weighted_r2": r["fidelity"]["Weighted R2"],
                "accuracy_overlap": r["accuracy"]["intersection"],
                "att_auroc": r["accuracy"]["att_auroc"],
                "faithfulness_top_minus_random": r["faithfulness"]["top_minus_random"],
                "faithfulness_p": r["faithfulness"]["empirical_p_top_vs_random"],
                "stability_jaccard": r["stability"]["jaccard_mean"],
                "stability_spearman": r["stability"]["spearman_mean"],
                "consistency_mean_variance": r["consistency"]["mean_variance"],
                "consistency_mean_sd": r["consistency"]["mean_sd"],
                "consistency_rank_spearman": r["consistency"]["rank_spearman_mean"],
            }
            for r in sorted(rows, key=lambda r: r["n"])
        ]
    )
    summary.to_csv(DEST / "summary_by_n.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
