"""How many donor draws does random-donor coordinate masking need? (check; replaces nothing)

Aligned Hull design (XGBoost on 30 predictors + BNG easting/northing, seed 42). Geo branch: N=4,000
coalitions of 5-10 LSOAs (seed 42), test-period rows; a masked LSOA's rows get the centroid of a
random other Hull LSOA; response = mean over K donor draws of W1 between log(1 + prediction) of the
affected rows before and after relocation. The log scale removes the Poisson (log-link) scaling of
absolute shifts with an area's crime level, so the coefficient measures proportional dependence on
location. Kernel-weighted Ridge with the usual 60/20/20 split and kernel menu.

For K in the grid: fidelity (dehghani_fidelity), consistency over 5 reseeded pools (seeds 100-104;
rank Spearman, mean SD), Spearman with the exact single-LSOA expectation over all 167 donors, and
convergence to the largest K (Spearman, top-20 Jaccard). Diagnostics at the chosen K: crime-level and
distance links, Moran's I, Old Town rank, agreement with the other designs.

Writes outputs/hull_geo_design_review/07_random_donor_ksweep/. Not causal.
"""
from __future__ import annotations

import json
import os
import time
from itertools import combinations
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import compute_geo_spatial_patterns as sp
from eval_by_n_perturb import dehghani_fidelity
from geosmile_hull_twobranch import SPATIAL_COLUMNS, apply_kernel, choose_feature_players, temporal_split
from model_factory import fit_model, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "hull_geo_design_review" / "07_random_donor_ksweep"
N, SEED, Y = 4000, 42, "target_next_month_crime"
K_GRID = (1, 2, 5, 10, 20, 30, 50, 100)
CONS_SEEDS = tuple(range(100, 105))


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


panel = pd.read_csv(OUT / "hull_lsoa_month_panel_with_covariates.csv")
panel["month"] = pd.to_datetime(panel["month"])
players = choose_feature_players(panel)
cols = players + SPATIAL_COLUMNS
train, val, test = temporal_split(panel)
model = make_model("xgboost", SEED, n_jobs=4)
fit_model("xgboost", model, panel.loc[train, cols], panel.loc[train, Y], panel.loc[val, cols], panel.loc[val, Y])
frame = panel.loc[test].reset_index(drop=True)
x = frame[cols].to_numpy(float)
ids = frame["lsoa21cd"].astype(str).to_numpy()
lsoas = np.array(sorted(set(ids)), dtype=object)
L = len(lsoas)
idx_by = [np.flatnonzero(ids == c) for c in lsoas]
cent = panel.drop_duplicates("lsoa21cd").set_index("lsoa21cd").loc[lsoas, SPATIAL_COLUMNS].to_numpy(float)
cix = [cols.index(c) for c in SPATIAL_COLUMNS]


def lp(rows: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(model.predict(pd.DataFrame(rows, columns=cols)), float), 0, None))


orig = lp(x)


def response(sel, rng, k_draws: int) -> float:
    aff = np.concatenate([idx_by[s] for s in sel])
    base = x[aff]
    blocks = []
    for _ in range(k_draws):
        p = base.copy()
        pos = 0
        for s in sel:
            donor = rng.integers(L - 1)
            donor += donor >= s
            p[pos:pos + len(idx_by[s]), cix] = cent[donor]
            pos += len(idx_by[s])
        blocks.append(p)
    pred = lp(np.vstack(blocks)).reshape(k_draws, len(aff))
    return float(np.mean([wasserstein_distance(orig[aff], pred[k]) for k in range(k_draws)]))


def pool(seed: int, k_draws: int):
    rng = np.random.default_rng(seed)
    drng = np.random.default_rng(seed + 7_000_003)
    M = np.zeros((N, L), np.int8)
    y = np.zeros(N)
    for j in range(N):
        sel = rng.choice(L, size=int(rng.integers(5, 11)), replace=False)
        M[j, sel] = 1
        y[j] = response(sel, drng, k_draws)
    d = M.sum(1) / L
    order = rng.permutation(N)
    tr, va = order[:int(.6 * N)], order[int(.6 * N):int(.8 * N)]
    sigma = max(float(np.median(d[tr])), 1 / L)
    sc = {k: r2_score(y[va], Ridge(alpha=1.0).fit(M[tr], y[tr], sample_weight=apply_kernel(d[tr], k, sigma)).predict(M[va]))
          for k in ("rbf", "laplace", "inverse")}
    best = max(sc, key=sc.get)
    trva = order[:int(.8 * N)]
    r = Ridge(alpha=1.0).fit(M[trva], y[trva], sample_weight=apply_kernel(d[trva], best, sigma))
    return pd.Series(r.coef_, index=lsoas), M, y, d


def exact_expectation() -> pd.Series:
    out = {}
    for i in range(L):
        rows = x[idx_by[i]]
        stack = []
        for dnr in range(L):
            if dnr != i:
                p = rows.copy()
                p[:, cix] = cent[dnr]
                stack.append(p)
        pred = lp(np.vstack(stack)).reshape(L - 1, len(rows))
        out[lsoas[i]] = float(np.mean([wasserstein_distance(orig[idx_by[i]], pred[k]) for k in range(L - 1)]))
    return pd.Series(out)


def top(s: pd.Series, k: int = 20) -> set:
    return set(s.sort_values(ascending=False).head(k).index)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    exact = exact_expectation()
    exact.rename("exact_log_w1").to_csv(DEST / "exact_single_lsoa_expectation.csv")
    log("exact single-LSOA expectation done")
    coefs, rows = {}, []
    for k in K_GRID:
        t0 = time.time()
        coef, M, y, d = pool(SEED, k)
        fid = dehghani_fidelity(M, y, d)
        runs = [pool(s, k)[0] for s in CONS_SEEDS]
        mat = pd.concat(runs, axis=1)
        rr = [spearmanr(mat.iloc[:, a], mat.iloc[:, b])[0] for a, b in combinations(range(len(runs)), 2)]
        coefs[k] = coef
        rows.append({"K": k, "fidelity_R2": fid["R2"], "fidelity_wR2": fid["Weighted R2"],
                     "consistency_rank_rho": float(np.mean(rr)), "consistency_mean_sd": float(mat.std(axis=1, ddof=1).mean()),
                     "rho_with_exact": float(spearmanr(coef, exact.loc[coef.index])[0]),
                     "top20_jaccard_with_exact": len(top(coef) & top(exact)) / len(top(coef) | top(exact)),
                     "mean_response": float(y.mean()), "minutes": (time.time() - t0) / 60})
        log(f"K={k}: R2={fid['R2']:.4f} cons rho={rows[-1]['consistency_rank_rho']:.3f} "
            f"rho exact={rows[-1]['rho_with_exact']:.3f} ({rows[-1]['minutes']:.1f} min)")
        pd.DataFrame(coefs).to_csv(DEST / "coefficients_by_K.csv")
    table = pd.DataFrame(rows)
    kmax = coefs[max(K_GRID)]
    table["rho_with_Kmax"] = [float(spearmanr(coefs[k], kmax)[0]) for k in table["K"]]
    table["top20_jaccard_with_Kmax"] = [len(top(coefs[k]) & top(kmax)) / len(top(coefs[k]) | top(kmax)) for k in table["K"]]
    # chosen K: smallest K within 0.005 of the best fidelity and consistency, and rho with K_max >= 0.99
    ok = table[(table["fidelity_R2"] >= table["fidelity_R2"].max() - 0.005)
               & (table["consistency_rank_rho"] >= table["consistency_rank_rho"].max() - 0.005)
               & (table["rho_with_Kmax"] >= 0.99)]
    k_star = int(ok["K"].min()) if len(ok) else int(max(K_GRID))
    table["chosen"] = table["K"] == k_star
    table.to_csv(DEST / "k_sweep_summary.csv", index=False)

    chosen = coefs[k_star]
    lvl = panel.loc[test].groupby("lsoa21cd")["crime_count"].mean()
    med = panel.loc[train, SPATIAL_COLUMNS].median().to_numpy(float)
    moved = pd.Series(np.hypot(*(cent - med).T) / 1000, index=lsoas)
    names = panel.drop_duplicates("lsoa21cd").set_index("lsoa21cd")["lsoa21nm"].astype(str)
    ranked = pd.DataFrame({"lsoa21cd": chosen.index, "geo_coefficient": chosen.values,
                           "exact_log_w1": exact.loc[chosen.index].values}).sort_values("geo_coefficient", ascending=False)
    ranked["geo_rank"] = np.arange(1, L + 1)
    ranked["lsoa21nm"] = ranked["lsoa21cd"].map(names)
    ranked.to_csv(DEST / f"geo_coefficients_K{k_star}.csv", index=False)
    sp.COEF_PATH, sp.VALUE_FIELD = DEST / f"geo_coefficients_K{k_star}.csv", "geo_coefficient"
    gdf = sp.attach_boundaries(sp.load_geo_frame())
    wq, islands = sp.queen_weights(gdf)
    _, _, gm, _, _ = sp.run_moran_lisa(gdf[~gdf["lsoa21cd"].isin(islands)].copy(), wq)
    other = {
        "area_data": pd.read_csv(OUT / "geo_area_data_eval" / "geo_coefficients_N4000.csv").set_index("lsoa21cd")["geo_coefficient"],
        "median_point": pd.read_csv(OUT / "geo_aligned_design" / "xgboost" / "geo_coefficients_N4000.csv").set_index("lsoa21cd")["geo_coefficient"],
        "random_donor_absolute_K10": pd.read_csv(OUT / "hull_geo_design_review" / "06_random_donor" / "geo_coefficients_N4000.csv").set_index("lsoa21cd")["geo_coefficient"],
    }
    diag = {
        "chosen_K": k_star,
        "selection_rule": "smallest K with fidelity and consistency within 0.005 of the best K and Spearman >= 0.99 with K=100",
        "rho_with_crime_level": float(spearmanr(chosen, lvl.loc[chosen.index])[0]),
        "top20_shared_with_highest_crime": len(top(chosen) & top(lvl)),
        "rho_with_distance_from_median_point": float(spearmanr(chosen, moved.loc[chosen.index])[0]),
        "moran_I": float(gm.I), "moran_p": float(gm.p_sim),
        "old_town_rank": int(ranked.set_index("lsoa21cd").loc["E01033104", "geo_rank"]),
        "agreement": {nm: {"spearman": float(spearmanr(chosen.loc[s.index], s)[0]), "top20_overlap": len(top(chosen) & top(s))}
                      for nm, s in other.items()},
        "top10": ranked.head(10)[["geo_rank", "lsoa21cd", "lsoa21nm", "geo_coefficient"]].to_dict("records"),
    }
    (DEST / "chosen_K_diagnostics.json").write_text(json.dumps(diag, indent=2, default=float), encoding="utf-8")
    print(table.round(4).to_string(index=False))
    print(json.dumps(diag, indent=1, default=float))


if __name__ == "__main__":
    main()
