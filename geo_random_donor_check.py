"""Random-donor coordinate masking for the aligned Hull design (check; replaces nothing).

Model: XGBoost on the 30 predictors + BNG easting/northing (as hull_aligned_design_eval.py).
Geo branch: as the aligned design (N=4,000 coalitions of 5-10 LSOAs, seed 42, test-period rows,
W1 on affected rows, kernel-weighted Ridge, 60/20/20 split), except that a masked LSOA's rows get
the centroid coordinates of a randomly drawn OTHER Hull LSOA instead of the single median point.
The response is the mean W1 over K=10 donor draws (each masked LSOA its own donor per draw), so
no single reference point is privileged ("how much do this area's predictions depend on knowing
where it is?"). Single-LSOA check: exact mean W1 over all 167 donors.

Also: 5 reseeded pools (consistency), intervention (top/bottom/1,000 random 20-sets), accuracy
against the expert regions, Moran's I, and distance-bias diagnostics. Writes
outputs/hull_geo_design_review/06_random_donor/. Not causal.
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
from scipy.stats import pearsonr, spearmanr, wasserstein_distance

import compute_geo_spatial_patterns as sp
from eval_by_n_perturb import dehghani_fidelity
from geosmile_hull_twobranch import SPATIAL_COLUMNS, apply_kernel, choose_feature_players, temporal_split
from hull_aligned_design_eval import REGIONS_CSV, geo_accuracy
from model_factory import fit_model, make_model
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "hull_geo_design_review" / "06_random_donor"
N, K, SEED, Y = 4000, 10, 42, "target_next_month_crime"


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
original = np.asarray(model.predict(pd.DataFrame(x, columns=cols)), float)
log(f"holdout R2 {r2_score(frame[Y], original):.4f}")
ids = frame["lsoa21cd"].astype(str).to_numpy()
lsoas = np.array(sorted(set(ids)), dtype=object)
L = len(lsoas)
idx_by = [np.flatnonzero(ids == c) for c in lsoas]
cent = panel.drop_duplicates("lsoa21cd").set_index("lsoa21cd").loc[lsoas, SPATIAL_COLUMNS].to_numpy(float)
cix = [cols.index(c) for c in SPATIAL_COLUMNS]


def predict(rows: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(pd.DataFrame(rows, columns=cols)), float)


def response(sel, rng) -> float:
    aff = np.concatenate([idx_by[s] for s in sel])
    base = x[aff]
    blocks = []
    for _ in range(K):
        p = base.copy()
        pos = 0
        for s in sel:
            donor = rng.integers(L - 1)
            donor += donor >= s                      # any LSOA except itself
            p[pos:pos + len(idx_by[s]), cix] = cent[donor]
            pos += len(idx_by[s])
        blocks.append(p)
    pred = predict(np.vstack(blocks)).reshape(K, len(aff))
    return float(np.mean([wasserstein_distance(original[aff], pred[k]) for k in range(K)]))


def pool(seed: int):
    rng = np.random.default_rng(seed)
    donor_rng = np.random.default_rng(seed + 7_000_003)
    M = np.zeros((N, L), np.int8)
    y = np.zeros(N)
    for j in range(N):
        sel = rng.choice(L, size=int(rng.integers(5, 11)), replace=False)
        M[j, sel] = 1
        y[j] = response(sel, donor_rng)
    d = M.sum(1) / L
    order = rng.permutation(N)
    tr, va, te = order[:int(.6 * N)], order[int(.6 * N):int(.8 * N)], order[int(.8 * N):]
    sigma = max(float(np.median(d[tr])), 1 / L)
    sc = {k: r2_score(y[va], Ridge(alpha=1.0).fit(M[tr], y[tr], sample_weight=apply_kernel(d[tr], k, sigma)).predict(M[va]))
          for k in ("rbf", "laplace", "inverse")}
    best = max(sc, key=sc.get)
    trva = np.concatenate([tr, va])
    r = Ridge(alpha=1.0).fit(M[trva], y[trva], sample_weight=apply_kernel(d[trva], best, sigma))
    coef = pd.DataFrame({"lsoa21cd": lsoas.astype(str), "geo_coefficient": r.coef_})
    return coef, M, y, d, best


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    coef, M, y, d, kernel = pool(SEED)
    fid = dehghani_fidelity(M, y, d)
    log(f"pool done ({(time.time() - t0) / 60:.1f} min): R2={fid['R2']:.4f} kernel={kernel} mean W1={y.mean():.3f}")

    # exact single-LSOA expectation over all 167 donors
    exact = []
    for i in range(L):
        rows = x[idx_by[i]]
        stack = []
        for dnr in range(L):
            if dnr == i:
                continue
            p = rows.copy()
            p[:, cix] = cent[dnr]
            stack.append(p)
        pred = predict(np.vstack(stack)).reshape(L - 1, len(rows))
        exact.append(float(np.mean([wasserstein_distance(original[idx_by[i]], pred[k]) for k in range(L - 1)])))
    coef["exact_w1"] = exact
    coef = coef.sort_values("geo_coefficient", ascending=False).reset_index(drop=True)
    coef["geo_rank"] = np.arange(1, L + 1)
    names = panel.drop_duplicates("lsoa21cd").set_index("lsoa21cd")["lsoa21nm"].astype(str)
    coef["lsoa21nm"] = coef["lsoa21cd"].map(names)
    coef.to_csv(DEST / "geo_coefficients_N4000.csv", index=False)

    # consistency (5 reseeded pools)
    runs = [coef.set_index("lsoa21cd")["geo_coefficient"]]
    for s in range(100, 105):
        c, *_ = pool(s)
        runs.append(c.set_index("lsoa21cd")["geo_coefficient"])
        log(f"consistency seed {s} done")
    mat = pd.concat(runs[1:], axis=1)
    var, sd = mat.var(axis=1, ddof=1), mat.std(axis=1, ddof=1)
    rr = [spearmanr(mat.iloc[:, a], mat.iloc[:, b])[0] for a, b in combinations(range(5), 2)]

    # intervention: top / bottom / 1,000 random 20-sets (exact K-draw response)
    rng = np.random.default_rng(SEED)
    pos = {c: i for i, c in enumerate(lsoas)}
    order = [pos[c] for c in coef["lsoa21cd"]]
    top_v, bot_v = response(order[:20], rng), response(order[-20:], rng)
    rand = np.array([response(rng.choice(L, 20, replace=False), rng) for _ in range(1000)])

    # comparisons and distance-bias diagnostics
    published = set(pd.read_csv(REGIONS_CSV)["lsoa21cd"].astype(str))
    acc = geo_accuracy(coef, published)
    c = coef.set_index("lsoa21cd")
    median_pt = pd.read_csv(OUT / "geo_aligned_design" / "xgboost" / "geo_coefficients_N4000.csv").set_index("lsoa21cd")["geo_coefficient"]
    area = pd.read_csv(OUT / "geo_area_data_eval" / "n_4000" / "geo_coefficients.csv").set_index("lsoa21cd")["geo_coefficient"]
    med = panel.loc[train, SPATIAL_COLUMNS].median().to_numpy(float)
    cdf = pd.DataFrame(cent, index=lsoas, columns=SPATIAL_COLUMNS)
    moved_median = np.hypot(*(cdf.to_numpy() - med).T) / 1000
    mean_donor_km = np.array([np.mean(np.delete(np.hypot(*(cent - cent[i]).T), i)) for i in range(L)]) / 1000
    sp.COEF_PATH, sp.VALUE_FIELD = DEST / "geo_coefficients_N4000.csv", "geo_coefficient"
    gdf = sp.attach_boundaries(sp.load_geo_frame())
    wq, islands = sp.queen_weights(gdf)
    order_m, _, gm, _, labels = sp.run_moran_lisa(gdf[~gdf["lsoa21cd"].isin(islands)].copy(), wq)
    lab = pd.Series(labels, index=order_m)
    top20 = coef.head(20)["lsoa21cd"]
    res = {
        "design": "random-donor coordinate masking (K=10 draws per perturbation; exact check over 167 donors)",
        "fidelity": fid, "best_kernel": kernel, "mean_W1_response": float(y.mean()),
        "signed_vs_exact_spearman": float(spearmanr(c["geo_coefficient"], c["exact_w1"])[0]),
        "abs_vs_exact_spearman": float(spearmanr(c["geo_coefficient"].abs(), c["exact_w1"])[0]),
        "exact_w1_median": float(c["exact_w1"].median()), "exact_w1_p90": float(c["exact_w1"].quantile(0.9)),
        "accuracy": acc,
        "intervention": {"top": top_v, "bottom": bot_v, "random_mean": float(rand.mean()),
                         "random_ci95": [float(v) for v in np.percentile(rand, [2.5, 97.5])],
                         "top_minus_random": float(top_v - rand.mean()),
                         "p": float((np.sum(rand >= top_v) + 1) / (len(rand) + 1))},
        "consistency": {"mean_variance": float(var.mean()), "median_variance": float(var.median()),
                        "mean_sd": float(sd.mean()), "max_sd": float(sd.max()), "max_sd_lsoa": str(sd.idxmax()),
                        "rank_spearman_mean": float(np.mean(rr)),
                        "spearman_seed42_vs_seed_mean": float(spearmanr(runs[0], mat.mean(axis=1).loc[runs[0].index])[0])},
        "moran": {"I": float(gm.I), "p_sim": float(gm.p_sim), "lisa_counts": lab.value_counts().to_dict()},
        "old_town_rank": int(c.loc["E01033104", "geo_rank"]),
        "top10": coef.head(10)[["geo_rank", "lsoa21cd", "lsoa21nm", "geo_coefficient", "exact_w1"]].to_dict("records"),
        "spearman_with_median_point_aligned": float(spearmanr(c["geo_coefficient"].loc[median_pt.index], median_pt)[0]),
        "top20_overlap_with_median_point": len(set(top20) & set(median_pt.sort_values(ascending=False).head(20).index)),
        "spearman_with_area_data": float(spearmanr(c["geo_coefficient"].loc[area.index], area)[0]),
        "top20_overlap_with_area_data": len(set(top20) & set(area.sort_values(ascending=False).head(20).index)),
        "distance_bias": {
            "rho_with_distance_from_median_point": float(spearmanr(c["geo_coefficient"].loc[lsoas], moved_median)[0]),
            "rho_with_mean_distance_to_donors": float(spearmanr(c["geo_coefficient"].loc[lsoas], mean_donor_km)[0]),
            "median_point_design_rho_with_distance": float(spearmanr(median_pt.loc[lsoas], moved_median)[0]),
        },
        "elapsed_min": (time.time() - t0) / 60,
    }
    (DEST / "random_donor_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("top10", "fidelity")}, indent=1, default=float))
    print("fidelity R2", fid["R2"], "wR2", fid["Weighted R2"])
    for r in res["top10"]:
        print(r)


if __name__ == "__main__":
    main()
