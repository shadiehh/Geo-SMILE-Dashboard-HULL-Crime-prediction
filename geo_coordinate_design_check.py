"""Hull Geo branch with coordinate-only masking (check; does not replace any published design).

Model: the 32-player XGBoost of geosmile_hull_feature_with_location.py (30 predictors + centroid
longitude/latitude), retrained identically. Geo coalitions: as in geo_area_data.area_geo_coefficients
(N=4,000, seed 42, 5-10 LSOAs per coalition, test-period rows). Masking an LSOA moves ONLY the
longitude/latitude of its rows to the single global training-median coordinate; every other input
is kept. Response: W1 on the affected rows' predictions. Weighted Ridge with the same kernel menu.
Compared with the published area-data GeoSensitivity (signed coefficient, N=4,000).
"""
import json, os, sys
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, wasserstein_distance, kendalltau
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score

from geosmile_hull_twobranch import RANDOM_SEED, apply_kernel, make_model, metrics, temporal_split
from geosmile_hull_feature_with_location import (LOCATION_PLAYERS, PANEL_PATH, add_lonlat,
                                                  choose_feature_players_with_location)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "geo_design_checks", "coordinate_masking_check")
os.makedirs(OUT, exist_ok=True)
N, SEED, Y = 4000, RANDOM_SEED, "target_next_month_crime"

panel = add_lonlat(pd.read_csv(PANEL_PATH).assign(month=lambda d: pd.to_datetime(d["month"])))
players = choose_feature_players_with_location(panel)
assert len(players) == 32
train, val, test = temporal_split(panel)
model = make_model(SEED)
model.fit(panel.loc[train, players], panel.loc[train, Y],
          eval_set=[(panel.loc[val, players], panel.loc[val, Y])], verbose=False)
frame = panel.loc[test].reset_index(drop=True)
x = frame[players].astype(float).to_numpy()
original = np.asarray(model.predict(pd.DataFrame(x, columns=players)), float)
holdout = metrics(frame[Y].to_numpy(float), original)
print("holdout (expect r2 0.895):", holdout, flush=True)

median_coord = panel.loc[train, LOCATION_PLAYERS].median()        # one global training-median point
cix = [players.index(c) for c in LOCATION_PLAYERS]
ids = frame["lsoa21cd"].astype(str).to_numpy()
lsoas = np.array(sorted(set(ids)), dtype=object)
idx_by = {c: np.flatnonzero(ids == c) for c in lsoas}

rng = np.random.default_rng(SEED)
M = np.zeros((N, len(lsoas)), np.int8); y = np.zeros(N)
for j in range(N):
    k = int(rng.integers(5, 11)); sel = rng.choice(len(lsoas), size=k, replace=False); M[j, sel] = 1
    aff = np.concatenate([idx_by[lsoas[s]] for s in sel])
    p = x.copy(); p[np.ix_(aff, cix)] = median_coord.to_numpy(float)
    pp = np.asarray(model.predict(pd.DataFrame(p, columns=players)), float)
    y[j] = wasserstein_distance(original[aff], pp[aff])
d = M.sum(1) / len(lsoas); order = rng.permutation(N)
tr, va, te = order[:int(.6 * N)], order[int(.6 * N):int(.8 * N)], order[int(.8 * N):]
sigma = max(float(np.median(d[tr])), 1 / len(lsoas)); scores = {}
for kname in ("rbf", "laplace", "inverse"):
    r = Ridge(alpha=1.0).fit(M[tr], y[tr], sample_weight=apply_kernel(d[tr], kname, sigma))
    scores[kname] = float(r2_score(y[va], r.predict(M[va])))
best = max(scores, key=scores.get); trva = np.concatenate([tr, va])
final = Ridge(alpha=1.0).fit(M[trva], y[trva], sample_weight=apply_kernel(d[trva], best, sigma))
coef = pd.DataFrame({"lsoa21cd": lsoas.astype(str), "geo_coefficient": final.coef_,
                     "geo_abs_coefficient": np.abs(final.coef_)}).set_index("lsoa21cd")
coef.to_csv(os.path.join(OUT, "geo_coordinate_masking_N4000.csv"))

# single-LSOA exact shift under coordinate masking (each LSOA alone), for the surrogate check
exact = {}
for c in lsoas:
    a = idx_by[c]; p = x.copy(); p[np.ix_(a, cix)] = median_coord.to_numpy(float)
    pp = np.asarray(model.predict(pd.DataFrame(p, columns=players)), float)
    exact[c] = wasserstein_distance(original[a], pp[a])
exact = pd.Series(exact)

pub = pd.read_csv("outputs/geo_area_data_eval/n_4000/geo_coefficients.csv").set_index("lsoa21cd")
pub = pub.loc[coef.index]
reg = set(pd.read_csv("outputs/geo_area_data_models/top50_maps/highlighted_regions_lsoas.csv")["lsoa21cd"].astype(str))
lab = pd.Series({c: int(c in reg) for c in coef.index})
def top(s, k=20): return set(s.sort_values(ascending=False).head(k).index)
def jacc(a, b): return len(a & b) / len(a | b)

res = {
    "model_holdout": holdout,
    "median_coordinate": median_coord.to_dict(),
    "surrogate": {"best_kernel": best, "validation_r2_by_kernel": scores,
                  "test_r2": float(r2_score(y[te], final.predict(M[te]))),
                  "mean_W1_response": float(y.mean()), "max_W1_response": float(y.max())},
    "surrogate_vs_exact_single_lsoa_shift": {
        "spearman_signed": float(spearmanr(coef["geo_coefficient"], exact[coef.index])[0]),
        "spearman_abs": float(spearmanr(coef["geo_abs_coefficient"], exact[coef.index])[0])},
    "vs_area_data_geo": {},
    "expert_regions_top20": {
        "coordinate_signed": int(lab[list(top(coef["geo_coefficient"]))].sum()),
        "coordinate_abs": int(lab[list(top(coef["geo_abs_coefficient"]))].sum()),
        "area_data_signed": int(lab[list(top(pub["geo_coefficient"]))].sum()),
        "chance": round(20 * lab.mean(), 1)},
}
for a in ("geo_coefficient", "geo_abs_coefficient"):
    for b in ("geo_coefficient", "geo_abs_coefficient"):
        res["vs_area_data_geo"][f"coord_{a}__vs__areadata_{b}"] = {
            "spearman": round(float(spearmanr(coef[a], pub[b])[0]), 3),
            "kendall": round(float(kendalltau(coef[a], pub[b])[0]), 3),
            "pearson": round(float(pearsonr(coef[a], pub[b])[0]), 3),
            "top20_overlap": len(top(coef[a]) & top(pub[b])),
            "top20_jaccard": round(jacc(top(coef[a]), top(pub[b])), 3)}
print(json.dumps(res, indent=2))
json.dump(res, open(os.path.join(OUT, "geo_coordinate_masking_N4000.json"), "w"), indent=2)
