"""Does Hull area-data GeoSensitivity mostly track crime level? (reviewer check)

1. Spearman of the published signed Geo coefficient with each LSOA's crime level and with how far
   its test-period profile sits from the training median.
2. Expert top-20 overlap of simple baselines (rank LSOAs by crime level / by distance from median).
3. Geo variants on the same fitted XGBoost, coalitions and kernel selection as the published run:
   mask ALL 30 inputs (reproduction), mask ONLY the 5 crime-history inputs, mask ONLY the 25 others.
"""
import json, os, sys
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from geosmile_hull_twobranch import choose_feature_players, temporal_split, RANDOM_SEED
from geo_area_data import apply_kernel
from hull_expert_topk import geo_expert_binary_frame
from hull_geo_area_data_eval import accuracy, PANEL_PATH, Y_COL
from model_factory import make_model, fit_model
os.makedirs("outputs/geo_design_checks/crime_history_check", exist_ok=True)

N, SEED = 4000, RANDOM_SEED
panel = pd.read_csv(PANEL_PATH); panel["month"] = pd.to_datetime(panel["month"])
players = choose_feature_players(panel)
train, val, test = temporal_split(panel)
base = panel.loc[train, players].median()
crime_cols = [c for c in players if c.startswith("crime_")]
other_cols = [c for c in players if c not in crime_cols]
print("crime-history inputs:", crime_cols)
hull = sorted(panel["lsoa21cd"].astype(str).unique())
expert = geo_expert_binary_frame(hull)[["lsoa21cd", "expert_top10_reference"]]
published = pd.read_csv("outputs/geo_area_data_eval/n_4000/geo_coefficients.csv")

model = make_model("xgboost", SEED, n_jobs=4)
fit_model("xgboost", model, panel.loc[train, players], panel.loc[train, Y_COL],
          panel.loc[val, players], panel.loc[val, Y_COL])

# ---- 1-2: crime level and profile distance --------------------------------------------------
sd = panel.loc[train, players].std().replace(0, 1)
te = panel.loc[test].copy()
z = ((te[players] - base) / sd).abs()
lev = pd.DataFrame({
    "lsoa21cd": te["lsoa21cd"].astype(str),
    "crime_test": te["crime_count"],
    "dist_crime": z[crime_cols].mean(axis=1),
    "dist_other": z[other_cols].mean(axis=1),
    "dist_all": z.mean(axis=1),
}).groupby("lsoa21cd").mean()
lev["crime_train"] = panel.loc[train].groupby("lsoa21cd")["crime_count"].mean()
g = published.set_index("lsoa21cd")["geo_coefficient"]
lev = lev.join(g)
res = {"spearman_with_signed_geo": {c: float(spearmanr(lev[c], lev["geo_coefficient"])[0])
                                     for c in ["crime_test", "crime_train", "dist_crime", "dist_other", "dist_all"]}}
def acc_of(series):
    frame = series.rename("geo_score").reset_index().rename(columns={"index": "lsoa21cd"})
    a = accuracy(frame, expert); return {"top20_in_expert": a["intersection"], "auroc": round(a["att_auroc"], 3)}
res["baseline_rankings"] = {c: acc_of(lev[c]) for c in ["crime_test", "crime_train", "dist_crime", "dist_all"]}
res["published_geo"] = acc_of(g)
top_geo = set(g.sort_values(ascending=False).head(20).index)
top_crime = set(lev["crime_test"].sort_values(ascending=False).head(20).index)
res["top20_overlap_geo_vs_crime_test"] = len(top_geo & top_crime)
res["expert_regions_vs_crime"] = {
    "n_expert_lsoas": int(expert["expert_top10_reference"].sum()),
    "mean_crime_in_expert": float(lev.loc[expert.set_index("lsoa21cd")["expert_top10_reference"] == 1, "crime_test"].mean()),
    "mean_crime_outside": float(lev.loc[expert.set_index("lsoa21cd")["expert_top10_reference"] == 0, "crime_test"].mean()),
}

# ---- 3: masking variants (same coalitions, same split, same kernel menu) ---------------------
frame = panel.loc[test].reset_index(drop=True)
x = frame[players].astype(float).to_numpy()
original = np.asarray(model.predict(pd.DataFrame(x, columns=players)), float)
ids = frame["lsoa21cd"].astype(str).to_numpy()
lsoas = np.array(sorted(set(ids)), dtype=object)
idx_by = {c: np.flatnonzero(ids == c) for c in lsoas}
col_ix = {c: i for i, c in enumerate(players)}

def variant(cols):
    rng = np.random.default_rng(SEED)
    cix = [col_ix[c] for c in cols]
    M = np.zeros((N, len(lsoas)), np.int8); y = np.zeros(N)
    for j in range(N):
        k = int(rng.integers(5, 11)); sel = rng.choice(len(lsoas), size=k, replace=False); M[j, sel] = 1
        aff = np.concatenate([idx_by[lsoas[s]] for s in sel])
        p = x.copy(); p[np.ix_(aff, cix)] = base[cols].to_numpy(float)
        pp = np.asarray(model.predict(pd.DataFrame(p, columns=players)), float)
        y[j] = wasserstein_distance(original[aff], pp[aff])
    d = M.sum(1) / len(lsoas); order = rng.permutation(N)
    tr, va, te_ = order[:int(.6 * N)], order[int(.6 * N):int(.8 * N)], order[int(.8 * N):]
    sigma = max(float(np.median(d[tr])), 1 / len(lsoas)); sc = {}
    for kname in ("rbf", "laplace", "inverse"):
        r = Ridge(alpha=1.0).fit(M[tr], y[tr], sample_weight=apply_kernel(d[tr], kname, sigma))
        sc[kname] = r2_score(y[va], r.predict(M[va]))
    best = max(sc, key=sc.get); trva = np.concatenate([tr, va])
    r = Ridge(alpha=1.0).fit(M[trva], y[trva], sample_weight=apply_kernel(d[trva], best, sigma))
    return pd.Series(r.coef_, index=lsoas), float(r2_score(y[te_], r.predict(M[te_]))), float(y.mean())

res["variants"] = {}
for name, cols in [("all_30", players), ("crime_history_only", crime_cols), ("non_crime_only", other_cols)]:
    coef, fid, ymean = variant(cols)
    res["variants"][name] = {
        "test_R2": round(fid, 3), "mean_W1_response": round(ymean, 3),
        "spearman_with_published": round(float(spearmanr(coef.loc[g.index], g)[0]), 3),
        "spearman_with_crime_test": round(float(spearmanr(coef.loc[lev.index], lev["crime_test"])[0]), 3),
        **acc_of(coef),
    }
    coef.rename("geo_coefficient").to_csv(f"outputs/geo_design_checks/crime_history_check/variant_{name}.csv")
    print(name, res["variants"][name], flush=True)
print(json.dumps(res, indent=2))
json.dump(res, open("outputs/geo_design_checks/crime_history_check/geo_crime_check.json", "w"), indent=2)
