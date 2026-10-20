"""Chapter-4 Geo numbers for the Seattle/Chicago-matched Hull design (check; publishes nothing).

Model: 32-player XGBoost (30 predictors + centroid longitude/latitude), trained as in
geosmile_hull_feature_with_location.py. Geo masking: the selected LSOAs' rows get ONLY their
longitude/latitude replaced by the single global training-median coordinate; all other inputs kept.

Every metric is computed by the published Chapter-4 functions (hull_geo_area_data_eval /
geo_area_data / eval_by_n_perturb / compute_geo_spatial_patterns). Those functions mask all the
columns they are given; `CoordOnly` turns that into coordinate-only masking: the non-coordinate
baselines are NaN sentinels, and the wrapper restores those cells from the unmasked frame of the
same shape (always predicted first) before calling XGBoost. Accuracy and ATT Faithfulness are not
computed. Writes outputs/geo_design_checks/coordinate_masking_eval/.

    python geo_coordinate_design_eval.py [N ...]   (default N grid 1,000 ... 10,000)
"""
import json, os, sys, time
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, os.getcwd())
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

import hull_geo_area_data_eval as ev
from eval_by_n_perturb import dehghani_fidelity, load_cached_neighbors
from geo_area_data import area_geo_coefficients, exact_area_shift
from geosmile_hull_twobranch import RANDOM_SEED, make_model, metrics, temporal_split
from geosmile_hull_feature_with_location import LOCATION_PLAYERS, PANEL_PATH, add_lonlat, choose_feature_players_with_location
from hull_dehghani_eval import sparsity_entropy_metrics
import compute_geo_spatial_patterns as sp

OUT = Path(__file__).resolve().parent / "outputs" / "geo_design_checks" / "coordinate_masking_eval"
OUT.mkdir(parents=True, exist_ok=True)
SEED, Y = RANDOM_SEED, "target_next_month_crime"
N_GRID = [int(a) for a in sys.argv[1:]] or [1000, 2000, 3000, 4000, 5000, 7500, 10000]


class CoordOnly:
    """Fill NaN (masked non-coordinate) cells from the unmasked frame of the same shape."""
    def __init__(self, model):
        self.model, self.orig = model, {}

    def predict(self, X):
        X = pd.DataFrame(X).astype(float)
        a = X.to_numpy()
        nan = np.isnan(a)
        if not nan.any():
            self.orig[a.shape] = a.copy()
        else:
            a = np.where(nan, self.orig[a.shape], a)
        return self.model.predict(pd.DataFrame(a, columns=X.columns))


def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

panel = add_lonlat(pd.read_csv(PANEL_PATH).assign(month=lambda d: pd.to_datetime(d["month"])))
players = choose_feature_players_with_location(panel)
train, val, test = temporal_split(panel)
xgb = make_model(SEED)
xgb.fit(panel.loc[train, players], panel.loc[train, Y], eval_set=[(panel.loc[val, players], panel.loc[val, Y])], verbose=False)
holdout = metrics(panel.loc[test, Y].to_numpy(float), np.asarray(xgb.predict(panel.loc[test, players]), float))
log(f"model holdout {holdout}")
model = CoordOnly(xgb)

baseline = pd.Series(np.nan, index=players)                   # NaN = keep the row's own value
baseline[LOCATION_PLAYERS] = panel.loc[train, LOCATION_PLAYERS].median()
hull = sorted(panel["lsoa21cd"].astype(str).unique())
names = panel.drop_duplicates("lsoa21cd").set_index("lsoa21cd")["lsoa21nm"].astype(str).to_dict()
neighbors, extra = load_cached_neighbors()
extra = add_lonlat(extra)[panel.columns]
X_test = panel.loc[test].reset_index(drop=True)

summary = []
for n in N_GRID:
    path = OUT / f"n_{n}.json"
    if path.exists():
        summary.append(json.loads(path.read_text())); log(f"N={n} cached"); continue
    t0 = time.time(); log(f"=== N={n}")
    coef, diag, M, y, d = area_geo_coefficients(model, panel, test, baseline, players, n, SEED, return_pool=True)
    fid = dehghani_fidelity(M, y, d)
    model.predict(X_test[players])                               # register the Hull test frame
    resp = ev.GeoResponder(model, X_test, players, baseline)
    inter = ev.faithfulness(resp, coef, SEED)                     # intervention impact (not ATT Faithfulness)
    stab = ev.stability(model, panel, test, baseline, players, hull, coef, neighbors, extra, n, SEED)
    cons = ev.consistency(model, panel, test, baseline, players, n, names)
    pay = {"n": n, "smile_diagnostics": diag, "fidelity": fid, "intervention": inter,
           "stability": stab, "consistency": cons, "mean_W1_response": float(y.mean())}
    if n == 4000:
        model.predict(X_test[players])
        exact = exact_area_shift(model, panel, test, baseline, players)
        both = coef.merge(exact, on="lsoa21cd")
        pay["exact_vs_smile_spearman_signed"] = float(spearmanr(both["geo_coefficient"], both["exact_w1"])[0])
        pay["exact_vs_smile_spearman_abs"] = float(spearmanr(both["geo_abs_coefficient"], both["exact_w1"])[0])
        pay["exact_w1_median"] = float(both["exact_w1"].median())
        pay["exact_w1_p90"] = float(both["exact_w1"].quantile(0.9))
        pay["concentration_exact_shift"] = sparsity_entropy_metrics(both["exact_w1"])
        r = ev.ranked(coef).merge(exact, on="lsoa21cd"); r["lsoa21nm"] = r["lsoa21cd"].map(names)
        pay["top10"] = r.head(10)[["geo_rank", "lsoa21cd", "lsoa21nm", "geo_coefficient", "exact_w1"]].to_dict("records")
        pay["n_negative_coefficients"] = int((coef["geo_coefficient"] < 0).sum())
        r.to_csv(OUT / "geo_coefficients_N4000.csv", index=False)
        fixed, _ = area_geo_coefficients(model, panel, test, baseline, players, n, SEED, kernel="inverse")
        pay["fixed_kernel_west"] = ev.stability(model, panel, test, baseline, players, hull, fixed, neighbors,
                                                extra, n, SEED, kernel="inverse")["directions"]["west"]
        # Moran / LISA on the signed coefficient, exactly as compute_geo_spatial_patterns --signed
        sp.COEF_PATH, sp.VALUE_FIELD = OUT / "geo_coefficients_N4000.csv", "geo_coefficient"
        gdf = sp.attach_boundaries(sp.load_geo_frame())
        w, islands = sp.queen_weights(gdf)
        order, yy, gm, lm, labels = sp.run_moran_lisa(gdf[~gdf["lsoa21cd"].isin(islands)].copy(), w)
        lab = pd.Series(labels, index=order)
        pay["moran"] = {"I": float(gm.I), "p_sim": float(gm.p_sim), "islands": list(islands),
                        "lisa_counts": lab.value_counts().to_dict(),
                        "HH_lsoas": [f"{c} {names.get(c, '')}" for c in lab[lab == "HH"].index]}
        pub = pd.read_csv("outputs/geo_area_data_eval/n_4000/geo_coefficients.csv").set_index("lsoa21cd")["geo_coefficient"]
        pay["spearman_with_area_data_geo"] = float(spearmanr(coef.set_index("lsoa21cd")["geo_coefficient"].loc[pub.index], pub)[0])
    pay["elapsed_min"] = (time.time() - t0) / 60
    path.write_text(json.dumps(ev.to_json(pay), indent=2))
    log(f"N={n}: R2={fid['R2']:.4f} interv={inter['top_minus_random']:.3f} p={inter['empirical_p_top_vs_random']:.3f} "
        f"stabJ={stab['jaccard_mean']:.3f} consVar={cons['mean_variance']:.5f} ({pay['elapsed_min']:.1f} min)")
    summary.append(ev.to_json(pay))

rows = [{"N": p["n"], "fid_R2": p["fidelity"]["R2"], "fid_wR2": p["fidelity"]["Weighted R2"],
         "interv_dW1": p["intervention"]["top_minus_random"], "interv_p": p["intervention"]["empirical_p_top_vs_random"],
         "stab_J": p["stability"]["jaccard_mean"], "stab_rho": p["stability"]["spearman_mean"],
         "cons_var": p["consistency"]["mean_variance"], "cons_sd": p["consistency"]["mean_sd"],
         "cons_rank_rho": p["consistency"]["rank_spearman_mean"]} for p in sorted(summary, key=lambda p: p["n"])]
pd.DataFrame(rows).to_csv(OUT / "summary_by_n.csv", index=False)
print(pd.DataFrame(rows).round(5).to_string(index=False))
