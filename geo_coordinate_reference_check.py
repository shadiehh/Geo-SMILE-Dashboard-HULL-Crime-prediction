"""Sensitivity of the coordinate-only Hull Geo branch to the reference point (check; publishes nothing).

Same model, coalitions and surrogate as geo_coordinate_design_check.py (32-player XGBoost, N=4,000,
seed 42, 5-10 LSOAs, test-period rows, W1 on affected rows, kernel-weighted Ridge). Only the point the
masked LSOAs' longitude/latitude are moved to changes: the training-median point (published check),
the four extreme points of the Hull boundary, a point 5 km beyond the west edge and a far point.
Each ranking is compared with the median-point ranking.

Writes outputs/geo_design_checks/coordinate_reference_check/. Not causal.
"""
import json, os, sys
os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from geosmile_hull_twobranch import RANDOM_SEED, apply_kernel, make_model, temporal_split
from geosmile_hull_feature_with_location import LOCATION_PLAYERS, PANEL_PATH, add_lonlat, choose_feature_players_with_location
from hull_boundaries import hull_outline, load_boundaries

OUT = os.path.join("outputs", "geo_design_checks", "coordinate_reference_check")
os.makedirs(OUT, exist_ok=True)
N, SEED, Y = 4000, RANDOM_SEED, "target_next_month_crime"

panel = add_lonlat(pd.read_csv(PANEL_PATH).assign(month=lambda d: pd.to_datetime(d["month"])))
players = choose_feature_players_with_location(panel)
train, val, test = temporal_split(panel)
model = make_model(SEED)
model.fit(panel.loc[train, players], panel.loc[train, Y], eval_set=[(panel.loc[val, players], panel.loc[val, Y])], verbose=False)

# Reference points (lon, lat)
codes = sorted(panel["lsoa21cd"].astype(str).unique())
bnd, _ = load_boundaries(codes)
outline = hull_outline(bnd.to_crs("EPSG:27700"))
from shapely import get_coordinates
xy = get_coordinates(outline)
import geopandas as gpd
pts27700 = {"border_north": xy[xy[:, 1].argmax()], "border_south": xy[xy[:, 1].argmin()],
            "border_east": xy[xy[:, 0].argmax()], "border_west": xy[xy[:, 0].argmin()]}
pts27700["outside_west_5km"] = pts27700["border_west"] - np.array([5000.0, 0.0])
ll = gpd.GeoSeries(gpd.points_from_xy([p[0] for p in pts27700.values()], [p[1] for p in pts27700.values()]),
                   crs="EPSG:27700").to_crs("EPSG:4326")
refs = {"median_point": tuple(panel.loc[train, LOCATION_PLAYERS].median())}
refs.update({k: (float(p.x), float(p.y)) for k, p in zip(pts27700, ll)})
refs["far_outside_leeds"] = (-1.55, 53.80)
lon_rng = (panel["centroid_longitude"].min(), panel["centroid_longitude"].max())
lat_rng = (panel["centroid_latitude"].min(), panel["centroid_latitude"].max())

frame = panel.loc[test].reset_index(drop=True)
x = frame[players].astype(float).to_numpy()
original = np.asarray(model.predict(pd.DataFrame(x, columns=players)), float)
ids = frame["lsoa21cd"].astype(str).to_numpy()
lsoas = np.array(sorted(set(ids)), dtype=object)
idx_by = {c: np.flatnonzero(ids == c) for c in lsoas}
cix = [players.index(c) for c in LOCATION_PLAYERS]


def run(point):
    rng = np.random.default_rng(SEED)
    M = np.zeros((N, len(lsoas)), np.int8); y = np.zeros(N)
    for j in range(N):
        k = int(rng.integers(5, 11)); sel = rng.choice(len(lsoas), size=k, replace=False); M[j, sel] = 1
        aff = np.concatenate([idx_by[lsoas[s]] for s in sel])
        p = x.copy(); p[np.ix_(aff, cix)] = point
        y[j] = wasserstein_distance(original[aff], np.asarray(model.predict(pd.DataFrame(p, columns=players)), float)[aff])
    d = M.sum(1) / len(lsoas); order = rng.permutation(N)
    tr, va, te = order[:int(.6 * N)], order[int(.6 * N):int(.8 * N)], order[int(.8 * N):]
    sigma = max(float(np.median(d[tr])), 1 / len(lsoas)); sc = {}
    for kname in ("rbf", "laplace", "inverse"):
        sc[kname] = r2_score(y[va], Ridge(alpha=1.0).fit(M[tr], y[tr], sample_weight=apply_kernel(d[tr], kname, sigma)).predict(M[va]))
    best = max(sc, key=sc.get); trva = np.concatenate([tr, va])
    r = Ridge(alpha=1.0).fit(M[trva], y[trva], sample_weight=apply_kernel(d[trva], best, sigma))
    exact = {}
    for c in lsoas:
        a = idx_by[c]; p = x.copy(); p[np.ix_(a, cix)] = point
        exact[c] = wasserstein_distance(original[a], np.asarray(model.predict(pd.DataFrame(p, columns=players)), float)[a])
    return pd.Series(r.coef_, index=lsoas), pd.Series(exact), float(r2_score(y[te], r.predict(M[te]))), float(y.mean())


reg = set(pd.read_csv("outputs/geo_area_data_models/top50_maps/highlighted_regions_lsoas.csv")["lsoa21cd"].astype(str))
area = pd.read_csv("outputs/geo_area_data_eval/n_4000/geo_coefficients.csv").set_index("lsoa21cd")["geo_coefficient"]
top = lambda s: set(s.sort_values(ascending=False).head(20).index)
coefs, rows = {}, []
for name, pt in refs.items():
    coef, exact, fid, ymean = run(np.array(pt, float))
    coefs[name] = coef
    rank = coef.rank(ascending=False)
    rows.append({"reference": name, "lon": round(pt[0], 4), "lat": round(pt[1], 4),
                 "inside_training_range": bool(lon_rng[0] <= pt[0] <= lon_rng[1] and lat_rng[0] <= pt[1] <= lat_rng[1]),
                 "test_R2": round(fid, 3), "mean_W1": round(ymean, 3),
                 "rho_signed_vs_exact": round(float(spearmanr(coef, exact[coef.index])[0]), 3),
                 "old_town_rank": int(rank["E01033104"]),
                 "top20_in_expert_regions": sum(c in reg for c in top(coef)),
                 "rho_with_area_data": round(float(spearmanr(coef.loc[area.index], area)[0]), 3)})
    print(rows[-1], flush=True)
base = coefs["median_point"]
for r in rows:
    c = coefs[r["reference"]]
    r["rho_with_median_point"] = round(float(spearmanr(c, base)[0]), 3)
    r["top20_overlap_with_median_point"] = len(top(c) & top(base))
table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT, "reference_point_summary.csv"), index=False)
pd.DataFrame(coefs).to_csv(os.path.join(OUT, "coefficients_by_reference.csv"))
cross = pd.DataFrame(coefs).corr(method="spearman").round(3)
cross.to_csv(os.path.join(OUT, "spearman_between_references.csv"))
print(table.to_string(index=False)); print(cross.to_string())
json.dump({"training_lon_range": lon_rng, "training_lat_range": lat_rng}, open(os.path.join(OUT, "ranges.json"), "w"))
