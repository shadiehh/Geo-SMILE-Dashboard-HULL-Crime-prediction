"""Chapter-4 numbers for the Seattle/Chicago-aligned Hull design (check; replaces nothing).

Design (as Seattle and Chicago):
  * ONE predictive model on all 32 inputs: the 30 Feature predictors + British National Grid
    centroid easting/northing (model_factory, seed 42; the Stage-1 "+ coordinates" model).
  * Feature branch: players = the 30 predictors; a masked predictor is set to its training median,
    the coordinates of every row are kept at their true value.
  * Geo branch: players = the 168 LSOAs; masking an LSOA moves ONLY the coordinates of its rows to
    the single global training-median coordinate; all 30 predictors are kept.

Every metric is produced by the published Chapter-4 functions (geosmile_hull_twobranch,
eval_by_n_perturb, geo_area_data, hull_geo_area_data_eval, compute_geo_spatial_patterns,
hull_dehghani_eval). Those functions pass either the 30 Feature columns or all 32 columns to
`model.predict`; `Aligned` turns that into the design above: 30-column frames get the rows' true
coordinates appended, and in 32-column frames the NaN baseline cells (non-coordinates) are restored
from the unmasked frame of the same shape, which those functions always predict first.

Part A (XGBoost, N grid): fidelity, accuracy, intervention impact, stability, consistency, and at
N=4,000 the global/local Feature results, concentration, single-LSOA shifts, Moran/LISA, rule
robustness and the fixed-kernel west run.
Part B (ten models, N=4,000): predictive performance, surrogate menus for both branches, accuracy,
and ATT Faithfulness = Pearson/Spearman across models of predictive R2 vs ATT accuracy (primary)
and ATT-AUC (secondary).

Writes outputs/geo_aligned_design/. Not causal.

    python hull_aligned_design_eval.py [--part A|B|both] [--n-grid 4000 1000 ...] [--models ...]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from itertools import combinations
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, pearsonr, spearmanr, wasserstein_distance
from shapely.geometry import Polygon
from shapely.ops import unary_union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

import compute_geo_spatial_patterns as sp
import hull_geo_area_data_eval as ev
from eval_by_n_perturb import att_from_importance, dehghani_fidelity, feature_responses, load_cached_neighbors
from geo_area_data import area_geo_coefficients, exact_area_shift, surrogate_menu
from geosmile_hull_twobranch import (SPATIAL_COLUMNS, choose_feature_players, local_feature_coefficients,
                                     smile_global_feature, temporal_split)
from hull_boundaries import load_boundaries
from hull_boundary_neighbors import expand_panel
from hull_dehghani_eval import sparsity_entropy_metrics
from hull_expert_topk import feature_expert_binary_frame
from model_factory import MODEL_NAMES, fit_model, make_model
from select_highlighted_region_lsoas import REGIONS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "geo_aligned_design"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"
REGIONS_CSV = OUT / "geo_area_data_models" / "top50_maps" / "highlighted_regions_lsoas.csv"
AREA_DATA_COEF = OUT / "geo_area_data_eval" / "n_4000" / "geo_coefficients.csv"
Y_COL, SEED = "target_next_month_crime", 42
N_GRID = (1000, 2000, 3000, 4000, 5000, 7500, 10000)
CONS_SEEDS = tuple(range(100, 105))
FEAT_K, GEO_K, N_RANDOM = 10, 20, 1000
DIRECTIONS = ("north", "east", "south", "west")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ev.to_json(obj), indent=2), encoding="utf-8")


class Aligned:
    """Model on players + coordinates; see module docstring."""

    def __init__(self, model, players: list[str]):
        self.model, self.players, self.cols = model, players, players + SPATIAL_COLUMNS
        self.coords: np.ndarray | None = None
        self.orig: dict = {}

    def set_coords(self, frame: pd.DataFrame) -> "Aligned":
        self.coords = frame[SPATIAL_COLUMNS].to_numpy(dtype=float)
        return self

    def predict(self, X):
        X = pd.DataFrame(X)
        if list(X.columns) == self.players:                     # Feature branch call
            assert self.coords is not None and len(self.coords) == len(X), "set_coords to the reference frame"
            a = np.column_stack([X.to_numpy(dtype=float), self.coords])
        else:                                                    # Geo branch call (all 32 columns)
            a = X[self.cols].to_numpy(dtype=float)
            nan = np.isnan(a)
            if nan.any():
                a = np.where(nan, self.orig[a.shape], a)
            else:
                self.orig[a.shape] = a.copy()
        return np.asarray(self.model.predict(pd.DataFrame(a, columns=self.cols)), float).ravel()


# ---- expert references --------------------------------------------------------------------------
def region_members(boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
    """Overlap of every LSOA with the traced expert regions (as select_highlighted_region_lsoas)."""
    regions = gpd.GeoDataFrame({"region": list(REGIONS)}, crs=4326,
                               geometry=[unary_union([Polygon(r) for r in s]) for s in REGIONS.values()]).to_crs(27700)
    g = boundaries[["lsoa21cd", "geometry"]].to_crs(27700)
    g["lsoa_km2"] = g.area / 1e6
    x = gpd.overlay(g, regions, how="intersection")
    x["inside_km2"] = x.area / 1e6
    x["frac"] = x["inside_km2"] / x["lsoa_km2"]
    return x.sort_values("frac", ascending=False).drop_duplicates("lsoa21cd")[["lsoa21cd", "region", "frac", "inside_km2"]]


def rule_sets(ov: pd.DataFrame, published: set[str]) -> dict[str, set[str]]:
    f, km = ov.set_index("lsoa21cd")["frac"], ov.set_index("lsoa21cd")["inside_km2"]
    return {
        "published (>=20% or >=0.1 km2; >=50% north)": published,
        ">=20% or >=0.1 km2, all regions": set(f[(f >= 0.2) | (km >= 0.1)].index),
        ">=10% of LSOA area": set(f[f >= 0.1].index),
        ">=50% of LSOA area": set(f[f >= 0.5].index),
        ">=75% of LSOA area": set(f[f >= 0.75].index),
    }


def geo_accuracy(coef: pd.DataFrame, ref: set[str], k: int = GEO_K) -> dict:
    s = coef.set_index("lsoa21cd")["geo_coefficient"]
    n = len(s)
    top = set(s.sort_values(ascending=False).head(k).index)
    tp = len(top & ref)
    tn = n - len(top | ref)
    lab = pd.Series([int(c in ref) for c in s.index], index=s.index)
    return {"k": k, "reference_size": len(ref), "intersection": tp, "accuracy": tp / k, "chance": len(ref) / n,
            "lift": (tp / k) / (len(ref) / n), "p_hypergeom": float(hypergeom.sf(tp - 1, n, len(ref), k)),
            "att_acc": (tp + tn) / n, "att_f1": 2 * tp / (len(ref) + k), "att_auroc": float(roc_auc_score(lab, s)),
            "precision": tp / k, "recall": tp / len(ref), "jaccard": tp / len(top | ref),
            "top_k_outside_reference": sorted(top - ref)}


# ---- Feature-branch metrics (mirror eval_by_n_perturb) ------------------------------------------
def feat_rank(imp: pd.DataFrame) -> pd.DataFrame:
    return imp.sort_values(["abs_coefficient", "feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def feature_intervention(w: Aligned, X: pd.DataFrame, players, baseline, imp, seed) -> dict:
    order = feat_rank(imp)["feature"].tolist()
    w.set_coords(X)
    orig = w.predict(X[players])
    rng = np.random.default_rng(seed)

    def resp(names):
        pert = X[players].copy()
        for nm in names:
            pert[nm] = baseline[nm]
        return float(wasserstein_distance(orig, w.predict(pert)))
    top_v, bot_v = resp(order[:FEAT_K]), resp(order[-FEAT_K:])
    rand = np.array([resp(list(rng.choice(order, size=FEAT_K, replace=False))) for _ in range(N_RANDOM)])
    lo, hi = np.percentile(rand, [2.5, 97.5])
    return {"k": FEAT_K, "top_response": top_v, "bottom_response": bot_v, "random_mean": float(rand.mean()),
            "random_ci95_low": float(lo), "random_ci95_high": float(hi), "top_minus_random": float(top_v - rand.mean()),
            "top_minus_bottom": float(top_v - bot_v),
            "empirical_p_top_vs_random": float((np.sum(rand >= top_v) + 1) / (len(rand) + 1)), "n_random": N_RANDOM}


def feature_stability(w, raw_model, panel, X_test, players, baseline, base_imp, neighbors, extra, n) -> dict:
    base = feat_rank(base_imp)
    base_top = set(base.head(FEAT_K)["feature"])
    base_abs = base_imp.set_index("feature")["abs_coefficient"]
    cols = players + SPATIAL_COLUMNS
    out, js, rs = {}, [], []
    for d in DIRECTIONS:
        code = str(neighbors.loc[neighbors["direction"] == d, "lsoa21cd"].iloc[0])
        expanded = expand_panel(panel, extra[extra["lsoa21cd"].astype(str) == code])
        _, _, et = temporal_split(expanded)
        X_exp = expanded.loc[et].reset_index(drop=True)
        w.set_coords(X_exp)
        imp, _, _, _, _ = smile_global_feature(w, X_exp, players, baseline, n, SEED)
        top = set(feat_rank(imp).head(FEAT_K)["feature"])
        j = len(top & base_top) / len(top | base_top)
        r = float(spearmanr(base_abs.loc[players], imp.set_index("feature")["abs_coefficient"].loc[players])[0])
        key = ["lsoa21cd", "month"]
        hull_exp = X_exp.merge(X_test[key], on=key)
        p_exp = raw_model.predict(hull_exp[cols].astype(float))
        p_base = raw_model.predict(X_test.merge(hull_exp[key], on=key)[cols].astype(float))
        out[d] = {"lsoa21cd": code, "jaccard": j, "spearman_abs": r, "top10_retained": len(top & base_top),
                  "prediction_spearman": float(spearmanr(p_base, p_exp)[0])}
        js.append(j); rs.append(r)
        log(f"    Feature stability {d}: J={j:.3f} rho={r:.4f}")
    return {"directions": out, "jaccard_mean": float(np.mean(js)), "spearman_mean": float(np.mean(rs))}


def feature_consistency(w, X_test, players, baseline, n) -> dict:
    runs = []
    for s in CONS_SEEDS:
        w.set_coords(X_test)
        imp, _, _, _, _ = smile_global_feature(w, X_test, players, baseline, n, s)
        runs.append(imp.set_index("feature")["coefficient"].loc[players])
    mat = pd.concat(runs, axis=1)
    var, sd = mat.var(axis=1, ddof=1), mat.std(axis=1, ddof=1)
    absm = mat.abs()
    rr, pr, jj = [], [], []
    for a, b in combinations(range(mat.shape[1]), 2):
        x, y = absm.iloc[:, a], absm.iloc[:, b]
        rr.append(spearmanr(x, y)[0]); pr.append(pearsonr(x, y)[0])
        ta, tb = set(x[x >= x.quantile(0.8)].index), set(y[y >= y.quantile(0.8)].index)
        jj.append(len(ta & tb) / len(ta | tb))
    top = absm.mean(axis=1).sort_values(ascending=False).head(3).index
    return {"seeds": list(CONS_SEEDS), "mean_variance": float(var.mean()), "median_variance": float(var.median()),
            "mean_sd": float(sd.mean()), "median_sd": float(sd.median()), "max_sd": float(sd.max()),
            "max_sd_feature": str(sd.idxmax()), "rank_spearman_mean": float(np.mean(rr)),
            "score_pearson_mean": float(np.mean(pr)), "top20pct_jaccard_mean": float(np.mean(jj)),
            "leading_features_mean_sd": {f: {"mean_coefficient": float(mat.loc[f].mean()), "sd": float(sd.loc[f])} for f in top}}


# ---- Part A --------------------------------------------------------------------------------------
def part_a(ctx: dict, n_grid) -> None:
    panel, players, cols, test, X_test = ctx["panel"], ctx["players"], ctx["cols"], ctx["test"], ctx["X_test"]
    fb, gb, w, raw = ctx["feat_baseline"], ctx["geo_baseline"], ctx["w"], ctx["raw"]
    neighbors, extra, hull, names = ctx["neighbors"], ctx["extra"], ctx["hull"], ctx["names"]
    for n in n_grid:
        path = DEST / "xgboost" / f"n_{n}.json"
        if path.exists():
            log(f"N={n}: cached"); continue
        t0 = time.time(); log(f"=== Part A, N={n}")
        # Feature branch
        w.set_coords(X_test)
        imp, fdiag, masks, pert, orig = smile_global_feature(w, X_test, players, fb, n, SEED)
        ffid = dehghani_fidelity(masks, feature_responses(masks, pert, orig), 1.0 - masks.mean(axis=1))
        facc = att_from_importance(imp, ctx["feat_expert"], id_col="feature", mag_col="abs_coefficient", k=FEAT_K)
        finter = feature_intervention(w, X_test, players, fb, imp, SEED)
        fstab = feature_stability(w, raw, panel, X_test, players, fb, imp, neighbors, extra, n)
        fcons = feature_consistency(w, X_test, players, fb, n)
        feature = {"fidelity": ffid, "smile_diagnostics": fdiag, "accuracy": facc, "intervention": finter,
                   "stability": fstab, "consistency": fcons,
                   "importance": feat_rank(imp)[["feature", "coefficient", "abs_coefficient"]].to_dict("records")}
        # Geo branch
        coef, gdiag, M, y, d = area_geo_coefficients(w, panel, test, gb, cols, n, SEED, return_pool=True)
        gfid = dehghani_fidelity(M, y, d)
        gacc = geo_accuracy(coef, ctx["ref"])
        w.predict(X_test[cols])
        ginter = ev.faithfulness(ev.GeoResponder(w, X_test, cols, gb), coef, SEED)
        gstab = ev.stability(w, panel, test, gb, cols, hull, coef, neighbors, extra, n, SEED)
        gcons = ev.consistency(w, panel, test, gb, cols, n, names)
        geo = {"fidelity": gfid, "smile_diagnostics": gdiag, "mean_W1_response": float(y.mean()), "accuracy": gacc,
               "intervention": ginter, "stability": gstab, "consistency": gcons}
        if n == 4000:
            feature["concentration"] = sparsity_entropy_metrics(imp["abs_coefficient"])
            local = local_feature_coefficients(masks, pert, orig, players, X_test["lsoa21cd"].astype(str).to_numpy(),
                                               fdiag["best_kernel"], SEED)
            local.to_csv(DEST / "xgboost" / "feature_local_coefficients_N4000.csv")
            feature["mapped_top6"] = feat_rank(imp).head(6)["feature"].tolist()
            feat_rank(imp).to_csv(DEST / "xgboost" / "feature_global_importance_N4000.csv", index=False)
            w.predict(X_test[cols])
            exact = exact_area_shift(w, panel, test, gb, cols)
            both = coef.merge(exact, on="lsoa21cd")
            r = ev.ranked(coef).merge(exact, on="lsoa21cd")
            r["lsoa21nm"] = r["lsoa21cd"].map(names)
            r.to_csv(DEST / "xgboost" / "geo_coefficients_N4000.csv", index=False)
            geo.update({
                "exact_vs_signed_spearman": float(spearmanr(both["geo_coefficient"], both["exact_w1"])[0]),
                "exact_vs_abs_spearman": float(spearmanr(both["geo_abs_coefficient"], both["exact_w1"])[0]),
                "exact_w1_median": float(both["exact_w1"].median()), "exact_w1_p90": float(both["exact_w1"].quantile(0.9)),
                "concentration_exact_shift": sparsity_entropy_metrics(both["exact_w1"]),
                "top10": r.head(10)[["geo_rank", "lsoa21cd", "lsoa21nm", "geo_coefficient", "exact_w1"]].to_dict("records"),
                "n_negative": int((coef["geo_coefficient"] < 0).sum()),
                "accuracy_k": {k: geo_accuracy(coef, ctx["ref"], k) for k in (42, 50)},
                "accuracy_rule_robustness": {nm: geo_accuracy(coef, s) for nm, s in ctx["rules"].items()},
                "spearman_with_area_data_geo": float(spearmanr(
                    coef.set_index("lsoa21cd")["geo_coefficient"].loc[ctx["area_data"].index], ctx["area_data"])[0]),
            })
            fixed, _ = area_geo_coefficients(w, panel, test, gb, cols, n, SEED, kernel="inverse")
            geo["fixed_kernel_west"] = ev.stability(w, panel, test, gb, cols, hull, fixed, neighbors, extra, n, SEED,
                                                    kernel="inverse")["directions"]["west"]
            sp.COEF_PATH, sp.VALUE_FIELD = DEST / "xgboost" / "geo_coefficients_N4000.csv", "geo_coefficient"
            gdf = sp.attach_boundaries(sp.load_geo_frame())
            wq, islands = sp.queen_weights(gdf)
            order, _, gm, _, labels = sp.run_moran_lisa(gdf[~gdf["lsoa21cd"].isin(islands)].copy(), wq)
            lab = pd.Series(labels, index=order)
            geo["moran"] = {"I": float(gm.I), "p_sim": float(gm.p_sim), "lisa_counts": lab.value_counts().to_dict(),
                            "HH": [f"{c} {names.get(c, '')}" for c in lab[lab == "HH"].index]}
        jdump(path, {"n": n, "model": "xgboost", "predictive": ctx["predictive"], "feature": feature, "geo": geo,
                     "elapsed_min": (time.time() - t0) / 60})
        log(f"N={n} done ({(time.time() - t0) / 60:.1f} min): Feature R2={ffid['R2']:.4f} acc={facc['intersection']}/10 | "
            f"Geo R2={gfid['R2']:.4f} acc={gacc['intersection']}/20")


# ---- Part B --------------------------------------------------------------------------------------
def part_b(ctx: dict, models) -> None:
    panel, players, cols, split = ctx["panel"], ctx["players"], ctx["cols"], ctx["split"]
    train, val, test = split
    X_test = ctx["X_test"]
    for name in models:
        path = DEST / "models" / f"{name}.json"
        if path.exists():
            log(f"[{name}] cached"); continue
        t0 = time.time()
        model = make_model(name, SEED, n_jobs=4)
        fit_model(name, model, panel.loc[train, cols], panel.loc[train, Y_COL], panel.loc[val, cols], panel.loc[val, Y_COL])
        pred = np.asarray(model.predict(panel.loc[test, cols].astype(float)), float).ravel()
        yt = panel.loc[test, Y_COL].to_numpy(float)
        perf = {"r2": float(r2_score(yt, pred)), "mae": float(mean_absolute_error(yt, pred)),
                "rmse": float(np.sqrt(mean_squared_error(yt, pred)))}
        log(f"[{name}] predictive R2 (30 + coordinates) = {perf['r2']:.4f}")
        w = Aligned(model, players).set_coords(X_test)
        imp, _, masks, pert, orig = smile_global_feature(w, X_test, players, ctx["feat_baseline"], 4000, SEED)
        fmenu = surrogate_menu(masks, feature_responses(masks, pert, orig), 1.0 - masks.mean(axis=1))
        facc = att_from_importance(imp, ctx["feat_expert"], id_col="feature", mag_col="abs_coefficient", k=FEAT_K)
        coef, _, M, y, d = area_geo_coefficients(w, panel, test, ctx["geo_baseline"], cols, 4000, SEED, return_pool=True)
        gmenu = surrogate_menu(M, y, d)
        gacc = geo_accuracy(coef, ctx["ref"])
        jdump(path, {"model": name, "predictive": perf, "feature_menu": fmenu, "geo_menu": gmenu,
                     "feature_accuracy": {k: facc[k] for k in ("intersection", "auroc", "precision", "recall", "f1", "jaccard_at_k")},
                     "geo_accuracy": gacc, "geo_mean_W1": float(y.mean()),
                     "geo_coefficients": coef[["lsoa21cd", "geo_coefficient"]].to_dict("list"),
                     "elapsed_min": (time.time() - t0) / 60})
        log(f"[{name}] Feature top-10 {facc['intersection']}/10 | Geo top-20 in regions {gacc['intersection']}/20 "
            f"({(time.time() - t0) / 60:.1f} min)")
    rows = []
    for name in MODEL_NAMES:
        p = DEST / "models" / f"{name}.json"
        if p.exists():
            r = json.loads(p.read_text())
            rows.append({"model": name, "r2": r["predictive"]["r2"], "mae": r["predictive"]["mae"], "rmse": r["predictive"]["rmse"],
                         "feat_overlap": r["feature_accuracy"]["intersection"], "feat_auc": r["feature_accuracy"]["auroc"],
                         "geo_overlap": r["geo_accuracy"]["intersection"], "geo_auc": r["geo_accuracy"]["att_auroc"],
                         "feat_ridge_r2": next(m["r2"] for m in r["feature_menu"] if m["surrogate"] == "Weighted Ridge"),
                         "geo_ridge_r2": next(m["r2"] for m in r["geo_menu"] if m["surrogate"] == "Weighted Ridge")})
    table = pd.DataFrame(rows)
    table.to_csv(DEST / "models" / "model_summary.csv", index=False)
    if len(table) >= 3:
        corr = []
        for subset, frame in (("all", table), ("without dl, cnn", table[~table["model"].isin(["dl", "cnn"])])):
            for branch, col, label in (("Feature", "feat_overlap", "ATT accuracy (top-10 overlap)"), ("Feature", "feat_auc", "ATT-AUC"),
                                       ("Geo", "geo_overlap", "ATT accuracy (top 20 in regions)"), ("Geo", "geo_auc", "ATT-AUC")):
                if frame[col].nunique() > 1:
                    r, p = pearsonr(frame["r2"], frame[col])
                    corr.append({"subset": subset, "branch": branch, "measure": label, "M": len(frame), "pearson_r": r,
                                 "p_two_sided": p, "spearman_rho": spearmanr(frame["r2"], frame[col])[0]})
        pd.DataFrame(corr).to_csv(DEST / "models" / "att_faithfulness.csv", index=False)
    menus = {}
    for name in table["model"]:
        r = json.loads((DEST / "models" / f"{name}.json").read_text())
        for b in ("feature", "geo"):
            for m in r[f"{b}_menu"]:
                menus.setdefault(b, {}).setdefault(m["surrogate"], {})[name] = m["r2"]
    for b, m in menus.items():
        pd.DataFrame(m).T.to_csv(DEST / "models" / f"{b}_surrogate_menu_r2.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=("A", "B", "both"), default="both")
    ap.add_argument("--n-grid", type=int, nargs="+", default=list(N_GRID))
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    cols = players + SPATIAL_COLUMNS
    split = temporal_split(panel)
    train, val, test = split
    med = panel.loc[train, cols].median()
    geo_baseline = pd.Series(np.nan, index=cols)
    geo_baseline[SPATIAL_COLUMNS] = med[SPATIAL_COLUMNS]
    hull = sorted(panel["lsoa21cd"].astype(str).unique())
    names = panel.drop_duplicates("lsoa21cd").set_index("lsoa21cd")["lsoa21nm"].astype(str).to_dict()
    neighbors, extra = load_cached_neighbors()
    boundaries, _ = load_boundaries(hull)
    published = set(pd.read_csv(REGIONS_CSV)["lsoa21cd"].astype(str))
    rules = rule_sets(region_members(boundaries), published)
    area = pd.read_csv(AREA_DATA_COEF).set_index("lsoa21cd")["geo_coefficient"]
    for nm, s in rules.items():   # the rule table must reproduce Chapter 4 for the area-data ranking
        a = geo_accuracy(area.reset_index(), s)
        log(f"rule check (area-data): {nm}: {len(s)} LSOAs, top-20 inside {a['intersection']}")
    ctx = {"panel": panel, "players": players, "cols": cols, "split": split, "test": test,
           "X_test": panel.loc[test].reset_index(drop=True), "feat_baseline": med[players], "geo_baseline": geo_baseline,
           "hull": hull, "names": names, "neighbors": neighbors, "extra": extra, "ref": published, "rules": rules,
           "area_data": area, "feat_expert": feature_expert_binary_frame(players)}
    jdump(DEST / "design.json", {"players": players, "coordinates": SPATIAL_COLUMNS,
                                 "median_coordinate": med[SPATIAL_COLUMNS].to_dict(),
                                 "rule_set_sizes": {k: len(v) for k, v in rules.items()}})
    if args.part in ("A", "both"):
        raw = make_model("xgboost", SEED, n_jobs=4)
        fit_model("xgboost", raw, panel.loc[train, cols], panel.loc[train, Y_COL], panel.loc[val, cols], panel.loc[val, Y_COL])
        pred = np.asarray(raw.predict(panel.loc[test, cols].astype(float)), float)
        yt = panel.loc[test, Y_COL].to_numpy(float)
        ctx["predictive"] = {"r2": float(r2_score(yt, pred)), "mae": float(mean_absolute_error(yt, pred)),
                             "rmse": float(np.sqrt(mean_squared_error(yt, pred)))}
        log(f"XGBoost (30 + coordinates) holdout: {ctx['predictive']}")
        ctx["raw"], ctx["w"] = raw, Aligned(raw, players)
        part_a(ctx, args.n_grid)
    if args.part in ("B", "both"):
        part_b(ctx, args.models)


if __name__ == "__main__":
    main()
