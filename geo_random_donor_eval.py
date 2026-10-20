"""Chapter-4 Geo-branch evaluation for the random-donor (absolute) Hull design.

Design: one model on 32 inputs (30 predictors + BNG easting/northing; model_factory, seed 42). Geo
branch: players = the 168 LSOAs; in each of N coalitions (5-10 LSOAs, seed 42) every masked LSOA's
test-period rows get the centroid coordinates of a randomly drawn OTHER Hull LSOA, all 30 predictors
kept; response = mean over K donor draws of W1 between the affected rows' original and relocated
predicted counts (crimes/month). Kernel-weighted Ridge, 60/20/20 split, RBF/Laplace/inverse.
The Feature branch is the aligned design's (outputs/geo_aligned_design/), unchanged.

Stage K : choose K (5, 10, 20, 50) on XGBoost at N=4,000 (fidelity, 5-seed consistency, Spearman with
          K=50).
Stage A : XGBoost, N grid: fidelity, accuracy (expert map 1; at N=4,000 also maps v2/v3, k=30/42/50 and
          rule robustness), intervention impact (top/bottom/1,000 population-area-matched random
          20-sets), ATT stability (one neighbouring LSOA per cardinal direction; fixed-kernel west at
          4,000), consistency (seeds 100-104); at N=4,000 the single-LSOA expectation over all donors,
          concentration, Moran/LISA and comparisons.
Stage B : ten models at N=4,000: Geo surrogate menu, weighted-Ridge metrics, accuracy, ATT
          Faithfulness (Geo here; Feature from the aligned design).

Writes outputs/geo_random_donor_design/. Not causal.
    python geo_random_donor_eval.py [--stage K A B] [--k K] [--n-grid ...] [--models ...]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from itertools import combinations
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import compute_geo_spatial_patterns as sp
import hull_geo_area_data_eval as ev
from eval_by_n_perturb import dehghani_fidelity, load_cached_neighbors
from geo_area_data import surrogate_menu
from geosmile_hull_twobranch import SPATIAL_COLUMNS, apply_kernel, choose_feature_players, temporal_split
from hull_aligned_design_eval import REGIONS_CSV, geo_accuracy, region_members, rule_sets
from hull_boundaries import load_boundaries
from hull_boundary_neighbors import expand_panel
from hull_dehghani_eval import sparsity_entropy_metrics
from model_factory import MODEL_NAMES, fit_model, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "geo_random_donor_design"
Y, SEED = "target_next_month_crime", 42
N_GRID = (1000, 2000, 3000, 4000, 5000, 7500, 10000)
CONS = tuple(range(100, 105))
DIRS = ("north", "east", "south", "west")
REVIEW = OUT / "hull_geo_design_review"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ev.to_json(obj), indent=2, default=float), encoding="utf-8")


class Ctx:
    """Rows to explain, LSOA order (core first, extras after) and donor centroids."""

    def __init__(self, model, panel: pd.DataFrame, cols: list[str], core: list[str], extras: list[str] = ()):
        _, _, test = temporal_split(panel)
        frame = panel.loc[test].reset_index(drop=True)
        self.model, self.cols = model, cols
        self.x = frame[cols].to_numpy(float)
        self.orig = self.predict(self.x)
        ids = frame["lsoa21cd"].astype(str).to_numpy()
        self.lsoas = np.array(list(core) + list(extras), dtype=object)
        self.n_core = len(core)
        self.idx = [np.flatnonzero(ids == c) for c in self.lsoas]
        cent = panel.drop_duplicates("lsoa21cd").set_index("lsoa21cd")
        self.cent = cent.loc[self.lsoas, SPATIAL_COLUMNS].to_numpy(float)
        self.cix = [cols.index(c) for c in SPATIAL_COLUMNS]

    def predict(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(pd.DataFrame(rows, columns=self.cols)), float).ravel()

    def response(self, sel, rng, k: int) -> float:
        aff = np.concatenate([self.idx[s] for s in sel])
        base, blocks, L = self.x[aff], [], len(self.lsoas)
        for _ in range(k):
            p = base.copy()
            pos = 0
            for s in sel:
                donor = rng.integers(L - 1)
                donor += donor >= s
                p[pos:pos + len(self.idx[s]), self.cix] = self.cent[donor]
                pos += len(self.idx[s])
            blocks.append(p)
        pred = self.predict(np.vstack(blocks)).reshape(k, len(aff))
        return float(np.mean([wasserstein_distance(self.orig[aff], pred[j]) for j in range(k)]))

    def pool(self, n: int, seed: int, k: int, kernel: str | None = None):
        rng, drng = np.random.default_rng(seed), np.random.default_rng(seed + 7_000_003)
        xrng = np.random.default_rng(seed + 1_000_003)
        L, nc = len(self.lsoas), self.n_core
        M, y = np.zeros((n, L), np.int8), np.zeros(n)
        for j in range(n):
            n_sel = int(rng.integers(5, 11))
            sel = list(rng.choice(nc, size=n_sel, replace=False))
            sel += [nc + e for e in range(L - nc) if xrng.random() < n_sel / nc]
            M[j, sel] = 1
            y[j] = self.response(sel, drng, k)
        d = M[:, :nc].sum(1) / nc
        order = rng.permutation(n)
        tr, va = order[:int(.6 * n)], order[int(.6 * n):int(.8 * n)]
        sigma = max(float(np.median(d[tr])), 1 / nc)
        sc = {kk: r2_score(y[va], Ridge(alpha=1.0).fit(M[tr], y[tr], sample_weight=apply_kernel(d[tr], kk, sigma)).predict(M[va]))
              for kk in ("rbf", "laplace", "inverse")}
        best = kernel or max(sc, key=sc.get)
        trva = order[:int(.8 * n)]
        r = Ridge(alpha=1.0).fit(M[trva], y[trva], sample_weight=apply_kernel(d[trva], best, sigma))
        coef = pd.DataFrame({"lsoa21cd": self.lsoas.astype(str), "geo_coefficient": r.coef_})
        coef["geo_abs_coefficient"] = coef["geo_coefficient"].abs()
        coef["geo_score"] = coef["geo_coefficient"]
        return coef, M, y, d, {"best_kernel": best, "validation_r2_by_kernel": sc}

    def exact(self) -> pd.Series:
        out, L = {}, len(self.lsoas)
        for i in range(self.n_core):
            rows, stack = self.x[self.idx[i]], []
            for dnr in range(L):
                if dnr != i:
                    p = rows.copy()
                    p[:, self.cix] = self.cent[dnr]
                    stack.append(p)
            pred = self.predict(np.vstack(stack)).reshape(L - 1, len(rows))
            out[self.lsoas[i]] = float(np.mean([wasserstein_distance(self.orig[self.idx[i]], pred[j]) for j in range(L - 1)]))
        return pd.Series(out)


class Responder:
    """Callable used by hull_geo_area_data_eval.faithfulness (intervention impact)."""

    def __init__(self, ctx: Ctx, k: int, seed: int):
        self.ctx, self.k, self.rng = ctx, k, np.random.default_rng(seed + 11)
        self.pos = {c: i for i, c in enumerate(ctx.lsoas)}

    def __call__(self, lsoas) -> float:
        return self.ctx.response([self.pos[str(c)] for c in lsoas], self.rng, self.k)


def consistency(ctx: Ctx, n: int, k: int, names: dict) -> dict:
    mat = pd.concat([ctx.pool(n, s, k)[0].set_index("lsoa21cd")["geo_coefficient"] for s in CONS], axis=1)
    var, sd = mat.var(axis=1, ddof=1), mat.std(axis=1, ddof=1)
    rr, pr, jj = [], [], []
    for a, b in combinations(range(mat.shape[1]), 2):
        x, y = mat.iloc[:, a], mat.iloc[:, b]
        rr.append(spearmanr(x, y)[0]); pr.append(pearsonr(x, y)[0])
        ta, tb = set(x[x >= x.quantile(.8)].index), set(y[y >= y.quantile(.8)].index)
        jj.append(len(ta & tb) / len(ta | tb))
    w = sd.idxmax()
    return {"mean_variance": float(var.mean()), "median_variance": float(var.median()), "mean_sd": float(sd.mean()),
            "median_sd": float(sd.median()), "max_sd": float(sd.max()), "max_sd_lsoa": str(w),
            "max_sd_lsoa_name": names.get(w, w), "max_sd_lsoa_mean_coefficient": float(mat.loc[w].mean()),
            "max_sd_pct_of_mean": float(100 * sd.max() / abs(mat.loc[w].mean())),
            "rank_spearman_mean": float(np.mean(rr)), "score_pearson_mean": float(np.mean(pr)),
            "top20pct_jaccard_mean": float(np.mean(jj))}


def stability(model, panel, cols, hull, base: pd.DataFrame, neighbors, extra, n, k, kernel=None) -> dict:
    btop = set(ev.top_k(base))
    out, js, rs = {}, [], []
    for dname in DIRS:
        code = str(neighbors.loc[neighbors["direction"] == dname, "lsoa21cd"].iloc[0])
        exp = expand_panel(panel, extra[extra["lsoa21cd"].astype(str) == code])
        c, *_ = Ctx(model, exp, cols, hull, [code]).pool(n, SEED, k, kernel)
        r = ev.ranked(c)
        top = set(ev.top_k(r))
        m = base[["lsoa21cd", "geo_score"]].merge(c[c.lsoa21cd.isin(hull)][["lsoa21cd", "geo_score"]], on="lsoa21cd")
        j = len(btop & top) / len(btop | top)
        s = float(spearmanr(m["geo_score_x"], m["geo_score_y"])[0])
        hit = r[r.lsoa21cd == code].iloc[0]
        out[dname] = {"lsoa21cd": code, "jaccard": j, "spearman": s, "top20_retained": len(btop & top),
                      "added_rank": int(hit["geo_rank"]), "added_coefficient": float(hit["geo_coefficient"]),
                      "added_in_top20": bool(hit["geo_rank"] <= 20)}
        js.append(j); rs.append(s)
        log(f"    stability {dname}: J={j:.3f} rho={s:.4f} added rank {int(hit['geo_rank'])}")
    return {"directions": out, "jaccard_mean": float(np.mean(js)), "spearman_mean": float(np.mean(rs))}


def setup():
    panel = pd.read_csv(OUT / "hull_lsoa_month_panel_with_covariates.csv")
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    cols = players + SPATIAL_COLUMNS
    tr, va, te = temporal_split(panel)
    hull = sorted(panel["lsoa21cd"].astype(str).unique())
    names = panel.drop_duplicates("lsoa21cd").set_index("lsoa21cd")["lsoa21nm"].astype(str).to_dict()
    b, _ = load_boundaries(hull)
    map1 = set(pd.read_csv(REGIONS_CSV)["lsoa21cd"].astype(str))
    refs = {"map1": map1,
            "map2": set(pd.read_csv(REVIEW / "08_expert_map_v2" / "lsoa_overlap_v2.csv", index_col=0).query("in_reference").index),
            "map3": set(pd.read_csv(REVIEW / "09_expert_map_v3" / "lsoa_overlap_v3.csv", index_col=0).query("in_primary").index)}
    return panel, cols, (tr, va, te), hull, names, refs, rule_sets(region_members(b), map1)


def fit(name, panel, cols, split):
    tr, va, te = split
    model = make_model(name, SEED, n_jobs=4)
    fit_model(name, model, panel.loc[tr, cols], panel.loc[tr, Y], panel.loc[va, cols], panel.loc[va, Y])
    return model


def stage_k(panel, cols, split, hull, names) -> int:
    path = DEST / "k_selection.json"
    if path.exists():
        return int(json.loads(path.read_text())["chosen_K"])
    ctx = Ctx(fit("xgboost", panel, cols, split), panel, cols, hull)
    coefs, rows = {}, []
    for k in (5, 10, 20, 50):
        t0 = time.time()
        coef, M, y, d, _ = ctx.pool(4000, SEED, k)
        cons = consistency(ctx, 4000, k, names)
        coefs[k] = coef.set_index("lsoa21cd")["geo_coefficient"]
        rows.append({"K": k, "fidelity_R2": dehghani_fidelity(M, y, d)["R2"], "consistency_rank_rho": cons["rank_spearman_mean"],
                     "consistency_mean_sd": cons["mean_sd"], "minutes": (time.time() - t0) / 60})
        log(f"K={k}: {rows[-1]}")
    t = pd.DataFrame(rows)
    t["rho_with_K50"] = [float(spearmanr(coefs[k], coefs[50])[0]) for k in t["K"]]
    ok = t[(t.fidelity_R2 >= t.fidelity_R2.max() - 0.005) & (t.consistency_rank_rho >= t.consistency_rank_rho.max() - 0.005)
           & (t.rho_with_K50 >= 0.99)]
    k_star = int(ok["K"].min()) if len(ok) else 50
    t.to_csv(DEST / "k_selection.csv", index=False)
    jdump(path, {"chosen_K": k_star, "rule": "smallest K with fidelity and consistency within 0.005 of the best and "
                 "Spearman >= 0.99 with K=50", "table": t.to_dict("records")})
    log(f"chosen K = {k_star}")
    return k_star


def stage_a(panel, cols, split, hull, names, refs, rules, k, n_grid) -> None:
    model = fit("xgboost", panel, cols, split)
    ctx = Ctx(model, panel, cols, hull)
    neighbors, extra = load_cached_neighbors()
    (DEST / "xgboost").mkdir(parents=True, exist_ok=True)
    for n in n_grid:
        path = DEST / "xgboost" / f"n_{n}.json"
        if path.exists():
            log(f"N={n} cached"); continue
        t0 = time.time(); log(f"=== Stage A, N={n}, K={k}")
        coef, M, y, d, diag = ctx.pool(n, SEED, k)
        res = {"n": n, "K": k, "smile_diagnostics": diag, "fidelity": dehghani_fidelity(M, y, d),
               "mean_W1_response": float(y.mean()), "accuracy": geo_accuracy(coef, refs["map1"]),
               "intervention": ev.faithfulness(Responder(ctx, k, SEED), coef, SEED),
               "stability": stability(model, panel, cols, hull, coef, neighbors, extra, n, k),
               "consistency": consistency(ctx, n, k, names)}
        if n == 4000:
            ex = ctx.exact()
            c = coef.set_index("lsoa21cd").join(ex.rename("exact_w1"))
            r = ev.ranked(coef).merge(ex.rename("exact_w1"), left_on="lsoa21cd", right_index=True)
            r["lsoa21nm"] = r["lsoa21cd"].map(names)
            r.to_csv(DEST / "xgboost" / "geo_coefficients_N4000.csv", index=False)
            fixed, *_ = ctx.pool(n, SEED, k, "inverse")
            res["fixed_kernel_west"] = stability(model, panel, cols, hull, fixed, neighbors, extra, n, k, "inverse")["directions"]["west"]
            sp.COEF_PATH, sp.VALUE_FIELD = DEST / "xgboost" / "geo_coefficients_N4000.csv", "geo_coefficient"
            gdf = sp.attach_boundaries(sp.load_geo_frame())
            wq, islands = sp.queen_weights(gdf)
            order, _, gm, _, labels = sp.run_moran_lisa(gdf[~gdf["lsoa21cd"].isin(islands)].copy(), wq)
            lab = pd.Series(labels, index=order)
            _, _, te = split
            crime = panel.loc[te].groupby("lsoa21cd")["crime_count"].mean()
            other = {"area_data": OUT / "geo_area_data_eval" / "n_4000" / "geo_coefficients.csv",
                     "median_point": OUT / "geo_aligned_design" / "xgboost" / "geo_coefficients_N4000.csv"}
            res.update({
                "exact_vs_signed_spearman": float(spearmanr(c["geo_coefficient"], c["exact_w1"])[0]),
                "exact_vs_abs_spearman": float(spearmanr(c["geo_coefficient"].abs(), c["exact_w1"])[0]),
                "exact_w1_median": float(ex.median()), "exact_w1_p90": float(ex.quantile(.9)),
                "concentration_exact_shift": sparsity_entropy_metrics(ex.to_numpy()),
                "n_negative": int((coef["geo_coefficient"] < 0).sum()),
                "top10": r.head(10)[["geo_rank", "lsoa21cd", "lsoa21nm", "geo_coefficient", "exact_w1"]].to_dict("records"),
                "moran": {"I": float(gm.I), "p_sim": float(gm.p_sim), "lisa_counts": lab.value_counts().to_dict(),
                          "HH": [f"{x} {names.get(x, '')}" for x in lab[lab == "HH"].index]},
                "accuracy_k": {f"map1_k{kk}": geo_accuracy(coef, refs["map1"], kk) for kk in (30, 42, 50)},
                "accuracy_rule_robustness": {nm: geo_accuracy(coef, s) for nm, s in rules.items()},
                "accuracy_other_maps": {f"{m}_k{kk}": geo_accuracy(coef, refs[m], kk) for m in ("map2", "map3") for kk in (20, 30, 42)},
                "rho_with_crime_level": float(spearmanr(c["geo_coefficient"], crime.loc[c.index])[0]),
                "comparisons": {nm: float(spearmanr(c["geo_coefficient"], pd.read_csv(p).set_index("lsoa21cd")["geo_coefficient"].loc[c.index])[0])
                                for nm, p in other.items()},
            })
        res["elapsed_min"] = (time.time() - t0) / 60
        jdump(path, res)
        log(f"N={n} done ({res['elapsed_min']:.1f} min): R2={res['fidelity']['R2']:.4f} acc={res['accuracy']['intersection']}/20 "
            f"interv={res['intervention']['top_minus_random']:.3f} stabJ={res['stability']['jaccard_mean']:.3f} "
            f"cons rho={res['consistency']['rank_spearman_mean']:.3f}")


def stage_b(panel, cols, split, hull, refs, k, models) -> None:
    for name in models:
        path = DEST / "models" / f"{name}.json"
        if path.exists():
            log(f"[{name}] cached"); continue
        t0 = time.time()
        ctx = Ctx(fit(name, panel, cols, split), panel, cols, hull)
        _, _, te = split
        r2 = float(r2_score(panel.loc[te, Y], ctx.orig))
        coef, M, y, d, _ = ctx.pool(4000, SEED, k)
        jdump(path, {"model": name, "predictive_r2": r2, "geo_menu": surrogate_menu(M, y, d),
                     "accuracy": {m: geo_accuracy(coef, s) for m, s in refs.items()}, "mean_W1": float(y.mean()),
                     "coefficients": coef[["lsoa21cd", "geo_coefficient"]].to_dict("list"), "minutes": (time.time() - t0) / 60})
        log(f"[{name}] R2={r2:.4f} Geo map1 {geo_accuracy(coef, refs['map1'])['intersection']}/20 ({(time.time() - t0) / 60:.1f} min)")
    feat = pd.read_csv(OUT / "geo_aligned_design" / "models" / "model_summary.csv").set_index("model")
    rows = []
    for name in MODEL_NAMES:
        p = DEST / "models" / f"{name}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        ridge = next(m for m in r["geo_menu"] if m["surrogate"] == "Weighted Ridge")
        row = {"model": name, "r2": r["predictive_r2"], "geo_ridge_r2": ridge["r2"]}
        for m in ("map1", "map2", "map3"):
            row[f"geo_overlap_{m}"] = r["accuracy"][m]["intersection"]
            row[f"geo_auc_{m}"] = r["accuracy"][m]["att_auroc"]
        row.update({"feat_overlap": feat.loc[name, "feat_overlap"], "feat_auc": feat.loc[name, "feat_auc"]})
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(DEST / "models" / "model_summary.csv", index=False)
    corr = []
    for subset, fr in (("all", t), ("without dl, cnn", t[~t.model.isin(["dl", "cnn"])])):
        for col, label in (("feat_overlap", "Feature ATT accuracy (top-10)"), ("feat_auc", "Feature ATT-AUC"),
                           ("geo_overlap_map1", "Geo ATT accuracy (top 20, map 1)"), ("geo_auc_map1", "Geo ATT-AUC (map 1)"),
                           ("geo_overlap_map2", "Geo ATT accuracy (top 20, map 2)"), ("geo_overlap_map3", "Geo ATT accuracy (top 20, map 3)")):
            if len(fr) >= 3 and fr[col].nunique() > 1:
                rr, pp = pearsonr(fr["r2"], fr[col])
                corr.append({"subset": subset, "measure": label, "M": len(fr), "pearson_r": rr, "p_two_sided": pp,
                             "spearman_rho": spearmanr(fr["r2"], fr[col])[0]})
    pd.DataFrame(corr).to_csv(DEST / "models" / "att_faithfulness.csv", index=False)
    menus = {}
    for name in t["model"]:
        for m in json.loads((DEST / "models" / f"{name}.json").read_text())["geo_menu"]:
            menus.setdefault(m["surrogate"], {})[name] = m
    jdump(DEST / "models" / "geo_menu_all.json", menus)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", nargs="+", default=["K", "A", "B"])
    ap.add_argument("--k", type=int)
    ap.add_argument("--n-grid", type=int, nargs="+", default=list(N_GRID))
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    a = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    panel, cols, split, hull, names, refs, rules = setup()
    k = a.k or (stage_k(panel, cols, split, hull, names) if "K" in a.stage else 10)
    if "A" in a.stage:
        stage_a(panel, cols, split, hull, names, refs, rules, k, a.n_grid)
    if "B" in a.stage:
        stage_b(panel, cols, split, hull, refs, k, a.models)


if __name__ == "__main__":
    main()
