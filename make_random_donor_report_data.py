"""Figures and table data for the random-donor (absolute) Chapter-4 report.

Reads outputs/geo_random_donor_design/ (Geo branch, geo_random_donor_eval.py) and
outputs/geo_aligned_design/ (Feature branch and predictive models of the same 32-input design), and
writes outputs/geo_random_donor_design/report/: figures (PNG) and report_data.json, which
build_random_donor_report.js turns into the Word report. Not causal.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import build_geo_design_review as b
from geosmile_hull_twobranch import FEATURE_LABELS, INK

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
GEO = OUT / "geo_random_donor_design"
FEAT = OUT / "geo_aligned_design"
REP = GEO / "report"
NS = (1000, 2000, 3000, 4000, 5000, 7500, 10000)
ORDER = ["gp", "transformer", "catboost", "lightgbm", "xgboost", "ann", "svr", "cnn", "dl", "mlp"]
LONG = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost", "gp": "Gaussian process", "svr": "SVR",
        "ann": "ANN", "cnn": "CNN", "mlp": "MLP", "transformer": "Transformer", "dl": "Keras (deep)"}
FAMILY = {"xgboost": "Tree ensemble", "lightgbm": "Tree ensemble", "catboost": "Tree ensemble", "gp": "Kernel", "svr": "Kernel",
          "ann": "Neural", "cnn": "Neural", "mlp": "Neural", "transformer": "Neural", "dl": "Neural"}
MENU = ["Weighted OLS", "Weighted Ridge", "Bayesian ridge", "Linear regression", "Ridge (unweighted)", "Elastic net (weak)",
        "Kernel ridge (RBF)", "Hist. grad. boosting", "Random forest", "Tweedie GLM", "Poisson GLM", "Unweighted OLS",
        "Ridge $\\alpha=0$", "Shallow tree", "Null, SD"]


def lab(f: str) -> str:
    return FEATURE_LABELS.get(f, f).replace("\n", " ")


def figures(g4: dict) -> dict:
    REP.mkdir(parents=True, exist_ok=True)
    figs = {}
    geo = pd.read_csv(GEO / "xgboost" / "geo_coefficients_N4000.csv", dtype={"lsoa21cd": str})
    gdf = b.geoframe(geo)
    outline = b.hull_outline(gdf)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    b.shade(fig, axes[0], gdf, "geo_coefficient", "Geo-SMILE geographic sensitivity coefficient", outline, top=10)
    axes[0].set_title("(a) Geo-SMILE geographic sensitivity coefficient", fontsize=12, fontweight="bold", color=INK)
    b.shade(fig, axes[1], gdf, "exact_w1", "W1, crimes per month", outline, start_at_min=False)
    axes[1].set_title("(b) Single-LSOA Wasserstein prediction shift\n(mean over 167 donor locations)", fontsize=12,
                      fontweight="bold", color=INK)
    reg = pd.read_csv(b.REGIONS_CSV, dtype={"lsoa21cd": str})[["lsoa21cd", "region"]]
    gdf.merge(reg, on="lsoa21cd").dissolve(by="region").boundary.plot(ax=axes[1], color=INK, linewidth=1.3)
    b.save(fig, REP, "geo_map", "Random-donor design: XGBoost on 30 predictors + BNG easting/northing; masking an LSOA "
           "relocates its rows to the centroids of random other Hull LSOAs (K = 50 draws), all predictors kept. N = 4,000, "
           "seed 42. Top 10 numbered in (a); (b) black outlines = expert regions (map 1). Sequential scales capped at the "
           "98th percentile; the sign is not a direction of change. Not causal effects.")
    figs["geo_map"] = str(REP / "geo_map.png")

    imp = pd.read_csv(FEAT / "xgboost" / "feature_global_importance_N4000.csv").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([lab(f) for f in imp["feature"]], imp["abs_coefficient"], color="#B5003C")
    for y, v in enumerate(imp["abs_coefficient"]):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_xlabel("|global Feature-SMILE coefficient|")
    ax.set_title("Leading global Feature-SMILE drivers (N = 4,000)", fontsize=12, fontweight="bold", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(REP / "feature_global.png", dpi=220, facecolor="white")
    plt.close(fig)
    figs["feature_global"] = str(REP / "feature_global.png")
    shutil.copy2(OUT / "hull_geo_design_review" / "05_aligned_design" / "feature_local_top6_N4000.png", REP / "feature_local.png")
    figs["feature_local"] = str(REP / "feature_local.png")
    shutil.copy2(OUT / "geo_area_data_models" / "top50_maps" / "highlighted_regions_check.png", REP / "expert_map1.png")
    shutil.copy2(OUT / "hull_geo_design_review" / "08_expert_map_v2" / "expert_map_v2_check.png", REP / "expert_map2.png")
    shutil.copy2(OUT / "hull_geo_design_review" / "09_expert_map_v3" / "expert_map_v3_check.png", REP / "expert_map3.png")
    figs.update({f"expert_map{i}": str(REP / f"expert_map{i}.png") for i in (1, 2, 3)})

    fx = {n: json.loads((FEAT / "xgboost" / f"n_{n}.json").read_text())["feature"] for n in NS}
    gx = {n: json.loads((GEO / "xgboost" / f"n_{n}.json").read_text()) for n in NS}
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    series = [("Fidelity R²", lambda d: d["fidelity"]["R2"]), ("Intervention ΔW1 (top − random)", lambda d: d["intervention"]["top_minus_random"]),
              ("Stability Jaccard", lambda d: d["stability"]["jaccard_mean"]), ("Consistency rank ρ", lambda d: d["consistency"]["rank_spearman_mean"])]
    for ax, (title, fn) in zip(axes, series):
        ax.plot(NS, [fn(fx[n]) for n in NS], "o-", color="#1E88E5", label="Feature")
        ax.plot(NS, [fn(gx[n]) for n in NS], "s-", color="#B5003C", label="Geo")
        ax.axvline(4000, color="#999", ls=":", lw=1)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("N")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Evaluation diagnostics across perturbation budgets (dotted line: N = 4,000)", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(REP / "metrics_by_n.png", dpi=220, facecolor="white")
    plt.close(fig)
    figs["metrics_by_n"] = str(REP / "metrics_by_n.png")
    return figs


def main() -> None:
    fx = {n: json.loads((FEAT / "xgboost" / f"n_{n}.json").read_text()) for n in NS}
    gx = {n: json.loads((GEO / "xgboost" / f"n_{n}.json").read_text()) for n in NS}
    fm = {m: json.loads((FEAT / "models" / f"{m}.json").read_text()) for m in ORDER}
    gm = {m: json.loads((GEO / "models" / f"{m}.json").read_text()) for m in ORDER}
    F, G = fx[4000]["feature"], gx[4000]
    data = {
        "K": json.loads((GEO / "k_selection.json").read_text()),
        "predictive": {m: fm[m]["predictive"] | {"family": FAMILY[m], "name": LONG[m]} for m in ORDER},
        "order": ORDER, "long": LONG,
        "feature_menu": {s: {m: next(x for x in fm[m]["feature_menu"] if x["surrogate"] == s) for m in ORDER} for s in MENU},
        "geo_menu": {s: {m: next(x for x in gm[m]["geo_menu"] if x["surrogate"] == s) for m in ORDER} for s in MENU},
        "feature": F, "geo": G,
        "feature_importance": [{"feature": lab(x["feature"]), "coef": x["coefficient"]} for x in F["importance"][:10]],
        "by_n": [{"n": n, "feature": {"R2": fx[n]["feature"]["fidelity"]["R2"], "wR2": fx[n]["feature"]["fidelity"]["Weighted R2"],
                                      "acc": fx[n]["feature"]["accuracy"]["intersection"], "dw": fx[n]["feature"]["intervention"]["top_minus_random"],
                                      "stab": fx[n]["feature"]["stability"]["jaccard_mean"], "var": fx[n]["feature"]["consistency"]["mean_variance"],
                                      "sd": fx[n]["feature"]["consistency"]["mean_sd"], "rho": fx[n]["feature"]["consistency"]["rank_spearman_mean"]},
                  "geo": {"R2": gx[n]["fidelity"]["R2"], "wR2": gx[n]["fidelity"]["Weighted R2"], "acc": gx[n]["accuracy"]["intersection"],
                          "dw": gx[n]["intervention"]["top_minus_random"], "stab": gx[n]["stability"]["jaccard_mean"],
                          "var": gx[n]["consistency"]["mean_variance"], "sd": gx[n]["consistency"]["mean_sd"],
                          "rho": gx[n]["consistency"]["rank_spearman_mean"]}} for n in NS],
        "models": pd.read_csv(GEO / "models" / "model_summary.csv").to_dict("records"),
        "faithfulness": pd.read_csv(GEO / "models" / "att_faithfulness.csv").to_dict("records"),
        "figures": figures(G),
    }
    (REP / "report_data.json").write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
    print("wrote", REP / "report_data.json")


if __name__ == "__main__":
    main()
