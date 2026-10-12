"""Assemble data.json for build_report.js from the project outputs.

Run from the repo root after select_highlighted_region_lsoas.py,
eval_expert_region_accuracy.py and geo_consistency_per_lsoa.py:

    python outputs/geo_area_data_models/report/build_report_data.py
    node outputs/geo_area_data_models/report/build_report.js [out.docx]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
ROOT = SRC.parent.parent
sys.path.insert(0, str(ROOT))
from map_geo_importance_ranks import rank_by  # noqa: E402

MODELS = ["xgboost", "lightgbm", "catboost", "svr", "gp", "mlp", "dl", "ann", "cnn", "transformer"]
FAMILY = {"xgboost": "Boosting", "lightgbm": "Boosting", "catboost": "Boosting", "svr": "Kernel", "gp": "Kernel"}
MENU_ORDER = ["Weighted Ridge", "Weighted OLS", "Ridge (α = 0)", "Bayesian ridge", "Linear regression",
              "Unweighted OLS", "Ridge (unweighted)", "Hist. grad. boosting", "Random forest", "Shallow tree",
              "Kernel ridge (RBF)", "Elastic net (weak)", "Poisson GLM", "Tweedie GLM", "Null, SD"]


def main() -> None:
    d: dict = {}
    table = pd.read_csv(SRC / "geo_model_table.csv").set_index("model")
    fid = pd.read_csv(SRC / "geo_weighted_ridge_fidelity.csv").set_index("model")
    menu = pd.read_csv(SRC / "geo_surrogate_menu_long.csv")
    wr = menu[menu["surrogate"] == "Weighted Ridge"].set_index("model")
    d["fidelity"] = [{"model": m, "family": FAMILY.get(m, "Neural"), "pred_r2": table.loc[m, "predictive_r2"],
                      "r2": fid.loc[m, "r2"], "wr2": wr.loc[m, "weighted_r2"], "mae": fid.loc[m, "mae"],
                      "wmae": fid.loc[m, "wmae"], "rmse": fid.loc[m, "rmse"], "wrmse": fid.loc[m, "wrmse"],
                      "kernel": wr.loc[m, "best_kernel"]} for m in MODELS]
    wide = pd.read_csv(SRC / "geo_surrogate_menu_r2_wide.csv").set_index("surrogate")
    wide.index = [re.sub(r"\$.*alpha=0\$", "(α = 0)", i) for i in wide.index]
    d["menu"] = {"models": MODELS, "rows": [{"surrogate": s, "vals": [wide.loc[s, m] for m in MODELS]} for s in MENU_ORDER]}

    acc = pd.read_csv(SRC / "expert_region_accuracy" / "accuracy_by_model_k.csv")
    summ = pd.read_csv(SRC / "expert_region_accuracy" / "accuracy_summary.csv").set_index("model")
    cols = ["hits", "accuracy", "lift", "p_hypergeom", "recall", "att_f1", "att_acc", "jaccard"]
    d["accuracy"] = [{"model": m, "pred_r2": summ.loc[m, "predictive_r2"], "auroc": summ.loc[m, "att_auroc"],
                      "k": {int(r.k): {c: r[c] for c in cols} for _, r in acc[acc.model == m].iterrows()}} for m in MODELS]
    d["faith"] = pd.read_csv(SRC / "expert_region_accuracy" / "faithfulness_correlation.csv").to_dict(orient="records")

    maps = SRC / "top50_maps"
    for k, name in ((50, "highlighted_regions_overlap_by_model.csv"), (42, "highlighted_regions_overlap_by_model_top42.csv")):
        o = pd.read_csv(maps / name)
        d[f"overlap{k}"] = {"cols": list(o.columns), "rows": o.values.tolist()}

    ref = pd.read_csv(maps / "highlighted_regions_lsoas.csv")
    n_all = 168
    KS = (20, 42, 50, len(ref))
    d["ref"] = ref.to_dict(orient="records")
    regions = (ref.groupby("region")
               .agg(n=("lsoa21cd", "size"), exp=("expert_top20", "sum"),
                    all10=("models_with_lsoa_in_top50", lambda s: int((s == 10).sum())))
               .reset_index().sort_values("n", ascending=False))
    d["ref_summary"] = regions.to_dict(orient="records")
    n_ref = len(ref)
    d["ref_meta"] = {
        "n": n_ref, "n_all": n_all, "chance": n_ref / n_all, "expert_top20_inside": int(ref["expert_top20"].sum()),
        "ks": list(KS),
        "expected_hits": {k: k * n_ref / n_all for k in KS},
        "max_p": max((a["k"][k]["p_hypergeom"], a["model"], k) for a in d["accuracy"] for k in KS),
    }

    # XGBoost narrative facts
    res = json.loads((SRC / "xgboost.json").read_text(encoding="utf-8"))
    ranked = rank_by(pd.DataFrame(res["coefficients"]), "geo_coefficient", "rank")
    m = ranked.merge(ref[["lsoa21cd", "region"]], on="lsoa21cd", how="left")
    d["xgb_facts"] = {
        "by_region": {k: m[(m["rank"] <= k) & m["region"].notna()]["region"].value_counts().to_dict() for k in (20, 42)},
        "top20_misses": m[(m["rank"] <= 20) & m["region"].isna()][["rank", "lsoa21cd"]].values.tolist(),
        "lowest_ref_ranks": sorted(m[m["region"].notna()]["rank"].nlargest(6).tolist()),
    }

    x = json.loads((ROOT / "outputs" / "geo_area_data_eval" / "n_4000" / "metrics.json").read_text(encoding="utf-8"))
    d["xgb"] = {k: x[k] for k in ("fidelity", "faithfulness", "stability", "consistency")}
    d["nsweep"] = pd.read_csv(ROOT / "outputs" / "geo_area_data_eval" / "summary_by_n.csv").to_dict(orient="records")

    c = pd.read_csv(ROOT / "outputs" / "geo_area_data_eval" / "n_4000" / "consistency_per_lsoa.csv")
    c["expert_region"] = c["lsoa21cd"].isin(set(ref["lsoa21cd"])).astype(int)  # current reference
    d["cons_top10"] = c.head(10)[["rank_by_mean", "lsoa21cd", "mean_coefficient", "sd", "variance",
                                   "sd_pct_of_abs_mean", "expert_region"]].to_dict(orient="records")
    groups = [("Top 20 (by mean coefficient)", c[c.rank_by_mean <= 20]), ("Top 42", c[c.rank_by_mean <= 42]),
              ("Ranks 43–168", c[c.rank_by_mean > 42]), ("Expert-region LSOAs", c[c.expert_region == 1]),
              ("Other LSOAs", c[c.expert_region == 0]), ("All LSOAs", c)]
    d["cons_groups"] = [{"group": g, "n": len(v), "mean_var": v.variance.mean(), "mean_sd": v.sd.mean(),
                         "median_sd": v.sd.median(), "max_sd": v.sd.max(), "med_rel": v.sd_pct_of_abs_mean.median()}
                        for g, v in groups]

    from PIL import Image
    d["images"] = {f: Image.open(HERE / f).size for f in ("regions.jpg", "top50_panel.jpg", "faithfulness.jpg", "xgb_top50.jpg")}

    (HERE / "data.json").write_text(json.dumps(d, default=float, indent=1), encoding="utf-8")
    print(f"data.json written: {n_ref} reference LSOAs, chance {n_ref / n_all:.3f}")


if __name__ == "__main__":
    main()
