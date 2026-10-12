"""Chapter-4 ten-model Geo comparisons under the area-data design.

For each candidate in model_factory.MODEL_NAMES: fit the 30-feature model
(seed 42) -- under this design the Geo branch explains the same model as the
Feature branch -- build one Geo pool (N=4,000, seed 42) with
geo_area_data.area_geo_coefficients, then
  * score the full Chapter-4 surrogate menu (Table "Geo branch fidelity"),
  * record the weighted-Ridge fidelity metrics (Table "Weighted Ridge ... all ten"),
  * score the signed Geo ranking against the fixed expert top-20
    (top-20 overlap and ATT-AUC).
Finally ATT Faithfulness = Pearson/Spearman correlation between each model's
predictive R2 and its Geo attribution quality, over all models.

Writes outputs/geo_area_data_models/. Per-model results are cached, so an
interrupted run resumes. Not causal.

    python hull_geo_area_data_models.py [--models xgboost gp ...] [--force]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

from geo_area_data import area_geo_coefficients, surrogate_menu
from geosmile_hull_twobranch import SPATIAL_COLUMNS, choose_feature_players, temporal_split
from hull_expert_topk import geo_expert_binary_frame
from hull_geo_area_data_eval import accuracy
from model_factory import MODEL_NAMES, fit_model, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "geo_area_data_models"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"
Y_COL = "target_next_month_crime"
SEED = 42
N_PERTURB = 4000
FAMILY = {
    "xgboost": "Tree ensemble", "lightgbm": "Tree ensemble", "catboost": "Tree ensemble",
    "gp": "Kernel", "svr": "Kernel",
    "ann": "Neural", "cnn": "Neural", "mlp": "Neural", "transformer": "Neural", "dl": "Neural",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_model(name, panel, players, split, baseline, expert, n_jobs) -> dict:
    train, validation, test = split
    t0 = time.time()
    model = make_model(name, SEED, n_jobs=n_jobs)
    fit_model(name, model, panel.loc[train, players], panel.loc[train, Y_COL],
              panel.loc[validation, players], panel.loc[validation, Y_COL])
    pred = np.asarray(model.predict(panel.loc[test, players].astype(float)), float)
    r2 = float(r2_score(panel.loc[test, Y_COL], pred))
    log(f"[{name}] predictive R2 (30-feature model) = {r2:.4f}; building Geo pool N={N_PERTURB}")
    coef, diag, M, y, d = area_geo_coefficients(model, panel, test, baseline, players, N_PERTURB, SEED, return_pool=True)
    menu = surrogate_menu(M, y, d)
    acc = accuracy(coef, expert)
    ridge = next(r for r in menu if r["surrogate"] == "Weighted Ridge")
    log(f"[{name}] Geo weighted-Ridge R2={ridge['r2']:.4f}  overlap {acc['intersection']}/20  "
        f"ATT-AUC={acc['att_auroc']:.4f}  ({(time.time() - t0) / 60:.1f} min)")
    return {
        "model": name,
        "family": FAMILY[name],
        "predictive_r2": r2,
        "smile_diagnostics": diag,
        "menu": menu,
        "accuracy": acc,
        "coefficients": coef[["lsoa21cd", "geo_coefficient"]].to_dict(orient="list"),
    }


def correlations(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for measure, label in (("overlap", "Top-20 reference overlap"), ("att_auc", "ATT-AUC")):
        x, y = table["predictive_r2"], table[measure]
        r, p = pearsonr(x, y)
        rows.append({
            "branch": "Geo",
            "measure": label,
            "M": len(table),
            "pearson_r": r,
            "p_two_sided": p,
            "spearman_rho": spearmanr(x, y).statistic,
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    p.add_argument("--force", action="store_true")
    p.add_argument("--n-jobs", type=int, default=4)
    args = p.parse_args()

    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    split = temporal_split(panel)
    baseline = panel.loc[split[0], players + SPATIAL_COLUMNS].median()
    expert = geo_expert_binary_frame(sorted(panel["lsoa21cd"].astype(str).unique()))[
        ["lsoa21cd", "expert_top10_reference"]
    ]
    DEST.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in args.models:
        path = DEST / f"{name}.json"
        if path.exists() and not args.force:
            log(f"[{name}] reusing {path}")
            results[name] = json.loads(path.read_text(encoding="utf-8"))
            continue
        try:
            res = run_model(name, panel, players, split, baseline, expert, args.n_jobs)
        except Exception as exc:  # keep the other models running
            log(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
            continue
        path.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
        results[name] = res

    menu = pd.DataFrame(
        [{"model": m, **row} for m, res in results.items() for row in res["menu"]]
    )
    menu.to_csv(DEST / "geo_surrogate_menu_long.csv", index=False)
    menu.pivot(index="surrogate", columns="model", values="r2").to_csv(DEST / "geo_surrogate_menu_r2_wide.csv")
    ridge = menu[menu["surrogate"] == "Weighted Ridge"][["model", "r2", "mae", "wmae", "rmse", "wrmse"]]
    ridge.to_csv(DEST / "geo_weighted_ridge_fidelity.csv", index=False)
    table = pd.DataFrame(
        [
            {
                "model": m,
                "family": res["family"],
                "predictive_r2": res["predictive_r2"],
                "overlap": res["accuracy"]["intersection"],
                "att_auc": res["accuracy"]["att_auroc"],
                "geo_ridge_r2": next(r["r2"] for r in res["menu"] if r["surrogate"] == "Weighted Ridge"),
            }
            for m, res in results.items()
        ]
    )
    table.to_csv(DEST / "geo_model_table.csv", index=False)
    if len(table) >= 3:
        corr = correlations(table)
        corr.to_csv(DEST / "geo_faithfulness_correlation.csv", index=False)
        print(corr.round(3).to_string(index=False))
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
