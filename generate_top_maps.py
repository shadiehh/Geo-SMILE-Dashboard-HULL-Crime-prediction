"""Top-20 Geo choropleth maps and top-10 Feature bar charts for the
"high-fidelity" candidate models, plus MLP/DL's Feature branch under their
best rescuing surrogate -- per explicit user request, extending
build_final_report.py's plotting (same save_feature_importance_ranking/
save_geo_rank_map helpers, same chart style) to the design-experiments
findings rather than just the original 7-model screen.

Three sources of coefficients, one per (model, branch):
  A) xgboost, lightgbm, catboost, svr, gp (both branches), dl (Geo only) --
     already saved by verify_metrics_all_models.py (standard kernel-weighted
     Ridge SMILE coefficient, N=1,024 Feature / 4,000 Geo, seed=42), reused
     as-is -- no recomputation.
  B) ann, cnn, transformer -- fresh Ridge SMILE fits at the same N/seed,
     since these three were only ever run through the fidelity-focused
     scripts (response_metric_comparison.py etc.), which measure fidelity
     from a mask/response pool but never save the coefficient dataframe
     itself.
  C) mlp, dl Feature branch -- their incumbent Ridge SMILE coefficient
     collapses (fidelity R2 0.09 / 0.12, see
     outputs/design_experiments/smile_design_experiments_section.tex).
     Uses feature_importances_ from the Random Forest surrogate found to
     rescue fidelity there instead: MLP at N=8,192 (kernel-weighted,
     R2=0.483), DL at N=4,096 (unweighted, R2=0.848). Magnitude-only
     (unsigned) and NOT part of the officially adopted GeoSMILE pipeline --
     tree ensembles have no linear-coefficient equivalent, so this is an
     exploratory alternative ranking, not a replacement attribution.

CNN's Geo branch is included despite failing the R2>0.7 predictive guard
(R2=0.661) -- forced through, same as its fidelity numbers, clearly labeled
in the figure title. MLP's Geo branch is excluded outright (R2=0.444,
correctly gated -- see the correction applied to model_screening_section.tex).

Writes outputs/top_maps/{figures,tables}/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from compute_att_stability import save_feature_importance_ranking, save_geo_rank_map
from eval_by_n_perturb import Y_COL, dehghani_fidelity
from geosmile_hull_twobranch import (
    FORBIDDEN_FEATURE_TOKENS,
    SPATIAL_COLUMNS,
    choose_feature_players,
    geo_smile_coefficients,
    smile_global_feature,
    temporal_split,
)
from hull_boundaries import load_boundaries
from hull_dehghani_eval import apply_kernel, split_masks
from model_factory import fit_model, make_model
from response_metric_comparison import SMILE_SEED, feature_pool_multi_response
from surrogate_comparison_extended import SPLIT_SEED

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "top_maps"
FIG_DIR = DEST / "figures"
TABLE_DIR = DEST / "tables"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"
PREDICTIVE_CSV = OUT / "eval_by_model" / "predictive_performance.csv"
VERIFY_DIR = OUT / "verify_metrics_all_models"

FEAT_N = 1024
GEO_N = 4000
R2_GATE = 0.7

SAVED_RIDGE_MODELS = ("xgboost", "lightgbm", "catboost", "svr", "gp", "dl")
FRESH_RIDGE_MODELS = ("ann", "cnn", "transformer")
TREE_FEATURE = {  # model -> (n, use kernel-weighted sample_weight)
    "mlp": (8192, True),
    "dl": (4096, False),
}
TREE_FEATURE_FIDELITY = {"mlp": 0.483, "dl": 0.848}  # best RF fidelity R2 found in the surrogate x N sweep
FORCE = {("cnn", "Geo")}
EXCLUDE = {("mlp", "Geo")}
ALL_MODELS = ("xgboost", "lightgbm", "catboost", "svr", "gp", "mlp", "dl", "ann", "cnn", "transformer")


def _log(msg: str) -> None:
    print(msg, flush=True)


def tree_feature_importance(model, X_test, players, baseline, n, seed, weighted) -> pd.DataFrame:
    masks, distances, resp_by_metric = feature_pool_multi_response(model, X_test, players, baseline, n, seed)
    responses = resp_by_metric["wasserstein"]
    tr, va, te = split_masks(len(masks), SPLIT_SEED)
    trva = np.concatenate([tr, va])
    sample_weight = None
    if weighted:
        ridge_fid = dehghani_fidelity(masks, responses, distances)
        sigma = max(float(np.median(distances[tr])), 1e-12)
        sample_weight = np.clip(apply_kernel(distances[trva], ridge_fid["best_kernel"], sigma), 1e-8, None)
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=SMILE_SEED, n_jobs=-1)
    rf.fit(masks[trva], responses[trva], sample_weight=sample_weight)
    imp = rf.feature_importances_.astype(float)
    return pd.DataFrame({"feature": players, "coefficient": imp, "abs_coefficient": imp})


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(exist_ok=True)

    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    bad = [p for p in players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise SystemExit(f"Spatial leak in Feature players: {bad}")
    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
    X_test = panel.loc[test].reset_index(drop=True)
    name_lookup = dict(zip(panel["lsoa21cd"].astype(str), panel["lsoa21nm"]))
    hull_lsoas = sorted(panel["lsoa21cd"].astype(str).unique())
    boundaries, _meta = load_boundaries(hull_lsoas)

    predictive = pd.read_csv(PREDICTIVE_CSV)

    def r2_of(model: str, branch: str) -> float:
        row = predictive[(predictive["model"] == model) & (predictive["branch"] == branch)]
        return float(row.iloc[0]["r2"]) if not row.empty else float("nan")

    qualifying = []
    for model_name in ALL_MODELS:
        model_feat = model_geo = None  # lazily fit only if a fresh SMILE run is needed
        for branch in ("Feature", "Geo"):
            key = (model_name, branch)
            if key in EXCLUDE:
                _log(f"[{model_name}/{branch}] EXCLUDED (correctly gated, see model_screening_section.tex)")
                continue
            r2 = r2_of(model_name, branch)
            forced = key in FORCE
            if not forced and not (np.isfinite(r2) and r2 > R2_GATE):
                _log(f"[{model_name}/{branch}] SKIPPED (R2={r2:.3f} <= {R2_GATE})")
                continue

            use_tree = model_name in TREE_FEATURE and branch == "Feature"
            if use_tree:
                n, weighted = TREE_FEATURE[model_name]
                if model_feat is None:
                    model_feat = make_model(model_name, SMILE_SEED, n_jobs=2)
                    fit_model(model_name, model_feat, panel.loc[train, players], panel.loc[train, Y_COL],
                              panel.loc[validation, players], panel.loc[validation, Y_COL])
                _log(f"[{model_name}/Feature] fitting Random Forest surrogate (n={n}, weighted={weighted})...")
                feat = tree_feature_importance(model_feat, X_test, players, baseline, n, SMILE_SEED, weighted)
                feat.to_csv(TABLE_DIR / f"feature_ranking_{model_name}_rf_surrogate.csv", index=False)
                save_feature_importance_ranking(
                    feat, FIG_DIR / f"feature_top10_{model_name}.png",
                    title=f"{model_name}: Feature ranking, Random Forest surrogate "
                          f"(fidelity R2={TREE_FEATURE_FIDELITY[model_name]:.3f})",
                    footer=f"Random Forest surrogate (n_estimators=300, max_depth=8), N={n} masks, seed=42. "
                           f"Magnitude-only (unsigned) -- not the officially adopted GeoSMILE attribution.",
                )
                qualifying.append({"model": model_name, "branch": branch, "predictive_r2": r2, "source": "rf_surrogate"})
                continue

            if model_name in SAVED_RIDGE_MODELS:
                src_dir = VERIFY_DIR / model_name
                csv_name = "feature_importance.csv" if branch == "Feature" else "geo_coefficients.csv"
                src_csv = src_dir / csv_name
                if not src_csv.exists():
                    _log(f"[{model_name}/{branch}] SKIPPED (no saved SMILE ranking at {src_csv})")
                    continue
                coef = pd.read_csv(src_csv)
                source_label = "saved (verify_metrics_all_models)"
            else:  # FRESH_RIDGE_MODELS: never saved, fit + run SMILE now
                if branch == "Feature":
                    if model_feat is None:
                        model_feat = make_model(model_name, SMILE_SEED, n_jobs=2)
                        fit_model(model_name, model_feat, panel.loc[train, players], panel.loc[train, Y_COL],
                                  panel.loc[validation, players], panel.loc[validation, Y_COL])
                    _log(f"[{model_name}/Feature] fresh Ridge SMILE fit (n={FEAT_N})...")
                    coef, _diag, *_ = smile_global_feature(model_feat, X_test[players], players, baseline, FEAT_N, SMILE_SEED)
                else:
                    if model_geo is None:
                        model_geo = make_model(model_name, SMILE_SEED, n_jobs=2)
                        fit_model(model_name, model_geo, panel.loc[train, SPATIAL_COLUMNS], panel.loc[train, Y_COL],
                                  panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, Y_COL])
                    _log(f"[{model_name}/Geo] fresh Ridge SMILE fit (n={GEO_N}){' [FORCED]' if forced else ''}...")
                    coef, _diag = geo_smile_coefficients(model_geo, panel, test, baseline, GEO_N, SMILE_SEED)
                    coef["lsoa21cd"] = coef["lsoa21cd"].astype(str)
                source_label = "fresh Ridge SMILE fit"

            if branch == "Feature":
                coef.to_csv(TABLE_DIR / f"feature_ranking_{model_name}.csv", index=False)
                title = f"{model_name}: Feature SMILE ranking (predictive R2={r2:.3f})"
                if forced:
                    title += " [forced past R2>0.7 guard]"
                save_feature_importance_ranking(
                    coef, FIG_DIR / f"feature_top10_{model_name}.png",
                    title=title,
                    footer=f"GeoSMILE Feature branch, N={FEAT_N} masks, seed=42. Model: {model_name}. "
                           f"Top-10 highlighted. Source: {source_label}.",
                )
            else:
                title = f"{model_name}: Geo SMILE top-20 (predictive R2={r2:.3f})"
                if forced:
                    title += " [forced past R2>0.7 guard]"
                coef.to_csv(TABLE_DIR / f"geo_ranking_{model_name}.csv", index=False)
                save_geo_rank_map(
                    coef, boundaries, FIG_DIR / f"geo_top20_{model_name}.png",
                    title=title,
                    footer=f"GeoSMILE Geo branch, N={GEO_N} perturbations, seed=42. Model: {model_name}. "
                           f"Source: {source_label}.",
                    name_lookup=name_lookup,
                )
            qualifying.append({"model": model_name, "branch": branch, "predictive_r2": r2, "source": source_label})

    pd.DataFrame(qualifying).to_csv(DEST / "qualifying_models.csv", index=False)
    _log(f"\nDone. {len(qualifying)} (model, branch) combos figured. Report: {DEST}")


if __name__ == "__main__":
    main()
