"""Assemble the final GeoSMILE model-comparison report: every summary table
produced today, plus a Feature SMILE ranking chart (top-10 highlighted) and a
Geo SMILE top-20 map for every (model, branch) whose predictive R2 exceeds
R2_THRESHOLD.

Reuses save_feature_importance_ranking()/save_geo_rank_map() from
compute_att_stability.py (same chart/map style as the rest of this pipeline)
rather than building new plotting code.

Source data: outputs/verify_metrics_all_models/<model>/{feature_importance,
geo_coefficients}.csv (N=1024 Feature / 4000 Geo, seed=42, this session's
7-model rigorous verification) and outputs/eval_by_model/predictive_performance.csv
for the R2 filter (same predictive fit regardless of SMILE N).

Writes outputs/geosmile_model_comparison_report/{tables,figures}/.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from compute_att_stability import save_feature_importance_ranking, save_geo_rank_map
from hull_boundaries import load_boundaries

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "geosmile_model_comparison_report"
FIG_DIR = DEST / "figures"
TABLE_DIR = DEST / "tables"

R2_THRESHOLD = 0.8

SOURCE_TABLES = [
    (OUT / "eval_by_model" / "predictive_performance.csv", "predictive_performance_all7.csv"),
    (OUT / "eval_by_model" / "model_comparison_summary.csv", "separate_models_full_suite_n2000_summary.csv"),
    (OUT / "eval_combined_model" / "combined_model_comparison_summary.csv", "combined_single_model_n2000_summary.csv"),
    (OUT / "verify_metrics_all_models" / "all_models_summary.csv", "rigorous_5metric_verification_n1024_4000_summary.csv"),
    (OUT / "hull_metrics_computed.json", "xgboost_published_design_reproduction_n1024_2000.json"),
    (OUT / "hull_consistency_per_player.csv", "xgboost_published_design_consistency_per_player.csv"),
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(exist_ok=True)

    for src, name in SOURCE_TABLES:
        if src.exists():
            shutil.copy(src, TABLE_DIR / name)
            _log(f"Copied {src.name} -> tables/{name}")
        else:
            _log(f"Skipped (not found): {src}")

    predictive_path = OUT / "eval_by_model" / "predictive_performance.csv"
    if not predictive_path.exists():
        raise SystemExit(f"Missing {predictive_path}")
    predictive = pd.read_csv(predictive_path)

    panel = pd.read_csv(OUT / "hull_lsoa_month_panel_with_covariates.csv")
    name_lookup = dict(zip(panel["lsoa21cd"].astype(str), panel["lsoa21nm"]))
    hull_lsoas = sorted(panel["lsoa21cd"].astype(str).unique())
    boundaries, _meta = load_boundaries(hull_lsoas)

    qualifying = []
    skipped_missing_source = []
    for _, row in predictive.iterrows():
        model, branch, r2 = row["model"], row["branch"], float(row["r2"])
        if r2 <= R2_THRESHOLD:
            continue
        src_dir = OUT / "verify_metrics_all_models" / model
        if branch == "Feature":
            feat_csv = src_dir / "feature_importance.csv"
            if not feat_csv.exists():
                skipped_missing_source.append((model, branch))
                continue
            feat = pd.read_csv(feat_csv)
            shutil.copy(feat_csv, TABLE_DIR / f"feature_ranking_{model}.csv")
            save_feature_importance_ranking(
                feat, FIG_DIR / f"feature_top10_{model}.png",
                title=f"{model}: Feature SMILE ranking (predictive R2={r2:.3f})",
                footer=f"GeoSMILE Feature branch, N=1024 masks, seed=42. Model: {model}. Top-10 highlighted.",
            )
            _log(f"[{model}/Feature] R2={r2:.3f} -> figure + table written")
        else:
            geo_csv = src_dir / "geo_coefficients.csv"
            if not geo_csv.exists():
                skipped_missing_source.append((model, branch))
                continue
            geo = pd.read_csv(geo_csv)
            geo["lsoa21cd"] = geo["lsoa21cd"].astype(str)
            shutil.copy(geo_csv, TABLE_DIR / f"geo_ranking_{model}.csv")
            save_geo_rank_map(
                geo, boundaries, FIG_DIR / f"geo_top20_{model}.png",
                title=f"{model}: Geo SMILE top-20 (predictive R2={r2:.3f})",
                footer=f"GeoSMILE Geo branch, N=4000 perturbations, seed=42. Model: {model}.",
                name_lookup=name_lookup,
            )
            _log(f"[{model}/Geo] R2={r2:.3f} -> figure + table written")
        qualifying.append({"model": model, "branch": branch, "predictive_r2": r2})

    pd.DataFrame(qualifying).to_csv(DEST / "qualifying_models_r2_above_0.8.csv", index=False)
    if skipped_missing_source:
        _log(f"\nSkipped (R2>0.8 but no source ranking yet, likely still running): {skipped_missing_source}")
    _log(f"\nDone. {len(qualifying)} qualifying (model, branch) combos figured. Report: {DEST}")


if __name__ == "__main__":
    main()
