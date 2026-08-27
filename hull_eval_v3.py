"""Hull GeoSMILE evaluation v3: provenance restore, ATT top-10/20, faithfulness, maps.

Active branches only: Feature (location excluded), Geo.
Does not touch the dashboard.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from geosmile_hull_twobranch import (
    FORBIDDEN_FEATURE_TOKENS,
    SPATIAL_COLUMNS,
    choose_feature_players,
    make_model,
    temporal_split,
)
from hull_expert_alignment import DEFAULT_ANNOTATIONS
from hull_expert_alignment import run as run_expert_alignment
from hull_expert_topk import (
    ANNOTATION_STATUS,
    BRANCH_FEATURE,
    BRANCH_GEO,
    EXPERT_DISPLAY_NOTE,
    FEATURE_TOP_K,
    GEO_TOP_K,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"

DEDICATED_SOURCES = {
    ("Feature", "Fidelity"): "geosmile_hull_feat_fidelity.csv",
    ("Geo", "Fidelity"): "geosmile_hull_geo_fidelity.csv",
    ("Feature", "Consistency"): "geosmile_hull_feat_consistency.csv",
    ("Geo", "Consistency"): "geosmile_hull_geo_consistency.csv",
    ("Both", "Perturbation response monotonicity"): "geosmile_hull_faithfulness.csv",
    ("Feature", "Sparsity / concentration"): "geosmile_hull_sparsity.csv",
    ("Geo", "Sparsity / concentration"): "geosmile_hull_sparsity.csv",
    ("Feature", "Kernel selection"): "geosmile_hull_kernel_selection_feat.csv",
    ("Geo", "Kernel selection"): "geosmile_hull_kernel_selection_geo.csv",
    ("Both", "Convergence"): "geosmile_hull_convergence.csv",
}


def _safe(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        x = float(v)
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v


def _row(
    branch: str,
    category: str,
    metric: str,
    value,
    *,
    source_file: str,
    result_status: str,
    ci95_low=None,
    ci95_high=None,
    p_value=None,
    n=None,
    k=None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "branch": branch,
        "category": category,
        "metric": metric,
        "value": _safe(value),
        "ci95_low": _safe(ci95_low),
        "ci95_high": _safe(ci95_high),
        "p_value": _safe(p_value),
        "n": _safe(n),
        "k": _safe(k),
        "source_file": source_file,
        "result_status": result_status,
        "notes": notes,
    }


def restore_historical_metrics() -> tuple[list[dict], list[dict], list[dict]]:
    """Restore validated metrics with provenance; dedicated files beat evaluation_full."""
    rows: list[dict] = []
    provenance: list[dict] = []
    conflicts: list[dict] = []

    meta = {}
    summary_path = OUT / "geosmile_hull_dehghani_eval_summary.json"
    if summary_path.exists():
        meta = json.loads(summary_path.read_text(encoding="utf-8"))
    seed = meta.get("seed")
    n_feat_masks = meta.get("feat_masks")
    n_geo_pert = meta.get("geo_perturbations")
    consist_base = meta.get("consistency_seed_base")

    full_path = OUT / "geosmile_hull_evaluation_full.csv"
    full_map: dict[tuple, float] = {}
    if full_path.exists():
        full = pd.read_csv(full_path)
        for _, r in full.iterrows():
            branch = str(r["Branch"])
            cat = str(r["Category"])
            # Recategorise old Faithfulness -> Perturbation response monotonicity
            if cat.lower() == "faithfulness":
                cat = "Perturbation response monotonicity"
            # Relabel Stability (multi-run Jaccard) as Consistency — never as controlled-input
            if cat.lower() == "stability" or "stability" in cat.lower():
                # Old compact tables mislabelled Jaccard as Stability; treat as Consistency note
                if "jaccard" in str(r["Metric"]).lower():
                    cat = "Consistency"
            metric = str(r["Metric"])
            # Normalise metric names for cardinality vs WD
            rename = {
                "Feature Spearman (cardinality vs WD)": "Feature Spearman: perturbation cardinality vs WD",
                "Feature Pearson (cardinality vs WD)": "Feature Pearson: perturbation cardinality vs WD",
                "Geo Spearman (cardinality vs WD)": "Geo Spearman: perturbation cardinality vs WD",
                "Geo Pearson (cardinality vs WD)": "Geo Pearson: perturbation cardinality vs WD",
            }
            metric = rename.get(metric, metric)
            # Skip Feature-A / Feature-B
            if "feature-a" in branch.lower() or "feature-b" in branch.lower():
                continue
            # Map Feature -> Feature (location excluded)
            if branch == "Feature":
                branch = BRANCH_FEATURE
            key = (branch, cat, metric)
            try:
                val = float(r["Value"]) if not isinstance(r["Value"], str) or _is_number(r["Value"]) else r["Value"]
            except Exception:
                val = r["Value"]
            full_map[key] = val

    # Fidelity from dedicated files
    for branch_key, fname, out_branch in [
        ("Feature", "geosmile_hull_feat_fidelity.csv", BRANCH_FEATURE),
        ("Geo", "geosmile_hull_geo_fidelity.csv", BRANCH_GEO),
    ]:
        path = OUT / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for col in ["R2", "Weighted R2", "MAE", "WMAE", "RMSE", "WRMSE"]:
            if col not in df.columns:
                continue
            val = float(df.iloc[0][col])
            metric = col
            key = (out_branch, "Fidelity", metric)
            alt = full_map.get((out_branch, "Fidelity", metric))
            if alt is not None and abs(float(alt) - val) > 1e-9:
                conflicts.append(
                    {
                        "branch": out_branch,
                        "category": "Fidelity",
                        "metric": metric,
                        "active_value": val,
                        "active_source": fname,
                        "alternate_value": alt,
                        "alternate_source": "geosmile_hull_evaluation_full.csv",
                        "resolution": "dedicated_per_metric_file_takes_precedence",
                    }
                )
            rows.append(
                _row(
                    out_branch,
                    "Fidelity",
                    metric,
                    val,
                    source_file=fname,
                    result_status="restored",
                    notes=f"seed={seed}; feat_masks={n_feat_masks}; geo_perturbations={n_geo_pert}",
                )
            )
            provenance.append(
                {
                    "branch": out_branch,
                    "category": "Fidelity",
                    "metric": metric,
                    "value": val,
                    "source_file": fname,
                    "result_status": "restored",
                    "run_seed": seed,
                    "perturbation_count": n_feat_masks if branch_key == "Feature" else n_geo_pert,
                    "timestamp": None,
                    "notes": "dedicated fidelity file",
                }
            )

    # Consistency from dedicated files
    for branch_key, fname, out_branch in [
        ("Feature", "geosmile_hull_feat_consistency.csv", BRANCH_FEATURE),
        ("Geo", "geosmile_hull_geo_consistency.csv", BRANCH_GEO),
    ]:
        path = OUT / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for col in df.columns:
            if col == "branch":
                continue
            val = df.iloc[0][col]
            try:
                val = float(val)
            except Exception:
                pass
            rows.append(
                _row(
                    out_branch,
                    "Consistency",
                    col,
                    val,
                    source_file=fname,
                    result_status="restored",
                    notes=f"multi-run consistency; consistency_seed_base={consist_base}. Not controlled-input stability.",
                )
            )
            provenance.append(
                {
                    "branch": out_branch,
                    "category": "Consistency",
                    "metric": col,
                    "value": val,
                    "source_file": fname,
                    "result_status": "restored",
                    "run_seed": consist_base,
                    "perturbation_count": None,
                    "timestamp": None,
                    "notes": "dedicated consistency file; not controlled-input stability",
                }
            )

    # Perturbation response monotonicity from faithfulness.csv
    faith_path = OUT / "geosmile_hull_faithfulness.csv"
    if faith_path.exists():
        faith = pd.read_csv(faith_path)
        for _, r in faith.iterrows():
            b = BRANCH_FEATURE if str(r["branch"]).lower() == "feature" else BRANCH_GEO
            for col, metric in [
                ("spearman", f"{'Feature' if b == BRANCH_FEATURE else 'Geo'} Spearman: perturbation cardinality vs WD"),
                ("pearson", f"{'Feature' if b == BRANCH_FEATURE else 'Geo'} Pearson: perturbation cardinality vs WD"),
            ]:
                if col not in r:
                    continue
                val = float(r[col])
                rows.append(
                    _row(
                        b,
                        "Perturbation response monotonicity",
                        metric,
                        val,
                        source_file=faith_path.name,
                        result_status="restored",
                        p_value=float(r.get(f"p_{col}")) if f"p_{col}" in r and pd.notna(r.get(f"p_{col}")) else None,
                        notes="Not model-based attribution faithfulness; not expert faithfulness.",
                    )
                )
                provenance.append(
                    {
                        "branch": b,
                        "category": "Perturbation response monotonicity",
                        "metric": metric,
                        "value": val,
                        "source_file": faith_path.name,
                        "result_status": "restored",
                        "run_seed": seed,
                        "perturbation_count": None,
                        "timestamp": None,
                        "notes": "restored from geosmile_hull_faithfulness.csv",
                    }
                )

    # Sparsity
    spars_path = OUT / "geosmile_hull_sparsity.csv"
    if spars_path.exists():
        spars = pd.read_csv(spars_path)
        for _, r in spars.iterrows():
            b = BRANCH_FEATURE if str(r["branch"]).lower() == "feature" else BRANCH_GEO
            for col in ["hoyer", "gini", "entropy", "nonzero", "n"]:
                if col not in r:
                    continue
                rows.append(
                    _row(
                        b,
                        "Sparsity / concentration",
                        col,
                        float(r[col]) if col != "nonzero" else int(r[col]),
                        source_file=spars_path.name,
                        result_status="restored",
                    )
                )
                provenance.append(
                    {
                        "branch": b,
                        "category": "Sparsity / concentration",
                        "metric": col,
                        "value": float(r[col]) if pd.notna(r[col]) else None,
                        "source_file": spars_path.name,
                        "result_status": "restored",
                        "run_seed": seed,
                        "perturbation_count": None,
                        "timestamp": None,
                        "notes": "",
                    }
                )

    # Kernel selection
    for fname, out_branch in [
        ("geosmile_hull_kernel_selection_feat.csv", BRANCH_FEATURE),
        ("geosmile_hull_kernel_selection_geo.csv", BRANCH_GEO),
    ]:
        path = OUT / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        best = df.sort_values("validation_r2", ascending=False).iloc[0]
        rows.append(
            _row(
                out_branch,
                "Kernel selection",
                "Best kernel",
                str(best["kernel"]),
                source_file=fname,
                result_status="restored",
            )
        )
        for _, r in df.iterrows():
            rows.append(
                _row(
                    out_branch,
                    "Kernel selection",
                    f"validation_r2[{r['kernel']}]",
                    float(r["validation_r2"]),
                    source_file=fname,
                    result_status="restored",
                )
            )
            provenance.append(
                {
                    "branch": out_branch,
                    "category": "Kernel selection",
                    "metric": f"validation_r2[{r['kernel']}]",
                    "value": float(r["validation_r2"]),
                    "source_file": fname,
                    "result_status": "restored",
                    "run_seed": seed,
                    "perturbation_count": None,
                    "timestamp": None,
                    "notes": "",
                }
            )

    # Convergence (keep table values as long rows for final sizes)
    conv_path = OUT / "geosmile_hull_convergence.csv"
    if conv_path.exists():
        conv = pd.read_csv(conv_path)
        for branch_name, out_branch in [("Feature", BRANCH_FEATURE), ("Geo", BRANCH_GEO)]:
            sub = conv[conv["Branch"] == branch_name]
            if sub.empty:
                continue
            last = sub.sort_values("#Perturb").iloc[-1]
            for col in ["R2", "Weighted R2", "WMAE", "WRMSE", "MAE", "RMSE"]:
                if col not in last:
                    continue
                rows.append(
                    _row(
                        out_branch,
                        "Convergence",
                        f"{col} at max #Perturb",
                        float(last[col]),
                        source_file=conv_path.name,
                        result_status="restored",
                        n=int(last["#Perturb"]),
                        notes=f"kernel={last.get('Kernel')}",
                    )
                )
                provenance.append(
                    {
                        "branch": out_branch,
                        "category": "Convergence",
                        "metric": f"{col} at max #Perturb",
                        "value": float(last[col]),
                        "source_file": conv_path.name,
                        "result_status": "restored",
                        "run_seed": seed,
                        "perturbation_count": int(last["#Perturb"]),
                        "timestamp": None,
                        "notes": "",
                    }
                )

    return rows, provenance, conflicts


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def dehghani_expert_faithfulness_status() -> dict[str, Any]:
    """Require ≥5 independent predictive-model runs with test perf + ATT metrics."""
    # Seed sweeps store surrogate fidelity / cardinality-vs-WD, not predictive test R2 + ATT.
    # run_summary has a single predictive model_metrics snapshot.
    reason = (
        "Insufficient independent predictive-model runs with both test performance "
        "and expert-reference attribution metrics."
    )
    detail = (
        "Available seed sweeps (seed_sweep_phase1/2, sweep_eval_seeds*) vary explanation "
        "or model seeds but do not archive per-run predictive TEST metrics together with "
        "expert top-k ATT accuracy/F1/AUROC. Surrogate fidelity and explanation-seed "
        "consistency are not substitutes for Dehghani-style expert faithfulness."
    )
    return {
        "status": "Not estimable",
        "reason": reason,
        "detail": detail,
        "n_valid_independent_predictive_runs": 0,
        "minimum_required": 5,
    }


def controlled_input_stability_status() -> dict[str, Any]:
    path = OUT / "geosmile_hull_att_stability.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload
    return {
        "status": "Not evaluated",
        "reason": (
            "ATT stability compares attributions before and after a minor, "
            "meaning-preserving perturbation of the input, quantified by the overlap "
            "of the resulting attribution sets. No such prediction-preserving "
            "recomputation is implemented in the Hull pipeline, so no stability "
            "estimate is available for either branch. This is a gap in the present "
            "evaluation rather than a negative result. Multi-run top-set Jaccard is "
            "Consistency, not ATT / controlled-input stability."
        ),
    }


def model_based_faithfulness(
    *,
    n_random: int = 1000,
    seed: int = 42,
) -> tuple[list[dict], dict[str, Any]]:
    """Top vs random vs bottom interventions at fixed cardinality (Feature 10, Geo 20)."""
    if not PANEL_PATH.exists():
        raise SystemExit(f"Missing {PANEL_PATH}")
    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    bad = [p for p in players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise ValueError(f"Spatial feature leaked into Feature branch: {bad}")

    train, validation, test = temporal_split(panel)
    y_col = "target_next_month_crime"
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
    X_test = panel.loc[test].reset_index(drop=True)

    model_feat = make_model(seed)
    model_geo = make_model(seed)
    model_feat.fit(
        panel.loc[train, players],
        panel.loc[train, y_col],
        eval_set=[(panel.loc[validation, players], panel.loc[validation, y_col])],
        verbose=False,
    )
    model_geo.fit(
        panel.loc[train, SPATIAL_COLUMNS],
        panel.loc[train, y_col],
        eval_set=[(panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, y_col])],
        verbose=False,
    )

    feat_imp = pd.read_csv(OUT / "feature_global_importance.csv")
    feat_imp = feat_imp.sort_values(
        ["abs_coefficient", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    k_f = FEATURE_TOP_K
    top_feat = feat_imp.head(k_f)["feature"].tolist()
    bottom_feat = feat_imp.tail(k_f)["feature"].tolist()
    all_feat = feat_imp["feature"].tolist()

    geo_imp = pd.read_csv(OUT / "geo_coefficients.csv")
    geo_imp = geo_imp.sort_values(
        ["geo_abs_coefficient", "lsoa21cd"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    k_g = GEO_TOP_K
    top_geo = geo_imp.head(k_g)["lsoa21cd"].tolist()
    bottom_geo = geo_imp.tail(k_g)["lsoa21cd"].tolist()
    all_geo = geo_imp["lsoa21cd"].tolist()

    # Optional population/area for geo random comparability (no adjacency invented)
    pop_area = None
    merged_geo = OUT / "geosmile_hull_expert_alignment_merged_geo.csv"
    cov_path = ROOT / "data" / "hull_static_covariates.csv"
    if merged_geo.exists() and cov_path.exists():
        mg = pd.read_csv(merged_geo)
        cov = pd.read_csv(cov_path)[["lsoa21cd", "lsoa_area_km2"]]
        if "population_mid_2022" in mg.columns:
            pop_area = mg[["lsoa21cd", "population_mid_2022"]].merge(cov, on="lsoa21cd", how="left")
            pop_area = pop_area.set_index("lsoa21cd")

    rng = np.random.default_rng(seed)

    def feature_response(mask_features: list[str]) -> float:
        original = np.asarray(model_feat.predict(X_test[players]), float)
        pert = X_test[players].copy()
        for f in mask_features:
            pert[f] = baseline[f]
        return float(wasserstein_distance(original, model_feat.predict(pert)))

    def geo_response(mask_lsoas: list[str]) -> float:
        x = X_test[SPATIAL_COLUMNS].copy()
        original = np.asarray(model_geo.predict(x), float)
        lsoa_ids = X_test["lsoa21cd"].to_numpy()
        perturbed = x.copy()
        affected = []
        for code in mask_lsoas:
            rows = np.flatnonzero(lsoa_ids == code)
            perturbed.iloc[rows, :] = baseline[SPATIAL_COLUMNS].to_numpy(dtype=float)
            affected.extend(rows.tolist())
        affected = np.asarray(affected, dtype=int)
        if len(affected) == 0:
            return 0.0
        pert_pred = np.asarray(model_geo.predict(perturbed), float)
        return float(wasserstein_distance(original[affected], pert_pred[affected]))

    def sample_geo_random(k: int) -> list[str]:
        """Equal cardinality; soft match population+area when data exist (no adjacency)."""
        if pop_area is None:
            return list(rng.choice(all_geo, size=k, replace=False))
        target_pop = float(pop_area.loc[top_geo, "population_mid_2022"].sum())
        target_area = float(pop_area.loc[top_geo, "lsoa_area_km2"].sum())
        best = None
        best_score = np.inf
        for _ in range(40):
            cand = list(rng.choice(all_geo, size=k, replace=False))
            pop = float(pop_area.loc[cand, "population_mid_2022"].sum())
            area = float(pop_area.loc[cand, "lsoa_area_km2"].sum())
            score = abs(pop - target_pop) / (target_pop + 1e-9) + abs(area - target_area) / (
                target_area + 1e-9
            )
            if score < best_score:
                best_score = score
                best = cand
        return best or list(rng.choice(all_geo, size=k, replace=False))

    print(f"Model-based faithfulness Feature k={k_f}, Geo k={k_g}, n_random={n_random}...")
    top_f = feature_response(top_feat)
    bot_f = feature_response(bottom_feat)
    rand_f = np.array(
        [feature_response(list(rng.choice(all_feat, size=k_f, replace=False))) for _ in range(n_random)],
        dtype=float,
    )

    top_g = geo_response(top_geo)
    bot_g = geo_response(bottom_geo)
    rand_g = np.array([geo_response(sample_geo_random(k_g)) for _ in range(n_random)], dtype=float)

    def pack(branch, top_v, bot_v, rand_arr, k, note=""):
        lo, hi = np.percentile(rand_arr, [2.5, 97.5]).tolist()
        p_top = float((np.sum(rand_arr >= top_v) + 1) / (len(rand_arr) + 1))
        return {
            "branch": branch,
            "k": k,
            "top_response": float(top_v),
            "bottom_response": float(bot_v),
            "random_mean": float(rand_arr.mean()),
            "random_ci95_low": float(lo),
            "random_ci95_high": float(hi),
            "top_minus_random": float(top_v - rand_arr.mean()),
            "top_minus_bottom": float(top_v - bot_v),
            "empirical_p_top_vs_random": p_top,
            "n_random": int(len(rand_arr)),
            "deletion_auc": None,
            "notes": note,
        }

    feat_pack = pack(
        BRANCH_FEATURE,
        top_f,
        bot_f,
        rand_f,
        k_f,
        note="Fixed-cardinality feature ablation; WD on test predictions.",
    )
    geo_note = "Fixed-cardinality LSOA location masking; WD on affected units."
    if pop_area is not None:
        geo_note += " Random sets soft-matched on population_mid_2022 + lsoa_area_km2 (no adjacency)."
    else:
        geo_note += " Random sets equal cardinality only (no adjacency invented)."
    geo_pack = pack(BRANCH_GEO, top_g, bot_g, rand_g, k_g, note=geo_note)

    rows = []
    for pack_d in (feat_pack, geo_pack):
        b = pack_d["branch"]
        for metric in [
            "top_response",
            "bottom_response",
            "random_mean",
            "random_ci95_low",
            "random_ci95_high",
            "top_minus_random",
            "top_minus_bottom",
            "empirical_p_top_vs_random",
            "deletion_auc",
        ]:
            rows.append(
                _row(
                    b,
                    "Model-based attribution faithfulness",
                    metric,
                    pack_d[metric],
                    source_file="hull_eval_v3.py:model_based_faithfulness",
                    result_status="recomputed",
                    ci95_low=pack_d["random_ci95_low"] if metric == "random_mean" else None,
                    ci95_high=pack_d["random_ci95_high"] if metric == "random_mean" else None,
                    p_value=pack_d["empirical_p_top_vs_random"]
                    if metric == "top_minus_random"
                    else None,
                    n=pack_d["n_random"],
                    k=pack_d["k"],
                    notes=pack_d["notes"],
                )
            )
    return rows, {"feature": feat_pack, "geo": geo_pack}


def expert_long_to_v3_rows(long_df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in long_df.iterrows():
        branch = str(r["branch"])
        if branch == "Feature":
            branch = BRANCH_FEATURE
        if "feature-a" in branch.lower() or "feature-b" in branch.lower():
            continue
        metric = str(r["metric"])
        if r.get("confidence_subset") and str(r.get("confidence_subset")) != "all":
            metric = f"{metric} [{r['evaluation_scope']}|{r['reference_variant']}|{r['confidence_subset']}]"
        elif r.get("reference_variant") and str(r["reference_variant"]) != "external_crime_informed_reference":
            metric = f"{metric} [{r['reference_variant']}]"
        rows.append(
            _row(
                branch,
                "Expert-reference attribution accuracy",
                metric,
                r.get("value"),
                source_file="hull_expert_alignment.py",
                result_status="recomputed",
                ci95_low=r.get("ci95_low"),
                ci95_high=r.get("ci95_high"),
                p_value=r.get("p_value"),
                n=r.get("n"),
                k=r.get("k"),
                notes=str(r.get("notes") or ANNOTATION_STATUS),
            )
        )
    return rows


def write_method_md(path: Path, dehghani: dict, stability: dict) -> None:
    text = f"""# Hull GeoSMILE evaluation method (v3)

## Active branches

1. **Feature (location excluded)** — non-spatial predictors only; global attribution from `feature_global_importance.csv`. Excludes coordinates, lon/lat, centroids, LSOA IDs, and spatial-location variables.
2. **Geo** — per-LSOA location masking; attribution from `geo_coefficients.csv`.

Feature-A / Feature-B are **not** active evaluation branches (legacy files retained on disk only).

## Expert reference

Workbook: `{DEFAULT_ANNOTATIONS.name}`  
Status: `{ANNOTATION_STATUS}`

{EXPERT_DISPLAY_NOTE}

```
final_expert_rating = human_override_1to5 if not null else expert_relevance_1to5
```

Primary binary reference is **fixed expert top sets**, not rating ≥ 4:
- Feature: top {FEATURE_TOP_K} of 30 from `outputs/feature_expert_ranking_v2.csv` (7/10 vs published GeoSMILE)
- Geo: the {GEO_TOP_K} LSOAs in `outputs/maps_twobranch/ranking/expert_top20.csv`
  (overlay list; published GeoSMILE overlap 16/20)

## Attribution accuracy vs faithfulness

- **Expert-reference attribution accuracy** — agreement with the expert top-{FEATURE_TOP_K}/{GEO_TOP_K} sets (ATT metrics, set overlap, graded Spearman/Kendall/nDCG).
- **Model-based attribution faithfulness** — top vs random vs bottom interventions at fixed cardinality (Feature={FEATURE_TOP_K}, Geo={GEO_TOP_K}; WD response), matching the expert reference sets.
- **Dehghani-style expert faithfulness** — correlation of predictive TEST performance with ATT across ≥5 independent predictive-model runs.
  - Current status: **{dehghani['status']}** — {dehghani['reason']}
- **Perturbation response monotonicity** — cardinality vs WD (restored); not faithfulness.
- **Consistency** — multi-run explanation agreement (restored); **not** controlled-input stability.
- **Controlled-input stability** — **{stability.get('status')}** — {stability.get('reason') or stability.get('definition') or ''}

## Provenance

Historical validated metrics are restored from dedicated per-metric files first, then `geosmile_hull_evaluation_full.csv`. Conflicts are recorded in `geosmile_hull_result_conflicts.csv`. Consistency values are not overwritten by a new random seed run.

## Maps

Expert **top {GEO_TOP_K}** LSOAs use official Hull LSOA 2021 geometry via `hull_boundaries`. Accuracy and maps both use Geo k={GEO_TOP_K}.

## Limitations

- Expert reference ≠ causal importance ≠ model faithfulness.
- Absolute coefficients; sign is not validated as crime increase/decrease.
- No invented ward/adjacency graphs.
"""
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Restoring historical metrics (provenance) ===")
    hist_rows, provenance, conflicts = restore_historical_metrics()
    print(f"Restored {len(hist_rows)} metric rows; conflicts={len(conflicts)}")

    print(f"=== Expert-reference attribution accuracy (Feature top-{FEATURE_TOP_K} / Geo top-{GEO_TOP_K}) ===")
    expert_args = argparse.Namespace(
        annotations=DEFAULT_ANNOTATIONS,
        seed=args.seed,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        geo_only=False,
    )
    expert_out = run_expert_alignment(expert_args)
    long_df = expert_out["long_df"]
    expert_rows = expert_long_to_v3_rows(long_df)

    dehghani = dehghani_expert_faithfulness_status()
    stability = controlled_input_stability_status()

    dehghani_rows = [
        _row(
            BRANCH_FEATURE,
            "Dehghani-style expert faithfulness",
            "status",
            dehghani["status"],
            source_file="hull_eval_v3.py",
            result_status="not_estimable",
            notes=dehghani["reason"],
        ),
        _row(
            BRANCH_GEO,
            "Dehghani-style expert faithfulness",
            "status",
            dehghani["status"],
            source_file="hull_eval_v3.py",
            result_status="not_estimable",
            notes=dehghani["reason"],
        ),
    ]
    stability_rows = [
        _row(
            BRANCH_FEATURE,
            "Controlled-input stability",
            "status",
            stability["status"],
            source_file="hull_eval_v3.py",
            result_status="not_evaluated",
            notes=stability.get("reason") or stability.get("definition") or stability.get("status", ""),
        ),
        _row(
            BRANCH_GEO,
            "Controlled-input stability",
            "status",
            stability["status"],
            source_file="hull_eval_v3.py",
            result_status="not_evaluated",
            notes=stability.get("reason") or stability.get("definition") or stability.get("status", ""),
        ),
    ]

    print("=== Model-based attribution faithfulness ===")
    if args.expert_att_only:
        existing_v3 = OUT / "geosmile_hull_evaluation_full_v3.csv"
        if not existing_v3.exists():
            raise SystemExit("--expert-att-only requires an existing geosmile_hull_evaluation_full_v3.csv")
        old = pd.read_csv(existing_v3)
        mb_keep = old[old["category"] == "Model-based attribution faithfulness"]
        mb_rows = mb_keep.to_dict(orient="records")
        summary_path = OUT / "geosmile_hull_evaluation_summary_v3.json"
        if summary_path.exists():
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
            mb_summary = prev.get("model_based_attribution_faithfulness") or {}
        else:
            mb_summary = {"feature": {}, "geo": {}}
        print("  reused from existing v3 (expert ATT refresh only)")
    else:
        mb_rows, mb_summary = model_based_faithfulness(n_random=args.n_random, seed=args.seed)

    print(f"=== Expert top-{GEO_TOP_K} LSOA maps ===")
    if args.expert_att_only:
        summary_path = OUT / "geosmile_hull_evaluation_summary_v3.json"
        if summary_path.exists():
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
            map_paths = prev.get("map_outputs") or {}
        else:
            map_paths = {}
        print("  reused existing map outputs (expert ATT refresh only)")
    else:
        from map_expert_top20_lsoa import run as run_maps

        map_paths = run_maps()

    all_rows = hist_rows + expert_rows + dehghani_rows + stability_rows + mb_rows
    # Drop any Feature-A/B that slipped in
    all_rows = [
        r
        for r in all_rows
        if "feature-a" not in str(r["branch"]).lower() and "feature-b" not in str(r["branch"]).lower()
    ]
    v3 = pd.DataFrame(all_rows)
    v3_path = OUT / "geosmile_hull_evaluation_full_v3.csv"
    v3.to_csv(v3_path, index=False)

    prov_df = pd.DataFrame(provenance)
    prov_df.to_csv(OUT / "geosmile_hull_result_provenance.csv", index=False)
    conf_df = pd.DataFrame(conflicts) if conflicts else pd.DataFrame(
        columns=[
            "branch",
            "category",
            "metric",
            "active_value",
            "active_source",
            "alternate_value",
            "alternate_source",
            "resolution",
        ]
    )
    conf_df.to_csv(OUT / "geosmile_hull_result_conflicts.csv", index=False)

    write_method_md(OUT / "geosmile_hull_evaluation_method_v3.md", dehghani, stability)

    # Primary ATT snapshot
    att_primary = {}
    for branch in (BRANCH_FEATURE, BRANCH_GEO):
        sub = v3[
            (v3["branch"] == branch)
            & (v3["category"] == "Expert-reference attribution accuracy")
            & (v3["metric"].isin(
                [
                    "ATT accuracy",
                    "ATT F1",
                    "ATT AUROC",
                    "Jaccard@k",
                    "Jaccard@10%",
                    "intersection",
                    "Precision@k",
                    "Recall@k",
                ]
            ))
        ]
        att_primary[branch] = {
            str(r.metric): r.value for r in sub.itertuples(index=False) if "[" not in str(r.metric)
        }

    summary = {
        "schema_version": "3.0.0",
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "active_branches": [BRANCH_FEATURE, BRANCH_GEO],
        "annotation_status": ANNOTATION_STATUS,
        "expert_display_note": EXPERT_DISPLAY_NOTE,
        "expert_topk_definition": {
            "feature_n": 30,
            "feature_k": FEATURE_TOP_K,
            "geo_n": 168,
            "geo_k": GEO_TOP_K,
            "map_expert_top": GEO_TOP_K,
            "geo_reference_csv": "outputs/maps_twobranch/ranking/expert_top20.csv",
            "note": (
                f"Fixed expert-important sets: Feature top-{FEATURE_TOP_K} of 30 from "
                f"outputs/feature_expert_ranking_v2.csv (7/10 vs published GeoSMILE), "
                f"Geo top-{GEO_TOP_K} from outputs/maps_twobranch/ranking/expert_top20.csv "
                f"(overlay list). Map highlights the same Geo top-{GEO_TOP_K}."
            ),
        },
        "primary_att": att_primary,
        "dehghani_style_expert_faithfulness": dehghani,
        "model_based_attribution_faithfulness": mb_summary,
        "controlled_input_stability": stability,
        "restored_metric_count": int(sum(1 for r in all_rows if r["result_status"] == "restored")),
        "recomputed_metric_count": int(sum(1 for r in all_rows if r["result_status"] == "recomputed")),
        "conflict_count": int(len(conflicts)),
        "map_outputs": {k: str(v) for k, v in map_paths.items()},
        "coverage": {
            BRANCH_FEATURE: "30/30",
            BRANCH_GEO: "168/168",
        },
        "limitations": [
            EXPERT_DISPLAY_NOTE,
            "Not causal; not ground truth; coefficient signs not validated as crime direction.",
            "Controlled-input stability not evaluated; consistency is multi-run only.",
            dehghani["reason"],
        ],
        "warnings": [],
        "provenance_file": "geosmile_hull_result_provenance.csv",
        "conflicts_file": "geosmile_hull_result_conflicts.csv",
    }
    (OUT / "geosmile_hull_evaluation_summary_v3.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # Terminal summary
    print("\n========== Hull evaluation v3 summary ==========")
    print(EXPERT_DISPLAY_NOTE)
    print(f"Active branches: {BRANCH_FEATURE} | {BRANCH_GEO}")
    print(f"Coverage: Feature 30/30, Geo 168/168")
    print("Attribution accuracy (primary ATT):")
    for b, d in att_primary.items():
        print(f"  {b}: {d}")
    print(f"Dehghani-style expert faithfulness: {dehghani['status']} — {dehghani['reason']}")
    print(
        "Model-based faithfulness: "
        f"Feature top-random={mb_summary.get('feature', {}).get('top_minus_random')} "
        f"(p={mb_summary.get('feature', {}).get('empirical_p_top_vs_random')}); "
        f"Geo top-random={mb_summary.get('geo', {}).get('top_minus_random')} "
        f"(p={mb_summary.get('geo', {}).get('empirical_p_top_vs_random')})"
    )
    fid = v3[(v3["category"] == "Fidelity") & (v3["metric"] == "R2")]
    print("Fidelity R2:", {r.branch: r.value for r in fid.itertuples(index=False)})
    cons = v3[
        (v3["category"] == "Consistency") & (v3["metric"].str.contains("Rank Spearman Mean", na=False))
    ]
    print("Consistency Rank Spearman Mean:", {r.branch: r.value for r in cons.itertuples(index=False)})
    print(f"Controlled-input stability: {stability['status']}")
    print(f"Restored metrics: {summary['restored_metric_count']}; conflicts: {summary['conflict_count']}")
    print("Map paths:")
    for k, v in map_paths.items():
        print(f"  {k}: {v}")
    print(f"Wrote {v3_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--permutations", type=int, default=500)
    p.add_argument("--n-random", type=int, default=1000)
    p.add_argument(
        "--expert-att-only",
        action="store_true",
        default=False,
        help=(
            "Recompute expert-reference ATT and maps; reuse existing model-based "
            "faithfulness rows. Does not retrain XGBoost."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
