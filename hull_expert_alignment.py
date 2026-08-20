"""Expert-reference attribution accuracy for Hull GeoSMILE (two-branch).

Active branches only:
  - Feature (location excluded) — feature_global_importance.csv
  - Geo — geo_coefficients.csv

Primary binary reference is fixed expert top sets (Feature k=10 of 30, Geo k=20 of 168), not rating>=4.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, kruskal, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from hull_expert_topk import (
    ANNOTATION_STATUS,
    BRANCH_FEATURE,
    BRANCH_GEO,
    EXPERT_DISPLAY_NOTE,
    FEATURE_TOP_K,
    GEO_TOP_K,
    assign_geosmile_ranks,
    ceil_pct_k,
    compute_att_suite,
    select_expert_top10_feature,
    select_expert_top10_geo,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEFAULT_ANNOTATIONS = ROOT / "hull_expert_top20.xlsx"

SCHEMA_VERSION = "3.0.0"
RANDOM_SEED = 42
BOOTSTRAP_REPS = 5000
PERMUTATION_REPS = 5000
TIE_METHOD = "magnitude_desc_then_id_asc"

CONFIDENCE_ORDER = {"High": 3, "Medium": 2, "Low": 1}


# ---------------------------------------------------------------------------
# Reference ratings
# ---------------------------------------------------------------------------


def resolve_reference_rating(
    expert_relevance: pd.Series,
    human_override: pd.Series | None = None,
) -> pd.Series:
    """final_expert_rating = override if non-null else expert_relevance_1to5."""
    expert = pd.to_numeric(expert_relevance, errors="coerce")
    if human_override is None:
        rating = expert.copy()
    else:
        override = pd.to_numeric(human_override, errors="coerce")
        rating = override.where(override.notna(), expert)
    if rating.isna().any():
        bad = rating[rating.isna()].index.tolist()
        raise ValueError(f"Missing reference ratings at rows: {bad[:10]}")
    if not np.all(np.isin(rating.to_numpy(), [1, 2, 3, 4, 5])):
        raise ValueError(
            f"Reference ratings must be integers 1–5; got {sorted(rating.unique().tolist())}"
        )
    if not np.all(rating == rating.astype(int)):
        raise ValueError("Reference ratings must be integers 1–5 (no fractional values).")
    return rating.astype(int)


def expert_positive_from_rating(rating: pd.Series) -> pd.Series:
    """Legacy helper (rating>=4). Primary binary is expert_top10_reference."""
    return (rating >= 4).astype(int)


# ---------------------------------------------------------------------------
# Attribution magnitude / joins / coverage
# ---------------------------------------------------------------------------


def detect_magnitude_column(frame: pd.DataFrame, *, kind: str) -> tuple[str, pd.Series]:
    """Return (source_label, magnitude series) for feature or geo attributions."""
    if kind == "feature":
        if "abs_coefficient" in frame.columns:
            return "abs_coefficient", pd.to_numeric(frame["abs_coefficient"], errors="coerce")
        if "abs_importance" in frame.columns:
            return "abs_importance", pd.to_numeric(frame["abs_importance"], errors="coerce")
        if "coefficient" in frame.columns:
            coef = pd.to_numeric(frame["coefficient"], errors="coerce")
            return "abs(coefficient)", coef.abs()
        raise ValueError("No feature magnitude column found (abs_coefficient/abs_importance/coefficient).")
    if kind == "geo":
        if "geo_abs_coefficient" in frame.columns:
            return "geo_abs_coefficient", pd.to_numeric(frame["geo_abs_coefficient"], errors="coerce")
        if "geo_coefficient" in frame.columns:
            coef = pd.to_numeric(frame["geo_coefficient"], errors="coerce")
            return "abs(geo_coefficient)", coef.abs()
        raise ValueError("No geo magnitude column found (geo_abs_coefficient/geo_coefficient).")
    raise ValueError(kind)


def coverage_report(
    attribution_ids: pd.Series,
    annotation_ids: pd.Series,
    *,
    id_name: str,
) -> dict[str, Any]:
    attr = attribution_ids.astype(str)
    ann = annotation_ids.astype(str)
    attr_set, ann_set = set(attr), set(ann)
    matched = attr_set & ann_set
    unmatched_attr = sorted(attr_set - ann_set)
    unmatched_ann = sorted(ann_set - attr_set)
    dup_attr = sorted(attr[attr.duplicated()].unique().tolist())
    dup_ann = sorted(ann[ann.duplicated()].unique().tolist())
    denom = max(len(ann_set), 1)
    return {
        "id_name": id_name,
        "n_attribution_rows": int(len(attr)),
        "n_annotation_rows": int(len(ann)),
        "n_matched": int(len(matched)),
        "n_unmatched_attribution": int(len(unmatched_attr)),
        "n_unmatched_annotation": int(len(unmatched_ann)),
        "unmatched_attribution_ids": unmatched_attr,
        "unmatched_annotation_ids": unmatched_ann,
        "duplicate_attribution_ids": dup_attr,
        "duplicate_annotation_ids": dup_ann,
        "coverage_pct": float(100.0 * len(matched) / denom),
    }


def assert_primary_coverage(cov: dict[str, Any], *, expected_n: int, label: str) -> None:
    if cov["duplicate_attribution_ids"] or cov["duplicate_annotation_ids"]:
        raise SystemExit(
            f"{label}: duplicate IDs detected. "
            f"attr={cov['duplicate_attribution_ids']} ann={cov['duplicate_annotation_ids']}"
        )
    if cov["n_matched"] != expected_n or cov["n_unmatched_attribution"] or cov["n_unmatched_annotation"]:
        raise SystemExit(
            f"{label}: incomplete coverage (expected {expected_n} matched). "
            f"matched={cov['n_matched']} "
            f"unmatched_attr={cov['unmatched_attribution_ids'][:10]} "
            f"unmatched_ann={cov['unmatched_annotation_ids'][:10]}"
        )


def merge_feature(
    attribution: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    branch: str,
    source_file: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mag_col, mag = detect_magnitude_column(attribution, kind="feature")
    attr = attribution.copy()
    attr["attribution_magnitude"] = mag
    if "coefficient" in attr.columns:
        attr["attribution_coefficient"] = pd.to_numeric(attr["coefficient"], errors="coerce")
    else:
        attr["attribution_coefficient"] = np.nan
    attr["feature"] = attr["feature"].astype(str)

    ann = annotations.copy()
    ann["feature"] = ann["feature"].astype(str)
    ann["final_expert_rating"] = resolve_reference_rating(
        ann["expert_relevance_1to5"], ann.get("human_override_1to5")
    )
    ann["reference_rating"] = ann["final_expert_rating"]
    keep_ann = [
        "feature",
        "final_expert_rating",
        "reference_rating",
        "confidence",
        "relevance_label",
        "expected_association",
        "expert_relevance_1to5",
        "human_override_1to5",
    ]
    if "mechanism_rationale" in ann.columns:
        keep_ann.append("mechanism_rationale")
    if "caveat" in ann.columns:
        keep_ann.append("caveat")

    cov = coverage_report(attr["feature"], ann["feature"], id_name="feature")
    merged = attr.merge(ann[keep_ann], on="feature", how="inner", validate="one_to_one")
    merged = select_expert_top10_feature(merged, k=FEATURE_TOP_K)
    k = min(FEATURE_TOP_K, len(merged))
    merged = merged.rename(columns={"attribution_magnitude": "geosmile_abs_attribution"})
    if "attribution_coefficient" in merged.columns:
        merged["geosmile_coefficient"] = merged["attribution_coefficient"]
    merged = assign_geosmile_ranks(
        merged, id_col="feature", mag_col="geosmile_abs_attribution", k=k
    )
    merged["attribution_magnitude"] = merged["geosmile_abs_attribution"]
    merged = rank_frame(merged, id_col="feature", mag_col="attribution_magnitude")
    merged["matched_top10"] = (
        (merged["expert_top10_reference"] == 1) & (merged["geosmile_top10_prediction"] == 1)
    ).astype(int)
    merged["source_branch"] = branch
    merged["attribution_source"] = source_file
    merged["magnitude_column"] = mag_col
    return merged, cov


def merge_geo(
    attribution: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    branch: str,
    source_file: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mag_col, mag = detect_magnitude_column(attribution, kind="geo")
    attr = attribution.copy()
    attr["attribution_magnitude"] = mag
    if "geo_coefficient" in attr.columns:
        attr["geo_coefficient"] = pd.to_numeric(attr["geo_coefficient"], errors="coerce")
    else:
        attr["geo_coefficient"] = np.nan
    attr["lsoa21cd"] = attr["lsoa21cd"].astype(str)

    ann = annotations.copy()
    ann["lsoa21cd"] = ann["lsoa21cd"].astype(str)
    ann["final_expert_rating"] = resolve_reference_rating(
        ann["expert_relevance_1to5"], ann.get("human_override_1to5")
    )
    ann["reference_rating"] = ann["final_expert_rating"]
    ann["no_crime_reference_rating"] = resolve_reference_rating(ann["no_crime_rating_1to5"], None)

    cov = coverage_report(attr["lsoa21cd"], ann["lsoa21cd"], id_name="lsoa21cd")
    keep_ann = [
        "lsoa21cd",
        "lsoa21nm",
        "final_expert_rating",
        "reference_rating",
        "no_crime_reference_rating",
        "confidence",
        "relevance_label",
        "expected_crime_context",
        "crime_domain_dependence",
        "expert_relevance_1to5",
        "expert_relevance_score",
        "no_crime_rating_1to5",
        "human_override_1to5",
    ]
    if "rationale" in ann.columns:
        keep_ann.append("rationale")
    if "population_mid_2022" in ann.columns:
        keep_ann.append("population_mid_2022")
    for col in (
        "prediction_expert_rank",
        "expert_top20_reference",
        "city_centre_priority",
        "persistent_crime_proxy",
        "prediction_selection_basis",
        "prediction_expert_comment",
    ):
        if col in ann.columns:
            keep_ann.append(col)
    merged = attr.merge(ann[keep_ann], on="lsoa21cd", how="inner", validate="one_to_one")
    if "lsoa21nm_x" in merged.columns:
        merged["lsoa21nm"] = merged["lsoa21nm_y"].fillna(merged["lsoa21nm_x"])
        merged = merged.drop(columns=[c for c in merged.columns if c.startswith("lsoa21nm_")])
    merged = select_expert_top10_geo(merged, k=GEO_TOP_K)
    k = min(GEO_TOP_K, len(merged))
    merged = merged.rename(columns={"attribution_magnitude": "geosmile_abs_attribution"})
    merged["geosmile_coefficient"] = merged.get("geo_coefficient", np.nan)
    merged = assign_geosmile_ranks(
        merged, id_col="lsoa21cd", mag_col="geosmile_abs_attribution", k=k
    )
    merged["attribution_magnitude"] = merged["geosmile_abs_attribution"]
    merged = rank_frame(merged, id_col="lsoa21cd", mag_col="attribution_magnitude")
    merged["matched_top10"] = (
        (merged["expert_top10_reference"] == 1) & (merged["geosmile_top10_prediction"] == 1)
    ).astype(int)
    merged["source_branch"] = branch
    merged["attribution_source"] = source_file
    merged["magnitude_column"] = mag_col
    return merged, cov


def rank_frame(frame: pd.DataFrame, *, id_col: str, mag_col: str) -> pd.DataFrame:
    out = frame.copy()
    # Attribution rank: higher magnitude = rank 1; ties broken by id ascending.
    order = out.sort_values([mag_col, id_col], ascending=[False, True], kind="mergesort")
    out.loc[order.index, "attribution_rank"] = np.arange(1, len(order) + 1)
    # Expert rating rank: higher rating = rank 1; ties broken by id.
    order_e = out.sort_values(["reference_rating", id_col], ascending=[False, True], kind="mergesort")
    out.loc[order_e.index, "expert_rating_rank"] = np.arange(1, len(order_e) + 1)
    out["attribution_rank"] = out["attribution_rank"].astype(int)
    out["expert_rating_rank"] = out["expert_rating_rank"].astype(int)
    return out


def topk_indices(magnitude: np.ndarray, ids: np.ndarray, k: int) -> np.ndarray:
    """Deterministic top-k: magnitude desc, then id asc."""
    k = int(max(0, min(k, len(magnitude))))
    if k == 0:
        return np.array([], dtype=int)
    order = np.lexsort((ids.astype(str), -magnitude.astype(float)))
    return order[:k]


def ceil_pct_k(n: int, pct: float) -> int:
    return int(max(1, math.ceil(n * pct))) if n else 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _safe_float(x) -> float | None:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def spearman_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, Any]:
    rho, p = spearmanr(x, y)
    boots = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2 and len(np.unique(x[idx])) < 2:
            continue
        r, _ = spearmanr(x[idx], y[idx])
        if np.isfinite(r):
            boots.append(float(r))
    lo = hi = None
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5]).tolist()
    return {
        "rho": _safe_float(rho),
        "p_value": _safe_float(p),
        "ci95_low": _safe_float(lo),
        "ci95_high": _safe_float(hi),
        "n": int(n),
        "n_valid_bootstrap": int(len(boots)),
    }


def kendall_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, Any]:
    tau, p = kendalltau(x, y, variant="b")
    boots = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        t, _ = kendalltau(x[idx], y[idx], variant="b")
        if np.isfinite(t):
            boots.append(float(t))
    lo = hi = None
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5]).tolist()
    return {
        "tau_b": _safe_float(tau),
        "p_value": _safe_float(p),
        "ci95_low": _safe_float(lo),
        "ci95_high": _safe_float(hi),
        "n": int(n),
        "n_valid_bootstrap": int(len(boots)),
    }


def graded_gains(ratings: np.ndarray) -> np.ndarray:
    return (2.0 ** (ratings.astype(float) - 1.0)) - 1.0


def ndcg_at_k(magnitude: np.ndarray, ratings: np.ndarray, ids: np.ndarray, k: int) -> float:
    gains = graded_gains(ratings)
    order = topk_indices(magnitude, ids, k)
    ideal = np.sort(gains)[::-1][:k]
    dcg = float(np.sum(gains[order] / np.log2(np.arange(2, len(order) + 2))))
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def ndcg_suite(magnitude: np.ndarray, ratings: np.ndarray, ids: np.ndarray) -> dict[str, Any]:
    n = len(magnitude)
    k10 = ceil_pct_k(n, 0.10)
    k20 = ceil_pct_k(n, 0.20)
    k_pos = int((ratings >= 4).sum())
    return {
        "ndcg_at_10pct": ndcg_at_k(magnitude, ratings, ids, k10),
        "ndcg_at_20pct": ndcg_at_k(magnitude, ratings, ids, k20),
        "ndcg_at_k_positive": ndcg_at_k(magnitude, ratings, ids, k_pos) if k_pos else None,
        "ndcg_full": ndcg_at_k(magnitude, ratings, ids, n),
        "k_10pct": k10,
        "k_20pct": k20,
        "k_positive": k_pos,
    }


def binary_discrimination(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    classes = np.unique(y_true)
    prevalence = float(y_true.mean()) if len(y_true) else None
    if len(classes) < 2:
        reason = "only_one_reference_class_present"
        return {
            "auroc": None,
            "auprc": None,
            "prevalence_expert_positive": prevalence,
            "null_reason": reason,
        }
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "prevalence_expert_positive": prevalence,
        "null_reason": None,
    }


def binary_with_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, Any]:
    base = binary_discrimination(y_true, y_score)
    if base["null_reason"]:
        return {
            **base,
            "auroc_ci95_low": None,
            "auroc_ci95_high": None,
            "auprc_ci95_low": None,
            "auprc_ci95_high": None,
            "n_valid_bootstrap": 0,
        }
    aurocs, auprcs = [], []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(float(roc_auc_score(y_true[idx], y_score[idx])))
        auprcs.append(float(average_precision_score(y_true[idx], y_score[idx])))
    out = dict(base)
    if aurocs:
        lo, hi = np.percentile(aurocs, [2.5, 97.5]).tolist()
        out["auroc_ci95_low"], out["auroc_ci95_high"] = float(lo), float(hi)
    else:
        out["auroc_ci95_low"] = out["auroc_ci95_high"] = None
    if auprcs:
        lo, hi = np.percentile(auprcs, [2.5, 97.5]).tolist()
        out["auprc_ci95_low"], out["auprc_ci95_high"] = float(lo), float(hi)
    else:
        out["auprc_ci95_low"] = out["auprc_ci95_high"] = None
    out["n_valid_bootstrap"] = int(len(aurocs))
    return out


def ndcg20_ci(
    magnitude: np.ndarray,
    ratings: np.ndarray,
    ids: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, Any]:
    n = len(magnitude)
    k20 = ceil_pct_k(n, 0.20)
    point = ndcg_at_k(magnitude, ratings, ids, k20)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(ndcg_at_k(magnitude[idx], ratings[idx], ids[idx], ceil_pct_k(n, 0.20)))
    lo, hi = np.percentile(boots, [2.5, 97.5]).tolist()
    return {
        "ndcg_at_20pct": float(point),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "k_20pct": k20,
        "n_valid_bootstrap": int(len(boots)),
    }


def topk_agreement(
    magnitude: np.ndarray,
    ratings: np.ndarray,
    ids: np.ndarray,
    k: int,
    *,
    expert_positive: np.ndarray | None = None,
) -> dict[str, Any]:
    """Set-overlap vs expert positives. Prefer expert_top10_reference when provided."""
    if expert_positive is None:
        # Legacy fallback only; primary path passes expert_top10_reference.
        y_true = (ratings >= 4).astype(int)
    else:
        y_true = np.asarray(expert_positive, dtype=int)
    pos = set(np.flatnonzero(y_true == 1).tolist())
    top = set(topk_indices(magnitude, ids, k).tolist())
    tp = len(top & pos)
    precision = tp / k if k else None
    recall = tp / len(pos) if pos else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    union = top | pos
    jaccard = (len(top & pos) / len(union)) if union else None
    prevalence = float(y_true.mean()) if len(y_true) else None
    lift = (precision / prevalence) if (precision is not None and prevalence and prevalence > 0) else None
    return {
        "k": int(k),
        "precision_at_k": _safe_float(precision),
        "recall_at_k": _safe_float(recall),
        "f1_at_k": _safe_float(f1),
        "jaccard_top_vs_expert_positive": _safe_float(jaccard),
        "lift_over_prevalence": _safe_float(lift),
        "tie_handling": TIE_METHOD,
    }


def attribution_by_rating(magnitude: np.ndarray, ratings: np.ndarray) -> dict[str, Any]:
    rows = []
    groups = []
    for r in range(1, 6):
        vals = magnitude[ratings == r]
        if len(vals):
            groups.append(vals)
        q1 = float(np.percentile(vals, 25)) if len(vals) else None
        q3 = float(np.percentile(vals, 75)) if len(vals) else None
        rows.append(
            {
                "rating": r,
                "count": int(len(vals)),
                "mean_magnitude": float(vals.mean()) if len(vals) else None,
                "median_magnitude": float(np.median(vals)) if len(vals) else None,
                "iqr": (q3 - q1) if (q1 is not None and q3 is not None) else None,
            }
        )
    if len(groups) >= 2 and all(len(g) > 0 for g in groups):
        stat, p = kruskal(*groups)
        kw = {"statistic": float(stat), "p_value": float(p), "n_groups": len(groups)}
    else:
        kw = {"statistic": None, "p_value": None, "n_groups": len(groups), "note": "insufficient_groups"}
    return {
        "by_rating": rows,
        "kruskal_wallis": kw,
        "label": "distributional_diagnostic_not_ranking_metric",
    }


def permutation_null(
    magnitude: np.ndarray,
    ratings: np.ndarray,
    *,
    rng: np.random.Generator,
    n_perm: int,
) -> dict[str, Any]:
    obs_rho, _ = spearmanr(magnitude, ratings)
    obs_tau, _ = kendalltau(magnitude, ratings, variant="b")
    rho_null, tau_null = [], []
    for _ in range(n_perm):
        shuffled = rng.permutation(magnitude)
        r, _ = spearmanr(shuffled, ratings)
        t, _ = kendalltau(shuffled, ratings, variant="b")
        if np.isfinite(r):
            rho_null.append(float(r))
        if np.isfinite(t):
            tau_null.append(float(t))
    def p_two_sided(obs, null):
        null = np.asarray(null, dtype=float)
        if null.size == 0 or obs is None or not np.isfinite(obs):
            return None
        return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (null.size + 1))

    return {
        "observed_spearman": _safe_float(obs_rho),
        "observed_kendall_tau_b": _safe_float(obs_tau),
        "spearman_null_mean": float(np.mean(rho_null)) if rho_null else None,
        "kendall_null_mean": float(np.mean(tau_null)) if tau_null else None,
        "spearman_permutation_p": p_two_sided(obs_rho, np.asarray(rho_null)),
        "kendall_permutation_p": p_two_sided(obs_tau, np.asarray(tau_null)),
        "n_permutations": int(n_perm),
        "n_valid_spearman": int(len(rho_null)),
        "n_valid_kendall": int(len(tau_null)),
    }


def msoa_block_from_name(name: str) -> str | None:
    """Extract MSOA numeric token from ONS LSOA name (e.g. '... 025A' -> '025')."""
    m = re.search(r"(\d{3})[A-Za-z]?$", str(name).strip())
    return m.group(1) if m else None


def spatial_block_bootstrap_spearman(
    magnitude: np.ndarray,
    ratings: np.ndarray,
    blocks: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, Any]:
    """Resample unique spatial blocks with replacement, then Spearman."""
    uniq = np.unique(blocks)
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(blocks == b) for b in chosen])
        if len(idx) < 3:
            continue
        r, _ = spearmanr(magnitude[idx], ratings[idx])
        if np.isfinite(r):
            boots.append(float(r))
    lo = hi = None
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5]).tolist()
    return {
        "method": "msoa_block_bootstrap_from_lsoa_name",
        "n_blocks": int(len(uniq)),
        "ci95_low": _safe_float(lo),
        "ci95_high": _safe_float(hi),
        "n_valid_bootstrap": int(len(boots)),
        "note": (
            "Primary geographic uncertainty: resample MSOA tokens embedded in ONS LSOA names. "
            "This is not a ward adjacency graph."
        ),
    }


def compute_alignment_metrics(
    frame: pd.DataFrame,
    *,
    id_col: str,
    rating_col: str = "final_expert_rating",
    seed: int = RANDOM_SEED,
    n_boot: int = BOOTSTRAP_REPS,
    n_perm: int = PERMUTATION_REPS,
    spatial_blocks: np.ndarray | None = None,
) -> dict[str, Any]:
    if "final_expert_rating" in frame.columns and rating_col == "final_expert_rating":
        ratings = frame["final_expert_rating"].to_numpy(dtype=int)
    else:
        ratings = frame[rating_col].to_numpy(dtype=int)
    mag = frame["attribution_magnitude"].to_numpy(dtype=float)
    ids = frame[id_col].astype(str).to_numpy()
    rng = np.random.default_rng(seed)

    # Primary: top-10% ATT suite when flags present
    att_block = None
    if "expert_top10_reference" in frame.columns and "geosmile_top10_prediction" in frame.columns:
        work = frame.copy()
        if "geosmile_abs_attribution" not in work.columns:
            work["geosmile_abs_attribution"] = work["attribution_magnitude"]
        att_block = compute_att_suite(work, id_col=id_col)

    spearman = spearman_with_ci(mag, ratings, rng=rng, n_boot=n_boot)
    kendall = kendall_with_ci(mag, ratings, rng=rng, n_boot=n_boot)
    ndcg = ndcg_suite(mag, ratings, ids)
    ndcg_ci = ndcg20_ci(mag, ratings, ids, rng=np.random.default_rng(seed + 1), n_boot=n_boot)

    expert_pos = (
        frame["expert_top10_reference"].to_numpy(dtype=int)
        if "expert_top10_reference" in frame.columns
        else (ratings >= 4).astype(int)
    )
    binary = binary_with_ci(expert_pos, mag, rng=np.random.default_rng(seed + 2), n_boot=n_boot)

    top = {
        "top_10pct": topk_agreement(
            mag, ratings, ids, ndcg["k_10pct"], expert_positive=expert_pos
        ),
        "top_20pct": topk_agreement(
            mag, ratings, ids, ndcg["k_20pct"], expert_positive=expert_pos
        ),
    }
    dist = attribution_by_rating(mag, ratings)
    perm = permutation_null(mag, ratings, rng=np.random.default_rng(seed + 3), n_perm=n_perm)

    spatial = None
    if spatial_blocks is not None:
        spatial = spatial_block_bootstrap_spearman(
            mag, ratings, spatial_blocks, rng=np.random.default_rng(seed + 4), n_boot=n_boot
        )
        ordinary = spearman_with_ci(mag, ratings, rng=np.random.default_rng(seed + 5), n_boot=n_boot)
        spearman["primary_uncertainty"] = "spatial_block_bootstrap"
        spearman["spatial_block_ci95_low"] = spatial["ci95_low"]
        spearman["spatial_block_ci95_high"] = spatial["ci95_high"]
        spearman["ordinary_bootstrap_exploratory"] = {
            "method": "ordinary_lsoa_bootstrap_exploratory",
            "ci95_low": ordinary["ci95_low"],
            "ci95_high": ordinary["ci95_high"],
            "n_valid_bootstrap": ordinary["n_valid_bootstrap"],
            "note": (
                "Exploratory only: ordinary unit bootstrap does not account for spatial dependence."
            ),
        }
    else:
        spearman["primary_uncertainty"] = "ordinary_unit_bootstrap"

    return {
        "primary_att": att_block,
        "ordinal_rank_alignment": {"spearman": spearman, "kendall_tau_b": kendall},
        "graded_ranking_quality": {**ndcg, "ndcg_at_20pct_bootstrap": ndcg_ci},
        "binary_discrimination": binary,
        "top_ranking_agreement": top,
        "attribution_distribution_by_expert_rating": dist,
        "permutation_null": perm,
        "spatial_uncertainty": spatial,
        "n": int(len(frame)),
        "tie_handling": TIE_METHOD,
        "primary_binary_definition": "expert_top10_reference",
    }


# ---------------------------------------------------------------------------
# Sensitivity filters
# ---------------------------------------------------------------------------


def filter_confidence(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    conf = frame["confidence"].astype(str)
    if subset == "all":
        return frame
    if subset == "high_medium":
        return frame[conf.isin(["High", "Medium"])].copy()
    if subset == "high":
        return frame[conf == "High"].copy()
    raise ValueError(subset)


def sufficient_for_metrics(frame: pd.DataFrame, rating_col: str = "reference_rating") -> tuple[bool, str]:
    if len(frame) < 5:
        return False, "n_lt_5"
    ratings = frame[rating_col]
    if ratings.nunique() < 2:
        return False, "single_rating_value"
    return True, ""


# ---------------------------------------------------------------------------
# Long-format rows / evaluation_full_v2
# ---------------------------------------------------------------------------


def _metric_rows(
    *,
    branch: str,
    evaluation_scope: str,
    reference_variant: str,
    confidence_subset: str,
    metrics: dict[str, Any],
    attribution_source: str,
    notes: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cat = "Expert-reference attribution accuracy"
    status = ANNOTATION_STATUS

    def add(metric, value, **kw):
        rows.append(
            {
                "branch": branch,
                "evaluation_scope": evaluation_scope,
                "reference_variant": reference_variant,
                "confidence_subset": confidence_subset,
                "category": cat,
                "metric": metric,
                "value": value,
                "ci95_low": kw.get("ci95_low"),
                "ci95_high": kw.get("ci95_high"),
                "p_value": kw.get("p_value"),
                "n": kw.get("n", metrics.get("n")),
                "k": kw.get("k"),
                "annotation_status": status,
                "attribution_source": attribution_source,
                "notes": kw.get("notes", notes),
            }
        )

    sp = metrics["ordinal_rank_alignment"]["spearman"]
    # Primary ATT metrics first
    att = (metrics.get("primary_att") or {}).get("att") or {}
    ov = (metrics.get("primary_att") or {}).get("overlap") or {}
    for key in [
        "ATT accuracy",
        "ATT precision",
        "ATT recall",
        "ATT F1",
        "ATT AUROC",
        "ATT AUPRC",
        "balanced accuracy",
        "MCC",
    ]:
        if key in att:
            add(
                key,
                att[key],
                k=att.get("k_true"),
                n=att.get("n"),
                notes=att.get("null_reason") or "primary_fixed_expert_topk_reference",
            )
    for key in [
        "intersection",
        "Precision@k",
        "Recall@k",
        "F1@k",
        "Jaccard@k",
        "overlap coefficient",
        "lift",
    ]:
        if key in ov:
            add(key, ov[key], k=ov.get("k_expert"), n=ov.get("n"), notes="equal_size_topk_sets")

    add("Spearman rho", sp["rho"], ci95_low=sp.get("ci95_low"), ci95_high=sp.get("ci95_high"), p_value=sp.get("p_value"), n=sp.get("n"), notes="secondary_graded")
    if sp.get("spatial_block_ci95_low") is not None:
        add(
            "Spearman rho spatial-block CI",
            sp["rho"],
            ci95_low=sp.get("spatial_block_ci95_low"),
            ci95_high=sp.get("spatial_block_ci95_high"),
            notes="primary geographic uncertainty",
        )
    kd = metrics["ordinal_rank_alignment"]["kendall_tau_b"]
    add("Kendall tau-b", kd["tau_b"], ci95_low=kd.get("ci95_low"), ci95_high=kd.get("ci95_high"), p_value=kd.get("p_value"), n=kd.get("n"), notes="secondary_graded")

    gq = metrics["graded_ranking_quality"]
    add("nDCG@10%", gq["ndcg_at_10pct"], k=gq["k_10pct"], notes="secondary_graded")
    add(
        "nDCG@20%",
        gq["ndcg_at_20pct"],
        k=gq["k_20pct"],
        ci95_low=gq["ndcg_at_20pct_bootstrap"]["ci95_low"],
        ci95_high=gq["ndcg_at_20pct_bootstrap"]["ci95_high"],
        notes="secondary_graded",
    )
    add("nDCG full", gq["ndcg_full"], k=gq.get("n", metrics["n"]), notes="secondary_graded")

    bd = metrics["binary_discrimination"]
    add(
        "AUROC (top10% reference scores)",
        bd["auroc"],
        ci95_low=bd.get("auroc_ci95_low"),
        ci95_high=bd.get("auroc_ci95_high"),
        notes=bd.get("null_reason") or notes,
    )
    add(
        "AUPRC (top10% reference scores)",
        bd["auprc"],
        ci95_low=bd.get("auprc_ci95_low"),
        ci95_high=bd.get("auprc_ci95_high"),
        notes=bd.get("null_reason") or notes,
    )

    for label, block in metrics["top_ranking_agreement"].items():
        if not block:
            continue
        add(f"Precision@{label}", block["precision_at_k"], k=block["k"])
        add(f"Recall@{label}", block["recall_at_k"], k=block["k"])
        add(f"F1@{label}", block["f1_at_k"], k=block["k"])
        add(f"Jaccard@{label}", block["jaccard_top_vs_expert_positive"], k=block["k"])
        add(f"Lift@{label}", block["lift_over_prevalence"], k=block["k"])

    perm = metrics["permutation_null"]
    add("Spearman permutation p", perm["spearman_permutation_p"], p_value=perm["spearman_permutation_p"])
    add("Kendall permutation p", perm["kendall_permutation_p"], p_value=perm["kendall_permutation_p"])

    kw = metrics["attribution_distribution_by_expert_rating"]["kruskal_wallis"]
    add(
        "Kruskal-Wallis p (distributional diagnostic)",
        kw.get("p_value"),
        p_value=kw.get("p_value"),
        notes="distributional_diagnostic_not_ranking_metric",
    )
    return rows


def build_evaluation_full_v2(
    existing_full: pd.DataFrame,
    expert_long: pd.DataFrame,
) -> pd.DataFrame:
    """Recategorise cardinality-vs-WD and append expert-reference metrics."""
    df = existing_full.copy()
    # Normalise column names
    colmap = {c: c for c in df.columns}
    for c in list(df.columns):
        cl = c.lower()
        if cl == "branch":
            colmap[c] = "Branch"
        elif cl == "category":
            colmap[c] = "Category"
        elif cl == "metric":
            colmap[c] = "Metric"
        elif cl == "value":
            colmap[c] = "Value"
    df = df.rename(columns=colmap)

    rename_map = {
        "Feature Spearman (cardinality vs WD)": "Feature Spearman: perturbation cardinality vs WD",
        "Feature Pearson (cardinality vs WD)": "Feature Pearson: perturbation cardinality vs WD",
        "Geo Spearman (cardinality vs WD)": "Geo Spearman: perturbation cardinality vs WD",
        "Geo Pearson (cardinality vs WD)": "Geo Pearson: perturbation cardinality vs WD",
    }
    mask_faith = df["Category"].astype(str).str.lower().eq("faithfulness")
    df.loc[mask_faith, "Category"] = "Perturbation response monotonicity"
    df["Metric"] = df["Metric"].replace(rename_map)

    # Also catch metric text that still says Faithfulness category elsewhere
    metric_mask = df["Metric"].astype(str).str.contains("cardinality vs WD", case=False, na=False)
    df.loc[metric_mask, "Category"] = "Perturbation response monotonicity"

    expert_rows = []
    for _, r in expert_long.iterrows():
        expert_rows.append(
            {
                "Branch": r["branch"],
                "Category": r["category"],
                "Metric": (
                    f"{r['metric']} [{r['evaluation_scope']}|{r['reference_variant']}|{r['confidence_subset']}]"
                ),
                "Value": r["value"],
            }
        )
    out = pd.concat([df, pd.DataFrame(expert_rows)], ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def load_annotations(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat = pd.read_excel(path, sheet_name="Feature_Annotations")
    geo = pd.read_excel(path, sheet_name="LSOA_Annotations")
    return feat, geo


def evaluate_feature_branch(
    attr_path: Path,
    annotations: pd.DataFrame,
    *,
    branch: str = BRANCH_FEATURE,
    evaluation_scope: str = "complete_30_feature",
    require_full: bool = True,
    expected_n: int | None = 30,
    seed: int,
    n_boot: int,
    n_perm: int,
) -> dict[str, Any]:
    attr = pd.read_csv(attr_path)
    if "local" in attr_path.name.lower() and "global" not in attr_path.name.lower():
        raise SystemExit(f"Refusing local-effect file as global attribution: {attr_path}")

    merged, cov = merge_feature(attr, annotations, branch=branch, source_file=attr_path.name)
    if require_full:
        assert_primary_coverage(cov, expected_n=expected_n or 30, label=branch)

    sensitivities = []
    long_rows: list[dict[str, Any]] = []
    for conf_subset in ["all", "high_medium", "high"]:
        sub = filter_confidence(merged, conf_subset)
        ok, reason = sufficient_for_metrics(sub, "final_expert_rating")
        entry = {
            "confidence_subset": conf_subset,
            "n": int(len(sub)),
            "reference_variant": "external_crime_informed_reference",
        }
        if not ok:
            entry["skipped"] = True
            entry["skip_reason"] = reason
            sensitivities.append(entry)
            continue
        # Recompute top-10% flags on subset relative to full-universe ranks already set;
        # for sensitivity subsets keep existing flags from full ranking.
        metrics = compute_alignment_metrics(
            sub,
            id_col="feature",
            rating_col="final_expert_rating",
            seed=seed,
            n_boot=n_boot,
            n_perm=n_perm,
        )
        entry["skipped"] = False
        entry["metrics"] = metrics
        sensitivities.append(entry)
        long_rows.extend(
            _metric_rows(
                branch=branch,
                evaluation_scope=evaluation_scope,
                reference_variant="external_crime_informed_reference",
                confidence_subset=conf_subset,
                metrics=metrics,
                attribution_source=attr_path.name,
            )
        )

    primary = next(s for s in sensitivities if s["confidence_subset"] == "all" and not s.get("skipped"))
    return {
        "branch": branch,
        "evaluation_scope": evaluation_scope,
        "attribution_source": attr_path.name,
        "coverage": cov,
        "primary_metrics": primary["metrics"],
        "sensitivity_analyses": sensitivities,
        "long_rows": long_rows,
        "merged": merged,
        "warnings": [],
    }


def evaluate_geo_branch(
    attr_path: Path,
    annotations: pd.DataFrame,
    *,
    seed: int,
    n_boot: int,
    n_perm: int,
) -> dict[str, Any]:
    attr = pd.read_csv(attr_path)
    if "local" in attr_path.name.lower() and "global" not in attr_path.name.lower():
        if "effects" in attr_path.name.lower():
            raise SystemExit(f"Refusing local-effect file as global attribution: {attr_path}")

    merged, cov = merge_geo(attr, annotations, branch=BRANCH_GEO, source_file=attr_path.name)
    assert_primary_coverage(cov, expected_n=168, label=BRANCH_GEO)

    blocks = np.array([msoa_block_from_name(x) or "UNK" for x in merged["lsoa21nm"]])

    sensitivities = []
    long_rows: list[dict[str, Any]] = []

    for conf_subset in ["all", "high_medium", "high"]:
        sub = filter_confidence(merged, conf_subset)
        ok, reason = sufficient_for_metrics(sub, "final_expert_rating")
        entry = {
            "confidence_subset": conf_subset,
            "n": int(len(sub)),
            "reference_variant": "external_crime_informed_reference",
        }
        if not ok:
            entry["skipped"] = True
            entry["skip_reason"] = reason
            sensitivities.append(entry)
            continue
        blk = np.array([msoa_block_from_name(x) or "UNK" for x in sub["lsoa21nm"]])
        metrics = compute_alignment_metrics(
            sub,
            id_col="lsoa21cd",
            rating_col="final_expert_rating",
            seed=seed,
            n_boot=n_boot,
            n_perm=n_perm,
            spatial_blocks=blk,
        )
        entry["skipped"] = False
        entry["metrics"] = metrics
        sensitivities.append(entry)
        long_rows.extend(
            _metric_rows(
                branch=BRANCH_GEO,
                evaluation_scope="complete_168_lsoa",
                reference_variant="external_crime_informed_reference",
                confidence_subset=conf_subset,
                metrics=metrics,
                attribution_source=attr_path.name,
            )
        )

    # Sensitivity: no_crime_domain_reference (graded only; top-10% flags stay crime-informed)
    work = merged.copy()
    work["final_expert_rating"] = work["no_crime_reference_rating"]
    work["reference_rating"] = work["no_crime_reference_rating"]
    metrics = compute_alignment_metrics(
        work,
        id_col="lsoa21cd",
        rating_col="final_expert_rating",
        seed=seed,
        n_boot=n_boot,
        n_perm=n_perm,
        spatial_blocks=blocks,
    )
    sensitivities.append(
        {
            "confidence_subset": "all",
            "n": int(len(work)),
            "reference_variant": "no_crime_domain_reference",
            "skipped": False,
            "metrics": metrics,
        }
    )
    long_rows.extend(
        _metric_rows(
            branch=BRANCH_GEO,
            evaluation_scope="complete_168_lsoa",
            reference_variant="no_crime_domain_reference",
            confidence_subset="all",
            metrics=metrics,
            attribution_source=attr_path.name,
            notes="Sensitivity excluding IoD Crime Domain from annotation rubric",
        )
    )

    primary = next(
        s
        for s in sensitivities
        if s.get("reference_variant") == "external_crime_informed_reference"
        and s.get("confidence_subset") == "all"
        and not s.get("filter")
        and not s.get("skipped")
    )
    return {
        "branch": BRANCH_GEO,
        "evaluation_scope": "complete_168_lsoa",
        "attribution_source": attr_path.name,
        "coverage": cov,
        "primary_metrics": primary["metrics"],
        "sensitivity_analyses": sensitivities,
        "long_rows": long_rows,
        "merged": merged,
        "warnings": [],
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return None  # dropped from JSON
    return obj


def run(args: argparse.Namespace) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    ann_path = Path(args.annotations)
    if not ann_path.exists():
        raise SystemExit(f"Missing annotations workbook: {ann_path}")

    feat_ann, geo_ann = load_annotations(ann_path)
    warnings_all: list[str] = []
    results = []

    feat_path = OUT / "feature_global_importance.csv"
    if not feat_path.exists():
        raise SystemExit(f"Missing {feat_path}")
    print(f"Evaluating {BRANCH_FEATURE} (complete 30-player set)...")
    feat_res = evaluate_feature_branch(
        feat_path,
        feat_ann,
        branch=BRANCH_FEATURE,
        evaluation_scope="complete_30_feature",
        require_full=True,
        expected_n=30,
        seed=args.seed,
        n_boot=args.bootstrap,
        n_perm=args.permutations,
    )
    results.append(feat_res)

    geo_path = OUT / "geo_coefficients.csv"
    if not geo_path.exists():
        raise SystemExit(f"Missing {geo_path}")
    print(f"Evaluating {BRANCH_GEO} (168 LSOAs)...")
    geo_res = evaluate_geo_branch(
        geo_path, geo_ann, seed=args.seed, n_boot=args.bootstrap, n_perm=args.permutations
    )
    results.append(geo_res)

    # Alignment table (per-unit)
    align_rows = []
    feat_merged = feat_res["merged"]
    for _, r in feat_merged.iterrows():
        align_rows.append(
            {
                "branch": BRANCH_FEATURE,
                "unit_id": r["feature"],
                "unit_name": r["feature"],
                "final_expert_rating": int(r["final_expert_rating"]),
                "expert_relevance_score": np.nan,
                "expert_rank": int(r["expert_rank"]),
                "expert_top10_reference": int(r["expert_top10_reference"]),
                "geosmile_coefficient": r.get("geosmile_coefficient", r.get("attribution_coefficient")),
                "geosmile_abs_attribution": float(r["geosmile_abs_attribution"]),
                "geosmile_rank": int(r["geosmile_rank"]),
                "geosmile_top10_prediction": int(r["geosmile_top10_prediction"]),
                "matched_top10": int(r["matched_top10"]),
                "confidence": r.get("confidence"),
                "rationale": r.get("mechanism_rationale"),
            }
        )
    geo_merged = geo_res["merged"]
    for _, r in geo_merged.iterrows():
        align_rows.append(
            {
                "branch": BRANCH_GEO,
                "unit_id": r["lsoa21cd"],
                "unit_name": r.get("lsoa21nm"),
                "final_expert_rating": int(r["final_expert_rating"]),
                "expert_relevance_score": r.get("expert_relevance_score"),
                "expert_rank": int(r["expert_rank"]),
                "expert_top10_reference": int(r["expert_top10_reference"]),
                "geosmile_coefficient": r.get("geosmile_coefficient", r.get("geo_coefficient")),
                "geosmile_abs_attribution": float(r["geosmile_abs_attribution"]),
                "geosmile_rank": int(r["geosmile_rank"]),
                "geosmile_top10_prediction": int(r["geosmile_top10_prediction"]),
                "matched_top10": int(r["matched_top10"]),
                "confidence": r.get("confidence"),
                "rationale": r.get("rationale"),
            }
        )
    align_df = pd.DataFrame(align_rows)
    align_df.to_csv(OUT / "geosmile_hull_expert_top10_alignment.csv", index=False)

    # Keep merged diagnostics (Feature + Geo only; no Feature-A/B)
    feat_merged[
        [
            c
            for c in [
                "feature",
                "geosmile_coefficient",
                "geosmile_abs_attribution",
                "final_expert_rating",
                "expert_rank",
                "expert_top10_reference",
                "geosmile_rank",
                "geosmile_top10_prediction",
                "matched_top10",
                "confidence",
                "relevance_label",
                "source_branch",
            ]
            if c in feat_merged.columns
        ]
    ].to_csv(OUT / "geosmile_hull_expert_alignment_merged_feature.csv", index=False)

    geo_keep = [
        c
        for c in [
            "lsoa21cd",
            "lsoa21nm",
            "geo_coefficient",
            "geosmile_abs_attribution",
            "final_expert_rating",
            "expert_relevance_score",
            "no_crime_reference_rating",
            "expert_rank",
            "expert_top10_reference",
            "geosmile_rank",
            "geosmile_top10_prediction",
            "matched_top10",
            "expected_crime_context",
            "confidence",
            "crime_domain_dependence",
            "rationale",
            "population_mid_2022",
        ]
        if c in geo_merged.columns
    ]
    geo_merged[geo_keep].to_csv(OUT / "geosmile_hull_expert_alignment_merged_geo.csv", index=False)

    long_df = pd.DataFrame([row for res in results for row in res["long_rows"]])
    long_df.to_csv(OUT / "geosmile_hull_expert_alignment_full.csv", index=False)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "annotation_filename": ann_path.name,
        "annotation_status": ANNOTATION_STATUS,
        "expert_display_note": EXPERT_DISPLAY_NOTE,
        "active_branches": [BRANCH_FEATURE, BRANCH_GEO],
        "expert_topk_definition": {
            "rule": "fixed_expert_important_sets",
            "feature_n": 30,
            "feature_k": FEATURE_TOP_K,
            "geo_n": 168,
            "geo_k": GEO_TOP_K,
            "map_geo_k": GEO_TOP_K,
            "geo_selection": (
                "expert_top20_reference"
                if "expert_top20_reference" in geo_ann.columns
                else "rating_score_confidence_rank"
            ),
            "primary_binary": "expert_top10_reference",
            "not_used_as_primary": "rating >= 4",
        },
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": args.seed,
        "bootstrap_repetitions": args.bootstrap,
        "permutation_repetitions": args.permutations,
        "limitations": [
            EXPERT_DISPLAY_NOTE,
            "Labels are not objective ground truth and do not establish causality.",
            "Absolute attribution magnitudes are used; coefficient sign is not interpreted as crime increase/decrease.",
            "Expert-reference attribution accuracy is not model faithfulness.",
            "Feature multicollinearity (lags/rolling means) can inflate or dilute individual feature attributions.",
            "Geo spatial dependence means ordinary LSOA bootstrap intervals are exploratory.",
        ],
        "warnings": warnings_all,
        "branches": [],
    }
    for res in results:
        summary["branches"].append(
            {
                "branch": res["branch"],
                "evaluation_scope": res["evaluation_scope"],
                "attribution_source": res["attribution_source"],
                "coverage": res["coverage"],
                "primary_metrics": _json_safe(res["primary_metrics"]),
                "sensitivity_analyses": _json_safe(
                    [
                        {k: v for k, v in s.items() if k != "metrics"}
                        | (
                            {"metrics": _json_safe(s["metrics"])}
                            if "metrics" in s and not s.get("skipped")
                            else {}
                        )
                        for s in res["sensitivity_analyses"]
                    ]
                ),
                "warnings": res.get("warnings", []),
            }
        )

    (OUT / "geosmile_hull_expert_alignment_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )

    print("\n=== Expert-reference attribution accuracy ===")
    print(EXPERT_DISPLAY_NOTE)
    for res in results:
        pm = res["primary_metrics"]
        att = (pm.get("primary_att") or {}).get("att") or {}
        ov = (pm.get("primary_att") or {}).get("overlap") or {}
        print(
            f"\n{res['branch']} coverage={res['coverage']['n_matched']}/"
            f"{res['coverage']['n_annotation_rows']}"
        )
        print(
            f"  ATT accuracy={att.get('ATT accuracy')}  F1={att.get('ATT F1')}  "
            f"AUROC={att.get('ATT AUROC')}  Jaccard@10%={ov.get('Jaccard@10%')}"
        )
        print(
            f"  expert_top10 count={(pm.get('primary_att') or {}).get('expert_ids') and len((pm.get('primary_att') or {}).get('expert_ids', []))}"
        )
    print(f"\nWrote alignment outputs under {OUT.resolve()}")
    return {"results": results, "summary": summary, "alignment": align_df, "long_df": long_df}


_METHOD_MD = """# Hull GeoSMILE expert-reference attribution accuracy

## Expert reference

Workbook: `hull_expert_top20.xlsx`  
Status: `expert_reference`

Primary evaluation uses fixed expert top sets:
- Feature: top 10 of 30
- Geo: top 20 of 168

## What this evaluation measures

**Expert-reference attribution accuracy** quantifies agreement between model attribution magnitudes and the expert reference.

It is distinct from:

- **Model faithfulness** — agreement with model behaviour under intervention
- **Perturbation response monotonicity** — correlation of mask cardinality with Wasserstein distance
- **Consistency** — repeated execution of the same explanation
- **Controlled-input stability** — prediction-preserving input changes (not evaluated here)

## Reference-rating construction

```
reference_rating = human_override_1to5
    if human_override_1to5 is non-null
    else expert_relevance_1to5
```

`expert_positive = 1` if `reference_rating >= 4`, else `0`.

## Why absolute coefficients

SMILE surrogates respond to a non-negative distributional distance (Wasserstein-1). Coefficient sign is not interpreted as increasing vs decreasing predicted crime.

## Joins and coverage

- Features join on exact `feature` IDs to `feature_global_importance.csv`.
- Geo joins on `lsoa21cd` to `geo_coefficients.csv` using `geo_abs_coefficient`.
- Primary Feature expects 30/30 coverage; primary Geo expects 168/168.

## Uncertainty

- Deterministic seed `42`, bootstrap and permutation intervals as configured.
- Geo geographic uncertainty uses an MSOA-block bootstrap derived from ONS LSOA names.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--permutations", type=int, default=PERMUTATION_REPS)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
