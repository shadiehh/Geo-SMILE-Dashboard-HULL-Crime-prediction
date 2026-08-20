"""Fixed expert-important sets and ATT metrics for Hull GeoSMILE.

Primary binary reference is fixed expert top sets (not rating >= 4):
  Feature: top 10 of 30
  Geo: top 20 of 168
GeoSMILE predictions use the same cardinalities.
Maps highlight the Geo expert top 20.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

CONFIDENCE_ORDER = {"High": 3, "Medium": 2, "Low": 1}
BRANCH_FEATURE = "Feature (location excluded)"
BRANCH_GEO = "Geo"
ANNOTATION_STATUS = "expert_reference"
EXPERT_DISPLAY_NOTE = "Expert reference."

# Fixed expert-important / GeoSMILE top-set sizes.
FEATURE_TOP_K = 10
GEO_TOP_K = 20
FEATURE_N = 30
GEO_N = 168


def ceil_pct_k(n: int, pct: float = 0.10) -> int:
    """Legacy helper for nDCG@10%/20%; not used for primary expert sets."""
    return int(math.ceil(n * pct)) if n else 0


def feature_top_k(n: int | None = None) -> int:
    return FEATURE_TOP_K


def geo_top_k(n: int | None = None) -> int:
    return GEO_TOP_K


def confidence_rank(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda x: CONFIDENCE_ORDER.get(x, 0)).astype(int)


def select_expert_top_feature(frame: pd.DataFrame, *, k: int | None = None) -> pd.DataFrame:
    """Sort Feature annotations; mark exactly k units as expert_top10_reference=1."""
    out = frame.copy()
    n = len(out)
    k = FEATURE_TOP_K if k is None else int(k)
    k = min(k, n)
    out["_conf_rank"] = confidence_rank(out["confidence"])
    order = out.sort_values(
        ["final_expert_rating", "_conf_rank", "feature"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    out.loc[order.index, "expert_rank"] = np.arange(1, n + 1)
    out["expert_rank"] = out["expert_rank"].astype(int)
    out["expert_top10_reference"] = (out["expert_rank"] <= k).astype(int)
    out = out.drop(columns=["_conf_rank"])
    if int(out["expert_top10_reference"].sum()) != k:
        raise AssertionError(
            f"Feature expert_top10_reference count={out['expert_top10_reference'].sum()} != k={k}"
        )
    return out


# Back-compat alias
select_expert_top10_feature = select_expert_top_feature


def select_expert_top_geo(frame: pd.DataFrame, *, k: int | None = None) -> pd.DataFrame:
    """Mark exactly k Geo LSOAs as expert_top10_reference=1.

    When ``expert_top20_reference`` is present (prediction-oriented workbook v2),
    use that fixed set and ``prediction_expert_rank`` for ordering. Otherwise fall
    back to rating/score/confidence ranking.
    """
    out = frame.copy()
    n = len(out)
    k = GEO_TOP_K if k is None else int(k)
    k = min(k, n)

    if "expert_top20_reference" in out.columns:
        ref = pd.to_numeric(out["expert_top20_reference"], errors="coerce").fillna(0).astype(int)
        if int(ref.sum()) != k:
            raise AssertionError(
                f"Geo expert_top20_reference count={ref.sum()} != k={k}"
            )
        out["expert_top10_reference"] = ref
        if "prediction_expert_rank" in out.columns:
            ranks = pd.to_numeric(out["prediction_expert_rank"], errors="coerce")
            out["expert_rank"] = ranks.fillna(n + 1).astype(int)
        else:
            selected = out.loc[ref == 1].sort_values("lsoa21cd", kind="mergesort")
            rank_map = {code: i + 1 for i, code in enumerate(selected["lsoa21cd"].astype(str))}
            out["expert_rank"] = out["lsoa21cd"].astype(str).map(rank_map).fillna(n + 1).astype(int)
        return out

    out["_conf_rank"] = confidence_rank(out["confidence"])
    if "expert_relevance_score" not in out.columns:
        out["expert_relevance_score"] = np.nan
    out["expert_relevance_score"] = pd.to_numeric(out["expert_relevance_score"], errors="coerce")
    order = out.sort_values(
        ["final_expert_rating", "expert_relevance_score", "_conf_rank", "lsoa21cd"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    out.loc[order.index, "expert_rank"] = np.arange(1, n + 1)
    out["expert_rank"] = out["expert_rank"].astype(int)
    out["expert_top10_reference"] = (out["expert_rank"] <= k).astype(int)
    out = out.drop(columns=["_conf_rank"])
    if int(out["expert_top10_reference"].sum()) != k:
        raise AssertionError(
            f"Geo expert_top10_reference count={out['expert_top10_reference'].sum()} != k={k}"
        )
    return out


select_expert_top10_geo = select_expert_top_geo


def select_expert_map_top20(frame: pd.DataFrame) -> pd.DataFrame:
    """Exactly 20 highest-ranked Geo LSOAs for map display (same as Geo expert set)."""
    ranked = select_expert_top_geo(frame, k=GEO_TOP_K)
    return ranked[ranked["expert_top10_reference"] == 1].sort_values("expert_rank").copy()


# Back-compat aliases
select_expert_map_top17 = select_expert_map_top20  # deprecated alias; design is Geo top-20/168


def select_expert_map_top10(frame: pd.DataFrame) -> pd.DataFrame:
    return select_expert_map_top20(frame)


def select_geosmile_topk(
    magnitude: np.ndarray,
    ids: np.ndarray,
    k: int,
) -> np.ndarray:
    """Deterministic top-k indices: magnitude desc, id asc."""
    k = int(max(0, min(k, len(magnitude))))
    if k == 0:
        return np.array([], dtype=int)
    order = np.lexsort((ids.astype(str), -np.asarray(magnitude, dtype=float)))
    return order[:k]


def assign_geosmile_ranks(frame: pd.DataFrame, *, id_col: str, mag_col: str, k: int) -> pd.DataFrame:
    out = frame.copy()
    order = out.sort_values([mag_col, id_col], ascending=[False, True], kind="mergesort")
    out.loc[order.index, "geosmile_rank"] = np.arange(1, len(order) + 1)
    out["geosmile_rank"] = out["geosmile_rank"].astype(int)
    out["geosmile_top10_prediction"] = (out["geosmile_rank"] <= k).astype(int)
    return out


def _safe_float(x) -> float | None:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def att_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    """ATT metrics with binary preds = GeoSMILE top-k and scores = abs attribution."""
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        matthews_corrcoef,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = (tp + tn) / n if n else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    classes = np.unique(y_true)
    if len(classes) < 2:
        auroc = auprc = None
        null_reason = "only_one_reference_class_present"
    else:
        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))
        null_reason = None
    try:
        bal = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        bal = None
    mcc = None
    mcc_note = None
    try:
        if len(classes) >= 2 and len(np.unique(y_pred)) >= 1:
            mcc = float(matthews_corrcoef(y_true, y_pred))
            if not np.isfinite(mcc):
                mcc = None
                mcc_note = "mcc_not_finite"
    except Exception as exc:  # pragma: no cover
        mcc_note = str(exc)

    return {
        "ATT accuracy": _safe_float(accuracy),
        "ATT precision": _safe_float(precision),
        "ATT recall": _safe_float(recall),
        "ATT F1": _safe_float(f1),
        "ATT AUROC": _safe_float(auroc),
        "ATT AUPRC": _safe_float(auprc),
        "balanced accuracy": _safe_float(bal),
        "MCC": _safe_float(mcc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "null_reason": null_reason,
        "mcc_note": mcc_note,
        "n": int(n),
        "k_true": int(y_true.sum()),
        "k_pred": int(y_pred.sum()),
    }


def set_overlap_metrics(expert_set: set, model_set: set, *, n_universe: int) -> dict[str, Any]:
    inter = expert_set & model_set
    union = expert_set | model_set
    k_e = len(expert_set)
    k_m = len(model_set)
    tp = len(inter)
    precision = tp / k_m if k_m else None
    recall = tp / k_e if k_e else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    jaccard = (tp / len(union)) if union else None
    overlap_coeff = tp / min(k_e, k_m) if min(k_e, k_m) else None
    expected = (k_e * k_m / n_universe) if n_universe else None
    lift = (tp / expected) if (expected and expected > 0) else None
    return {
        "intersection": int(tp),
        "Precision@k": _safe_float(precision),
        "Recall@k": _safe_float(recall),
        "F1@k": _safe_float(f1),
        "Jaccard@k": _safe_float(jaccard),
        "Precision@10%": _safe_float(precision),  # alias for dashboard/compat
        "Recall@10%": _safe_float(recall),
        "F1@10%": _safe_float(f1),
        "Jaccard@10%": _safe_float(jaccard),
        "overlap coefficient": _safe_float(overlap_coeff),
        "lift": _safe_float(lift),
        "k_expert": int(k_e),
        "k_geosmile": int(k_m),
        "n": int(n_universe),
        "equal_size": k_e == k_m,
    }


def compute_att_suite(frame: pd.DataFrame, *, id_col: str) -> dict[str, Any]:
    """Primary ATT + set-overlap for a merged ranked frame."""
    y_true = frame["expert_top10_reference"].to_numpy(dtype=int)
    y_pred = frame["geosmile_top10_prediction"].to_numpy(dtype=int)
    y_score = frame["geosmile_abs_attribution"].to_numpy(dtype=float)
    att = att_binary_metrics(y_true, y_pred, y_score)
    expert_ids = set(frame.loc[frame["expert_top10_reference"] == 1, id_col].astype(str))
    model_ids = set(frame.loc[frame["geosmile_top10_prediction"] == 1, id_col].astype(str))
    overlap = set_overlap_metrics(expert_ids, model_ids, n_universe=len(frame))
    return {"att": att, "overlap": overlap, "expert_ids": sorted(expert_ids), "model_ids": sorted(model_ids)}
