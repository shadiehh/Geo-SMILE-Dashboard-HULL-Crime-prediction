"""Compact JSON-safe evidence packages for the LLM. Never send raw CSVs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dashboard.data.loader import HullBundle, compact_evaluation, magnitude_class, top_features_for_lsoa
from dashboard.data.schema import LISA_LABELS, LSOA_ID, LSOA_NAME, MAP_VIEWS, STAKEHOLDERS, human_label
from dashboard.llm.prompt import SYSTEM_PROMPT_VERSION


def _num(value: Any):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_num(v) if not isinstance(v, str) else v for v in value]
    return value


def _global_geo(lsoa: pd.DataFrame) -> dict[str, Any]:
    abs_vals = lsoa["geo_sensitivity_abs"].to_numpy(dtype=float)
    return {
        "n_lsoas": int(len(lsoa)),
        "geo_sensitivity_mean_abs": _num(float(np.mean(abs_vals))),
        "geo_sensitivity_median_abs": _num(float(np.median(abs_vals))),
        "geo_sensitivity_p90_abs": _num(float(np.quantile(abs_vals, 0.9))),
        "geo_sensitivity_max_abs": _num(float(np.max(abs_vals))),
        "geo_sensitivity_min_abs": _num(float(np.min(abs_vals))),
        "note": (
            "GeoSensitivity is the Geo-branch location-masking SMILE coefficient, "
            "not latitude or longitude feature attribution."
        ),
    }


def _spatial_global(pattern: dict) -> dict[str, Any]:
    g = pattern.get("GLOBAL_SPATIAL_PATTERN") or {}
    local = pattern.get("LOCAL_SPATIAL_PATTERN") or {}
    return {
        "global_pattern": {
            "moran_i": _num(g.get("Moran_I")),
            "p_value": _num(g.get("p_value")),
            "interpretation": g.get("interpretation"),
            "n_permutations": g.get("n_permutations"),
            "weights": pattern.get("weights"),
            "value_field": pattern.get("value_field"),
        },
        "lisa_counts": local.get("counts"),
        "pattern_scale": "not established",
        "note": (
            "LISA classes describe clustering of |GeoSensitivity|. "
            "They are not crime hotspots and not latitude/longitude effects."
        ),
    }


def _cluster_extent(row: pd.Series, lsoa: pd.DataFrame) -> dict[str, Any] | None:
    members = row.get("cluster_members")
    if not isinstance(members, (list, tuple)) or not members:
        return None
    sub = lsoa[lsoa[LSOA_ID].isin([str(c) for c in members])]
    area = float(sub["lsoa_area_km2"].sum()) if "lsoa_area_km2" in sub.columns else None
    coverage = float(len(sub) / len(lsoa)) if len(lsoa) else None
    return {
        "cluster_id": row.get("cluster_id"),
        "n_lsoas": int(row.get("cluster_size") or len(members)),
        "area_km2": _num(area),
        "coverage": _num(coverage),
    }


def _top_global_features(bundle: HullBundle, k: int = 10) -> list[dict[str, Any]]:
    frame = bundle.global_features.copy()
    if "abs_coefficient" not in frame.columns:
        frame["abs_coefficient"] = frame["coefficient"].abs()
    frame = frame.sort_values("abs_coefficient", ascending=False, kind="mergesort")
    out = []
    for i, rec in enumerate(frame.head(k).itertuples(index=False), start=1):
        name = str(getattr(rec, "feature"))
        out.append(
            {
                "feature": name,
                "feature_label": human_label(name),
                "value": _num(getattr(rec, "coefficient")),
                "abs_value": _num(getattr(rec, "abs_coefficient")),
                "rank": i,
                "source": "published Feature branch (location excluded)",
            }
        )
    return out


def build_evidence(
    bundle: HullBundle,
    *,
    stakeholder_id: str,
    current_map_view: str,
    scope: str,
    selected_lsoa: str | None,
    user_question: str | None = None,
) -> dict[str, Any]:
    lsoa = bundle.lsoa
    geo_stats = _global_geo(lsoa)
    spatial = _spatial_global(bundle.spatial_pattern)
    median = float(geo_stats["geo_sensitivity_median_abs"])
    p90 = float(geo_stats["geo_sensitivity_p90_abs"])

    evidence: dict[str, Any] = {
        "application": {
            "domain": "crime prediction",
            "study_area": "Hull",
            "spatial_unit": "LSOA",
            "target": "next-month crime count",
        },
        "stakeholder": {
            "id": stakeholder_id,
            "label": STAKEHOLDERS[stakeholder_id],
        },
        "current_map_view": current_map_view,
        "current_map_view_label": MAP_VIEWS[current_map_view],
        "scope": scope,
        "prompt_version": SYSTEM_PROMPT_VERSION,
        "semantic_separation": {
            "geo_sensitivity": "Geo branch location-masking SMILE",
            "latitude_effect": "latitude as a Feature SMILE player",
            "longitude_effect": "longitude as a Feature SMILE player",
            "spatial_pattern": "LISA/Moran on |GeoSensitivity|",
            "not_equivalent": [
                "GeoSensitivity != latitude feature effect",
                "GeoSensitivity != longitude feature effect",
                "cluster != high magnitude",
                "rank != strength",
                "attribution != causation",
            ],
        },
        "global_geo": geo_stats,
        "global_spatial_pattern": spatial,
        "top_global_features": _top_global_features(bundle),
        "evaluation": compact_evaluation(bundle.evaluation),
        "limitations": [
            "Not causal.",
            "No RAG / no invented local context.",
            "pattern_scale is not established in the computed outputs.",
            "local_p_adjusted is not available in the published LISA table.",
        ],
    }

    if user_question:
        evidence["user_question"] = user_question

    if scope == "lsoa" and selected_lsoa:
        match = lsoa[lsoa[LSOA_ID] == str(selected_lsoa)]
        if match.empty:
            evidence["selected_lsoa_error"] = "LSOA not found in published results"
            return evidence
        row = match.iloc[0]
        abs_geo = float(row["geo_sensitivity_abs"])
        rank = int(row["geo_rank"])
        modest = magnitude_class(abs_geo, median, p90) == "modest"
        high_rank = rank <= 20
        lisa = str(row["lisa_class"])
        selected = {
            "id": str(row[LSOA_ID]),
            "name": str(row[LSOA_NAME]),
            "geo_sensitivity": _num(row["geo_sensitivity"]),
            "geo_sensitivity_abs": _num(abs_geo),
            "geo_rank": rank,
            "geo_rank_of": int(len(lsoa)),
            "geo_magnitude_class": magnitude_class(abs_geo, median, p90),
            "rank_high_magnitude_modest": bool(high_rank and modest),
            "latitude_effect": _num(row["latitude_effect"]),
            "longitude_effect": _num(row["longitude_effect"]),
            "spatial_pattern": {
                "class": lisa,
                "class_label": LISA_LABELS.get(lisa, lisa),
                "local_moran_i": _num(row["local_moran_i"]),
                "local_p": _num(row["local_p"]),
                "cluster_id": row["cluster_id"] if pd.notna(row["cluster_id"]) else None,
                "pattern_scale": "not established",
            },
            "top_features": top_features_for_lsoa(bundle.local_features, str(row[LSOA_ID])),
        }
        extent = _cluster_extent(row, lsoa)
        if extent:
            selected["cluster_extent"] = extent
        evidence["selected_lsoa"] = selected
        if high_rank and modest:
            evidence["magnitude_instruction"] = (
                "This LSOA ranks highly relative to other Hull LSOAs, but its "
                "absolute GeoSensitivity remains modest. Say that explicitly. "
                "Do not say geography strongly drives predictions there."
            )
        elif abs_geo >= p90:
            evidence["magnitude_instruction"] = (
                "Absolute GeoSensitivity is in the upper tail for Hull. "
                "Still do not claim causation, and still distinguish latitude/longitude."
            )
        else:
            evidence["magnitude_instruction"] = (
                "Treat GeoSensitivity magnitude as comparative within Hull. "
                "Do not inflate rank into strength."
            )
    elif scope == "lsoa":
        evidence["selected_lsoa_error"] = "Select an LSOA on the map before requesting a local explanation."

    return evidence


def user_payload(evidence: dict[str, Any], user_question: str | None = None) -> str:
    import json

    task = (
        "Explain the supplied Hull GeoSMILE evidence for the selected stakeholder. "
        "Use only this JSON. Keep GeoSensitivity, latitude effect, longitude effect, "
        "and spatial clustering conceptually separate."
    )
    if user_question:
        task += f"\nFollow-up question: {user_question}"
    return task + "\n\nEVIDENCE_JSON:\n" + json.dumps(evidence, ensure_ascii=True, sort_keys=True)
