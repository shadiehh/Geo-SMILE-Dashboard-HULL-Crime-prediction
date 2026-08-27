"""Compact evidence summary of the Local Attribution Patterns dashboard results,
so LLM-generated explanations can reference them.

Independent of GeoSensitivity (Geo branch); the compute function is pure
(no Streamlit dependency, directly unit-testable) and the cached wrapper
below it is what the explanation panel actually calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.data.attribution_patterns import (
    attribution_archetypes,
    dominant_local_feature,
    local_attribution_concentration,
)
from dashboard.data.schema import LSOA_ID, human_label
from dashboard.data.spatial_stats import build_queen_weights, local_moran

DEFAULT_ARCHETYPE_K = 4


def _round(value, digits: int = 4):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), digits)


def compute_local_attribution_evidence(
    *,
    local_features: pd.DataFrame,
    global_features: pd.DataFrame,
    geometry,
    feature_names: list[str],
    selected_lsoa: str | None,
) -> dict:
    """Pure: no Streamlit dependency, directly unit-testable."""
    dominant = dominant_local_feature(local_features, feature_names)
    concentration = local_attribution_concentration(local_features, feature_names)
    labels_df, profiles = attribution_archetypes(local_features, feature_names, k=DEFAULT_ARCHETYPE_K)

    top_counts = dominant["dominant_feature_label"].value_counts().head(5)
    dominant_block = {
        "definition": (
            "Per LSOA, the non-location Feature-branch local feature with the "
            "largest |local SMILE coefficient|."
        ),
        "hull_wide_top_features": [
            {"feature_label": str(lbl), "n_lsoas": int(n)} for lbl, n in top_counts.items()
        ],
    }
    concentration_block = {
        "definition": (
            "Normalized entropy (0-1) of |local coefficient| shares across the "
            "non-location features per LSOA. Near 1 = many features contribute "
            "about equally (diffuse); near 0 = one or two features dominate "
            "(concentrated)."
        ),
        "hull_wide_mean": _round(concentration["entropy_norm"].mean(), 3),
        "hull_wide_min": _round(concentration["entropy_norm"].min(), 3),
        "hull_wide_max": _round(concentration["entropy_norm"].max(), 3),
    }
    archetype_block = {
        "definition": (
            "K-means clusters on each LSOA's standardized local Feature-branch "
            "coefficient vector; groups areas with a similar explanation profile, "
            "independent of geographic adjacency."
        ),
        "k": DEFAULT_ARCHETYPE_K,
        "clusters": [
            {"cluster": int(r.cluster), "n_lsoas": int(r.n_lsoas), "label": str(r.label)}
            for r in profiles.itertuples()
        ],
    }

    top_global = global_features.copy()
    if "abs_coefficient" not in top_global.columns:
        top_global["abs_coefficient"] = top_global["coefficient"].abs()
    top_row = top_global.sort_values("abs_coefficient", ascending=False, kind="mergesort").iloc[0]
    top_feat = str(top_row["feature"])
    w, _islands = build_queen_weights(geometry, LSOA_ID)
    values = dict(zip(local_features[LSOA_ID].astype(str), local_features[top_feat]))
    moran_df, g_i, g_p = local_moran(values, w)
    lisa_counts = moran_df["lisa"].value_counts()
    clustering_block = {
        "definition": (
            "Local Moran's I / LISA of the single globally most important "
            "Feature-branch driver's LOCAL coefficient across Hull LSOAs. This is "
            "spatial clustering of a NAMED FEATURE's attribution, not of "
            "GeoSensitivity — keep it distinct from global_spatial_pattern."
        ),
        "feature_label": human_label(top_feat),
        "morans_i": _round(g_i, 3),
        "p_value": _round(g_p, 3),
        "significant_clustering_p05": bool(g_p < 0.05),
        "lisa_counts": {k: int(lisa_counts.get(k, 0)) for k in ("HH", "LL", "HL", "LH", "NS")},
    }

    evidence: dict = {
        "note": (
            "Dashboard-derived analyses over per-LSOA local Feature-branch "
            "coefficients (Local Attribution Patterns section). Not published "
            "GeoSMILE artefacts. Independent of GeoSensitivity."
        ),
        "dominant_local_feature": dominant_block,
        "explanation_concentration": concentration_block,
        "explanation_archetypes": archetype_block,
        "dominant_global_feature_clustering": clustering_block,
    }

    if selected_lsoa:
        sid = str(selected_lsoa)
        row = dominant.loc[dominant[LSOA_ID] == sid]
        conc_row = concentration.loc[concentration[LSOA_ID] == sid]
        arch_row = labels_df.loc[labels_df[LSOA_ID] == sid]
        if not row.empty:
            r = row.iloc[0]
            evidence["dominant_local_feature"]["selected_lsoa"] = {
                "feature_label": str(r["dominant_feature_label"]),
                "signed_value": _round(r["dominant_signed_value"]),
            }
        if not conc_row.empty:
            r = conc_row.iloc[0]
            evidence["explanation_concentration"]["selected_lsoa"] = {
                "entropy_norm": _round(r["entropy_norm"], 3),
                "effective_n_features": _round(r["effective_n_features"], 1),
            }
        if not arch_row.empty:
            cluster_id = int(arch_row.iloc[0]["cluster"])
            label_map = dict(zip(profiles["cluster"], profiles["label"]))
            evidence["explanation_archetypes"]["selected_lsoa_archetype"] = label_map.get(cluster_id)

    return evidence


@st.cache_data(show_spinner="Summarising local attribution patterns for the explanation…")
def build_local_attribution_evidence(
    local_features: pd.DataFrame,
    global_features: pd.DataFrame,
    _geometry,
    feature_names: tuple[str, ...],
    selected_lsoa: str | None,
) -> dict:
    return compute_local_attribution_evidence(
        local_features=local_features,
        global_features=global_features,
        geometry=_geometry,
        feature_names=list(feature_names),
        selected_lsoa=selected_lsoa,
    )
