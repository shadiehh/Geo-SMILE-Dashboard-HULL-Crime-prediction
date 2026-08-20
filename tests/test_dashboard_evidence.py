"""Evidence-package tests."""

from __future__ import annotations

from dashboard.data.loader import load_hull_bundle
from dashboard.llm.evidence_builder import build_evidence


def test_selected_lsoa_evidence_contains_required_fields():
    bundle = load_hull_bundle()
    evidence = build_evidence(
        bundle,
        stakeholder_id="domain",
        current_map_view="geo_sensitivity",
        scope="lsoa",
        selected_lsoa="E01012891",
    )
    selected = evidence["selected_lsoa"]
    assert selected["id"] == "E01012891"
    assert selected["geo_sensitivity"] is not None
    assert selected["geo_rank"] == 14
    assert selected["latitude_effect"] is not None
    assert selected["longitude_effect"] is not None
    assert selected["spatial_pattern"]["class"] == "HH"
    assert selected["spatial_pattern"]["pattern_scale"] == "not established"
    assert len(selected["top_features"]) == 10
    assert selected["top_features"][0]["feature_rank"] == 1
    assert "centroid_longitude" not in {r["feature"] for r in selected["top_features"]}
    assert evidence["semantic_separation"]["not_equivalent"]


def test_hh_cluster_is_not_automatically_large_magnitude():
    bundle = load_hull_bundle()
    evidence = build_evidence(
        bundle,
        stakeholder_id="technical",
        current_map_view="spatial_pattern",
        scope="lsoa",
        selected_lsoa="E01012858",
    )
    selected = evidence["selected_lsoa"]
    assert selected["spatial_pattern"]["class"] == "HH"
    assert selected["geo_magnitude_class"] in {"modest", "moderate"}
    assert selected["geo_sensitivity_abs"] < 5


def test_overall_scope_has_no_selected_lsoa():
    bundle = load_hull_bundle()
    evidence = build_evidence(
        bundle,
        stakeholder_id="public",
        current_map_view="geo_sensitivity",
        scope="overall",
        selected_lsoa=None,
    )
    assert "selected_lsoa" not in evidence
    assert evidence["global_spatial_pattern"]["pattern_scale"] == "not established"
    assert evidence["global_geo"]["n_lsoas"] == 168
