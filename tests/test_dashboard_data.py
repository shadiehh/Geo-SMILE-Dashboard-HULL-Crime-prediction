"""Data-layer tests for the Hull stakeholder dashboard."""

from __future__ import annotations

from dashboard.data.loader import load_hull_bundle, run_diagnostics, top_features_for_lsoa
from dashboard.data.schema import LSOA_ID, is_location_feature


def test_lsoa_join_is_one_to_one():
    bundle = load_hull_bundle()
    assert len(bundle.lsoa) == 168
    assert bundle.lsoa[LSOA_ID].is_unique
    assert bundle.geometry[LSOA_ID].is_unique
    assert set(bundle.lsoa[LSOA_ID]) == set(bundle.geometry[LSOA_ID].astype(str))
    assert bundle.diagnostics["unmatched_results"] == []
    assert bundle.diagnostics["unmatched_polygons"] == []
    assert bundle.diagnostics["negative_geo_preserved"] is True


def test_top10_excludes_coordinates_and_ranks_by_abs():
    bundle = load_hull_bundle()
    code = str(bundle.lsoa.iloc[0][LSOA_ID])
    top = top_features_for_lsoa(bundle.local_features, code, k=10)
    assert len(top) == 10
    assert [r["feature_rank"] for r in top] == list(range(1, 11))
    abs_vals = [r["abs_value"] for r in top]
    assert abs_vals == sorted(abs_vals, reverse=True)
    assert not any(is_location_feature(r["feature"]) for r in top)
    assert all("centroid_longitude" != r["feature"] for r in top)
    assert all("centroid_latitude" != r["feature"] for r in top)
    signs = [r["feature_value"] for r in top]
    assert any(v < 0 for v in signs) or any(v > 0 for v in signs)


def test_pattern_parsing_hh_cluster():
    bundle = load_hull_bundle()
    # Area-data design: city-centre HH cluster around Hull 029E.
    row = bundle.lsoa[bundle.lsoa[LSOA_ID] == "E01033104"].iloc[0]
    assert row["lisa_class"] == "HH"
    assert row["cluster_id"] == "HH-1"
    assert int(row["cluster_size"]) == 9
    assert "E01012854" in row["cluster_members"]
    assert row["locality_label"] == "Old Town"
    global_p = bundle.spatial_pattern["GLOBAL_SPATIAL_PATTERN"]
    assert global_p["Moran_I"] > 0
    assert "p_value" in global_p


def test_lat_lon_are_not_geo_sensitivity():
    bundle = load_hull_bundle()
    row = bundle.lsoa[bundle.lsoa[LSOA_ID] == "E01033104"].iloc[0]
    assert abs(float(row["geo_sensitivity"]) - float(row["latitude_effect"])) > 1
    assert abs(float(row["geo_sensitivity"]) - float(row["longitude_effect"])) > 1
    assert int(row["geo_rank"]) == 1


def test_diagnostics_reports_missing_adjusted_p():
    diag = run_diagnostics()
    assert diag["local_p_adjusted"] == "not available in published LISA table"
    assert diag["pattern_scale"] == "not established"
    assert diag["sample_top10_location_leak"] == []
