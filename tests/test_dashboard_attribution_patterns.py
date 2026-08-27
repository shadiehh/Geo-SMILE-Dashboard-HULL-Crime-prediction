"""Tests for the new GeoSensitivity / local-attribution derived results.

GeoSensitivity (Geo branch) and local feature attribution (Feature branch)
must stay conceptually separate: these functions only ever map one of them
or compare their magnitudes, never blend them into a single score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dashboard.data.attribution_patterns import (
    attribution_archetypes,
    dominant_local_feature,
    local_attribution_concentration,
    top_categories_with_other,
)
from dashboard.data.loader import load_hull_bundle
from dashboard.data.schema import LSOA_ID

ROOT_BUNDLE_AVAILABLE = True
try:
    _BUNDLE = load_hull_bundle()
except FileNotFoundError:
    ROOT_BUNDLE_AVAILABLE = False
    _BUNDLE = None

pytestmark = pytest.mark.skipif(not ROOT_BUNDLE_AVAILABLE, reason="published Hull artefacts not generated yet")


def _synthetic_local_features() -> tuple[pd.DataFrame, list[str]]:
    feature_names = ["f1", "f2", "f3"]
    frame = pd.DataFrame(
        {
            LSOA_ID: ["A", "B", "C"],
            "f1": [5.0, -0.1, 1.0],
            "f2": [0.2, 3.0, 1.0],
            "f3": [-0.1, 0.2, 1.0],
        }
    )
    return frame, feature_names


def test_dominant_local_feature_picks_largest_abs_and_keeps_sign():
    frame, names = _synthetic_local_features()
    out = dominant_local_feature(frame, names)
    row = out.set_index(LSOA_ID)
    assert row.loc["A", "dominant_feature"] == "f1"
    assert row.loc["A", "dominant_signed_value"] == pytest.approx(5.0)
    assert row.loc["B", "dominant_feature"] == "f2"
    assert row.loc["B", "dominant_signed_value"] == pytest.approx(3.0)


def test_concentration_bounds_and_extremes():
    frame, names = _synthetic_local_features()
    out = local_attribution_concentration(frame, names)
    assert out["entropy_norm"].between(0.0, 1.0).all()
    row = out.set_index(LSOA_ID)
    # LSOA A: one feature dominates -> low entropy (concentrated)
    # LSOA C: three equal features -> entropy at its max (fully diffuse)
    assert row.loc["A", "entropy_norm"] < row.loc["C", "entropy_norm"]
    assert row.loc["C", "entropy_norm"] == pytest.approx(1.0, abs=1e-9)


def test_top_categories_with_other_folds_long_tail():
    series = pd.Series(["a"] * 5 + ["b"] * 4 + ["c"] * 3 + ["d"] * 2 + ["e"] * 1)
    out = top_categories_with_other(series, max_categories=2)
    assert set(out.unique()) == {"a", "b", "Other"}
    assert (out == "Other").sum() == 6


def test_attribution_archetypes_shapes():
    labels, profiles = attribution_archetypes(_BUNDLE.local_features, _BUNDLE.feature_names, k=4)
    assert len(labels) == len(_BUNDLE.local_features)
    assert set(labels["cluster"].unique()).issubset(set(range(4)))
    assert len(profiles) == 4
    assert profiles["n_lsoas"].sum() == len(_BUNDLE.local_features)
    assert all(len(p) <= 3 for p in profiles["top_features"])


def test_dominant_feature_never_a_location_feature():
    out = dominant_local_feature(_BUNDLE.local_features, _BUNDLE.feature_names)
    assert len(out) == len(_BUNDLE.lsoa)
    from dashboard.data.schema import is_location_feature

    assert not out["dominant_feature"].map(is_location_feature).any()


def test_real_data_concentration_matches_synthetic_bounds():
    out = local_attribution_concentration(_BUNDLE.local_features, _BUNDLE.feature_names)
    assert len(out) == 168
    assert out["entropy_norm"].between(0.0, 1.0).all()
    assert out["effective_n_features"].between(1.0, len(_BUNDLE.feature_names)).all()
