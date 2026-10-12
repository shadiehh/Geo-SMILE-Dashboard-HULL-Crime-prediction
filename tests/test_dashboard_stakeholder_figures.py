"""Every stakeholder's evidence figures build for any LSOA, with captions from the data."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from dashboard.data.stakeholder_data import load_stakeholder_data
from dashboard.views.stakeholder_figures import STAKEHOLDER_PANELS, is_path


@pytest.fixture(scope="module")
def data():
    return load_stakeholder_data()


def test_thesis_case_values(data):
    assert data.default_contrast("E01033104") == "E01012877"  # 029E -> 003C, as in Chapter 6
    caption = STAKEHOLDER_PANELS["public"](data, "E01033104", "E01012877")[0]()[2]
    assert "rank 1 of 168" in caption and "22.33" in caption


@pytest.mark.parametrize("lsoa", ["E01033104", "E01012891", "E01012786", "E01012877"])
@pytest.mark.parametrize("stakeholder", list(STAKEHOLDER_PANELS))
def test_panels_build_for_any_lsoa(data, stakeholder, lsoa):
    contrast = data.default_contrast(lsoa)
    assert contrast != lsoa
    for build in STAKEHOLDER_PANELS[stakeholder](data, lsoa, contrast):
        obj, title, caption = build()
        assert title and caption and "nan" not in caption.lower()
        if is_path(obj):
            assert obj.exists()
        else:
            plt.close(obj)
