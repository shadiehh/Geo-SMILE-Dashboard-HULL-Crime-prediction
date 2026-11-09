"""The dashboard shows the Chapter 4 results, with the Chapter 3 local sign."""

from __future__ import annotations

import pandas as pd
import pytest

from dashboard.data.loader import compact_evaluation, load_hull_bundle
from dashboard.data.schema import FEATURE_LOCAL_PUBLISHED_CSV
from dashboard.data.stakeholder_data import load_stakeholder_data
from dashboard.data.sweep_loader import load_sweep_bundle
from dashboard.llm import card_mode as cm


@pytest.fixture(scope="module")
def bundle():
    return load_hull_bundle()


@pytest.fixture(scope="module")
def data():
    return load_stakeholder_data()


def test_local_sign_is_perturbed_minus_original(data):
    assert data.local.loc["E01033104", "crime_rolling_mean_6"] == pytest.approx(50.40, abs=0.01)
    assert data.local.loc["E01012889", "crime_rolling_mean_6"] == pytest.approx(32.05, abs=0.01)


def test_local_positive_shares_match_chapter_4(data):
    assert 100 * (data.local["long_term_sick_disabled_pct"] > 0).mean() == pytest.approx(19.6, abs=0.05)
    assert 100 * (data.local["crime_lag_1"] > 0).mean() == pytest.approx(27.4, abs=0.05)


def test_feature_panel_uses_the_same_run(bundle, data):
    panel = bundle.local_features.set_index("lsoa21cd")
    assert panel.loc["E01033104", "crime_rolling_mean_6"] == data.local.loc["E01033104", "crime_rolling_mean_6"]
    assert FEATURE_LOCAL_PUBLISHED_CSV.startswith("outputs/dashboard_final/")


def test_global_ranking_matches_chapter_4(bundle):
    top = bundle.global_features.sort_values("abs_coefficient", ascending=False).head(10)
    assert top["abs_coefficient"].iloc[0] == pytest.approx(5.3204, abs=1e-4)
    assert list(top["feature"].iloc[-2:]) == ["road_length_km", "private_rented_pct"]


def test_evaluation_matches_chapter_4(bundle):
    ev = compact_evaluation(bundle.evaluation)
    assert ev["feature_holdout_r2"] == pytest.approx(0.8952, abs=1e-4)
    assert ev["feature_surrogate_test_r2"] == pytest.approx(0.9850, abs=1e-4)
    assert ev["geo_fidelity_r2"] == pytest.approx(0.9296, abs=1e-4)
    assert ev["consistency"]["feature"]["rank_spearman_mean"] == pytest.approx(0.915, abs=1e-3)
    assert ev["consistency"]["geo"]["rank_spearman_mean"] == pytest.approx(0.904, abs=1e-3)
    assert ev["feature_faithfulness_top_minus_random"] == pytest.approx(4.4986, abs=1e-4)
    assert ev["geo_faithfulness_top_minus_random"] == pytest.approx(1.2728, abs=1e-4)


def test_faithfulness_table_matches_chapter_4():
    table = pd.read_csv("outputs/dashboard_final/att_faithfulness.csv").set_index(["branch", "measure"])["pearson_r"]
    assert table["Feature", "att_accuracy"] == pytest.approx(-0.01, abs=0.005)
    assert table["Feature", "att_auc"] == pytest.approx(0.17, abs=0.005)
    assert table["Geo", "att_accuracy"] == pytest.approx(0.32, abs=0.005)
    assert table["Geo", "att_auc"] == pytest.approx(0.28, abs=0.005)


def test_sweep_matches_chapter_4():
    metrics = load_sweep_bundle().metrics
    published = metrics[metrics["is_published"]].set_index("branch")
    assert published.loc["Feature", "surrogate_fidelity_r2"] == pytest.approx(0.9850, abs=1e-4)
    assert published.loc["Feature", "expert_overlap_count"] == 7
    assert published.loc["Geo", "expert_overlap_count"] == 16


def test_manager_contrast_follows_its_rule(data):
    lead = data.local.abs().idxmax(axis=1)
    not_crime = lead[~lead.str.startswith("crime_")].index
    assert cm.MANAGER_CONTRAST == data.geo.loc[not_crime, "geo_rank"].idxmin()


@pytest.mark.parametrize("role", sorted(cm.SPECS))
def test_published_card_passed_its_checks(role):
    import json
    from pathlib import Path

    path = Path(f"outputs/dashboard_final/cards/{role}_card_llm.json")
    run = json.loads(path.read_text(encoding="utf-8"))["runs"]["openai"]
    assert run["ok"] and run["checks"]["ok"], run.get("checks")
    assert run["model"] == "gpt-4.1"
