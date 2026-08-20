"""Unit tests for Hull expert-reference attribution accuracy."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hull_expert_alignment import (
    assert_primary_coverage,
    binary_discrimination,
    build_evaluation_full_v2,
    compute_alignment_metrics,
    coverage_report,
    detect_magnitude_column,
    expert_positive_from_rating,
    merge_feature,
    merge_geo,
    ndcg_at_k,
    resolve_reference_rating,
    spearman_with_ci,
)

ROOT = Path(__file__).resolve().parent


def test_human_override_replaces_rating():
    expert = pd.Series([5, 2, 3])
    override = pd.Series([np.nan, 4, np.nan])
    got = resolve_reference_rating(expert, override)
    assert got.tolist() == [5, 4, 3]


def test_binary_threshold_ge_4_legacy_helper():
    from hull_expert_alignment import expert_positive_from_rating

    rating = pd.Series([1, 3, 4, 5])
    assert expert_positive_from_rating(rating).tolist() == [0, 0, 1, 1]


def test_resolve_rating_rejects_out_of_range():
    with pytest.raises(ValueError):
        resolve_reference_rating(pd.Series([1, 6]), None)


def test_feature_join_exact_ids_not_row_order():
    attr = pd.DataFrame(
        {
            "feature": ["b", "a"],
            "coefficient": [0.2, -1.0],
            "abs_coefficient": [0.2, 1.0],
        }
    )
    ann = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "expert_relevance_1to5": [5, 2],
            "human_override_1to5": [np.nan, np.nan],
            "confidence": ["High", "Low"],
            "relevance_label": ["Very high", "Low"],
            "expected_association": ["x", "y"],
        }
    )
    merged, cov = merge_feature(attr, ann, branch="Feature", source_file="t.csv")
    assert cov["n_matched"] == 2
    row_a = merged.loc[merged["feature"] == "a"].iloc[0]
    assert row_a["geosmile_abs_attribution"] == 1.0
    assert int(row_a["final_expert_rating"]) == 5
    assert "expert_top10_reference" in merged.columns


def test_geo_join_on_lsoa21cd_not_name():
    attr = pd.DataFrame(
        {
            "lsoa21cd": ["E1", "E2"],
            "lsoa21nm": ["Wrong Name", "Other"],
            "geo_coefficient": [1.0, -2.0],
            "geo_abs_coefficient": [1.0, 2.0],
        }
    )
    ann = pd.DataFrame(
        {
            "lsoa21cd": ["E2", "E1"],
            "lsoa21nm": ["B", "A"],
            "expert_relevance_1to5": [4, 1],
            "no_crime_rating_1to5": [3, 2],
            "human_override_1to5": [np.nan, np.nan],
            "confidence": ["High", "High"],
            "relevance_label": ["High", "Very low"],
            "expected_crime_context": ["Elevated", "Low"],
            "crime_domain_dependence": ["Low", "Low"],
            "expert_relevance_score": [0.8, 0.2],
        }
    )
    merged, cov = merge_geo(attr, ann, branch="Geo", source_file="g.csv")
    assert cov["n_matched"] == 2
    assert merged.loc[merged["lsoa21cd"] == "E2", "geosmile_abs_attribution"].iloc[0] == 2.0


def test_duplicate_id_detection_and_incomplete_coverage_failure():
    cov = coverage_report(
        pd.Series(["a", "a", "b"]),
        pd.Series(["a", "b", "c"]),
        id_name="feature",
    )
    assert cov["duplicate_attribution_ids"] == ["a"]
    assert cov["n_unmatched_annotation"] == 1
    with pytest.raises(SystemExit):
        assert_primary_coverage(cov, expected_n=3, label="Feature")


def test_magnitude_column_fallback_abs_coefficient_then_abs_importance_then_abs():
    mag_col, mag = detect_magnitude_column(
        pd.DataFrame({"feature": ["a"], "abs_coefficient": [3.0]}), kind="feature"
    )
    assert mag_col == "abs_coefficient" and mag.iloc[0] == 3.0

    mag_col, mag = detect_magnitude_column(
        pd.DataFrame({"feature": ["a"], "abs_importance": [2.5], "coefficient": [-2.5]}),
        kind="feature",
    )
    assert mag_col == "abs_importance" and mag.iloc[0] == 2.5

    mag_col, mag = detect_magnitude_column(
        pd.DataFrame({"feature": ["a"], "coefficient": [-1.5]}), kind="feature"
    )
    assert mag_col == "abs(coefficient)" and mag.iloc[0] == 1.5

    mag_col, mag = detect_magnitude_column(
        pd.DataFrame({"lsoa21cd": ["E1"], "geo_coefficient": [-4.0]}), kind="geo"
    )
    assert mag_col == "abs(geo_coefficient)" and mag.iloc[0] == 4.0


def test_perfect_ranking_spearman_and_ndcg_one():
    n = 10
    ratings = np.arange(1, n + 1)  # 1..10 then clipped conceptually; use 1-5 repeated ordered
    ratings = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    magnitude = ratings.astype(float)  # perfect alignment
    ids = np.array([f"f{i:02d}" for i in range(n)])
    from scipy.stats import spearmanr

    rho, _ = spearmanr(magnitude, ratings)
    assert rho == pytest.approx(1.0)
    assert ndcg_at_k(magnitude, ratings, ids, n) == pytest.approx(1.0)


def test_reversed_ranking_worse_than_perfect():
    ratings = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    ids = np.array([f"f{i}" for i in range(10)])
    perfect = ratings.astype(float)
    reversed_mag = (6 - ratings).astype(float)
    assert ndcg_at_k(perfect, ratings, ids, 10) > ndcg_at_k(reversed_mag, ratings, ids, 10)
    from scipy.stats import spearmanr

    assert spearmanr(perfect, ratings).correlation > spearmanr(reversed_mag, ratings).correlation


def test_random_ranking_near_baseline():
    rng = np.random.default_rng(0)
    ratings = np.array([1, 2, 3, 4, 5] * 20)
    ids = np.array([f"i{i}" for i in range(len(ratings))])
    mag = rng.random(len(ratings))
    score = ndcg_at_k(mag, ratings, ids, len(ratings))
    # Random full-ranking nDCG should not be near perfect.
    assert 0.2 < score < 0.95


def test_auroc_auprc_null_for_one_class():
    y = np.ones(10, dtype=int)
    score = np.linspace(0, 1, 10)
    out = binary_discrimination(y, score)
    assert out["auroc"] is None and out["auprc"] is None
    assert out["null_reason"] == "only_one_reference_class_present"


def test_bootstrap_deterministic_under_seed():
    x = np.linspace(0, 1, 30)
    y = np.array([1, 2, 3, 4, 5] * 6)
    a = spearman_with_ci(x, y, rng=np.random.default_rng(42), n_boot=200)
    b = spearman_with_ci(x, y, rng=np.random.default_rng(42), n_boot=200)
    assert a["ci95_low"] == b["ci95_low"]
    assert a["ci95_high"] == b["ci95_high"]
    assert a["n_valid_bootstrap"] == b["n_valid_bootstrap"]


def test_cardinality_vs_wd_recategorised_not_deleted():
    existing = pd.DataFrame(
        [
            {"Branch": "Feature", "Category": "Fidelity", "Metric": "R2", "Value": 0.98},
            {
                "Branch": "Both",
                "Category": "Faithfulness",
                "Metric": "Feature Spearman (cardinality vs WD)",
                "Value": 0.46,
            },
            {
                "Branch": "Feature",
                "Category": "Consistency (multi-run)",
                "Metric": "Rank Spearman Mean",
                "Value": 0.75,
            },
        ]
    )
    expert_long = pd.DataFrame(
        [
            {
                "branch": "Feature",
                "evaluation_scope": "complete_30_feature",
                "reference_variant": "external_crime_informed_reference",
                "confidence_subset": "all",
                "category": "Expert-reference attribution accuracy",
                "metric": "Spearman rho",
                "value": 0.12,
            }
        ]
    )
    v2 = build_evaluation_full_v2(existing, expert_long)
    assert (v2["Category"] == "Fidelity").any()
    assert (v2["Category"] == "Consistency (multi-run)").any()
    mono = v2[v2["Category"] == "Perturbation response monotonicity"]
    assert len(mono) == 1
    assert float(mono["Value"].iloc[0]) == pytest.approx(0.46)
    assert not (v2["Category"] == "Faithfulness").any()
    assert (v2["Category"] == "Expert-reference attribution accuracy").any()


def test_fidelity_consistency_values_unchanged_by_v2_builder():
    existing = pd.DataFrame(
        [
            {"Branch": "Feature", "Category": "Fidelity", "Metric": "R2", "Value": 0.9853},
            {
                "Branch": "Geo",
                "Category": "Consistency (multi-run)",
                "Metric": "Top-20% Jaccard Mean",
                "Value": 0.6233,
            },
            {
                "Branch": "Both",
                "Category": "Faithfulness",
                "Metric": "Geo Pearson (cardinality vs WD)",
                "Value": 0.0113,
            },
        ]
    )
    v2 = build_evaluation_full_v2(existing, pd.DataFrame())
    assert float(v2.loc[v2["Metric"] == "R2", "Value"].iloc[0]) == pytest.approx(0.9853)
    assert float(
        v2.loc[v2["Metric"] == "Top-20% Jaccard Mean", "Value"].iloc[0]
    ) == pytest.approx(0.6233)
