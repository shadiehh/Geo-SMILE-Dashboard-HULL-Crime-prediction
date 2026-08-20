"""Tests for Hull GeoSMILE evaluation v3 (Feature top-10 / Geo top-20)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geosmile_hull_twobranch import FORBIDDEN_FEATURE_TOKENS, choose_feature_players
from hull_eval_v3 import (
    controlled_input_stability_status,
    dehghani_expert_faithfulness_status,
    restore_historical_metrics,
)
from hull_expert_alignment import (
    assert_primary_coverage,
    coverage_report,
    merge_feature,
    merge_geo,
    resolve_reference_rating,
)
from hull_expert_topk import (
    BRANCH_FEATURE,
    BRANCH_GEO,
    FEATURE_TOP_K,
    GEO_TOP_K,
    att_binary_metrics,
    select_expert_map_top20,
    select_expert_top_feature,
    select_expert_top_geo,
    set_overlap_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PANEL = OUT / "hull_lsoa_month_panel_with_covariates.csv"


def test_only_two_active_branch_labels():
    assert BRANCH_FEATURE == "Feature (location excluded)"
    assert BRANCH_GEO == "Geo"
    assert FEATURE_TOP_K == 10
    assert GEO_TOP_K == 20
    assert FEATURE_TOP_K / 30 == 10 / 30
    assert GEO_TOP_K / 168 == 20 / 168


def test_feature_excludes_coordinates():
    if not PANEL.exists():
        pytest.skip("panel missing")
    panel = pd.read_csv(PANEL)
    players = choose_feature_players(panel)
    bad = [p for p in players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    assert bad == []
    assert len(players) == 30


def test_feature_expert_topk_equals_10():
    rows = [
        {
            "feature": f"f{i:02d}",
            "final_expert_rating": (i % 5) + 1,
            "confidence": ["High", "Medium", "Low"][i % 3],
        }
        for i in range(30)
    ]
    out = select_expert_top_feature(pd.DataFrame(rows))
    assert int(out["expert_top10_reference"].sum()) == 10
    assert FEATURE_TOP_K == 10



def test_geo_expert_topk_equals_20():
    rows = [
        {
            "lsoa21cd": f"E{i:08d}",
            "final_expert_rating": (i % 5) + 1,
            "expert_relevance_score": float(i),
            "confidence": ["High", "Medium", "Low"][i % 3],
        }
        for i in range(168)
    ]
    out = select_expert_top_geo(pd.DataFrame(rows))
    assert int(out["expert_top10_reference"].sum()) == 20


def test_geo_expert_topk_from_v2_reference_flag():
    v2 = ROOT / "geosmile_hull_prediction_expert_top20_v2.xlsx"
    if not v2.exists():
        pytest.skip("v2 workbook missing")
    geo = pd.read_excel(v2, sheet_name="LSOA_Annotations")
    geo["lsoa21cd"] = geo["lsoa21cd"].astype(str)
    geo["final_expert_rating"] = resolve_reference_rating(
        geo["expert_relevance_1to5"], geo.get("human_override_1to5")
    )
    out = select_expert_top_geo(geo)
    assert int(out["expert_top10_reference"].sum()) == 20
    assert out["expert_top10_reference"].equals(
        pd.to_numeric(geo["expert_top20_reference"], errors="coerce").fillna(0).astype(int)
    )


def test_map_selects_exactly_20():
    rows = [
        {
            "lsoa21cd": f"E{i:08d}",
            "final_expert_rating": (i % 5) + 1,
            "expert_relevance_score": float(168 - i),
            "confidence": "High",
        }
        for i in range(168)
    ]
    top = select_expert_map_top20(pd.DataFrame(rows))
    assert len(top) == 20


def test_human_override_replaces_rating():
    got = resolve_reference_rating(pd.Series([5, 2, 3]), pd.Series([np.nan, 4, np.nan]))
    assert got.tolist() == [5, 4, 3]


def test_deterministic_tie_handling_feature():
    frame = pd.DataFrame(
        {
            "feature": ["b", "a", "c"],
            "final_expert_rating": [5, 5, 5],
            "confidence": ["High", "High", "High"],
        }
    )
    out = select_expert_top_feature(frame, k=1)
    assert out.loc[out["expert_top10_reference"] == 1, "feature"].iloc[0] == "a"


def test_exact_id_joins():
    attr = pd.DataFrame(
        {"feature": ["b", "a"], "coefficient": [0.2, -1.0], "abs_coefficient": [0.2, 1.0]}
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
    merged, cov = merge_feature(attr, ann, branch=BRANCH_FEATURE, source_file="t.csv")
    assert cov["n_matched"] == 2
    assert int(merged.loc[merged["feature"] == "a", "final_expert_rating"].iloc[0]) == 5


def test_duplicate_coverage_failure():
    cov = coverage_report(pd.Series(["a", "a", "b"]), pd.Series(["a", "b", "c"]), id_name="feature")
    with pytest.raises(SystemExit):
        assert_primary_coverage(cov, expected_n=3, label="Feature")


def test_att_perfect_and_disjoint():
    att = att_binary_metrics(
        np.array([1, 1, 0, 0]),
        np.array([1, 1, 0, 0]),
        np.array([3.0, 2.0, 0.1, 0.0]),
    )
    assert att["ATT accuracy"] == pytest.approx(1.0)
    assert att["ATT F1"] == pytest.approx(1.0)
    ov = set_overlap_metrics({"a", "b"}, {"a", "b"}, n_universe=4)
    assert ov["Jaccard@k"] == pytest.approx(1.0)
    assert ov["Precision@k"] == ov["Recall@k"]
    ov0 = set_overlap_metrics({"a", "b"}, {"c", "d"}, n_universe=4)
    assert ov0["Jaccard@k"] == pytest.approx(0.0)


def test_att_auroc():
    att = att_binary_metrics(
        np.array([1, 0, 1, 0]),
        np.array([1, 0, 0, 1]),
        np.array([0.9, 0.1, 0.8, 0.2]),
    )
    assert att["ATT AUROC"] == pytest.approx(1.0)


def test_dehghani_gate():
    status = dehghani_expert_faithfulness_status()
    assert status["status"] == "Not estimable"
    assert status["n_valid_independent_predictive_runs"] < 5


def test_model_based_cardinality_constants():
    assert FEATURE_TOP_K == 10
    assert GEO_TOP_K == 20


def test_cardinality_vs_wd_preserved():
    rows, _, _ = restore_historical_metrics()
    mono = [
        r
        for r in rows
        if r["category"] == "Perturbation response monotonicity"
        and "Spearman" in r["metric"]
        and "Feature" in r["metric"]
    ]
    assert mono
    assert abs(float(mono[0]["value"]) - 0.4624799677336309) < 1e-6


def test_dedicated_precedence():
    rows, provenance, conflicts = restore_historical_metrics()
    assert provenance
    fid = [r for r in rows if r["category"] == "Fidelity" and r["metric"] == "R2"]
    assert fid
    assert "fidelity" in fid[0]["source_file"]
    assert isinstance(conflicts, list)


def test_stability_distinct_from_consistency():
    st = controlled_input_stability_status()
    assert st.get("status") in {"Evaluated", "Not evaluated"}
    rows, _, _ = restore_historical_metrics()
    cats = {r["category"] for r in rows}
    assert "Consistency" in cats


def test_no_feature_ab_in_v3_when_present():
    path = OUT / "geosmile_hull_evaluation_full_v3.csv"
    if not path.exists():
        pytest.skip("v3 not generated yet")
    branches = set(pd.read_csv(path)["branch"].astype(str))
    assert not any("Feature-A" in b or "Feature-B" in b for b in branches)
    assert any("location excluded" in b for b in branches)
    assert BRANCH_GEO in branches


def test_map_outputs_top20_when_generated():
    paths = [
        OUT / "geosmile_hull_expert_top20_lsoa_overview.png",
        OUT / "geosmile_hull_expert_top20_lsoa_overview.pdf",
        OUT / "geosmile_hull_expert_top20_lsoa_interactive.html",
        OUT / "geosmile_hull_expert_vs_geosmile_top20.png",
        OUT / "geosmile_hull_expert_vs_geosmile_top20.html",
        OUT / "geosmile_hull_expert_top20_lsoa_map_data.csv",
    ]
    if not paths[0].exists():
        pytest.skip("maps not generated yet")
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
    data = pd.read_csv(paths[-1])
    assert int((data["expert_map_top20"] == 1).sum()) == 20
