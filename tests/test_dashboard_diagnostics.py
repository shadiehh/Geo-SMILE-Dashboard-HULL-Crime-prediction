"""Tests for the Model Diagnostics / Robustness sweep loader and views.

Expert-reference overlap here is computed from the raw exported rankings; it
must never depend on accuracy.json's intersection/accuracy/precision/recall/
f1/overlap_ratio/jaccard_at_k fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dashboard.data.schema import EVAL_SWEEP_DIR
from dashboard.data.sweep_loader import (
    FEATURE_TOP_K,
    GEO_TOP_K,
    SweepBundle,
    expert_overlap_count,
    load_sweep_bundle,
    topk_ranking,
)

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / EVAL_SWEEP_DIR

pytestmark = pytest.mark.skipif(not SWEEP_DIR.exists(), reason="eval_by_n_perturb sweep not generated yet")


@pytest.fixture(scope="module")
def bundle() -> SweepBundle:
    return load_sweep_bundle()


def test_sweep_loader_never_reads_accuracy_json():
    source = (ROOT / "dashboard" / "data" / "sweep_loader.py").read_text(encoding="utf-8")
    assert "accuracy.json" not in source
    diag_source = (ROOT / "dashboard" / "views" / "diagnostics.py").read_text(encoding="utf-8")
    assert "accuracy.json" not in diag_source


def test_metrics_has_both_branches_never_merged(bundle: SweepBundle):
    branches = set(bundle.metrics["branch"])
    assert branches == {"Feature", "Geo"}
    assert not any(c.lower() in {"overall", "combined", "merged"} for c in bundle.metrics.columns)


def test_published_n_markers(bundle: SweepBundle):
    assert bundle.published_n == {"feature": 1024, "geo": 2000}
    pub = bundle.metrics[bundle.metrics["is_published"]]
    feat_n = pub.loc[pub["branch"] == "Feature", "n_perturb"].iloc[0]
    geo_n = pub.loc[pub["branch"] == "Geo", "n_perturb"].iloc[0]
    assert int(feat_n) == 1024
    assert int(geo_n) == 2000


def test_three_fidelity_figures_are_distinct_columns(bundle: SweepBundle):
    cols = {"surrogate_fidelity_r2", "kernel_diagnostic_r2", "consistency_fidelity_r2_mean"}
    assert cols.issubset(bundle.metrics.columns)
    row = bundle.metrics[(bundle.metrics["branch"] == "Geo") & (~bundle.metrics["is_published"])].iloc[0]
    values = {row[c] for c in cols}
    assert len(values) == 3, "the three fidelity figures must never be blended into one number"


def test_expert_overlap_matches_manual_verification_at_published_n(bundle: SweepBundle):
    feat = bundle.published_rankings["feature"]
    geo = bundle.published_rankings["geo"]
    assert expert_overlap_count(feat, k=FEATURE_TOP_K) == 7
    assert expert_overlap_count(geo, k=GEO_TOP_K) == 16


def test_topk_ranking_sizes(bundle: SweepBundle):
    assert len(topk_ranking(bundle, branch="feature", n_perturb=None)) == FEATURE_TOP_K
    assert len(topk_ranking(bundle, branch="geo", n_perturb=None)) == GEO_TOP_K
    assert len(topk_ranking(bundle, branch="feature", n_perturb=1000)) == FEATURE_TOP_K
    assert len(topk_ranking(bundle, branch="geo", n_perturb=1000)) == GEO_TOP_K


def test_geo_topk_labels_use_lsoa_names_not_codes(bundle: SweepBundle):
    frame = topk_ranking(bundle, branch="geo", n_perturb=None)
    assert (frame["label"] != frame["lsoa21cd"]).any()


def test_rank_is_deterministic_and_covers_full_universe(bundle: SweepBundle):
    feat = bundle.rankings[(1000, "feature")]
    geo = bundle.rankings[(1000, "geo")]
    assert len(feat) == 30
    assert len(geo) == 168
    assert feat["rank"].tolist() == list(range(1, 31))
    assert geo["rank"].tolist() == list(range(1, 169))
