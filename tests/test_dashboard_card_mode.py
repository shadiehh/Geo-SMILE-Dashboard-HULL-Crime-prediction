"""Planner short/comprehensive narrative modes (dashboard.llm.card_mode)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dashboard.data.stakeholder_data import load_stakeholder_data
from dashboard.llm import card_mode as cm
from dashboard.llm.openai_provider import OpenAIProvider
from dashboard.llm.prompt import GEOSMILE_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
CARD_JSON = ROOT / "outputs" / "stakeholder_narratives" / "planner_card_llm.json"


@pytest.fixture(scope="module")
def data():
    return load_stakeholder_data()


def test_category_shares_sum_to_one(data):
    assert cm.category_shares(data).sum() == pytest.approx(1.0)
    assert cm.category_shares(data, cm.CENTRE).sum() == pytest.approx(1.0)
    built = cm.built_feature_shares(data)
    assert built.sum() == pytest.approx(cm.category_shares(data)["built"])


def test_block_matches_published_card_evidence(data):
    if not CARD_JSON.exists():
        pytest.skip("card record not generated")
    recorded = json.loads(CARD_JSON.read_text(encoding="utf-8"))["evidence"]["built_environment_share"]
    assert cm.built_environment_block(data) == recorded


def test_published_card_passes_checks():
    if not CARD_JSON.exists():
        pytest.skip("card record not generated")
    record = json.loads(CARD_JSON.read_text(encoding="utf-8"))
    assert cm.check(record["runs"]["openai"]["content"], record["evidence"], "short")["ok"]


def test_check_flags_jargon_unsupported_and_unmarked_figures():
    evidence = {"share": 0.139}
    card = {"headline": "h", "footer": "f", "tiles": [{"value": "14%", "label": "x"}] * 4,
            "section_01": "Mild clustering, Moran's I [[n:0.11]] with [[t:retail]] and [[t:roads]].",
            "section_02": "About 37 areas, [[n:14%]], [[t:green space]] and [[t:transit]]."}
    problems = " ".join(cm.check(card, evidence, "short")["problems"])
    assert "test-statistic wording" in problems
    assert "figure not in evidence: '0.11'" in problems
    assert "section_02: unmarked figure" in problems


def test_comprehensive_allows_statistics():
    evidence = {"moran_i": 0.219, "p": 0.001}
    text = ("[[t:Old Town]] leads; [[t:clustered]] (Moran's I [[n:0.22]], p = [[n:0.001]]). "
            "\n\n[[t:crime history]] and [[t:retail]].")
    assert cm.check(text, evidence, "comprehensive")["ok"]


def test_marked_html_escapes_and_highlights():
    out = cm.marked_html("<b>x</b> [[n:14%]] and [[t:green space]]\n\nnext")
    assert "&lt;b&gt;" in out and "<b>" not in out
    assert out.count(cm.PINK) == 2
    assert out.count("<p") == 2


def test_marked_tex_uses_card_macros():
    assert cm.marked_tex("[[n:14%]] in [[t:Old Town]]") == r"\pv{14\%} in \kt{Old Town}"


def test_police_block_is_consistent(data):
    block = cm.police_block(data)
    sig, trend = block["crime_signals"], block["recorded_trend"]
    assert sig["highest_recent_share_lsoas"][0]["id"] == cm.CENTRE
    shares = cm.crime_signal_shares(data)
    assert (shares["recent_share"] + shares["long_run_share"] <= shares["crime_history_share"] + 1e-9).all()
    assert trend["n_rising_over_quarter"] + trend["n_falling_over_quarter"] <= trend["n_assessed"] <= data.n
    assert all(r["previous_mean_monthly_crime"] >= cm.TREND_LABEL_MIN for r in trend["largest_rises"])
    assert "crime_rolling_mean_6" not in json.dumps(block)  # plain labels only


def test_every_card_stakeholder_has_figures_and_both_modes():
    from dashboard.views.stakeholder_figures import STAKEHOLDER_PANELS

    for stakeholder, spec in cm.SPECS.items():
        assert set(spec["addenda"]) == set(cm.MODES) == set(spec["version"])
        assert f"{stakeholder}_short" in STAKEHOLDER_PANELS
        assert cm.user_suffix("short", stakeholder) and cm.user_suffix("comprehensive", stakeholder) is None


def test_published_police_card_passes_checks():
    path = CARD_JSON.with_name("police_card_llm.json")
    if not path.exists():
        pytest.skip("police card record not generated")
    record = json.loads(path.read_text(encoding="utf-8"))
    limits = cm.SPECS["police"]["limits"]
    assert cm.check(record["runs"]["openai"]["content"], record["evidence"], "short", limits)["ok"]


def test_tex_escape_handles_comparison_signs():
    assert cm.tex_escape("up >25%") == r"up $>$25\%"


def test_provider_sends_master_prompt_then_addendum(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = MagicMock()
    client.responses.create.return_value = MagicMock(output_text="{}", usage=None)
    with patch("openai.OpenAI", return_value=client):
        OpenAIProvider(model="m").generate(GEOSMILE_SYSTEM_PROMPT, {"k": 1}, {"id": "planner"},
                                           addendum=cm.CARD_MODE_ADDENDUM, user_suffix="\n\nEXAMPLE")
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["instructions"] == GEOSMILE_SYSTEM_PROMPT + "\n" + cm.CARD_MODE_ADDENDUM
    assert kwargs["input"].endswith("\n\nEXAMPLE")
