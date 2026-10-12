"""LLM provider tests with mocked SDKs."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from dashboard.data.loader import load_hull_bundle
from dashboard.llm.anthropic_provider import AnthropicProvider
from dashboard.llm.attribution_evidence import compute_local_attribution_evidence
from dashboard.llm.base import LLMResponse
from dashboard.llm.evidence_builder import build_evidence, user_payload
from dashboard.llm.openai_provider import OpenAIProvider
from dashboard.llm.prompt import GEOSMILE_SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION, assert_prompt_semantics

ROOT = Path(__file__).resolve().parents[1]


def test_prompt_forbids_semantic_collapses():
    assert_prompt_semantics()
    text = GEOSMILE_SYSTEM_PROMPT
    assert text.startswith("# GEOSMILE EXPLAIN — SYSTEM POLICY")
    assert "## 1. ROLE" in text
    assert "## 4. INPUT / EVIDENCE CONTRACT" in text
    assert "## 8. SPATIAL-PATTERN SEMANTICS" in text
    assert "Global Moran versus Local Moran" in text
    assert SYSTEM_PROMPT_VERSION == "geosmile-hull-v8"


def test_dashboard_prompt_contains_full_policy_not_a_loader():
    source = (ROOT / "dashboard" / "llm" / "prompt.py").read_text(encoding="utf-8")
    assert "importlib" not in source
    assert "GEOSMILE EXPLAIN — SYSTEM POLICY" in source
    assert "## 16. SELF-CHECK" in source


def test_repo_root_prompt_matches_dashboard_prompt():
    path = ROOT / "llm" / "prompt.py"
    spec = importlib.util.spec_from_file_location("geosmile_root_prompt", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SYSTEM_PROMPT_VERSION == SYSTEM_PROMPT_VERSION
    assert mod.GEOSMILE_SYSTEM_PROMPT == GEOSMILE_SYSTEM_PROMPT


def test_openai_and_anthropic_share_identical_master_prompt():
    from dashboard.llm.openai_provider import GEOSMILE_SYSTEM_PROMPT as openai_prompt
    from dashboard.llm.anthropic_provider import GEOSMILE_SYSTEM_PROMPT as anthropic_prompt

    assert openai_prompt is GEOSMILE_SYSTEM_PROMPT
    assert anthropic_prompt is GEOSMILE_SYSTEM_PROMPT
    assert openai_prompt == anthropic_prompt


def test_missing_openai_key_does_not_crash(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = OpenAIProvider().generate(GEOSMILE_SYSTEM_PROMPT, {"a": 1}, {"id": "researcher"})
    assert response.ok is False
    assert "API key is not configured for this provider." in response.error


def test_missing_anthropic_key_does_not_crash(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    response = AnthropicProvider().generate(GEOSMILE_SYSTEM_PROMPT, {"a": 1}, {"id": "researcher"})
    assert response.ok is False
    assert "API key is not configured for this provider." in response.error


def test_compare_sends_identical_evidence():
    bundle = load_hull_bundle()
    evidence = build_evidence(
        bundle,
        stakeholder_id="planner",
        current_map_view="geo_sensitivity",
        scope="lsoa",
        selected_lsoa="E01012891",
    )
    payload = user_payload(evidence)
    seen = []

    class Fake:
        def __init__(self, name, model):
            self.name = name
            self.model = model

        def generate(self, system_prompt, ev, stakeholder, user_question=None):
            seen.append(
                {
                    "name": self.name,
                    "system": system_prompt,
                    "evidence": ev,
                    "stakeholder": stakeholder,
                    "question": user_question,
                    "payload": user_payload(ev, user_question),
                }
            )
            return LLMResponse(True, "ok", self.name, self.model, "now")

    openai = Fake("openai", "gpt-test")
    anthropic = Fake("anthropic", "claude-test")
    stakeholder = {"id": "planner", "label": "City Planner / Local Authority"}
    for provider in (openai, anthropic):
        provider.generate(GEOSMILE_SYSTEM_PROMPT, evidence, stakeholder, None)
    assert seen[0]["evidence"] == seen[1]["evidence"]
    assert seen[0]["system"] == seen[1]["system"]
    assert seen[0]["stakeholder"] == seen[1]["stakeholder"]
    assert seen[0]["payload"] == seen[1]["payload"] == payload


def test_provider_failure_is_isolated():
    class Boom:
        name = "openai"
        model = "x"

        def generate(self, *args, **kwargs):
            raise RuntimeError("should be caught by provider, not UI")

    class Ok:
        name = "anthropic"
        model = "y"

        def generate(self, *args, **kwargs):
            return LLMResponse(True, "survived", "anthropic", "y", "now")

    caught = OpenAIProvider()
    with patch.object(caught, "generate", side_effect=lambda *a, **k: LLMResponse(
        False, "", "openai", "x", "now", error="OpenAI request failed (RuntimeError)."
    )):
        openai_result = caught.generate("s", {}, {"id": "researcher"})
    anthropic_result = Ok().generate("s", {}, {"id": "researcher"})
    assert openai_result.ok is False
    assert anthropic_result.ok is True
    assert anthropic_result.text == "survived"


def test_prompt_enforces_word_limit_and_documents_new_evidence():
    text = GEOSMILE_SYSTEM_PROMPT
    assert "Maximum 70 words" in text
    assert "comparison_table_mode" in text and "dashboard_mode" in text
    assert "90–120 words" in text
    assert "local_attribution_patterns" in text
    assert "place_context" in text


def test_prompt_defines_six_role_based_stakeholders():
    from dashboard.data.schema import STAKEHOLDERS

    text = GEOSMILE_SYSTEM_PROMPT
    assert "## 9. SIX STAKEHOLDER CONFIGURATIONS" in text
    assert len(STAKEHOLDERS) == 6
    for stakeholder_id, label in STAKEHOLDERS.items():
        assert f"{label} (id: {stakeholder_id})" in text
    assert "Stakeholder separation rule" in text
    assert "Selected-area integration rule" in text
    assert "scope: overall Hull, or selected LSOA" not in text


def test_local_attribution_evidence_hull_wide_shape():
    bundle = load_hull_bundle()
    evidence = compute_local_attribution_evidence(
        local_features=bundle.local_features,
        global_features=bundle.global_features,
        geometry=bundle.geometry,
        feature_names=bundle.feature_names,
        selected_lsoa=None,
    )
    assert "selected_lsoa" not in evidence["dominant_local_feature"]
    assert len(evidence["dominant_local_feature"]["hull_wide_top_features"]) <= 5
    assert 0.0 <= evidence["explanation_concentration"]["hull_wide_mean"] <= 1.0
    assert len(evidence["explanation_archetypes"]["clusters"]) == 4
    clustering = evidence["dominant_global_feature_clustering"]
    assert clustering["feature_label"]
    assert set(clustering["lisa_counts"]) == {"HH", "LL", "HL", "LH", "NS"}
    assert "GeoSensitivity" not in evidence["note"] or "not" in evidence["note"].lower()


def test_local_attribution_evidence_selected_lsoa_adds_fields():
    bundle = load_hull_bundle()
    lsoa_id = str(bundle.lsoa.iloc[0]["lsoa21cd"])
    evidence = compute_local_attribution_evidence(
        local_features=bundle.local_features,
        global_features=bundle.global_features,
        geometry=bundle.geometry,
        feature_names=bundle.feature_names,
        selected_lsoa=lsoa_id,
    )
    assert "selected_lsoa" in evidence["dominant_local_feature"]
    assert "selected_lsoa" in evidence["explanation_concentration"]
    assert "selected_lsoa_archetype" in evidence["explanation_archetypes"]


def test_explanation_panel_merges_local_attribution_patterns_into_evidence():
    bundle = load_hull_bundle()
    evidence = build_evidence(
        bundle,
        stakeholder_id="researcher",
        current_map_view="geo_sensitivity",
        scope="overall",
        selected_lsoa=None,
    )
    assert "local_attribution_patterns" not in evidence  # build_evidence itself stays unchanged
    from dashboard.llm.attribution_evidence import build_local_attribution_evidence

    evidence["local_attribution_patterns"] = build_local_attribution_evidence(
        bundle.local_features,
        bundle.global_features,
        bundle.geometry,
        tuple(bundle.feature_names),
        None,
    )
    assert "local_attribution_patterns" in evidence
    assert "dominant_local_feature" in evidence["local_attribution_patterns"]


def test_public_error_does_not_echo_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value-should-not-leak")
    provider = OpenAIProvider()
    fake_client = MagicMock()
    fake_client.responses.create.side_effect = RuntimeError("invalid api key sk-secret-value-should-not-leak")
    fake_client.chat.completions.create.side_effect = RuntimeError(
        "invalid api key sk-secret-value-should-not-leak"
    )
    with patch("openai.OpenAI", return_value=fake_client):
        response = provider.generate("sys", {"k": 1}, {"id": "researcher"})
    assert response.ok is False
    assert "sk-secret-value-should-not-leak" not in (response.error or "")
    assert "sk-" not in (response.error or "")
