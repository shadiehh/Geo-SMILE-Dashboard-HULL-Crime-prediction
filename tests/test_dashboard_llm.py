"""LLM provider tests with mocked SDKs."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from dashboard.data.loader import load_hull_bundle
from dashboard.llm.anthropic_provider import AnthropicProvider
from dashboard.llm.base import LLMResponse
from dashboard.llm.evidence_builder import build_evidence, user_payload
from dashboard.llm.openai_provider import OpenAIProvider
from dashboard.llm.prompt import GEOSMILE_SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION, assert_prompt_semantics


def test_prompt_forbids_semantic_collapses():
    assert_prompt_semantics()
    text = GEOSMILE_SYSTEM_PROMPT
    assert text.startswith("# GEOSMILE EXPLAIN — SYSTEM POLICY")
    assert "## 1. ROLE" in text
    assert "## 4. INPUT / EVIDENCE CONTRACT" in text
    assert "## 8. SPATIAL-PATTERN SEMANTICS" in text
    assert "Global Moran versus Local Moran" in text
    assert SYSTEM_PROMPT_VERSION == "geosmile-hull-v1"


def test_openai_and_anthropic_share_identical_master_prompt():
    from dashboard.llm.openai_provider import GEOSMILE_SYSTEM_PROMPT as openai_prompt
    from dashboard.llm.anthropic_provider import GEOSMILE_SYSTEM_PROMPT as anthropic_prompt

    assert openai_prompt is GEOSMILE_SYSTEM_PROMPT
    assert anthropic_prompt is GEOSMILE_SYSTEM_PROMPT
    assert openai_prompt == anthropic_prompt


def test_missing_openai_key_does_not_crash(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = OpenAIProvider().generate(GEOSMILE_SYSTEM_PROMPT, {"a": 1}, {"id": "technical"})
    assert response.ok is False
    assert "API key is not configured for this provider." in response.error


def test_missing_anthropic_key_does_not_crash(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    response = AnthropicProvider().generate(GEOSMILE_SYSTEM_PROMPT, {"a": 1}, {"id": "technical"})
    assert response.ok is False
    assert "API key is not configured for this provider." in response.error


def test_compare_sends_identical_evidence():
    bundle = load_hull_bundle()
    evidence = build_evidence(
        bundle,
        stakeholder_id="domain",
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
    stakeholder = {"id": "domain", "label": "Domain / Operational Expert"}
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
        openai_result = caught.generate("s", {}, {"id": "technical"})
    anthropic_result = Ok().generate("s", {}, {"id": "technical"})
    assert openai_result.ok is False
    assert anthropic_result.ok is True
    assert anthropic_result.text == "survived"


def test_public_error_does_not_echo_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value-should-not-leak")
    provider = OpenAIProvider()
    fake_client = MagicMock()
    fake_client.responses.create.side_effect = RuntimeError("invalid api key sk-secret-value-should-not-leak")
    fake_client.chat.completions.create.side_effect = RuntimeError(
        "invalid api key sk-secret-value-should-not-leak"
    )
    with patch("openai.OpenAI", return_value=fake_client):
        response = provider.generate("sys", {"k": 1}, {"id": "technical"})
    assert response.ok is False
    assert "sk-secret-value-should-not-leak" not in (response.error or "")
    assert "sk-" not in (response.error or "")
