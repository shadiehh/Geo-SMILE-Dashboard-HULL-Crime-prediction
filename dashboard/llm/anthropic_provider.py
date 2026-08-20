"""Anthropic provider using the official Messages API."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from dashboard.llm.base import LLMResponse
from dashboard.llm.evidence_builder import user_payload
from dashboard.llm.prompt import SYSTEM_PROMPT_VERSION

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"


def _secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is not None and name in secrets:
            return str(secrets[name])
    except Exception:
        return None
    return None


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or _secret("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL

    def generate(
        self,
        system_prompt: str,
        evidence: dict,
        stakeholder: dict,
        user_question: str | None = None,
    ) -> LLMResponse:
        key = _secret("ANTHROPIC_API_KEY")
        stamp = datetime.now(timezone.utc).isoformat()
        if not key:
            return LLMResponse(
                ok=False,
                text="",
                provider=self.name,
                model=self.model,
                timestamp=stamp,
                error="API key is not configured for this provider.",
                prompt_version=SYSTEM_PROMPT_VERSION,
            )
        user_content = user_payload(evidence, user_question)
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=key, timeout=60.0)
            response = client.messages.create(
                model=self.model,
                max_tokens=2500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            text = "".join(
                getattr(block, "text", "")
                for block in (response.content or [])
                if getattr(block, "type", None) == "text"
            ).strip()
            usage = None
            if getattr(response, "usage", None) is not None:
                usage = {
                    "input_tokens": getattr(response.usage, "input_tokens", None),
                    "output_tokens": getattr(response.usage, "output_tokens", None),
                }
            if not text:
                return LLMResponse(
                    ok=False,
                    text="",
                    provider=self.name,
                    model=self.model,
                    timestamp=stamp,
                    error="The model returned an empty response.",
                    usage=usage,
                    prompt_version=SYSTEM_PROMPT_VERSION,
                )
            return LLMResponse(
                ok=True,
                text=text,
                provider=self.name,
                model=self.model,
                timestamp=stamp,
                usage=usage,
                prompt_version=SYSTEM_PROMPT_VERSION,
            )
        except Exception as exc:
            return LLMResponse(
                ok=False,
                text="",
                provider=self.name,
                model=self.model,
                timestamp=stamp,
                error=_public_error(exc),
                prompt_version=SYSTEM_PROMPT_VERSION,
            )


def _public_error(exc: Exception) -> str:
    name = type(exc).__name__
    message = str(exc)
    lowered = message.lower()
    if "api key" in lowered or "authentication" in lowered or "unauthorized" in lowered:
        return "The Anthropic API key was rejected. Check ANTHROPIC_API_KEY."
    if "rate" in lowered:
        return "Anthropic rate limit reached. Try again shortly."
    if "timeout" in lowered:
        return "Anthropic request timed out."
    if "model" in lowered and ("not found" in lowered or "not_found" in lowered):
        return f"Anthropic model is unavailable ({os.getenv('ANTHROPIC_MODEL') or DEFAULT_ANTHROPIC_MODEL})."
    if "sk-ant" in message:
        return f"Anthropic request failed ({name})."
    return f"Anthropic request failed ({name})."
