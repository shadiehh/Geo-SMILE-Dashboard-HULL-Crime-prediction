"""OpenAI provider using the Responses API with a Chat Completions fallback."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from dashboard.llm.base import LLMResponse
from dashboard.llm.evidence_builder import user_payload
from dashboard.llm.prompt import SYSTEM_PROMPT_VERSION

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


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


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or _secret("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL

    def generate(
        self,
        system_prompt: str,
        evidence: dict,
        stakeholder: dict,
        user_question: str | None = None,
    ) -> LLMResponse:
        key = _secret("OPENAI_API_KEY")
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
            from openai import OpenAI

            client = OpenAI(api_key=key, timeout=60.0)
            text, usage = self._call(client, system_prompt, user_content)
            if not text.strip():
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
                text=text.strip(),
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

    def _call(self, client, system_prompt: str, user_content: str) -> tuple[str, dict | None]:
        try:
            response = client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=user_content,
            )
            text = getattr(response, "output_text", None) or ""
            usage = _usage(getattr(response, "usage", None))
            return str(text), usage
        except Exception:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            text = (response.choices[0].message.content or "") if response.choices else ""
            usage = _usage(getattr(response, "usage", None))
            return str(text), usage


def _usage(usage) -> dict | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    try:
        return json.loads(usage.json())
    except Exception:
        return None


def _public_error(exc: Exception) -> str:
    name = type(exc).__name__
    message = str(exc)
    lowered = message.lower()
    if "api key" in lowered or "authentication" in lowered or "unauthorized" in lowered:
        return "The OpenAI API key was rejected. Check OPENAI_API_KEY."
    if "rate" in lowered:
        return "OpenAI rate limit reached. Try again shortly."
    if "timeout" in lowered:
        return "OpenAI request timed out."
    if "model" in lowered and ("not found" in lowered or "does not exist" in lowered):
        return f"OpenAI model is unavailable ({os.getenv('OPENAI_MODEL') or DEFAULT_OPENAI_MODEL})."
    if "sk-" in message or "key" in lowered:
        return f"OpenAI request failed ({name})."
    return f"OpenAI request failed ({name})."
