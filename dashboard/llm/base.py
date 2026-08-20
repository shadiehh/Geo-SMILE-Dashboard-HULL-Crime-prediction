"""Provider-agnostic LLM response object."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass
class LLMResponse:
    ok: bool
    text: str
    provider: str
    model: str
    timestamp: str
    error: str | None = None
    usage: dict[str, Any] | None = None
    prompt_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(
        self,
        system_prompt: str,
        evidence: dict,
        stakeholder: dict,
        user_question: str | None = None,
    ) -> LLMResponse:
        ...
