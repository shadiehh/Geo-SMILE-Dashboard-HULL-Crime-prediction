"""Hash-based cache so explanations are not regenerated on every Streamlit rerun."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dashboard.llm.base import LLMResponse
from dashboard.llm.prompt import SYSTEM_PROMPT_VERSION

CACHE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "dashboard_llm_cache"


def evidence_hash(
    *,
    provider: str,
    model: str,
    stakeholder: str,
    scope: str,
    selected_lsoa: str | None,
    current_map: str,
    evidence: dict[str, Any],
    user_question: str | None,
    prompt_version: str = SYSTEM_PROMPT_VERSION,
) -> str:
    payload = {
        "provider": provider,
        "model": model,
        "stakeholder": stakeholder,
        "scope": scope,
        "selected_lsoa": selected_lsoa,
        "current_map": current_map,
        "structured_evidence": evidence,
        "system_prompt_version": prompt_version,
        "user_question": user_question or "",
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_get(key: str) -> LLMResponse | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return LLMResponse(**data)


def cache_put(key: str, response: LLMResponse) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps(response.to_dict(), indent=2), encoding="utf-8"
    )
