"""Usage limits and a usage log for live AI generation on a public deployment.

Limits come from the environment or Streamlit secrets (LLM_DAILY_LIMIT, LLM_SESSION_LIMIT). Cached answers cost
nothing and are not counted. The daily counter lives in the server process, so it restarts with the app; the log
file records every paid call for cost monitoring.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import streamlit as st

from dashboard.llm.cache import CACHE_DIR
from dashboard.llm.openai_provider import _secret

USAGE_LOG = CACHE_DIR / "usage.jsonl"


def _limit(name: str, default: int) -> int:
    try:
        return int(_secret(name) or default)
    except ValueError:
        return default


@st.cache_resource(show_spinner=False)
def _counter() -> dict:
    return {"day": None, "calls": 0, "tokens": 0}


def _today() -> dict:
    counter = _counter()
    day = datetime.now(timezone.utc).date().isoformat()
    if counter["day"] != day:
        counter.update(day=day, calls=0, tokens=0)
    return counter


def refusal() -> str | None:
    """Why a new paid call is not allowed now, or None."""
    if _today()["calls"] >= _limit("LLM_DAILY_LIMIT", 300):
        return "The daily limit for live AI explanations has been reached. Please try again tomorrow."
    if st.session_state.get("llm_calls", 0) >= _limit("LLM_SESSION_LIMIT", 30):
        return "This session has reached its limit for live AI explanations."
    return None


def record(usage: dict | None, model: str, stakeholder: str, mode: str | None) -> None:
    counter = _today()
    tokens = int((usage or {}).get("total_tokens") or 0)
    counter["calls"] += 1
    counter["tokens"] += tokens
    st.session_state["llm_calls"] = st.session_state.get("llm_calls", 0) + 1
    try:
        USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with USAGE_LOG.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "model": model,
                                  "stakeholder": stakeholder, "mode": mode, "tokens": tokens}) + "\n")
    except OSError:
        pass


def summary() -> dict:
    counter = _today()
    return {"calls": counter["calls"], "tokens": counter["tokens"], "daily_limit": _limit("LLM_DAILY_LIMIT", 300),
            "session_limit": _limit("LLM_SESSION_LIMIT", 30)}
