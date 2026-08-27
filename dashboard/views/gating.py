"""Shared persona-gating for optional dashboard sections."""

from __future__ import annotations

import streamlit as st


def persona_gated(
    stakeholder_id: str,
    *,
    default_personas: set[str],
    toggle_label: str,
    toggle_key: str,
    help_text: str,
) -> bool:
    """True if the caller should render its section body.

    Visible by default for stakeholder_id in default_personas; gated behind
    an explicit checkbox (default unchecked) for everyone else.
    """
    if stakeholder_id in default_personas:
        return True
    return st.checkbox(toggle_label, value=False, key=toggle_key, help=help_text)
