"""Stakeholder explanation panel: generate only on explicit user action."""

from __future__ import annotations

import streamlit as st

from dashboard.data.loader import HullBundle
from dashboard.data.schema import PROVIDERS, STAKEHOLDER_HELP, STAKEHOLDERS
from dashboard.llm.anthropic_provider import AnthropicProvider
from dashboard.llm.attribution_evidence import build_local_attribution_evidence
from dashboard.llm.base import LLMResponse
from dashboard.llm.cache import cache_get, cache_put, evidence_hash
from dashboard.llm.evidence_builder import build_evidence
from dashboard.llm.openai_provider import OpenAIProvider
from dashboard.llm.prompt import GEOSMILE_SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION


def _providers_for(choice: str) -> list:
    if choice == "anthropic":
        return [AnthropicProvider()]
    return [OpenAIProvider()]


def _generate(
    provider,
    evidence: dict,
    stakeholder: dict,
    user_question: str | None,
    *,
    force: bool,
) -> LLMResponse:
    key = evidence_hash(
        provider=provider.name,
        model=provider.model,
        stakeholder=stakeholder["id"],
        scope=evidence.get("scope", ""),
        selected_lsoa=(evidence.get("selected_lsoa") or {}).get("id"),
        current_map=evidence.get("current_map_view", ""),
        evidence=evidence,
        user_question=user_question,
        prompt_version=SYSTEM_PROMPT_VERSION,
    )
    if not force:
        cached = cache_get(key)
        if cached is not None:
            return cached
    response = provider.generate(
        GEOSMILE_SYSTEM_PROMPT,
        evidence,
        stakeholder,
        user_question,
    )
    if response.ok:
        cache_put(key, response)
    return response


def _show_response(response: LLMResponse, heading: str) -> None:
    st.markdown(f"**{heading}**")
    if not response.ok:
        st.warning(response.error or "Explanation could not be generated.")
        return
    st.markdown(response.text)
    st.caption(
        f"{response.provider} · {response.model} · {response.timestamp} · "
        f"prompt {response.prompt_version or SYSTEM_PROMPT_VERSION}"
    )
    if response.usage:
        with st.expander("Diagnostics"):
            st.json(response.usage)


def render_explanation_panel(
    bundle: HullBundle,
    *,
    stakeholder_id: str,
    provider_choice: str,
    scope: str,
    current_map_view: str,
    selected_lsoa: str | None,
) -> None:
    st.subheader("LLM stakeholder explanation")
    st.caption(
        "The models explain the structured evidence package. They do not "
        "recompute GeoSMILE or invent spatial patterns from the map."
    )

    if scope == "lsoa" and not selected_lsoa:
        st.info("Select an LSOA on the map before generating a local explanation.")

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        generate = st.button("Generate explanation", type="primary")
    with col_b:
        regenerate = st.button("Regenerate")
    follow_up = st.text_input("Ask about this result...", value="", placeholder="Optional follow-up")

    evidence = build_evidence(
        bundle,
        stakeholder_id=stakeholder_id,
        current_map_view=current_map_view,
        scope=scope,
        selected_lsoa=selected_lsoa,
        user_question=follow_up.strip() or None,
    )
    evidence["local_attribution_patterns"] = build_local_attribution_evidence(
        bundle.local_features,
        bundle.global_features,
        bundle.geometry,
        tuple(bundle.feature_names),
        selected_lsoa if scope == "lsoa" else None,
    )
    with st.expander("Evidence package sent to the LLM"):
        st.json(evidence)

    if not (generate or regenerate):
        previous = st.session_state.get("last_explanations")
        if previous:
            _render_result(previous, provider_choice)
        return

    if scope == "lsoa" and not selected_lsoa:
        st.warning("Select an LSOA first.")
        return

    stakeholder = {"id": stakeholder_id, "label": STAKEHOLDERS[stakeholder_id]}
    results: dict[str, LLMResponse] = {}
    for provider in _providers_for(provider_choice):
        results[provider.name] = _generate(
            provider,
            evidence,
            stakeholder,
            follow_up.strip() or None,
            force=regenerate,
        )
    st.session_state["last_explanations"] = results
    st.session_state["last_evidence"] = evidence
    _render_result(results, provider_choice)


def _render_result(results: dict[str, LLMResponse], provider_choice: str) -> None:
    name = "anthropic" if provider_choice == "anthropic" else "openai"
    heading = PROVIDERS[name] + " explanation"
    if name in results:
        _show_response(results[name], heading)


def stakeholder_selector() -> str:
    choice = st.selectbox(
        "Explain for",
        list(STAKEHOLDERS.keys()),
        format_func=lambda k: STAKEHOLDERS[k],
    )
    st.caption(STAKEHOLDER_HELP[choice])
    return choice
