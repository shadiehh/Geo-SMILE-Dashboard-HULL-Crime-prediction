"""Stakeholder explanation panel: generate only on explicit user action."""

from __future__ import annotations

import html
import os
import re

import streamlit as st

from dashboard.data.loader import HullBundle
from dashboard.data.schema import PROVIDERS, STAKEHOLDER_HELP, STAKEHOLDERS
from dashboard.llm.anthropic_provider import AnthropicProvider
from dashboard.llm import card_mode as cm
from dashboard.llm.attribution_evidence import build_local_attribution_evidence
from dashboard.llm.base import LLMResponse
from dashboard.llm.cache import cache_get, cache_put, evidence_hash
from dashboard.llm.evidence_builder import build_evidence
from dashboard.llm.openai_provider import OpenAIProvider
from dashboard.llm.prompt import GEOSMILE_SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION
from dashboard.views.stakeholder_evidence import (
    contrast_selector,
    figure_evidence,
    focus_lsoa,
    render_emphasis,
    render_evidence_column,
    stakeholder_data,
    stakeholder_panels,
)


def _providers_for(choice: str, mode: str | None = None) -> list:
    if choice == "anthropic":
        return [AnthropicProvider()]
    # Card-mode (short/comprehensive) narratives need the markup rules followed; the Chapter 6
    # cards were generated with gpt-4.1 (gpt-4.1-mini missed them), so default to it here.
    if mode:
        return [OpenAIProvider(model=os.getenv("CARD_OPENAI_MODEL") or os.getenv("PLANNER_OPENAI_MODEL", "gpt-4.1"))]
    return [OpenAIProvider()]


def _generate(
    provider,
    evidence: dict,
    stakeholder: dict,
    user_question: str | None,
    *,
    force: bool,
    mode: str | None = None,
) -> LLMResponse:
    spec = cm.SPECS.get(stakeholder["id"]) if mode else None
    version = f"{SYSTEM_PROMPT_VERSION}+{spec['version'][mode]}" if spec else SYSTEM_PROMPT_VERSION
    key = evidence_hash(
        provider=provider.name,
        model=provider.model,
        stakeholder=stakeholder["id"],
        scope=evidence.get("scope", ""),
        selected_lsoa=(evidence.get("selected_lsoa") or {}).get("id"),
        current_map=evidence.get("current_map_view", ""),
        evidence=evidence,
        user_question=user_question,
        prompt_version=version,
    )
    if not force:
        cached = cache_get(key)
        if cached is not None:
            return cached
    extra = ({"addendum": spec["addenda"][mode], "user_suffix": cm.user_suffix(mode, stakeholder["id"])}
             if spec else {})
    response = provider.generate(
        GEOSMILE_SYSTEM_PROMPT,
        evidence,
        stakeholder,
        user_question,
        **extra,
    )
    response.prompt_version = version
    if mode and response.ok:
        response.usage = {**(response.usage or {}), "checks": _checks(response.text, evidence, mode, spec)}
    if response.ok:
        cache_put(key, response)
    return response


def _checks(text: str, evidence: dict, mode: str, spec: dict | None = None) -> dict:
    try:
        content = cm.parse_card(text) if mode == "short" else text
    except ValueError as exc:
        return {"ok": False, "problems": [f"not valid JSON: {exc}"]}
    limits = (spec or {}).get("limits" if mode == "short" else "comprehensive_limits")
    return cm.check(content, evidence, mode, limits)


def _show_card(response: LLMResponse, mode: str, stakeholder_id: str) -> None:
    if mode == "short":
        try:
            card = cm.parse_card(response.text)
        except ValueError:
            st.warning("The model did not return a valid card.")
            st.code(response.text)
            return
        st.markdown(cm.tiles_html(card.get("tiles", [])), unsafe_allow_html=True)
        for title, key in zip(cm.SPECS[stakeholder_id]["titles"], ("section_01", "section_02")):
            st.markdown(f"<div style='font-size:0.8rem; font-weight:700; opacity:0.7;'>{title}</div>"
                        f"{cm.marked_html(card.get(key, ''))}", unsafe_allow_html=True)
        st.caption(card.get("footer", ""))
    else:
        st.markdown(cm.marked_html(response.text), unsafe_allow_html=True)
    checks = (response.usage or {}).get("checks")
    if checks and checks["ok"]:
        st.caption("Checks passed: every figure traced to the evidence; length and highlight limits met.")
    elif checks:
        st.warning("Checks failed: " + "; ".join(checks["problems"]))


def _show_response(response: LLMResponse, heading: str, mode: str | None = None,
                   stakeholder_id: str = "") -> None:
    st.markdown(f"**{heading}**")
    if not response.ok:
        st.warning(response.error or "Explanation could not be generated.")
        return
    if mode:
        _show_card(response, mode, stakeholder_id)
    else:
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
    st.subheader(f"Stakeholder explanation: {STAKEHOLDERS[stakeholder_id]}")
    st.caption(
        "Left: evidence figures chosen for this stakeholder and drawn for the selected LSOA. "
        "Right: the LLM narrative, which explains the same structured evidence package. "
        "The models do not recompute GeoSMILE or invent spatial patterns from the map."
    )

    if scope == "lsoa" and not selected_lsoa:
        st.info("Select an LSOA on the map before generating a local explanation.")

    mode = None
    if stakeholder_id in cm.SPECS:
        mode = st.radio("Narrative", list(cm.MODES), format_func=cm.MODES.get, horizontal=True,
                        key=f"{stakeholder_id}_mode",
                        help="Short: the stakeholder card (Hull-wide, plain language). Comprehensive: a "
                             "longer briefing with feature charts and maps, for the selected LSOA.")
    if mode == "short":
        # The short card is Hull-wide, like the Chapter 6 card.
        scope, selected_lsoa = "overall", None
        st.caption("Short card: Hull-wide. Choose Comprehensive for the selected LSOA, feature charts and maps.")

    data = stakeholder_data()
    focus, is_fallback = focus_lsoa(data, scope, selected_lsoa)
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        generate = st.button("Generate explanation", type="primary")
    with col_b:
        regenerate = st.button("Regenerate")
    with col_c:
        contrast = (data.default_contrast(focus) if mode == "short"
                    else contrast_selector(data, stakeholder_id, focus))
    follow_up = st.text_input("Ask about this result...", value="", placeholder="Optional follow-up")
    panels = stakeholder_panels(f"{stakeholder_id}_short" if mode == "short" else stakeholder_id, focus, contrast)
    note = (f"No LSOA selected: figures describe the rank-1 area, {data.short(focus)}."
            if is_fallback and mode != "short" else None)

    evidence = build_evidence(
        bundle,
        stakeholder_id=stakeholder_id,
        current_map_view=current_map_view,
        scope=scope,
        selected_lsoa=selected_lsoa,
        user_question=follow_up.strip() or (cm.SPECS[stakeholder_id]["question"] if mode else None),
    )
    if mode:
        cm.add_evidence(evidence, data, stakeholder_id, mode)
    evidence["local_attribution_patterns"] = build_local_attribution_evidence(
        bundle.local_features,
        bundle.global_features,
        bundle.geometry,
        tuple(bundle.feature_names),
        selected_lsoa if scope == "lsoa" else None,
    )
    evidence["dashboard_figures"] = {
        "lsoa": focus,
        "contrast_lsoa": contrast if stakeholder_id == "planner" else None,
        "figures": figure_evidence(panels),
    }
    with st.expander("Evidence package sent to the LLM"):
        st.json(evidence)

    left, right = st.columns([0.42, 0.58], gap="large")
    with left:
        render_evidence_column(stakeholder_id, panels, fallback_note=note)
    with right:
        st.markdown("<span style='font-size:0.75rem; font-weight:700; letter-spacing:0.04em;'>"
                    "STAKEHOLDER-SPECIFIC NARRATIVE</span>", unsafe_allow_html=True)
        _narrative_column(evidence, stakeholder_id, provider_choice, scope, selected_lsoa,
                          follow_up, generate, regenerate, mode)
        render_emphasis(stakeholder_id)


def _narrative_column(evidence, stakeholder_id, provider_choice, scope, selected_lsoa,
                      follow_up, generate, regenerate, mode=None) -> None:
    if not (generate or regenerate):
        previous = st.session_state.get("last_explanations")
        if (previous and st.session_state.get("last_stakeholder") == stakeholder_id
                and st.session_state.get("last_mode") == mode):
            _render_result(previous, provider_choice, mode, stakeholder_id)
        else:
            st.caption("Press Generate explanation to write the narrative for these figures.")
        return

    if scope == "lsoa" and not selected_lsoa:
        st.warning("Select an LSOA first.")
        return

    stakeholder = {"id": stakeholder_id, "label": STAKEHOLDERS[stakeholder_id]}
    results: dict[str, LLMResponse] = {}
    for provider in _providers_for(provider_choice, mode):
        results[provider.name] = _generate(
            provider,
            evidence,
            stakeholder,
            follow_up.strip() or (cm.SPECS[stakeholder_id]["question"] if mode else None),
            force=regenerate,
            mode=mode,
        )
    st.session_state["last_explanations"] = results
    st.session_state["last_evidence"] = evidence
    st.session_state["last_stakeholder"] = stakeholder_id
    st.session_state["last_mode"] = mode
    _render_result(results, provider_choice, mode, stakeholder_id)


def _render_result(results: dict[str, LLMResponse], provider_choice: str, mode: str | None = None,
                   stakeholder_id: str = "") -> None:
    name = "anthropic" if provider_choice == "anthropic" else "openai"
    heading = PROVIDERS[name] + " explanation" + (f" ({cm.MODES[mode].lower()})" if mode else "")
    if name in results:
        _show_response(results[name], heading, mode, stakeholder_id)


def _generate_all_stakeholders(
    bundle: HullBundle,
    *,
    current_map_view: str,
    scope: str,
    selected_lsoa: str | None,
    user_question: str | None,
    force: bool,
) -> dict[str, LLMResponse]:
    """Generate one narrative per stakeholder, all via OpenAI, for the same evidence context."""
    provider = OpenAIProvider()
    local_patterns = build_local_attribution_evidence(
        bundle.local_features,
        bundle.global_features,
        bundle.geometry,
        tuple(bundle.feature_names),
        selected_lsoa if scope == "lsoa" else None,
    )
    results: dict[str, LLMResponse] = {}
    for stakeholder_id in STAKEHOLDERS:
        evidence = build_evidence(
            bundle,
            stakeholder_id=stakeholder_id,
            current_map_view=current_map_view,
            scope=scope,
            selected_lsoa=selected_lsoa,
            user_question=user_question,
            output_mode="comparison_table_mode",
        )
        evidence["local_attribution_patterns"] = local_patterns
        data = stakeholder_data()
        focus, _ = focus_lsoa(data, scope, selected_lsoa)
        contrast = data.default_contrast(focus)
        evidence["dashboard_figures"] = {
            "lsoa": focus,
            "contrast_lsoa": contrast if stakeholder_id == "planner" else None,
            "figures": figure_evidence(stakeholder_panels(stakeholder_id, focus, contrast)),
        }
        stakeholder = {"id": stakeholder_id, "label": STAKEHOLDERS[stakeholder_id]}
        results[stakeholder_id] = _generate(
            provider, evidence, stakeholder, user_question, force=force
        )
    return results


def _narrative_html(text: str) -> str:
    escaped = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def _render_stakeholder_table(results: dict[str, LLMResponse]) -> None:
    rows = []
    for stakeholder_id, label in STAKEHOLDERS.items():
        response = results.get(stakeholder_id)
        if response is None:
            continue
        if response.ok:
            body = _narrative_html(response.text)
            meta = f"{response.model} &middot; {html.escape(response.timestamp)}"
        else:
            body = (
                '<span style="color:#d92d20;">'
                f'{html.escape(response.error or "Explanation could not be generated.")}'
                "</span>"
            )
            meta = html.escape(response.model)
        rows.append(
            f"""<tr>
<td style="padding:10px; border-bottom:1px solid rgba(128,128,128,0.25); vertical-align:top; width:20%;">
<strong>{html.escape(label)}</strong><br>
<span style="opacity:0.65; font-size:0.78rem;">{html.escape(STAKEHOLDER_HELP[stakeholder_id])}</span>
</td>
<td style="padding:10px; border-bottom:1px solid rgba(128,128,128,0.25); vertical-align:top; line-height:1.5;">
{body}<br><span style="opacity:0.55; font-size:0.72rem;">{meta}</span>
</td>
</tr>"""
        )
    table_html = f"""<table style="width:100%; border-collapse:collapse; font-size:0.92rem;">
<thead>
<tr>
<th style="text-align:left; padding:8px 10px; border-bottom:2px solid rgba(128,128,128,0.4);">Stakeholder</th>
<th style="text-align:left; padding:8px 10px; border-bottom:2px solid rgba(128,128,128,0.4);">Narrative (OpenAI)</th>
</tr>
</thead>
<tbody>
{"".join(rows)}
</tbody>
</table>"""
    st.markdown(table_html, unsafe_allow_html=True)


def render_stakeholder_table_section(
    bundle: HullBundle,
    *,
    scope: str,
    current_map_view: str,
    selected_lsoa: str | None,
) -> None:
    st.subheader("Stakeholder narrative comparison (OpenAI)")
    st.caption(
        "One short narrative per stakeholder persona for the current scope, map view, "
        "and LSOA selection — same evidence package, six audiences. Generated with OpenAI."
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        generate = st.button("Generate stakeholder table", type="primary", key="gen_table")
    with col_b:
        regenerate = st.button("Regenerate table", key="regen_table")
    follow_up = st.text_input(
        "Ask about this result...",
        value="",
        placeholder="Optional follow-up (applied to all six stakeholders)",
        key="table_follow_up",
    )

    if scope == "lsoa" and not selected_lsoa:
        st.info("Select an LSOA on the map before generating a local comparison table.")

    if not (generate or regenerate):
        previous = st.session_state.get("stakeholder_table")
        if previous:
            _render_stakeholder_table(previous)
        return

    if scope == "lsoa" and not selected_lsoa:
        st.warning("Select an LSOA first.")
        return

    with st.spinner("Generating narratives for all six stakeholders…"):
        results = _generate_all_stakeholders(
            bundle,
            current_map_view=current_map_view,
            scope=scope,
            selected_lsoa=selected_lsoa,
            user_question=follow_up.strip() or None,
            force=regenerate,
        )
    st.session_state["stakeholder_table"] = results
    _render_stakeholder_table(results)


def stakeholder_selector() -> str:
    choice = st.selectbox(
        "Explain for",
        list(STAKEHOLDERS.keys()),
        format_func=lambda k: STAKEHOLDERS[k],
    )
    st.caption(STAKEHOLDER_HELP[choice])
    return choice
