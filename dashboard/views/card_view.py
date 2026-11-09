"""The six Chapter 6 stakeholder cards, as published, and the comparison across roles.

A card is the output of generate_card_narrative.py: GPT-4.1 at temperature 0, master prompt plus the
role's card addendum, checked figure by figure against the evidence package. Viewers see the archived
run, so opening the dashboard costs no API credit and every viewer reads the same text.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from dashboard.data.schema import CARDS_DIR, STAKEHOLDERS
from dashboard.llm import card_mode as cm
from dashboard.views.stakeholder_evidence import stakeholder_data, stakeholder_panels

HULL_ROOT = Path(__file__).resolve().parents[2]

# Chapter 6 layout: subtitle, panel headings, text headings, tiles shown, footer note.
CARD_LAYOUT = {
    "public": ("What the model pays attention to, across Hull and in a typical area",
               ("01 · Across Hull", "02 · A typical area: Stoneferry"), ("", ""), 4,
               "Panels are the dashboard's public view."),
    "planner": ("Important areas and built-environment reliance",
                ("01 · Most important areas", "02 · Built-environment share"), ("", ""), 4,
                "Map 01: signed GeoSensitivity (sign is not a crime direction). Map 02: share of local "
                "importance, not an effect on crime."),
    "police": ("Sensitivity, recent or long-run signals and recorded change",
               ("01 · Recent or long-run signal", "02 · Recorded change"), ("", ""), 4,
               "Panels are the dashboard's police view for Old Town (029E)."),
    "manager": ("What drives the model, where it is sensitive, and what can be set aside",
                ("01 · Most sensitive areas", "02 · What drives the model"),
                ("Drivers and exceptions", "What can be set aside, and confidence"), 4,
                "Map 01: Geo branch. Panel 02: Feature branch, share of local importance, not an effect on crime."),
    "researcher": ("How Geo-SMILE behaves across Hull, and how far its explanations can be trusted",
                   ("01 · Surrogate against exact masking", "02 · Top feature, local effect"),
                   ("Explanation behaviour across Hull", "Fidelity, intervention and reproducibility"), 4,
                   "Panels are the dashboard's Hull-wide researcher view."),
    "governance": ("Provenance, reproducibility and the limitations to review before reliance",
                   ("01 · Reproducibility across seeds", "02 · Stability under neighbour additions"),
                   ("Provenance and reproducibility", "Limitations requiring review"), 2,
                   "Panels are the dashboard's governance view."),
}


@st.cache_data(show_spinner=False)
def published_card(stakeholder: str) -> dict | None:
    path = HULL_ROOT / CARDS_DIR / f"{stakeholder}_card_llm.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    run = (record.get("runs") or {}).get("openai") or {}
    return run if run.get("ok") else None


def _tiles(tiles: list[dict]) -> str:
    cells = "".join(
        f"<div class='gs-tile'><div class='gs-tile-value'>{html.escape(str(t.get('value', '')))}</div>"
        f"<div class='gs-tile-label'>{html.escape(str(t.get('label', '')))}</div></div>"
        for t in tiles)
    return f"<div class='gs-tiles'>{cells}</div>"


def _inline(text: str) -> str:
    return cm.marked_html(text).replace("<p style='margin:0 0 0.6rem 0; line-height:1.55;'>", "").replace("</p>", "")


def render_card(stakeholder: str) -> None:
    run = published_card(stakeholder)
    if run is None:
        st.warning("This card has not been published yet. Run generate_card_narrative.py.")
        return
    card = run["content"]
    subtitle, panels_titles, heads, n_tiles, note = CARD_LAYOUT[stakeholder]
    d = stakeholder_data()
    panels = stakeholder_panels(f"{stakeholder}_short", cm.CENTRE, d.default_contrast(cm.CENTRE))

    with st.container(border=True):
        st.markdown(
            f"<div class='gs-kicker'>{html.escape(STAKEHOLDERS[stakeholder])} · {html.escape(subtitle)}</div>"
            f"<div class='gs-headline'>{_inline(card['headline'])}</div>", unsafe_allow_html=True)
        left, right = st.columns(2, gap="large")
        for col, title, head, key, (image, _, caption) in zip((left, right), panels_titles, heads,
                                                              ("section_01", "section_02"), panels):
            with col:
                st.markdown(f"<div class='gs-kicker'>{html.escape(title)}</div>", unsafe_allow_html=True)
                st.image(image, width="stretch")
                st.markdown(f"<div class='gs-caption'>{html.escape(caption)}</div>", unsafe_allow_html=True)
                if head:
                    st.markdown(f"<div class='gs-kicker'>{html.escape(head)}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='gs-body'>{cm.marked_html(card[key])}</div>", unsafe_allow_html=True)
        st.markdown(_tiles(card["tiles"][:n_tiles]), unsafe_allow_html=True)
        checks = "Checks passed" if run["checks"]["ok"] else "Checks failed: " + "; ".join(run["checks"]["problems"])
        st.markdown(
            f"<div class='gs-foot'>{html.escape(card['footer'])} {html.escape(note)}<br>"
            f"<span class='gs-meta'>Text generated by {html.escape(run['model'])} at temperature 0 "
            f"· prompt {html.escape(run['prompt_version'])} + card addendum "
            f"· {html.escape(run['timestamp'][:10])} · {html.escape(checks)}</span></div>", unsafe_allow_html=True)


def render_comparison() -> None:
    st.markdown("<div class='gs-lede'>One model and one evidence base, six explanatory roles. "
                "Each role receives different evidence, not simpler or harder wording.</div>", unsafe_allow_html=True)
    cols = st.columns(2, gap="medium")
    for i, (stakeholder, label) in enumerate(STAKEHOLDERS.items()):
        run = published_card(stakeholder)
        with cols[i % 2], st.container(border=True):
            st.markdown(f"<div class='gs-kicker'>{html.escape(label)}</div>", unsafe_allow_html=True)
            if run is None:
                st.caption("Not published yet.")
                continue
            n_tiles = CARD_LAYOUT[stakeholder][3]
            st.markdown(f"<div class='gs-headline gs-headline-small'>{_inline(run['content']['headline'])}</div>"
                        + _tiles(run["content"]["tiles"][:n_tiles]), unsafe_allow_html=True)
