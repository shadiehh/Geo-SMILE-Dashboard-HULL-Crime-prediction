"""Each stakeholder's picture, title and key questions, as in the thesis stakeholder figure (Chapter 6)."""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "stakeholders"
PROFILES = {
    "public": ("Residents / Public", "Understand what the model focuses on in my area", "#E07020",
               ("What does the model pay attention to across Hull?", "What does it weigh in a place like mine?")),
    "planner": ("City Planners / Local Authority", "Identify where and what to prioritise in spatial planning", "#3A9A4A",
                ("Which areas matter most to the model?", "Which built-environment features does it rely on, and where?")),
    "police": ("Police / Community Safety", "Read recent and persistent signals for operational insight", "#C8323C",
               ("Does the model read an area through recent months or its long-run crime level?",
                "Which areas show high geographic sensitivity?")),
    "manager": ("Managers / Strategic Decision Makers", "Strategic overview with key exceptions and confidence", "#7A4BB5",
                ("What drives the model, and where is it sensitive?",
                 "What should be taken into consideration, and how far can the results be trusted?")),
    "researcher": ("GeoAI Researchers / Model Analysts", "Understand and validate model behaviour", "#2E6DB4",
                   ("How does Geo-SMILE behave across Hull?", "How accurate, consistent and reproducible are its explanations?")),
    "governance": ("Governance / Assurance", "Ensure responsible, transparent and well-bounded use", "#B8860B",
                   ("Is the explanation traceable and reproducible?",
                    "Which limitations need explicit review before the evidence is relied upon?")),
}


def role_image(role: str) -> str:
    return str(ASSETS / f"{role}.png")


def render_role_banner(role: str) -> None:
    title, subtitle, colour, questions = PROFILES[role]
    picture, text = st.columns([1, 6], gap="medium", vertical_alignment="center")
    picture.image(role_image(role), width="stretch")
    items = "".join(f"<li>{html.escape(q)}</li>" for q in questions)
    text.markdown(
        f"<div style='border-left:5px solid {colour};padding:.2rem 0 .2rem .9rem;'>"
        f"<div style='font-size:1.25rem;font-weight:700;color:#12171C;'>{html.escape(title)}</div>"
        f"<div style='color:#57646D;margin-bottom:.35rem;'>{html.escape(subtitle)}</div>"
        f"<div style='font-size:.78rem;font-weight:700;letter-spacing:.06em;color:{colour};'>KEY QUESTIONS</div>"
        f"<ul style='margin:.15rem 0 0 0;padding-left:1.1rem;'>{items}</ul></div>",
        unsafe_allow_html=True)
