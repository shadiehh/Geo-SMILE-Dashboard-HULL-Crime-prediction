"""Geo-SMILE: stakeholder-adaptive explanation dashboard for Hull next-month crime prediction.

    streamlit run dashboard/app.py

Reads published Geo-SMILE results only; it never recomputes an explanation. The six stakeholder
cards are the archived runs of generate_card_narrative.py, so viewing them costs no API credit.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    pass

import streamlit as st

from dashboard.data.loader import compact_evaluation, load_hull_bundle
from dashboard.data.schema import LSOA_ID, LSOA_NAME, MAP_VIEWS, PROVIDERS, STAKEHOLDERS
from dashboard.views.attribution_patterns import render_attribution_patterns_section
from dashboard.views.card_view import render_card, render_comparison
from dashboard.views.diagnostics import render_diagnostics_section
from dashboard.views.explanation_panel import render_explanation_panel
from dashboard.views.feature_panel import render_evaluation, render_feature_panel, render_global_features
from dashboard.views.maps import render_map

ROLE_LABELS = {"public": "Resident", "planner": "Planner", "police": "Police", "manager": "Manager",
               "researcher": "Researcher", "governance": "Governance"}
ROLE_QUESTIONS = {
    "public": "Which measured factors matter most, and least, to the model's prediction?",
    "planner": "Where is GeoSensitivity strongest, and how does reliance on built-environment features vary across Hull?",
    "police": "Which recent and persistent model signals make an area stand out, and how does recorded change compare across Hull?",
    "manager": "What drives the model city-wide, what can be deprioritised, what is exceptional, and how robust are the main findings?",
    "researcher": "How is Geo-SMILE behaving technically, and how faithful, stable and reproducible is the explanation?",
    "governance": "Is the explanation traceable, robust, appropriately bounded and defensible?",
}
PAGES = ("Stakeholder card", "Explore the map", "Compare roles", "Model diagnostics", "How to read this")

STYLE = """
<style>
.block-container {max-width: 1240px; padding-top: 2.2rem;}
.gs-hero h1 {font-size: 1.9rem; font-weight: 700; margin: 0; padding: 0; letter-spacing: -0.01em;}
.gs-hero p {color: #57646D; margin: 0.25rem 0 0 0; max-width: 62rem; line-height: 1.5;}
.gs-stats {display: flex; flex-wrap: wrap; gap: 0.5rem 2.25rem; margin: 0.9rem 0 0.4rem 0;}
.gs-stat b {font-size: 1.15rem; color: #12171C; font-variant-numeric: tabular-nums;}
.gs-stat span {display: block; font-size: 0.76rem; color: #57646D;}
.gs-kicker {font-size: 0.78rem; font-weight: 700; color: #57646D; margin: 0.15rem 0 0.35rem 0;}
.gs-headline {font-size: 1.28rem; font-weight: 650; line-height: 1.38; margin: 0 0 0.9rem 0; color: #12171C;}
.gs-headline-small {font-size: 1.02rem; margin-bottom: 0.6rem;}
.gs-question {font-size: 0.95rem; color: #57646D; margin: 0.15rem 0 0.8rem 0;}
.gs-caption {font-size: 0.76rem; color: #57646D; line-height: 1.4; margin: -0.3rem 0 0.75rem 0;}
.gs-body {font-size: 0.96rem;}
.gs-tiles {display: grid; grid-template-columns: repeat(auto-fit, minmax(6.6rem, 1fr)); gap: 0.55rem; margin: 0.35rem 0 0.75rem 0;}
.gs-tile {background: #FCE4EC; border-radius: 6px; padding: 0.55rem 0.7rem;}
.gs-tile-value {color: #8A0033; font-weight: 700; font-size: 1.2rem; font-variant-numeric: tabular-nums;}
.gs-tile-label {color: #57646D; font-size: 0.78rem; line-height: 1.3;}
.gs-foot {border-top: 1px solid #D9DEE2; padding-top: 0.5rem; font-size: 0.8rem; color: #57646D; line-height: 1.45;}
.gs-meta {font-size: 0.73rem;}
.gs-lede {font-size: 1rem; color: #57646D; margin-bottom: 0.8rem;}
</style>
"""


@st.cache_resource(show_spinner="Loading Hull Geo-SMILE results…")
def _bundle():
    return load_hull_bundle(ROOT)


def _header(bundle) -> None:
    ev = compact_evaluation(bundle.evaluation)
    stats = [
        (f"{len(bundle.lsoa)}", "Hull LSOAs explained"),
        (f"{ev['feature_holdout_r2']:.3f}", "held-out R² of the XGBoost model"),
        (f"{ev['feature_surrogate_test_r2']:.3f}", "Feature surrogate fidelity"),
        (f"{ev['geo_fidelity_r2']:.3f}", "Geo surrogate fidelity"),
        ("4,000", "perturbations per branch"),
    ]
    cells = "".join(f"<div class='gs-stat'><b>{v}</b><span>{k}</span></div>" for v, k in stats)
    st.markdown(
        "<div class='gs-hero'><h1>Geo-SMILE for Hull crime prediction</h1>"
        "<p>How one next-month crime model behaves across Kingston upon Hull, explained for six stakeholder "
        "roles. These are attributions of a fitted model, not causes of crime and not advice on where to "
        f"deploy resources.</p></div><div class='gs-stats'>{cells}</div>", unsafe_allow_html=True)


def _selector(key: str, options, labels: dict, default: str):
    """A segmented control that cannot be left empty."""
    choice = st.segmented_control(" ", options, format_func=labels.get, default=default, key=key,
                                  label_visibility="collapsed")
    if choice is None:
        choice = st.session_state.get(f"{key}_last", default)
    st.session_state[f"{key}_last"] = choice
    return choice


def _page_card(role: str) -> None:
    st.markdown(f"<div class='gs-question'>{ROLE_QUESTIONS[role]}</div>", unsafe_allow_html=True)
    render_card(role)


def _lsoa_picker(bundle) -> str | None:
    labels = {str(code): f"{name.replace('Kingston upon Hull ', '')} · {place}" if place else
              str(name).replace("Kingston upon Hull ", "")
              for code, name, place in zip(bundle.lsoa[LSOA_ID].astype(str), bundle.lsoa[LSOA_NAME].astype(str),
                                           bundle.lsoa["locality_label"].fillna(""))}
    options = [""] + sorted(labels, key=labels.get)
    current = st.session_state.get("selected_lsoa") or ""
    picked = st.selectbox("Area", options, index=options.index(current) if current in options else 0,
                          format_func=lambda v: labels.get(v, "Choose an area, or click the map"))
    if picked != current:
        st.session_state["selected_lsoa"] = picked or None
    return st.session_state.get("selected_lsoa")


def _page_map(bundle, role: str) -> None:
    controls, mapper, detail = st.columns([0.95, 3.1, 1.55], gap="medium")
    with controls:
        map_view = st.radio("Map layer", list(MAP_VIEWS), format_func=MAP_VIEWS.get, key="map_view")
        selected = _lsoa_picker(bundle)
        provider = st.radio("Briefing model", list(PROVIDERS), format_func=PROVIDERS.get, key="provider_choice")
    with mapper:
        if map_view in ("latitude_effect", "longitude_effect", "overlay", "top50_rank"):
            st.caption("Coordinate-inclusive diagnostic run. Latitude and longitude effects are not GeoSensitivity.")
        clicked = render_map(bundle, map_view, selected)
        if clicked and clicked != selected:
            st.session_state["selected_lsoa"] = clicked
            st.rerun()
    with detail:
        render_feature_panel(bundle, selected)
        render_global_features(bundle)
        render_evaluation(bundle)
    st.divider()
    render_explanation_panel(
        bundle, stakeholder_id=role, provider_choice=provider, scope="lsoa" if selected else "overall",
        current_map_view=map_view, selected_lsoa=selected, force_mode="comprehensive",
    )


def _page_about() -> None:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            """
#### What each quantity means
**GeoSensitivity.** How much predictions depend on knowing where an area is. An area's coordinates are moved
to the centroids of other randomly drawn Hull areas (50 draws) and all other inputs are kept. Areas are ranked
by the signed coefficient. Its sign is not a direction of crime.

**Single-area shift.** The prediction change, in crimes per month, when one area alone is relocated. It reports
size, which rank does not.

**Global Feature importance.** Ranked by coefficient magnitude. Because the response is a distance, the sign
does not say whether a predictor raises or lowers predicted crime.

**Local Feature effect.** Directional. Pink means the area's own value raises its prediction relative to the
training-median reference, and blue means it lowers it.

**Spatial pattern.** Whether signed GeoSensitivity values cluster with their neighbours. A cluster is not a
crime hotspot.
"""
        )
    with right:
        st.markdown(
            """
#### Limits
- Attributions describe a fitted model. They do not identify causes of crime.
- Nothing here is advice on deployment, surveillance, targeting or resource allocation.
- Because the model predicts counts, GeoSensitivity is larger where predicted crime is higher.
- Local LISA p-values are not adjusted for multiple testing.
- Census characteristics can act as proxies for protected groups.
- Expert agreement reflects one expert's judgement and is not validation of causes.

#### Where the text comes from
The language model only communicates structured results. It does not see maps, compute statistics or find
patterns. Every figure on a card is checked against the evidence it was given. The stakeholder cards are
archived runs of GPT-4.1 at temperature 0.
"""
        )


def main() -> None:
    st.set_page_config(page_title="Geo-SMILE · Hull crime prediction", page_icon="🗺️", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.session_state.setdefault("selected_lsoa", None)
    st.session_state.setdefault("map_view", "geo_sensitivity")
    bundle = _bundle()

    _header(bundle)
    page = _selector("page", list(PAGES), {p: p for p in PAGES}, PAGES[0])
    role = "public"
    if page in ("Stakeholder card", "Explore the map"):
        st.markdown("<div class='gs-kicker'>Explain for</div>", unsafe_allow_html=True)
        role = _selector("role", list(STAKEHOLDERS), ROLE_LABELS, "public")

    if page == "Stakeholder card":
        _page_card(role)
    elif page == "Explore the map":
        _page_map(bundle, role)
    elif page == "Compare roles":
        render_comparison()
    elif page == "Model diagnostics":
        render_diagnostics_section("researcher")
        render_attribution_patterns_section(bundle, "researcher")
    else:
        _page_about()


if __name__ == "__main__":
    main()
