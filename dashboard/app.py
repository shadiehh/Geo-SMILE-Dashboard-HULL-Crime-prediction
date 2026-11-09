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
from dashboard.data.schema import LSOA_ID, LSOA_NAME, PROVIDERS, STAKEHOLDERS, human_label
from dashboard.llm import card_mode as cm
from dashboard.views.attribution_patterns import render_attribution_patterns_section
from dashboard.views.area_panel import render_area_panel
from dashboard.views.role_banner import render_role_banner
from dashboard.views.decision_support import decision_table, export_buttons, render_compare, render_filters, render_shortlist
from dashboard.views.card_view import render_area_card, render_area_detailed, render_card, render_comparison, render_detailed
from dashboard.views.diagnostics import render_diagnostics_section
from dashboard.views.explanation_panel import render_explanation_panel
from dashboard.views.maps import prediction_months
from dashboard.views.feature_panel import render_evaluation, render_global_features
from dashboard.llm.anthropic_provider import _secret as _anthropic_secret
from dashboard.llm.openai_provider import _secret as _openai_secret
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
PAGES = ("Map & Analysis", "Stakeholder Cards", "Area Comparison", "Role Comparison", "Spatial Patterns",
         "Model & Explanations", "Data & Governance")
LAYERS = {"geo_sensitivity": "GeoSensitivity", "feature_local": "Local features", "top50_rank": "Top 50"}

STYLE = """
<style>
[data-testid="stHeader"] {display: none;}
.block-container {max-width: 1640px; padding: 0 1.4rem 2rem 1.4rem;}
[data-testid="stSidebar"] {background: #1F2A44;}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {color: #F2F4F8;}
[data-testid="stSidebar"] [data-baseweb="select"] > div {background: #2B3858; border-color: #4A5778;}
[data-testid="stSidebar"] [role="radiogroup"] {gap: .25rem;}
[data-testid="stSidebar"] [role="radiogroup"] label {padding: .5rem .7rem; border-radius: 8px; width: 100%;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background: #4F4FA8;}
[data-testid="stSidebar"] [data-testid="stRadioOption"] div:has(> [data-testid="stMarkdownContainer"]) > div:first-child:not([data-testid]) {display: none !important;}
[data-testid="stSidebar"] summary, [data-testid="stSidebar"] summary:hover {background: #2B3858 !important;}
[data-testid="stSidebar"] summary p, [data-testid="stSidebar"] summary svg {color: #F2F4F8 !important; fill: #F2F4F8;}
[data-testid="stSidebar"] button {background: #2B3858; border-color: #4A5778; color: #F2F4F8;}
[data-testid="stSidebar"] [data-testid="stExpander"] details {border-color: #4A5778;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color: #C9D0DE;}
[data-testid="stSidebar"] [data-baseweb="select"] span, [data-testid="stSidebar"] [data-baseweb="select"] div {color: #12171C;}
[data-testid="stMainBlockContainer"] {padding-top: 0 !important;}
.gs-top {background: #1F2A44; color: #fff; margin: -1.1rem -1.4rem .9rem -1.4rem; padding: .85rem 1.4rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: .5rem;}
.gs-top h1 {color: #fff; font-size: 1.55rem; font-weight: 700; margin: 0; padding: 0;}
.gs-top p {color: #D5DAE5; margin: .1rem 0 0 0; font-size: .92rem;}
.gs-pill {border: 1px solid #5A6788; background: #2B3858; border-radius: 999px; padding: .3rem .85rem; font-size: .85rem; margin-left: .4rem; color: #fff;}
.gs-kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: .7rem; margin-bottom: .9rem;}
.gs-kpi {border: 1px solid #DDE1E8; border-radius: 10px; padding: .55rem .9rem; background: #F8F9FB;}
.gs-kpi span {font-size: .8rem; color: #57646D; display: block;}
.gs-kpi b {font-size: 1.28rem; color: #12171C;}
.gs-area {font-size: 1.3rem; font-weight: 700; color: #12171C;}
.gs-area span {font-size: .9rem; font-weight: 400; color: #57646D;}
.gs-card-row {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .5rem; margin-bottom: .6rem;}
.gs-card {border: 1px solid #DDE1E8; border-radius: 10px; padding: .6rem .75rem; background: #fff; height: 100%;}
.gs-card-title {font-weight: 650; font-size: .86rem; color: #12171C;}
.gs-card-value {font-weight: 700; font-size: 1.22rem; color: #12171C; line-height: 1.35;}
.gs-card-note {font-size: .74rem; color: #57646D; line-height: 1.3;}
.gs-chip {background: #ECEBFA; color: #3B37A8; border-radius: 5px; font-size: .72rem; padding: .08rem .35rem; font-weight: 700; white-space: nowrap;}
.gs-bar-row {display: grid; grid-template-columns: minmax(0, 1.5fr) 1fr 2.4rem; gap: .35rem; align-items: center; font-size: .78rem; margin-top: .3rem;}
.gs-bar {background: #EEF0F5; border-radius: 3px; height: .6rem;}
.gs-bar div {background: #7C78DA; height: 100%; border-radius: 3px;}
.gs-insight {background: #F0EFFB; border: 1px solid #DCDAF5; border-radius: 10px; padding: .6rem .85rem; margin: .55rem 0; font-size: .86rem;}
.gs-insight ul {margin: .25rem 0 0 0; padding-left: 1.1rem;}
.gs-insight li {margin-bottom: .1rem;}
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
[class*="st-key-gs_features"] button {justify-content: flex-start; text-align: left; min-height: 1.9rem; padding: 0.1rem 0.4rem;}
[class*="st-key-gs_features"] button p {font-size: 0.88rem;}
[class*="st-key-gs_features"] [data-testid="stHorizontalBlock"] {gap: 0.25rem;}
[class*="st-key-gs_features"] [data-testid="stVerticalBlock"] {gap: 0.1rem;}
.gs-lede {font-size: 1rem; color: #57646D; margin-bottom: 0.8rem;}
</style>
"""


def _live_briefings() -> bool:
    """Live LLM briefings need an API key; the public deployment has none, so viewers cannot spend credit."""
    return bool(_openai_secret("OPENAI_API_KEY") or _anthropic_secret("ANTHROPIC_API_KEY"))


@st.cache_resource(show_spinner="Loading Hull Geo-SMILE results…")
def _bundle():
    return load_hull_bundle(ROOT)


def _header(bundle) -> None:
    ev = compact_evaluation(bundle.evaluation)
    st.markdown(
        "<div class='gs-top'><div><h1>Geo-SMILE | Hull Urban Safety</h1>"
        "<p>Crime prediction · Geographic explainability · Stakeholder decision support</p></div>"
        "<div><span class='gs-pill'>Web GIS (Leaflet)</span><span class='gs-pill'>Hull, UK</span></div></div>",
        unsafe_allow_html=True)
    kpis = [("Hull LSOAs", f"{len(bundle.lsoa)}"), ("Prediction horizon", "Next month"),
            ("Model", f"XGBoost · R² {ev['feature_holdout_r2']:.3f}"),
            ("Explanation", f"Geo-SMILE · {(ev.get('geo_design') or {}).get('n_perturbations', 0):,} perturbations")]
    cells = "".join(f"<div class='gs-kpi'><span>{k}</span><b>{v}</b></div>" for k, v in kpis)
    st.markdown(f"<div class='gs-kpis'>{cells}</div>", unsafe_allow_html=True)


def _page_card(bundle, role: str) -> None:
    render_role_banner(role)
    length = st.segmented_control("Length", ["Short card", "Detailed narrative"], default="Short card",
                                  key="narrative_length", label_visibility="collapsed")
    area_card = role in cm.AREA_ROLES
    picker, chooser = st.columns([1, 2.4], gap="medium")
    with picker:
        selected = _lsoa_picker(bundle)
        st.caption("Choose an area here or click it on the map. Map: GeoSensitivity; the chosen area is outlined.")
        if not area_card:
            st.caption("This role's evidence is city-level, so its card is Hull-wide. Area cards are written for the "
                       "Resident, Planner and Police roles.")
    with chooser:
        clicked = render_map(bundle, "geo_sensitivity", selected, show_labels=False, key="card_area_map", height=430)
    if clicked and clicked != selected:
        st.session_state["selected_lsoa"] = clicked
        st.rerun()
    if length == "Detailed narrative":
        if area_card and selected and _openai_secret("OPENAI_API_KEY"):
            render_area_detailed(bundle, role, selected)
            return
        if area_card and selected:
            st.info("Area narratives are unavailable here because no API key is configured. Showing the Hull-wide narrative.")
        elif area_card:
            st.caption("Choose an area to generate its detailed narrative. Showing the archived Hull-wide narrative.")
        render_detailed(role)
        return
    if role not in cm.AREA_ROLES:
        render_card(role)
        return
    if selected and _openai_secret("OPENAI_API_KEY"):
        render_area_card(bundle, role, selected)
        return
    if selected:
        st.info("Area cards are unavailable here because no API key is configured. Showing the Hull-wide card.")
    else:
        st.caption("Choose an area to generate its card. Showing the archived Hull-wide card.")
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


def _layer_picker() -> str:
    st.session_state.setdefault("map_view", "geo_sensitivity")
    if st.session_state["map_view"] not in LAYERS:
        st.session_state["map_view"] = "geo_sensitivity"
    choice = st.segmented_control("Map layer", list(LAYERS), format_func=LAYERS.get, key="map_view", label_visibility="collapsed")
    if choice is None:
        choice = st.session_state.get("map_view_last", "geo_sensitivity")
    st.session_state["map_view_last"] = choice
    return choice


def _page_map(bundle, role: str) -> None:
    table = decision_table(bundle)
    mapper, panel = st.columns([1.15, 1], gap="medium")
    with mapper:
        map_view = _layer_picker()
        map_slot = st.container()
    bar = st.columns([1.4, 1.4, 1.1, 1, 1], gap="small", vertical_alignment="bottom")
    month = feature = None
    with bar[0]:
        if map_view == "crime_prediction":
            month = st.selectbox("Time (prediction)", prediction_months(), key="map_month")
        elif map_view == "feature_local":
            st.session_state.setdefault("map_feature", "crime_rolling_mean_6")
            feature = st.selectbox("Feature", sorted(bundle.feature_names, key=human_label), format_func=human_label, key="map_feature")
    with bar[1]:
        selected = _lsoa_picker(bundle)
    opacity = bar[2].slider("Layer opacity", 0.2, 1.0, 0.8, 0.05, key="map_opacity")
    show_labels = bar[3].toggle("Show area labels", True, key="map_labels")
    with st.sidebar:
        matches = render_filters(table)
    with bar[4], st.popover("Export", width="stretch"):
        export_buttons(bundle, table, matches, "bar")
    code = selected or str(table.index[0])
    with map_slot:
        if map_view in ("latitude_effect", "longitude_effect", "overlay", "top50_rank"):
            st.caption("Coordinate-inclusive diagnostic run. Latitude and longitude effects are not GeoSensitivity.")
        elif map_view == "feature_local":
            st.caption("Pink raises and blue lowers the prediction relative to the masking reference. Model attribution, not causes of crime.")
        elif map_view == "crime_prediction":
            st.caption("Hold-out predictions of the model Geo-SMILE explains. For evaluation, not an operational forecast.")
        if matches is not None:
            st.caption(f"{len(matches)} areas meet the filters; the others are dimmed.")
        clicked = render_map(bundle, map_view, code, feature, opacity, matches, month, show_labels)
    if clicked and clicked != selected:
        st.session_state["selected_lsoa"] = clicked
        st.rerun()
    with panel:
        render_area_panel(bundle, table, code, role, ROLE_LABELS[role], month)
    if not selected:
        st.caption("Showing the highest-GeoSensitivity area until you choose one.")
    st.divider()
    render_shortlist(table, matches)
    if role in cm.AREA_ROLES and _live_briefings() and st.toggle("Detailed AI briefing for this area", key="map_briefing"):
        provider = st.radio("Briefing model", list(PROVIDERS), format_func=PROVIDERS.get, key="provider_choice", horizontal=True)
        render_explanation_panel(bundle, stakeholder_id=role, provider_choice=provider, scope="lsoa", current_map_view=map_view,
                                 selected_lsoa=code, force_mode="comprehensive")


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
- ATT Accuracy compares the highest-ranked players with one expert's reference. It does not validate causes.

#### Where the text comes from
The language model only communicates structured results. It does not see maps, compute statistics or find
patterns. Every figure on a card is checked against the evidence it was given. The stakeholder cards are
archived runs of GPT-4.1 at temperature 0.
"""
        )


def _sidebar() -> tuple[str, str]:
    with st.sidebar:
        role = st.selectbox("Select stakeholder view", list(STAKEHOLDERS), format_func=ROLE_LABELS.get, key="role")
        st.write("")
        page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
        st.divider()
        st.caption("Hull Geo-SMILE. XGBoost + Geo-SMILE, LSOA (2021), monthly prediction. Model attributions, not causes of "
                   "crime and not advice on where to deploy resources.")
    return role, page


def main() -> None:
    st.set_page_config(page_title="Geo-SMILE · Hull Urban Safety", page_icon="🗺️", layout="wide", initial_sidebar_state="expanded")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.session_state.setdefault("selected_lsoa", None)
    bundle = _bundle()
    role, page = _sidebar()
    _header(bundle)
    if page == "Map & Analysis":
        _page_map(bundle, role)
    elif page == "Stakeholder Cards":
        _page_card(bundle, role)
    elif page == "Area Comparison":
        render_compare(decision_table(bundle))
    elif page == "Role Comparison":
        render_comparison()
    elif page == "Spatial Patterns":
        render_attribution_patterns_section(bundle, "researcher")
    elif page == "Model & Explanations":
        render_diagnostics_section("researcher")
        render_global_features(bundle)
        render_evaluation(bundle)
    else:
        _page_about()


if __name__ == "__main__":
    main()
