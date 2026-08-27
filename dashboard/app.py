"""GeoSMILE — Hull crime prediction stakeholder-adaptive explanation dashboard.

Run from the hull folder:

    streamlit run dashboard/app.py

Uses published GeoSMILE artefacts only. Does not recompute GeoSMILE.
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

from dashboard.data.loader import load_hull_bundle
from dashboard.data.schema import LSOA_ID, LSOA_NAME, MAP_VIEWS, PROVIDERS
from dashboard.views.attribution_patterns import render_attribution_patterns_section
from dashboard.views.diagnostics import render_diagnostics_section
from dashboard.views.explanation_panel import render_explanation_panel, stakeholder_selector
from dashboard.views.feature_panel import (
    render_evaluation,
    render_feature_panel,
    render_global_features,
)
from dashboard.views.maps import render_map

st.set_page_config(
    page_title="GeoSMILE — Hull Crime Prediction",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading Hull GeoSMILE results…")
def _bundle():
    return load_hull_bundle(ROOT)


def _init_state() -> None:
    st.session_state.setdefault("selected_lsoa", None)
    st.session_state.setdefault("map_view", "geo_sensitivity")


def main() -> None:
    _init_state()
    bundle = _bundle()

    st.title("GeoSMILE — Hull Crime Prediction")
    st.caption("Stakeholder-adaptive spatial model explanation. Not causal.")

    with st.expander("How to read these views", expanded=False):
        st.markdown(
            """
**GeoSensitivity**
Measures how sensitive the Geo-branch explanation is to geographic context
(location-masking SMILE). This is not latitude or longitude attribution.

**Latitude / Longitude feature effects**
Show the contribution associated with coordinate features included in the
predictive Feature SMILE run. These are not GeoSensitivity.

**Lat / lon / GeoSensitivity overlay**
RGB composite of independently scaled absolute values: red =
|GeoSensitivity|, green = |latitude effect|, blue = |longitude effect|.
This is a visualisation only. It does not merge the three quantities.

**GeoSensitivity top 50 + lat/lon overlay**
The importance set is the top 50 |GeoSensitivity| LSOAs. Shade is
GeoSensitivity rank (darker = rank 1). Latitude and longitude overlay
only on that set. An LSOA that is top-50 lat/lon but not top-50
GeoSensitivity is not treated as important.

**Spatial pattern**
Tests whether |GeoSensitivity| values form spatial clusters or outliers
(Queen-contiguity LISA). A cluster is not the same as a large effect.

**Top features**
Shows which non-location variables contribute most strongly to the selected
LSOA's Feature SMILE explanation. Rank is not the same as strength.
"""
        )

    controls, mapper, evidence = st.columns([1.05, 2.35, 1.25], gap="medium")

    with controls:
        st.markdown("**Controls**")
        stakeholder_id = stakeholder_selector()
        map_view = st.radio(
            "Spatial explanation view",
            list(MAP_VIEWS.keys()),
            format_func=lambda k: MAP_VIEWS[k],
            key="map_view",
        )
        provider_choice = st.radio(
            "Explanation model",
            list(PROVIDERS.keys()),
            format_func=lambda k: PROVIDERS[k],
            key="provider_choice",
        )
        scope = st.radio(
            "Scope",
            ["overall", "lsoa"],
            format_func=lambda k: "Overall Hull" if k == "overall" else "Selected LSOA",
            key="scope",
        )
        labels = {
            str(code): f"{code} — {name}"
            for code, name in zip(
                bundle.lsoa[LSOA_ID].astype(str), bundle.lsoa[LSOA_NAME].astype(str)
            )
        }
        options = [""] + list(labels)
        current = st.session_state.selected_lsoa or ""
        picked = st.selectbox(
            "LSOA",
            options,
            index=options.index(current) if current in options else 0,
            format_func=lambda v: labels.get(v, "Select an LSOA"),
        )
        if picked and picked != st.session_state.selected_lsoa:
            st.session_state.selected_lsoa = picked
        elif not picked:
            pass
        st.caption("Changing the map view keeps the selected LSOA.")
        with st.expander("Data diagnostics"):
            st.json(bundle.diagnostics)

    with mapper:
        st.markdown(f"**{MAP_VIEWS[map_view]}**")
        if map_view == "overlay":
            st.caption(
                "Red = |GeoSensitivity|. Green = |latitude feature effect|. "
                "Blue = |longitude feature effect|. Each channel uses its own "
                "98th-percentile scale. Not a single merged effect. Not causal."
            )
        if map_view == "top50_rank":
            st.caption(
                "Importance = top 50 |GeoSensitivity|. Shade = that rank. "
                "Outline = also top-50 latitude and/or longitude. "
                "Lat/lon-only LSOAs stay grey. Not causal."
            )
        clicked = render_map(bundle, map_view, st.session_state.selected_lsoa)
        if clicked and clicked != st.session_state.selected_lsoa:
            st.session_state.selected_lsoa = clicked
            st.rerun()

    with evidence:
        render_feature_panel(bundle, st.session_state.selected_lsoa)
        render_global_features(bundle)
        render_evaluation(bundle)

    st.divider()
    render_explanation_panel(
        bundle,
        stakeholder_id=stakeholder_id,
        provider_choice=provider_choice,
        scope=scope,
        current_map_view=map_view,
        selected_lsoa=st.session_state.selected_lsoa,
    )

    render_diagnostics_section(stakeholder_id)
    render_attribution_patterns_section(bundle, stakeholder_id)


if __name__ == "__main__":
    main()
