"""Streamlit wrapper for the stakeholder evidence figures (Chapter 6 two-column design).

Figures are built on demand for whichever LSOA is selected and cached per
(stakeholder, panel, LSOA, contrast), so switching areas or stakeholders is cheap.
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import streamlit as st

from dashboard.data.schema import STAKEHOLDERS
from dashboard.data.stakeholder_data import StakeholderData, load_stakeholder_data
from dashboard.views.stakeholder_figures import STAKEHOLDER_EMPHASIS, STAKEHOLDER_PANELS, is_path


@st.cache_resource(show_spinner="Loading stakeholder evidence…")
def stakeholder_data() -> StakeholderData:
    return load_stakeholder_data()


@st.cache_data(show_spinner=False, max_entries=512)
def _panel(stakeholder: str, index: int, lsoa: str, contrast: str) -> tuple[bytes | str, str, str]:
    d = stakeholder_data()
    obj, title, caption = STAKEHOLDER_PANELS[stakeholder](d, lsoa, contrast)[index]()
    if is_path(obj):
        return str(obj), title, caption
    buf = io.BytesIO()
    obj.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=obj.get_facecolor())
    plt.close(obj)
    return buf.getvalue(), title, caption


def focus_lsoa(d: StakeholderData, scope: str, selected: str | None) -> tuple[str, bool]:
    """The LSOA the figures describe; falls back to the rank-1 area for the Hull-wide scope."""
    if selected and selected in d.geo.index:
        return selected, False
    return str(d.geo.sort_values("geo_rank").index[0]), True


def contrast_selector(d: StakeholderData, stakeholder: str, lsoa: str) -> str:
    default = d.default_contrast(lsoa)
    if stakeholder != "planner":
        return default
    codes = [c for c in d.geo.sort_values("geo_rank").index if c != lsoa]
    labels = {c: f"{d.short(c)} — rank {int(d.geo.loc[c, 'geo_rank'])}"
                 + (f" ({d.place(c)})" if d.place(c) else "") for c in codes}
    return st.selectbox("Compare with (planner contrast)", codes, index=codes.index(default),
                        format_func=labels.get, key=f"contrast_{lsoa}",
                        help="Default: a low-sensitivity Low-Low cluster case for a high area, "
                             "or the rank-1 area for a low one.")


def stakeholder_panels(stakeholder: str, lsoa: str, contrast: str) -> list[tuple[bytes | str, str, str]]:
    n = len(STAKEHOLDER_PANELS[stakeholder](stakeholder_data(), lsoa, contrast))
    return [_panel(stakeholder, i, lsoa, contrast) for i in range(n)]


def figure_evidence(panels) -> list[dict]:
    """Caption facts sent to the LLM so the narrative and the figures agree."""
    return [{"title": t.rstrip("."), "caption": c} for _, t, c in panels]


def render_evidence_column(stakeholder: str, panels, *, fallback_note: str | None = None) -> None:
    st.markdown("<span style='font-size:0.75rem; font-weight:700; letter-spacing:0.04em;'>DASHBOARD EVIDENCE</span>",
                unsafe_allow_html=True)
    if fallback_note:
        st.caption(fallback_note)
    for image, title, caption in panels:
        st.image(image, width="stretch")
        st.markdown(f"<div style='font-size:0.78rem; opacity:0.7; margin:-0.4rem 0 0.9rem 0;'>"
                    f"<b>{title}</b> {caption}</div>", unsafe_allow_html=True)


def render_emphasis(stakeholder: str) -> None:
    st.markdown(f"<div style='font-size:0.78rem; border-top:1px solid rgba(128,128,128,0.35); padding-top:0.4rem;'>"
                f"<b>Stakeholder emphasis ({STAKEHOLDERS[stakeholder]}).</b> {STAKEHOLDER_EMPHASIS[stakeholder]}"
                f"</div>", unsafe_allow_html=True)
