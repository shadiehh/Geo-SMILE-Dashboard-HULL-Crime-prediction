"""The right-hand panel of the Web GIS: one area's evidence, in four views.

Every value comes from the published tables through decision_table. Wording stays with what they measure:
predictions are model outputs, GeoSensitivity's sign is not a direction of crime, and a spatial cluster of
GeoSensitivity is not a crime hotspot.
"""

from __future__ import annotations

import html
import io

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap, Normalize

from dashboard.data.loader import HullBundle
from dashboard.data.schema import LISA_LABELS, human_label
from dashboard.llm import card_mode as cm
from dashboard.views.decision_support import open_card
from dashboard.views.feature_panel import render_feature_panel
from dashboard.views.maps import predictions_for, load_predictions
from dashboard.views.stakeholder_evidence import stakeholder_data

SENSITIVITY = LinearSegmentedColormap.from_list("sensitivity", ["#FFFFFF", "#FF0051", "#8A0033"])
TABS = ("Overview", "Explanations", "Neighbourhood", "Trends")


def _step(table: pd.DataFrame, code: str, by: int) -> None:
    codes = list(table.index)
    st.session_state["selected_lsoa"] = codes[(codes.index(code) + by) % len(codes)]


def _card(title: str, value: str, badge: str, note: str) -> str:
    chip = f"<span class='gs-chip'>{html.escape(badge)}</span>" if badge else ""
    return (f"<div class='gs-card'><div class='gs-card-title'>{html.escape(title)}</div>"
            f"<div class='gs-card-value'>{html.escape(value)} {chip}</div>"
            f"<div class='gs-card-note'>{html.escape(note)}</div></div>")


def _feature_bars(code: str) -> str:
    local = stakeholder_data().abs_local(code)
    shares = (local / local.sum()).head(5)
    top = float(shares.iloc[0])
    rows = "".join(
        f"<div class='gs-bar-row'><span>{html.escape(human_label(f))}</span>"
        f"<div class='gs-bar'><div style='width:{100 * v / top:.0f}%'></div></div><b>{100 * v:.0f}%</b></div>"
        for f, v in shares.items())
    return f"<div class='gs-card-title'>Top contributing features</div><div class='gs-card-note'>Feature branch, share of local importance</div>{rows}"


@st.cache_data(show_spinner=False, max_entries=200)
def _inset(code: str) -> bytes:
    """GeoSensitivity of the area and its surroundings, drawn from the LSOA values themselves."""
    d = stakeholder_data()
    g = d.geometry
    target = g[g["lsoa21cd"] == code]
    x0, y0, x1, y1 = target.total_bounds
    pad = max(x1 - x0, y1 - y0) * 0.7
    half_w, half_h = (x1 - x0) / 2 + pad, (y1 - y0) / 2 + pad
    half_w, half_h = max(half_w, half_h * 1.35), max(half_h, half_w / 1.35)  # a fixed 4:3 window
    mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
    norm = Normalize(float(g["geo_coefficient"].min()), float(g["geo_coefficient"].abs().quantile(0.98)))
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=170)
    g.plot(ax=ax, column="geo_coefficient", cmap=SENSITIVITY, norm=norm, edgecolor="#C9CED6", linewidth=0.4)
    target.boundary.plot(ax=ax, color="#3B37A8", linewidth=2)
    ax.set_xlim(mid_x - half_w, mid_x + half_w)
    ax.set_ylim(mid_y - half_h, mid_y + half_h)
    ax.set_axis_off()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return buffer.getvalue()


def _insights(table: pd.DataFrame, code: str, role: str, predicted: float, average: float) -> list[str]:
    row = table.loc[code]
    place = row["locality_label"] or row["area"]
    direction = "above" if predicted >= average else "below"
    lines = [f"The model predicts {predicted:.1f} crimes a month here, {direction} the Hull area average of {average:.1f}."]
    if role == "planner":
        lines.append(f"Built-environment features are {100 * row['built_share']:.0f}% of the local explanation, led by "
                     f"{row['dominant_built_feature']}.")
    elif role == "police":
        relation = cm._versus_median(row["recent_share"], table["recent_share"].median())
        lines.append(f"Recent months are {100 * row['recent_share']:.0f}% of the local explanation, {relation} the Hull median "
                     f"of {100 * table['recent_share'].median():.0f}%.")
        lines.append("Recorded change was not assessed: under five crimes a month." if pd.isna(row["change"]) else
                     f"Recorded crime changed by {100 * row['change']:+.0f}% in the last three months against the six before.")
    else:
        lines.append(f"{human_label(row['leading_feature'])} leads the local explanation, and crime history is "
                     f"{100 * row['crime_history_share']:.0f}% of it.")
    lines.append(f"{place} ranks {int(row['geo_rank'])} of {len(table)} for GeoSensitivity, how far predictions depend on "
                 "knowing the area's location.")
    lines.append("These describe what the model uses, not causes of crime or where to act.")
    return lines


def _overview(table: pd.DataFrame, code: str, role: str, role_label: str, month: str | None) -> None:
    predictions = predictions_for(month)
    predicted = float(predictions.loc[code, "predicted"])
    average = float(predictions["predicted"].mean())
    change = 100 * (predicted / average - 1)
    row = table.loc[code]
    rank = int(row["geo_rank"])
    cards = (_card("Predicted crime", f"{predicted:.1f}", f"{change:+.0f}%", f"vs. Hull area average ({average:.1f})")
             + _card("GeoSensitivity", f"{row['geo_sensitivity']:+.2f}", f"Rank {rank}/{len(table)}", "Sign is not a direction of crime")
             + _card("Spatial pattern", LISA_LABELS.get(row["lisa_class"], row["lisa_class"]), "",
                     "Clustering of GeoSensitivity, not a crime hotspot"))
    st.markdown(f"<div class='gs-card-row'>{cards}</div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="small")
    left.markdown(f"<div class='gs-card'>{_feature_bars(code)}</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='gs-card-title'>Geographic explanation</div>"
                    "<div class='gs-card-note'>Geo branch: this area outlined, darker is more location-sensitive</div>",
                    unsafe_allow_html=True)
        st.image(_inset(code), width="stretch")
    bullets = "".join(f"<li>{html.escape(line)}</li>" for line in _insights(table, code, role, predicted, average))
    st.markdown(f"<div class='gs-insight'><b>{html.escape(role_label)} insight</b><ul>{bullets}</ul></div>", unsafe_allow_html=True)
    st.button("Open this area's stakeholder card", on_click=open_card, args=(code,), width="stretch")


def _neighbourhood(table: pd.DataFrame, code: str) -> None:
    g = stakeholder_data().geometry.set_index("lsoa21cd")
    shape = g.loc[code, "geometry"]
    around = [str(c) for c in g.index[g.geometry.touches(shape)]]
    rows = table.loc[[c for c in around if c in table.index]].sort_values("geo_rank")
    st.caption(f"{len(rows)} areas share a boundary with this one.")
    st.dataframe(pd.DataFrame({
        "Area": rows["area"], "Locality": rows["locality_label"], "Predicted": rows["predicted_mean"].round(1),
        "Geo rank": rows["geo_rank"].astype(int), "Spatial pattern": rows["lisa_class"].map(lambda c: LISA_LABELS.get(c, c)),
    }), hide_index=True, width="stretch")


def _trends(code: str) -> None:
    crime = stakeholder_data().crime
    series = crime[crime["lsoa21cd"].astype(str) == code].assign(when=lambda f: pd.to_datetime(f["month"])).sort_values("when")
    held = load_predictions()
    held = held[held["lsoa21cd"] == code].assign(when=lambda f: pd.to_datetime(f["target_month"])).sort_values("when")
    fig = go.Figure()
    fig.add_scatter(x=series["when"], y=series["crime_count"], name="Recorded", mode="lines", line=dict(color="#57646D", width=1.8))
    fig.add_scatter(x=held["when"], y=held["predicted"], name="Predicted (hold-out)", mode="lines+markers",
                    line=dict(color="#4F46B8", width=2.4))
    fig.update_layout(height=340, margin=dict(l=8, r=8, t=8, b=8), legend=dict(orientation="h", y=1.1),
                      yaxis_title="Crimes a month", plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_area_panel(bundle: HullBundle, table: pd.DataFrame, code: str, role: str, role_label: str, month: str | None) -> None:
    row = table.loc[code]
    title, back, forward = st.columns([5, 0.7, 0.7], vertical_alignment="center")
    place = html.escape(str(row["locality_label"] or row["area"]))
    title.markdown(f"<div class='gs-area'>{place} <span>({html.escape(code)})</span></div>"
                   f"<div class='gs-card-note'>LSOA {html.escape(str(row['area']))} · Hull</div>", unsafe_allow_html=True)
    back.button("‹", key="area_prev", on_click=_step, args=(table, code, -1), width="stretch", help="Next higher GeoSensitivity")
    forward.button("›", key="area_next", on_click=_step, args=(table, code, 1), width="stretch", help="Next lower GeoSensitivity")
    tab = st.segmented_control("Area view", TABS, default=TABS[0], key="area_tab", label_visibility="collapsed") or TABS[0]
    if tab == "Explanations":
        render_feature_panel(bundle, code)
    elif tab == "Neighbourhood":
        _neighbourhood(table, code)
    elif tab == "Trends":
        _trends(code)
    else:
        _overview(table, code, role, role_label, month)
