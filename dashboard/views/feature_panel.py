"""Selected-LSOA evidence panel and top-10 non-location features."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data.loader import HullBundle, magnitude_class, top_features_for_lsoa
from dashboard.data.schema import LISA_LABELS, LSOA_ID, LSOA_NAME, human_label


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "not available"
    if isinstance(value, (int,)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def render_feature_panel(bundle: HullBundle, selected_lsoa: str | None) -> None:
    st.subheader("Selected Area")
    if not selected_lsoa:
        st.info("Click an LSOA on the map, or choose one in the list.")
        return

    match = bundle.lsoa[bundle.lsoa[LSOA_ID] == selected_lsoa]
    if match.empty:
        st.warning("That LSOA is not in the published results.")
        return
    row = match.iloc[0]
    median = float(bundle.lsoa["geo_sensitivity_abs"].median())
    p90 = float(bundle.lsoa["geo_sensitivity_abs"].quantile(0.9))
    mag = magnitude_class(float(row["geo_sensitivity_abs"]), median, p90)

    st.markdown(f"**{row[LSOA_NAME]}**")
    st.caption(str(row[LSOA_ID]))
    st.markdown(
        f"""
| Measure | Value |
|---|---|
| GeoSensitivity | `{_fmt(row['geo_sensitivity'])}` |
| Geo rank | `{int(row['geo_rank'])}` of {len(bundle.lsoa)} ({mag} magnitude) |
| Latitude feature effect | `{_fmt(row['latitude_effect'])}` |
| Longitude feature effect | `{_fmt(row['longitude_effect'])}` |
| Spatial-pattern class | {LISA_LABELS.get(str(row['lisa_class']), row['lisa_class'])} |
"""
    )
    if mag == "modest" and int(row["geo_rank"]) <= 20:
        st.caption(
            "High rank relative to Hull, but absolute GeoSensitivity is modest. "
            "Rank is comparative, not a claim of strong geographic drive."
        )

    st.markdown("**Top 10 features for this LSOA**")
    st.caption(
        "Non-location Feature SMILE local coefficients from the run that also "
        "attributes longitude and latitude. Location variables are excluded here. "
        "Not causal."
    )
    features = top_features_for_lsoa(bundle.local_features, selected_lsoa, k=10)
    if not features:
        st.error(
            "Per-LSOA feature results are missing for this area. "
            "Global importance is not used as a substitute."
        )
        return

    mode = st.radio("Display", ["Bar chart", "Ranked list"], horizontal=True, key="feat_display")
    if mode == "Ranked list":
        for rec in features:
            st.markdown(
                f"{rec['feature_rank']}. **{rec['feature_label']}** — "
                f"signed `{rec['feature_value']:.4f}` · "
                f"|coef| `{rec['abs_value']:.4f}`"
            )
        return

    labels = [r["feature_label"] for r in features][::-1]
    values = [r["feature_value"] for r in features][::-1]
    colors = ["#FF0051" if v >= 0 else "#1E88E5" for v in values]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>signed: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(280, 28 * len(features) + 80),
        margin=dict(l=10, r=10, t=8, b=24),
        xaxis_title="Signed local SMILE coefficient",
        yaxis=dict(automargin=True),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    if features and features[0]["abs_value"] < 0.25:
        st.caption(
            f"The highest-ranked local feature is {features[0]['feature_label']}, "
            "but its absolute magnitude is small. Rank is not the same as strength."
        )


def render_global_features(bundle: HullBundle) -> None:
    with st.expander("Overall model feature importance (Feature branch, location excluded)"):
        st.caption(
            "Global SMILE ranking of the published 30-player Feature branch. "
            "This is not the top-feature list for the selected LSOA, and it does "
            "not include latitude or longitude."
        )
        frame = bundle.global_features.copy()
        frame["label"] = frame["feature"].map(human_label)
        frame = frame.sort_values("abs_coefficient", ascending=False, kind="mergesort")
        st.dataframe(
            frame[["label", "coefficient", "abs_coefficient"]].rename(
                columns={
                    "label": "Feature",
                    "coefficient": "SMILE coefficient",
                    "abs_coefficient": "|coefficient|",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
