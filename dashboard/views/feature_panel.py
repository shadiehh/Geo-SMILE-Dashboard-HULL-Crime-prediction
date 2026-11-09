"""Selected-LSOA evidence panel and top-10 non-location features."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data.loader import HullBundle, compact_evaluation, magnitude_class, top_features_for_lsoa
from dashboard.data.schema import LISA_LABELS, LSOA_ID, LSOA_NAME, human_label


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "not available"
    if isinstance(value, (int,)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _show_feature(feature: str) -> None:
    st.session_state["map_feature"] = feature
    st.session_state["map_view"] = "feature_local"


def _feature_rows(rows: list[tuple[str, str, str, float | None]], prefix: str) -> None:
    """One clickable row per feature: name, and its value (pink raises, blue lowers, grey undirected)."""
    active = st.session_state.get("map_feature") if st.session_state.get("map_view") == "feature_local" else None
    with st.container(key=f"gs_features_{prefix}"):
        for feature, label, value, signed in rows:
            name, number = st.columns([3.4, 1], gap="small", vertical_alignment="center")
            name.button(label, key=f"feat_{prefix}_{feature}", on_click=_show_feature, args=(feature,),
                        type="primary" if feature == active else "tertiary", width="stretch")
            colour = "#57646D" if signed is None else ("#C2003F" if signed >= 0 else "#1565C0")
            number.markdown(f"<div style='text-align:right; font-size:0.85rem; color:{colour}; "
                            f"font-variant-numeric:tabular-nums;'>{value}</div>", unsafe_allow_html=True)


def render_feature_panel(bundle: HullBundle, selected_lsoa: str | None) -> None:
    if not selected_lsoa:
        st.subheader("Features across Hull")
        st.caption("Ranked by global importance. Click a feature to map its local effect. "
                   "Click an area on the map for that area's own ranking.")
        frame = bundle.global_features.sort_values("abs_coefficient", ascending=False, kind="mergesort").head(10)
        _feature_rows([(r["feature"], human_label(r["feature"]), f"{r['abs_coefficient']:.2f}", None)
                       for _, r in frame.iterrows()], "global")
        return

    st.subheader("Selected Area")

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

    st.markdown("**Top 10 features for this area**")
    st.caption(
        "Click a feature to map it. Pink raises and blue lowers this area's prediction relative to the "
        "masking reference. Model attribution, not causes of crime."
    )
    features = top_features_for_lsoa(bundle.local_features, selected_lsoa, k=10)
    if not features:
        st.error(
            "Per-LSOA feature results are missing for this area. "
            "Global importance is not used as a substitute."
        )
        return

    _feature_rows([(r["feature"], r["feature_label"], f"{r['feature_value']:+.2f}", r["feature_value"])
                   for r in features], "local")
    if features and features[0]["abs_value"] < 0.25:
        st.caption(
            f"The highest-ranked local feature is {features[0]['feature_label']}, "
            "but its absolute magnitude is small. Rank is not the same as strength."
        )


def render_global_features(bundle: HullBundle) -> None:
    with st.expander("Overall feature importance (Feature branch)"):
        st.caption(
            "Global ranking of the 30 Feature players (Chapter 4, N = 4,000), by coefficient magnitude; "
            "the sign is not a direction of prediction change. "
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


def render_evaluation(bundle: HullBundle) -> None:
    ev = compact_evaluation(bundle.evaluation)
    consist = ev.get("consistency") or {}
    with st.expander("Explanation evaluation (includes Consistency)"):
        st.caption(
            "Published GeoSMILE evaluation. Consistency is multi-run seed "
            "agreement. ATT stability is a separate controlled-input test. "
            "Not causal."
        )
        feat = consist.get("feature") or {}
        geo = consist.get("geo") or {}
        st.markdown("**Consistency** (five independently seeded runs)")
        st.markdown(
            f"""
| Branch | Rank Spearman | Score Pearson | Top-20% Jaccard |
|---|---:|---:|---:|
| Feature | {_fmt(feat.get('rank_spearman_mean'), 3)} | {_fmt(feat.get('score_pearson_mean'), 3)} | {_fmt(feat.get('top20pct_jaccard_mean'), 3)} |
| Geo | {_fmt(geo.get('rank_spearman_mean'), 3)} | {_fmt(geo.get('score_pearson_mean'), 3)} | {_fmt(geo.get('top20pct_jaccard_mean'), 3)} |
"""
        )
        st.markdown("**ATT stability** (controlled input; not Consistency)")
        st.markdown(
            f"Feature Jaccard `{_fmt(ev.get('att_stability_feature_jaccard'), 3)}` · "
            f"Geo Jaccard `{_fmt(ev.get('att_stability_geo_jaccard'), 3)}`"
        )
        st.markdown("**Intervention impact** (top minus random)")
        st.markdown(
            f"Feature `{_fmt(ev.get('feature_intervention_top_minus_random'))}` "
            f"(p `{_fmt(ev.get('feature_intervention_empirical_p'), 3)}`) · "
            f"Geo `{_fmt(ev.get('geo_intervention_top_minus_random'))}` "
            f"(p `{_fmt(ev.get('geo_intervention_empirical_p'), 3)}`)"
        )
