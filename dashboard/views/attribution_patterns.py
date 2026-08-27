"""Local Attribution Patterns: new derived results combining GeoSensitivity
with per-LSOA local feature attribution.

Separate from the published GeoSMILE artefacts and from Model Diagnostics.
GeoSensitivity (Geo branch) and local feature attribution (Feature branch)
stay conceptually separate objects everywhere here: views either map one of
them alone, or compare their magnitudes side by side — never merge them into
one score.

Categorical colors follow the dataviz skill's validated 8-hue categorical
order (fixed order, not reassigned per filter); past 8 distinct categories
the long tail folds into a neutral "Other" so color never has to carry more
identities than it can reliably distinguish. Every fill still discloses its
exact value via hover text.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data.attribution_patterns import (
    attribution_archetypes,
    dominant_local_feature,
    local_attribution_concentration,
    top_categories_with_other,
)
from dashboard.data.loader import HullBundle
from dashboard.data.schema import LSOA_ID, LSOA_NAME, human_label
from dashboard.data.spatial_stats import build_queen_weights, local_moran
from dashboard.views.gating import persona_gated
from dashboard.views.maps import DIVERGING, LISA_COLOURS

GLOBAL_TOP_N = 10

CATEGORICAL_SLOTS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
OTHER_COLOR = "#B7C0C7"
SEQUENTIAL_BLUE = [
    [0.0, "#cde2fb"],
    [0.2, "#9ec5f4"],
    [0.4, "#6da7ec"],
    [0.6, "#3987e5"],
    [0.8, "#256abf"],
    [1.0, "#0d366b"],
]

MAP_CENTER = {"lat": 53.767, "lon": -0.327}
DEFAULT_PERSONAS = {"technical", "domain", "governance"}


@st.cache_data(show_spinner=False)
def _cached_dominant(local_features: pd.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame:
    return dominant_local_feature(local_features, list(feature_names))


@st.cache_data(show_spinner=False)
def _cached_concentration(local_features: pd.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame:
    return local_attribution_concentration(local_features, list(feature_names))


@st.cache_data(show_spinner="Clustering explanation archetypes…")
def _cached_archetypes(
    local_features: pd.DataFrame, feature_names: tuple[str, ...], k: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return attribution_archetypes(local_features, list(feature_names), k=k)


@st.cache_resource(show_spinner=False)
def _cached_weights(_geometry: pd.DataFrame):
    return build_queen_weights(_geometry, LSOA_ID)


@st.cache_data(show_spinner=False)
def _cached_local_moran(_w, islands: tuple[str, ...], values: dict):
    return local_moran(values, _w)


def _global_top_features(bundle: HullBundle) -> pd.DataFrame:
    g = bundle.global_features.copy()
    if "abs_coefficient" not in g.columns:
        g["abs_coefficient"] = g["coefficient"].abs()
    return g.sort_values("abs_coefficient", ascending=False, kind="mergesort").head(GLOBAL_TOP_N).reset_index(
        drop=True
    )


def _geojson(gdf: pd.DataFrame) -> dict:
    return json.loads(gdf.to_json())


def _categorical_color_map(categories: list[str]) -> dict[str, str]:
    colors: dict[str, str] = {}
    slot = 0
    for cat in categories:
        if cat == "Other":
            colors[cat] = OTHER_COLOR
            continue
        colors[cat] = CATEGORICAL_SLOTS[slot % len(CATEGORICAL_SLOTS)]
        slot += 1
    return colors


def _swatch_legend_html(colors: dict[str, str]) -> str:
    """Explicit HTML colour-guide, since a Plotly legend on a mapbox figure is
    easy to miss (small, low-contrast, can sit under the map body)."""
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;margin:2px 14px 2px 0;'
        f'font-size:0.85rem;color:inherit;">'
        f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;'
        f'background:{color};margin-right:5px;border:1px solid rgba(120,120,120,0.35);">'
        f"</span>{label}</span>"
        for label, color in colors.items()
    )
    return f'<div style="margin:2px 0 10px 0;">{items}</div>'


def build_categorical_choropleth(
    gdf: pd.DataFrame, *, value_col: str, hover_col: str
) -> tuple[go.Figure, dict[str, str]]:
    fig = go.Figure()
    categories = list(gdf[value_col].value_counts().index)
    if "Other" in categories:
        categories = [c for c in categories if c != "Other"] + ["Other"]
    colors = _categorical_color_map(categories)
    for cat in categories:
        sub = gdf[gdf[value_col] == cat]
        if sub.empty:
            continue
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=_geojson(sub[[LSOA_ID, "geometry"]]),
                locations=sub[LSOA_ID],
                z=[1] * len(sub),
                featureidkey="properties.lsoa21cd",
                colorscale=[[0, colors[cat]], [1, colors[cat]]],
                showscale=False,
                name=str(cat),
                hovertext=sub[hover_col],
                hoverinfo="text",
                marker_line_width=0.35,
                marker_line_color="white",
            )
        )
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=10.6,
        mapbox_center=MAP_CENTER,
        margin=dict(l=0, r=0, t=8, b=0),
        height=560,
        showlegend=False,
        paper_bgcolor="white",
    )
    return fig, colors


def build_sequential_choropleth(
    gdf: pd.DataFrame, *, value_col: str, hover_col: str, legend_title: str
) -> go.Figure:
    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=_geojson(gdf[[LSOA_ID, "geometry"]]),
            locations=gdf[LSOA_ID],
            z=gdf[value_col].to_numpy(dtype=float),
            featureidkey="properties.lsoa21cd",
            zmin=0.0,
            zmax=1.0,
            colorscale=SEQUENTIAL_BLUE,
            colorbar=dict(title=legend_title, thickness=12, len=0.72),
            hovertext=gdf[hover_col],
            hoverinfo="text",
            marker_line_width=0.35,
            marker_line_color="white",
        )
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=10.6,
        mapbox_center=MAP_CENTER,
        margin=dict(l=0, r=0, t=8, b=0),
        height=560,
        paper_bgcolor="white",
    )
    return fig


def _limit(values) -> float:
    import numpy as np

    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return max(float(np.percentile(np.abs(v), 98)), 1e-9)


def build_diverging_choropleth(gdf: pd.DataFrame, *, value_col: str, hover_col: str, legend_title: str) -> go.Figure:
    z = gdf[value_col].to_numpy(dtype=float)
    vmax = _limit(z)
    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=_geojson(gdf[[LSOA_ID, "geometry"]]),
            locations=gdf[LSOA_ID],
            z=z,
            featureidkey="properties.lsoa21cd",
            zmid=0.0,
            zmin=-vmax,
            zmax=vmax,
            colorscale=DIVERGING,
            colorbar=dict(title=legend_title, thickness=12, len=0.72),
            hovertext=gdf[hover_col],
            hoverinfo="text",
            marker_line_width=0.35,
            marker_line_color="white",
        )
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=10.6,
        mapbox_center=MAP_CENTER,
        margin=dict(l=0, r=0, t=8, b=0),
        height=480,
        paper_bgcolor="white",
    )
    return fig


def build_lisa_choropleth(gdf: pd.DataFrame, *, hover_col: str) -> tuple[go.Figure, dict[str, str]]:
    fig = go.Figure()
    present = [lab for lab in LISA_COLOURS if lab in set(gdf["lisa"])]
    colors = {lab: LISA_COLOURS[lab] for lab in present}
    for label, colour in colors.items():
        sub = gdf[gdf["lisa"] == label]
        if sub.empty:
            continue
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=_geojson(sub[[LSOA_ID, "geometry"]]),
                locations=sub[LSOA_ID],
                z=[1] * len(sub),
                featureidkey="properties.lsoa21cd",
                colorscale=[[0, colour], [1, colour]],
                showscale=False,
                name=label,
                hovertext=sub[hover_col],
                hoverinfo="text",
                marker_line_width=0.35,
                marker_line_color="white",
            )
        )
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=10.6,
        mapbox_center=MAP_CENTER,
        margin=dict(l=0, r=0, t=8, b=0),
        height=480,
        showlegend=False,
        paper_bgcolor="white",
    )
    return fig, colors


def build_geo_vs_feature_scatter(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=frame["geo_sensitivity_abs"],
            y=frame["dominant_abs_value"],
            mode="markers",
            marker=dict(size=9, color="#2a78d6", line=dict(width=1, color="white"), opacity=0.85),
            customdata=list(
                zip(frame[LSOA_NAME], frame["dominant_feature_label"], frame["geo_rank"])
            ),
            hovertemplate=(
                "%{customdata[0]}<br>|GeoSensitivity|: %{x:.4f} (rank %{customdata[2]})<br>"
                "dominant local feature: %{customdata[1]}<br>|coefficient|: %{y:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis_title="|GeoSensitivity| (Geo branch)",
        yaxis_title="|dominant local feature coefficient| (Feature branch)",
        height=460,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _render_dominant_feature_map(gdf: pd.DataFrame) -> None:
    st.markdown("**Dominant local feature**")
    st.caption(
        "For each LSOA, the non-location feature with the largest |local SMILE "
        "coefficient|. The most frequent features get distinct colors; the rest "
        "are grouped as “Other” — hover always names the exact feature."
    )
    work = gdf.copy()
    work["category"] = top_categories_with_other(work["dominant_feature_label"], max_categories=8)
    work["hover"] = (
        work[LSOA_NAME]
        + "<br>Dominant local feature: "
        + work["dominant_feature_label"]
        + "<br>signed coefficient: "
        + work["dominant_signed_value"].round(4).astype(str)
    )
    fig, colors = build_categorical_choropleth(work, value_col="category", hover_col="hover")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(_swatch_legend_html(colors), unsafe_allow_html=True)

    counts = (
        gdf["dominant_feature_label"]
        .value_counts()
        .rename_axis("Feature")
        .reset_index(name="LSOAs where it dominates")
    )
    st.dataframe(counts, hide_index=True, use_container_width=True)


def _render_scatter(gdf: pd.DataFrame) -> None:
    st.markdown("**GeoSensitivity vs. feature magnitude**")
    st.caption(
        "Each point is one LSOA: |GeoSensitivity| (Geo branch) against the "
        "magnitude of its dominant local feature (Feature branch). A side-by-side "
        "magnitude comparison, not a merged score — the two branches stay "
        "conceptually separate."
    )
    fig = build_geo_vs_feature_scatter(gdf)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    corr = gdf[["geo_sensitivity_abs", "dominant_abs_value"]].corr().iloc[0, 1]
    st.caption(
        f"Pearson correlation of the two magnitudes across all {len(gdf)} LSOAs: {corr:.3f}. "
        "Not causal; describes whether geography-driven and feature-driven areas overlap."
    )


def _render_archetypes(bundle: HullBundle, gdf: pd.DataFrame) -> None:
    st.markdown("**Explanation archetypes**")
    st.caption(
        "K-means clusters on each LSOA's standardized local feature coefficient "
        "vector (30 non-location features), grouping areas whose explanations "
        "share a similar feature profile — independent of geographic adjacency "
        "(distinct from the LISA clustering of |GeoSensitivity| elsewhere in the app)."
    )
    k = st.slider("Number of archetypes (k)", min_value=2, max_value=8, value=4, key="attr_archetype_k")
    feature_names = tuple(bundle.feature_names)
    labels_df, profiles = _cached_archetypes(bundle.local_features, feature_names, k)
    work = gdf.merge(labels_df, on=LSOA_ID, how="left")
    label_map = dict(zip(profiles["cluster"], profiles["label"]))
    work["archetype_label"] = work["cluster"].map(label_map)
    work["hover"] = work[LSOA_NAME] + "<br>Archetype: " + work["archetype_label"].astype(str)
    fig, colors = build_categorical_choropleth(work, value_col="archetype_label", hover_col="hover")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(_swatch_legend_html(colors), unsafe_allow_html=True)

    display = profiles[["cluster", "n_lsoas", "label"]].rename(
        columns={
            "cluster": "Archetype",
            "n_lsoas": "LSOAs",
            "label": "Top driving features (by mean coefficient)",
        }
    )
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.caption(
        "Unsupervised and re-fit on every k change; cluster numbering is not "
        "stable across different k."
    )


def _render_concentration_map(gdf: pd.DataFrame) -> None:
    st.markdown("**Explanation concentration**")
    st.caption(
        "Normalized entropy of |coefficient| shares across the 30 non-location "
        "features per LSOA. Near 1 = many features contribute roughly equally "
        "(diffuse explanation); near 0 = one or two features dominate "
        "(concentrated explanation)."
    )
    work = gdf.copy()
    work["hover"] = (
        work[LSOA_NAME]
        + "<br>Concentration (entropy, 0–1): "
        + work["entropy_norm"].round(3).astype(str)
        + "<br>Effective number of features: "
        + work["effective_n_features"].round(1).astype(str)
    )
    fig = build_sequential_choropleth(
        work, value_col="entropy_norm", hover_col="hover", legend_title="Concentration (entropy)"
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_dominant_global_features(bundle: HullBundle, gdf: pd.DataFrame) -> None:
    st.markdown("**Dominant global features**")
    st.caption(
        "The top globally important Feature-branch drivers (by |global coefficient|), "
        "each mapped by its own local coefficient across Hull, then tested for spatial "
        "clustering — the same method used for GeoSensitivity elsewhere, applied here to "
        "a feature's attribution instead."
    )
    st.caption(
        "★ **Moran's I** — global spatial-autocorrelation statistic (roughly −1 to +1). "
        "Positive = the feature's local coefficients cluster geographically (similar "
        "values sit near each other); near 0 = no spatial pattern; negative = a "
        "checkerboard-like dispersion.  \n"
        "★ **LISA** (Local Indicators of Spatial Association) — classifies each LSOA by "
        "its relation to its neighbours: **HH** = high surrounded by high (hot cluster), "
        "**LL** = low surrounded by low (cold cluster), **HL**/**LH** = a local outlier "
        "surrounded by the opposite, **NS** = not significant at p<0.05."
    )
    top = _global_top_features(bundle)
    w, islands = _cached_weights(bundle.geometry)

    rows = []
    results: dict[str, tuple[pd.DataFrame, float, float]] = {}
    for _, row in top.iterrows():
        feat = row["feature"]
        values = dict(zip(bundle.local_features[LSOA_ID], bundle.local_features[feat]))
        result, g_i, g_p = _cached_local_moran(w, tuple(islands), values)
        results[feat] = (result, g_i, g_p)
        counts = result["lisa"].value_counts()
        rows.append(
            {
                "Feature": human_label(feat),
                "Moran's I": round(g_i, 3),
                "p-value": round(g_p, 3),
                "Clustered (p<0.05)": "yes" if g_p < 0.05 else "no",
                "HH": int(counts.get("HH", 0)),
                "LL": int(counts.get("LL", 0)),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    labels = top["feature"].map(human_label).tolist()
    chosen_label = st.selectbox("Feature to map", labels, key="attr_global_feat")
    chosen = top.loc[labels.index(chosen_label), "feature"]
    result, g_i, g_p = results[chosen]
    work = gdf.merge(result, on=LSOA_ID, how="left")
    work["hover_val"] = (
        work[LSOA_NAME] + "<br>local coefficient: " + work["value"].round(4).astype(str)
    )
    work["hover_lisa"] = (
        work[LSOA_NAME]
        + "<br>LISA: "
        + work["lisa"].astype(str)
        + "<br>local I: "
        + work["local_I"].round(3).astype(str)
    )
    left, right = st.columns(2)
    with left:
        st.markdown(f"**{chosen_label} — local coefficient**")
        fig = build_diverging_choropleth(
            work, value_col="value", hover_col="hover_val", legend_title="Local coefficient"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown(f"**{chosen_label} — LISA clusters**")
        fig2, lisa_colors = build_lisa_choropleth(work, hover_col="hover_lisa")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown(_swatch_legend_html(lisa_colors), unsafe_allow_html=True)
    st.caption(
        f"Global Moran's I = {g_i:.3f} (p={g_p:.3f}), "
        f"{'significant clustering' if g_p < 0.05 else 'not significant'} at p<0.05. "
        "Queen contiguity, 249 permutations, seed 42. Not causal."
    )


def render_attribution_patterns_section(bundle: HullBundle, stakeholder_id: str) -> None:
    show = persona_gated(
        stakeholder_id,
        default_personas=DEFAULT_PERSONAS,
        toggle_label="Show local attribution patterns",
        toggle_key="show_attribution_patterns_toggle",
        help_text=(
            "New derived views combining GeoSensitivity with per-LSOA local "
            "feature attribution: dominant local feature, magnitude comparison, "
            "explanation archetypes, explanation concentration, and dominant "
            "global features with spatial clustering. Shown by default for "
            "Technical, Domain, and Governance personas."
        ),
    )
    if not show:
        return

    st.divider()
    st.subheader("Local Attribution Patterns")
    st.caption(
        "New derived results, not published GeoSMILE artefacts on their own. "
        "GeoSensitivity and local feature attribution stay conceptually separate "
        "objects; the views below compare or aggregate their magnitudes, they "
        "never merge them into one score."
    )

    feature_names = tuple(bundle.feature_names)
    dominant = _cached_dominant(bundle.local_features, feature_names)
    concentration = _cached_concentration(bundle.local_features, feature_names)
    gdf = bundle.geometry.merge(dominant, on=LSOA_ID, how="left").merge(
        concentration, on=LSOA_ID, how="left"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Dominant local feature",
            "GeoSensitivity vs. feature magnitude",
            "Explanation archetypes",
            "Explanation concentration",
            "Dominant global features",
        ]
    )
    with tab1:
        _render_dominant_feature_map(gdf)
    with tab2:
        _render_scatter(gdf)
    with tab3:
        _render_archetypes(bundle, gdf)
    with tab4:
        _render_concentration_map(gdf)
    with tab5:
        _render_dominant_global_features(bundle, gdf)
