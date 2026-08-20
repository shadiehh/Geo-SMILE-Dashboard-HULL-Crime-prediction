"""Interactive Hull choropleths. Values come only from published GeoSMILE tables."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data.loader import HullBundle
from dashboard.data.schema import LISA_LABELS, LSOA_ID, LSOA_NAME, MAP_VIEWS

LISA_COLOURS = {
    "HH": "#C62828",
    "LL": "#1565C0",
    "HL": "#EF6C00",
    "LH": "#00838F",
    "NS": "#E8ECF0",
    "ISLAND": "#BDBDBD",
}

VIEW_FIELDS = {
    "geo_sensitivity": ("geo_sensitivity", "GeoSensitivity (signed)"),
    "latitude_effect": ("latitude_effect", "Latitude feature effect"),
    "longitude_effect": ("longitude_effect", "Longitude feature effect"),
    "spatial_pattern": ("lisa_class", "Spatial pattern (LISA of |GeoSensitivity|)"),
}

DIVERGING = [
    [0.0, "#1E88E5"],
    [0.5, "#FFFFFF"],
    [1.0, "#FF0051"],
]


def _geojson(gdf) -> dict:
    return json.loads(gdf.to_json())


def _limit(values: np.ndarray) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return max(float(np.percentile(np.abs(v), 98)), 1e-9)


def _hover(row: pd.Series, view: str) -> str:
    name = f"{row[LSOA_NAME]}<br>{row[LSOA_ID]}"
    if view == "geo_sensitivity":
        return (
            f"{name}<br>"
            f"GeoSensitivity: {row['geo_sensitivity']:.4f}<br>"
            f"|GeoSensitivity|: {row['geo_sensitivity_abs']:.4f}<br>"
            f"Geo rank: {int(row['geo_rank'])} of 168<br>"
            f"LISA: {LISA_LABELS.get(str(row['lisa_class']), row['lisa_class'])}"
        )
    if view == "latitude_effect":
        return (
            f"{name}<br>Latitude feature effect: {row['latitude_effect']:.4f}<br>"
            f"GeoSensitivity (separate): {row['geo_sensitivity']:.4f}"
        )
    if view == "longitude_effect":
        return (
            f"{name}<br>Longitude feature effect: {row['longitude_effect']:.4f}<br>"
            f"GeoSensitivity (separate): {row['geo_sensitivity']:.4f}"
        )
    lisa = str(row["lisa_class"])
    cid = row["cluster_id"] if pd.notna(row.get("cluster_id")) else "none"
    return (
        f"{name}<br>"
        f"LISA: {LISA_LABELS.get(lisa, lisa)}<br>"
        f"Local Moran I: {row['local_moran_i']:.4f}<br>"
        f"p-value: {row['local_p']:.3f}<br>"
        f"Adjusted p: not available<br>"
        f"Cluster id: {cid}"
    )


def build_map_figure(bundle: HullBundle, view: str, selected_lsoa: str | None):
    gdf = bundle.geometry.copy()
    field, legend = VIEW_FIELDS[view]
    geojson = _geojson(gdf[[LSOA_ID, "geometry"]])
    hover = [_hover(row, view) for _, row in gdf.iterrows()]

    fig = go.Figure()
    if view == "spatial_pattern":
        for cls, colour in LISA_COLOURS.items():
            sub = gdf[gdf["lisa_class"] == cls]
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
                    name=LISA_LABELS.get(cls, cls),
                    hovertext=[_hover(r, view) for _, r in sub.iterrows()],
                    hoverinfo="text",
                    marker_line_width=0.4,
                    marker_line_color="white",
                    customdata=sub[LSOA_ID].astype(str).tolist(),
                )
            )
    else:
        z = gdf[field].to_numpy(dtype=float)
        vmax = _limit(z)
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=geojson,
                locations=gdf[LSOA_ID],
                z=z,
                featureidkey="properties.lsoa21cd",
                zmid=0.0,
                zmin=-vmax,
                zmax=vmax,
                colorscale=DIVERGING,
                colorbar=dict(title=legend, thickness=12, len=0.72),
                hovertext=hover,
                hoverinfo="text",
                marker_line_width=0.35,
                marker_line_color="white",
                customdata=gdf[LSOA_ID].astype(str).tolist(),
                name=MAP_VIEWS[view],
            )
        )

    if selected_lsoa:
        sel = gdf[gdf[LSOA_ID] == selected_lsoa]
        if not sel.empty:
            fig.add_trace(
                go.Choroplethmapbox(
                    geojson=_geojson(sel[[LSOA_ID, "geometry"]]),
                    locations=sel[LSOA_ID],
                    z=[1],
                    featureidkey="properties.lsoa21cd",
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                    showscale=False,
                    hoverinfo="skip",
                    marker_line_width=2.4,
                    marker_line_color="#12171C",
                    name="Selected LSOA",
                )
            )

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=10.6,
        mapbox_center={"lat": 53.767, "lon": -0.327},
        margin=dict(l=0, r=0, t=8, b=0),
        height=620,
        legend=dict(orientation="h", y=0.01, x=0.01, bgcolor="rgba(255,255,255,0.75)"),
        paper_bgcolor="white",
    )
    return fig


def clicked_lsoa(event) -> str | None:
    if event is None:
        return None
    selection = getattr(event, "selection", None)
    if selection is None:
        return None
    points = getattr(selection, "points", None) or []
    if not points:
        return None
    pt = points[0]
    if isinstance(pt, dict):
        custom = pt.get("customdata")
        if isinstance(custom, (list, tuple)) and custom:
            return str(custom[0])
        if isinstance(custom, str):
            return custom
        loc = pt.get("location")
        return str(loc) if loc else None
    custom = getattr(pt, "customdata", None)
    if custom is not None:
        if isinstance(custom, (list, tuple)) and custom:
            return str(custom[0])
        return str(custom)
    loc = getattr(pt, "location", None)
    return str(loc) if loc else None


def render_map(bundle: HullBundle, view: str, selected_lsoa: str | None) -> str | None:
    fig = build_map_figure(bundle, view, selected_lsoa)
    try:
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key=f"hull_map_{view}",
            config={"displayModeBar": False},
        )
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        return None
    return clicked_lsoa(event)
