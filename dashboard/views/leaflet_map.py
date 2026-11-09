"""Draw the dashboard's choropleth figures in Leaflet, which does not need WebGL.

The map views build Plotly Choroplethmap figures that hold the colour scales, hover text and overlays.
render_figure reads those traces and draws them unchanged, so a layer means the same in either renderer.
"""

from __future__ import annotations

import html

import folium
import numpy as np
import streamlit as st
from folium.plugins import Fullscreen, MeasureControl, MousePosition, Search
from plotly.colors import sample_colorscale
from streamlit_folium import st_folium

CENTER = (53.767, -0.327)
NO_DATA = "#D9DEE2"
LSOA_KEY = "lsoa21cd"
SELECTED_TRACE = "Selected LSOA"
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services"
BASEMAPS = {
    "Light grey": (f"{ESRI}/Canvas/World_Light_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}",
                   "Esri, HERE, Garmin, (c) OpenStreetMap contributors"),
    "Streets": ("https://tile.openstreetmap.org/{z}/{x}/{y}.png", "(c) OpenStreetMap contributors"),
    "Satellite": (f"{ESRI}/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}", "Esri, Maxar, Earthstar Geographics"),
}


def _clear(colour: str) -> bool:
    compact = str(colour).replace(" ", "")
    return compact.startswith("rgba") and compact.endswith(",0)")


def _fill_colours(trace) -> list[str | None]:
    """One colour per location; None for an outline-only trace."""
    scale = [(float(stop), colour) for stop, colour in trace.colorscale]
    z = np.asarray(trace.z, float)
    if _clear(scale[0][1]):
        return [None] * len(z)
    if len(scale) == 2 * len(z):  # one colour band per location (the RGB overlay)
        return [scale[2 * i][1] for i in range(len(z))]
    finite = z[np.isfinite(z)]
    low = float(trace.zmin) if trace.zmin is not None else float(finite.min())
    high = float(trace.zmax) if trace.zmax is not None else float(finite.max())
    span = high - low
    shares = np.clip((np.nan_to_num(z, nan=low) - low) / span, 0, 1) if span > 0 else np.zeros(len(z))
    colours = sample_colorscale([[s, c] for s, c in scale], shares.tolist())
    return [colour if np.isfinite(value) else NO_DATA for colour, value in zip(colours, z)]


def _features(trace, names: dict[str, str] | None) -> list[dict]:
    """The trace's polygons, each carrying its own colour, hover text and LSOA code."""
    locations = [str(code) for code in trace.locations]
    colours = _fill_colours(trace)
    tips = list(trace.hovertext) if trace.hovertext is not None and trace.hoverinfo != "skip" else [None] * len(locations)
    position = {code: i for i, code in enumerate(locations)}
    out = []
    for feature in trace.geojson["features"]:
        code = str(feature["properties"][LSOA_KEY])
        i = position.get(code)
        if i is None:
            continue
        out.append({"type": "Feature", "geometry": feature["geometry"],
                    "properties": {"_id": code, "_fill": colours[i], "_tip": tips[i],
                                   "_name": (names or {}).get(code, code)}})
    return out


def _style(line_colour: str, line_width: float, opacity: float, highlight: set[str] | None):
    def style(feature):
        props = feature["properties"]
        if props["_fill"] is None:
            return {"fillOpacity": 0, "color": line_colour, "weight": line_width}
        dimmed = highlight is not None and props["_id"] not in highlight
        return {"fillColor": props["_fill"], "fillOpacity": opacity * (0.15 if dimmed else 1.0),
                "color": line_colour, "weight": line_width, "opacity": 0.3 if dimmed else 0.9}
    return style


def _legend(fig) -> str:
    """A colour bar for scaled layers and swatches for categorical ones, drawn outside the map."""
    parts = []
    for trace in fig.data:
        scale = trace.colorscale
        if trace.showscale is False:
            if scale and not _clear(scale[0][1]) and trace.name and scale[0][1] == scale[-1][1]:
                parts.append(f"<span style='display:inline-flex;align-items:center;margin:2px 14px 2px 0;'>"
                             f"<span style='width:12px;height:12px;border-radius:2px;background:{scale[0][1]};"
                             f"margin-right:5px;border:1px solid rgba(120,120,120,.35);'></span>"
                             f"{html.escape(str(trace.name))}</span>")
            continue
        z = np.asarray(trace.z, float)
        low = trace.zmin if trace.zmin is not None else np.nanmin(z)
        high = trace.zmax if trace.zmax is not None else np.nanmax(z)
        stops = ",".join(f"{colour} {100 * float(stop):.0f}%" for stop, colour in scale)
        title = trace.colorbar.title.text or ""
        parts.append(f"<div style='max-width:420px;'><div>{html.escape(str(title))}</div>"
                     f"<div style='height:10px;border-radius:2px;background:linear-gradient(to right,{stops});"
                     f"border:1px solid rgba(120,120,120,.35);'></div>"
                     f"<div style='display:flex;justify-content:space-between;'><span>{low:.3g}</span>"
                     f"<span>{high:.3g}</span></div></div>")
    return f"<div style='font-size:.8rem;color:#57646D;margin:.1rem 0 .5rem 0;'>{''.join(parts)}</div>" if parts else ""


def build_map(fig, *, opacity: float, highlight: set[str] | None, names: dict[str, str] | None, tools: bool,
              labels: list[tuple[float, float, str]] | None = None) -> tuple[folium.Map, folium.FeatureGroup]:
    zoom = 12 if tools else 11
    m = folium.Map(location=CENTER, zoom_start=zoom, tiles=None, control_scale=True, prefer_canvas=False)
    for i, (label, (url, attribution)) in enumerate(BASEMAPS.items()):
        folium.TileLayer(url, attr=attribution, name=label, show=i == 0, max_zoom=19).add_to(m)
    selected = folium.FeatureGroup(name="Selected area")
    everything = []
    for trace in fig.data:
        features = _features(trace, names)
        if not features:
            continue
        data = {"type": "FeatureCollection", "features": features}
        if trace.name == SELECTED_TRACE:
            folium.GeoJson(data, style_function=_style("#12171C", 3, 0, None), interactive=False).add_to(selected)
            continue
        everything += features
        line = trace.marker.line
        outline_only = features[0]["properties"]["_fill"] is None
        tooltip = None
        if features[0]["properties"]["_tip"] is not None:
            tooltip = folium.GeoJsonTooltip(fields=["_tip"], labels=False, sticky=True)
        kwargs = {"interactive": False} if outline_only else {}
        layer = folium.GeoJson(data, name=str(trace.name or "Layer"), tooltip=tooltip,
                               style_function=_style(line.color or "white", float(line.width or 0.4), opacity, highlight),
                               highlight_function=None if outline_only else (lambda _: {"weight": 2.2, "color": "#12171C"}),
                               **kwargs)
        layer.add_to(m)
    if tools:
        unique = list({f["properties"]["_id"]: f for f in everything}.values())
        index = folium.GeoJson({"type": "FeatureCollection", "features": unique}, control=False, interactive=False,
                               style_function=lambda _: {"fillOpacity": 0, "weight": 0, "opacity": 0})
        index.add_to(m)
        Search(layer=index, search_label="_name", placeholder="Search area, e.g. 029E", collapsed=False,
               search_zoom=14, position="topright").add_to(m)
        Fullscreen().add_to(m)
        MeasureControl(position="bottomleft", primary_length_unit="kilometers", primary_area_unit="sqkilometers").add_to(m)
        MousePosition(position="bottomright", separator=", ", num_digits=4).add_to(m)
    for lat, lon, text in labels or []:
        folium.Marker((lat, lon), interactive=False, icon=folium.DivIcon(
            icon_size=(0, 0), html=f"<div class='gs-place' style='display:inline-block;transform:translate(-50%,-50%);"
                                   f"background:#fff;border:1px solid #AEB6C2;border-radius:4px;padding:1px 6px;"
                                   f"font:600 11px sans-serif;color:#12171C;white-space:nowrap;'>{html.escape(text)}</div>")).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m, selected


def render_figure(fig, key: str, *, height: int = 620, opacity: float = 0.8, highlight: set[str] | None = None,
                  names: dict[str, str] | None = None, tools: bool = False, clickable: bool = False,
                  legend: bool = True, labels: list[tuple[float, float, str]] | None = None) -> str | None:
    """Draw the figure; return the LSOA code of a new click, if clickable."""
    m, selected = build_map(fig, opacity=opacity, highlight=highlight, names=names, tools=tools, labels=labels)
    result = st_folium(m, key=key, height=height, use_container_width=True, feature_group_to_add=selected,
                       returned_objects=["last_active_drawing"] if clickable else [])
    if legend:
        st.markdown(_legend(fig), unsafe_allow_html=True)
    drawing = (result or {}).get("last_active_drawing") if clickable else None
    code = ((drawing or {}).get("properties") or {}).get("_id")
    seen = f"{key}_last_click"
    if code and st.session_state.get(seen) != code:
        st.session_state[seen] = code
        return str(code)
    return None
