"""Build static socioeconomic and built-environment covariates for Hull LSOAs.

Sources
-------
* ONS Census 2021 topic-summary bulk files distributed by Nomis.
* OpenStreetMap current snapshot queried through OSMnx/Overpass.

The output has exactly one row per LSOA and contains no coordinate columns.
"""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, Point, Polygon

from hull_boundaries import load_boundaries

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT = DATA_DIR / "hull_static_covariates.csv"
METADATA = DATA_DIR / "hull_static_covariates_metadata.json"
PANEL = ROOT / "outputs" / "hull_lsoa_month_panel.csv"
OSM_FEATURE_CACHE = DATA_DIR / "hull_osm_features.geojson"
OSM_ROADS_CACHE = DATA_DIR / "hull_osm_drive_edges.geojson"
OSM_INTERSECTIONS_CACHE = DATA_DIR / "hull_osm_intersections.geojson"

CENSUS_BASE = "https://www.nomisweb.co.uk/output/census/2021"
CENSUS_CACHE = DATA_DIR / "census_cache"
USER_AGENT = "HullGeoSMILEThesis/1.0 (academic research)"
OSM_MAP_API = "https://api.openstreetmap.org/api/0.6/map"
OVERPASS_URLS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]


def _download_lsoa_table(table: str) -> pd.DataFrame:
    CENSUS_CACHE.mkdir(parents=True, exist_ok=True)
    cache = CENSUS_CACHE / f"census2021-{table}-lsoa.csv"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    url = f"{CENSUS_BASE}/census2021-{table}.zip"
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=180)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        member = f"census2021-{table}-lsoa.csv"
        frame = pd.read_csv(archive.open(member), low_memory=False)
    frame.to_csv(cache, index=False)
    return frame


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return np.where(denominator > 0, 100.0 * numerator / denominator, np.nan)


def census_covariates(expected_codes: list[str]) -> pd.DataFrame:
    density = _download_lsoa_table("ts006")
    age = _download_lsoa_table("ts007a")
    tenure = _download_lsoa_table("ts054")
    activity = _download_lsoa_table("ts066")
    qualification = _download_lsoa_table("ts067")

    code = "geography code"
    out = density[[code]].copy().rename(columns={code: "lsoa21cd"})
    out["population_density_per_km2"] = density[
        "Population Density: Persons per square kilometre; measures: Value"
    ]

    age_total = age["Age: Total"]
    age_15_34 = age[
        [
            "Age: Aged 15 to 19 years",
            "Age: Aged 20 to 24 years",
            "Age: Aged 25 to 29 years",
            "Age: Aged 30 to 34 years",
        ]
    ].sum(axis=1)
    age_65_plus = age[
        [
            "Age: Aged 65 to 69 years",
            "Age: Aged 70 to 74 years",
            "Age: Aged 75 to 79 years",
            "Age: Aged 80 to 84 years",
            "Age: Aged 85 years and over",
        ]
    ].sum(axis=1)
    age_features = pd.DataFrame(
        {
            "lsoa21cd": age[code],
            "young_adult_15_34_pct": _ratio(age_15_34, age_total),
            "older_adult_65_plus_pct": _ratio(age_65_plus, age_total),
        }
    )

    tenure_total = tenure["Tenure of household: Total: All households"]
    tenure_features = pd.DataFrame(
        {
            "lsoa21cd": tenure[code],
            "social_rented_pct": _ratio(
                tenure["Tenure of household: Social rented"], tenure_total
            ),
            "private_rented_pct": _ratio(
                tenure["Tenure of household: Private rented"], tenure_total
            ),
            "owner_occupied_pct": _ratio(
                tenure["Tenure of household: Owned"], tenure_total
            ),
        }
    )

    activity_total = activity[
        "Economic activity status: Total: All usual residents aged 16 years and over"
    ]
    unemployed = (
        activity[
            "Economic activity status: Economically active (excluding full-time students): Unemployed"
        ]
        + activity[
            "Economic activity status: Economically active and a full-time student: Unemployed"
        ]
    )
    activity_features = pd.DataFrame(
        {
            "lsoa21cd": activity[code],
            "unemployed_pct_age16plus": _ratio(unemployed, activity_total),
            "economically_inactive_pct": _ratio(
                activity["Economic activity status: Economically inactive"],
                activity_total,
            ),
            "long_term_sick_disabled_pct": _ratio(
                activity[
                    "Economic activity status: Economically inactive: Long-term sick or disabled"
                ],
                activity_total,
            ),
        }
    )

    qual_total = qualification[
        "Highest level of qualification: Total: All usual residents aged 16 years and over"
    ]
    qualification_features = pd.DataFrame(
        {
            "lsoa21cd": qualification[code],
            "no_qualifications_pct": _ratio(
                qualification["Highest level of qualification: No qualifications"],
                qual_total,
            ),
            "level4_plus_qualification_pct": _ratio(
                qualification[
                    "Highest level of qualification: Level 4 qualifications and above"
                ],
                qual_total,
            ),
        }
    )

    for frame in (
        age_features,
        tenure_features,
        activity_features,
        qualification_features,
    ):
        out = out.merge(frame, on="lsoa21cd", how="left", validate="one_to_one")

    return out[out["lsoa21cd"].isin(expected_codes)].reset_index(drop=True)


def _overpass_json(query: str) -> dict:
    """Run one bounded query, failing over across public Overpass mirrors."""
    errors = []
    for url in OVERPASS_URLS:
        for attempt in range(2):
            try:
                print(f"Querying {url} (attempt {attempt + 1}) ...", flush=True)
                response = requests.post(
                    url,
                    data={"data": query},
                    headers={"User-Agent": USER_AGENT},
                    timeout=240,
                )
                response.raise_for_status()
                payload = response.json()
                if "elements" not in payload:
                    raise ValueError("Overpass response has no elements")
                return payload
            except (requests.RequestException, ValueError) as exc:
                errors.append(f"{url}#{attempt + 1}: {type(exc).__name__}: {exc}")
                print(f"OSM mirror failed; trying next: {errors[-1]}", flush=True)
    raise RuntimeError("All Overpass mirrors failed:\n" + "\n".join(errors))


def _overpass_bbox(bbox: tuple[float, float, float, float]) -> str:
    left, bottom, right, top = bbox
    return f"{bottom},{left},{top},{right}"


def _overpass_poly(geom) -> str | None:
    """Compact Overpass poly string (lat lon pairs) for one polygon."""
    g = geom
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda part: part.area)
    if g is None or g.is_empty:
        return None
    simple = g.simplify(0.0004, preserve_topology=True)
    coords = list(simple.exterior.coords)
    if len(coords) > 70:
        idx = np.linspace(0, len(coords) - 1, 70, dtype=int)
        coords = [coords[int(i)] for i in idx]
    return " ".join(f"{lat:.5f} {lon:.5f}" for lon, lat in coords)


def _overpass_spatial_filter(boundary_wgs84: gpd.GeoDataFrame) -> str:
    if len(boundary_wgs84) == 1:
        poly = _overpass_poly(boundary_wgs84.geometry.iloc[0])
        if poly:
            return f'(poly:"{poly}")'
    left, bottom, right, top = boundary_wgs84.total_bounds
    return f"({bottom},{left},{top},{right})"


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf is None or gdf.empty:
        path.write_text(
            '{"type":"FeatureCollection","features":[]}', encoding="utf-8"
        )
        return
    gdf.to_file(path, driver="GeoJSON")


def _padded_wgs84_bbox(boundary_wgs84: gpd.GeoDataFrame, pad_deg: float = 0.0006) -> tuple[float, float, float, float]:
    left, bottom, right, top = boundary_wgs84.total_bounds
    return (
        float(left) - pad_deg,
        float(bottom) - pad_deg,
        float(right) + pad_deg,
        float(top) + pad_deg,
    )


def _osm_map_xml(bbox: tuple[float, float, float, float]) -> bytes:
    """Fetch one OSM map API bbox (minlon, minlat, maxlon, maxlat)."""
    params = {"bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"}
    print(f"Querying OSM map API bbox={params['bbox']} ...", flush=True)
    response = requests.get(
        OSM_MAP_API,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=180,
    )
    if response.status_code == 400 and "too many nodes" in response.text.lower():
        raise ValueError("OSM map API bbox has too many nodes")
    response.raise_for_status()
    return response.content


def _parse_osm_xml(xml_bytes: bytes) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return (tagged features, roads, intersections) in WGS84 from OSM XML."""
    root = ET.fromstring(xml_bytes)
    node_xy: dict[int, tuple[float, float]] = {}
    node_tags: dict[int, dict[str, str]] = {}
    for node in root.findall("node"):
        nid = int(node.get("id"))
        node_xy[nid] = (float(node.get("lon")), float(node.get("lat")))
        tags = {tag.get("k"): tag.get("v") for tag in node.findall("tag")}
        if tags:
            node_tags[nid] = tags

    feature_records = []
    road_records = []
    node_use: dict[int, int] = {}
    highway_values = {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "unclassified",
        "residential",
        "living_street",
        "service",
    }
    for nid, tags in node_tags.items():
        lon, lat = node_xy[nid]
        feature_records.append(
            {
                "osm_type": "node",
                "osm_id": nid,
                "amenity": tags.get("amenity"),
                "shop": tags.get("shop"),
                "highway": tags.get("highway"),
                "public_transport": tags.get("public_transport"),
                "railway": tags.get("railway"),
                "leisure": tags.get("leisure"),
                "landuse": tags.get("landuse"),
                "natural": tags.get("natural"),
                "geometry": Point(lon, lat),
            }
        )

    for way in root.findall("way"):
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
        nds = [int(nd.get("ref")) for nd in way.findall("nd")]
        coords = [node_xy[i] for i in nds if i in node_xy]
        if len(coords) < 2:
            continue
        wid = int(way.get("id"))
        geom = Polygon(coords) if len(coords) >= 4 and coords[0] == coords[-1] else LineString(coords)
        feature_records.append(
            {
                "osm_type": "way",
                "osm_id": wid,
                "amenity": tags.get("amenity"),
                "shop": tags.get("shop"),
                "highway": tags.get("highway"),
                "public_transport": tags.get("public_transport"),
                "railway": tags.get("railway"),
                "leisure": tags.get("leisure"),
                "landuse": tags.get("landuse"),
                "natural": tags.get("natural"),
                "geometry": geom,
            }
        )
        if tags.get("highway") in highway_values:
            road_records.append({"osm_id": wid, "geometry": LineString(coords)})
            for nid in set(nds):
                if nid in node_xy:
                    node_use[nid] = node_use.get(nid, 0) + 1

    features = (
        gpd.GeoDataFrame(feature_records, geometry="geometry", crs="EPSG:4326")
        if feature_records
        else gpd.GeoDataFrame(
            {"osm_type": pd.Series(dtype=str), "osm_id": pd.Series(dtype="int64")},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
    )
    if feature_records:
        features = features.drop_duplicates(["osm_type", "osm_id"])
    roads = (
        gpd.GeoDataFrame(road_records, geometry="geometry", crs="EPSG:4326")
        if road_records
        else gpd.GeoDataFrame(
            {"osm_id": pd.Series(dtype="int64")},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
    )
    intersection_records = [
        {"osm_node_id": nid, "geometry": Point(node_xy[nid])}
        for nid, count in node_use.items()
        if count >= 2 and nid in node_xy
    ]
    intersections = gpd.GeoDataFrame(
        intersection_records, geometry="geometry", crs="EPSG:4326"
    )
    return features, roads, intersections


def _load_osm_via_map_api(
    boundary_wgs84: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    bbox = _padded_wgs84_bbox(boundary_wgs84)
    parts = _osm_map_parts(bbox, depth=0)
    feat_parts = [p[0] for p in parts if p[0] is not None and not p[0].empty]
    road_parts = [p[1] for p in parts if p[1] is not None and not p[1].empty]
    int_parts = [p[2] for p in parts if p[2] is not None and not p[2].empty]
    features = (
        gpd.GeoDataFrame(
            pd.concat(feat_parts, ignore_index=True), geometry="geometry", crs="EPSG:4326"
        ).drop_duplicates(["osm_type", "osm_id"])
        if feat_parts
        else gpd.GeoDataFrame(
            {"osm_type": pd.Series(dtype=str), "osm_id": pd.Series(dtype="int64")},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
    )
    roads = (
        gpd.GeoDataFrame(
            pd.concat(road_parts, ignore_index=True), geometry="geometry", crs="EPSG:4326"
        ).drop_duplicates(["osm_id"])
        if road_parts
        else gpd.GeoDataFrame(
            {"osm_id": pd.Series(dtype="int64")},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
    )
    intersections = (
        gpd.GeoDataFrame(
            pd.concat(int_parts, ignore_index=True), geometry="geometry", crs="EPSG:4326"
        ).drop_duplicates(["osm_node_id"])
        if int_parts
        else gpd.GeoDataFrame(
            {"osm_node_id": pd.Series(dtype="int64")},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
    )
    return features, roads, intersections


def _osm_map_parts(
    bbox: tuple[float, float, float, float], depth: int
) -> list[tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]]:
    try:
        return [_parse_osm_xml(_osm_map_xml(bbox))]
    except ValueError as exc:
        if "too many nodes" not in str(exc).lower() or depth >= 6:
            raise
        print(f"  tiling OSM bbox at depth {depth + 1}: {bbox}")
        minx, miny, maxx, maxy = bbox
        midx, midy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        tiles = [
            (minx, miny, midx, midy),
            (midx, miny, maxx, midy),
            (minx, midy, midx, maxy),
            (midx, midy, maxx, maxy),
        ]
        parts: list[tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]] = []
        for tile in tiles:
            parts.extend(_osm_map_parts(tile, depth + 1))
        return parts


def _feature_gdf(payload: dict) -> gpd.GeoDataFrame:
    records = []
    for element in payload["elements"]:
        tags = element.get("tags", {})
        geometry = None
        if element["type"] == "node" and "lat" in element and "lon" in element:
            geometry = Point(element["lon"], element["lat"])
        elif element.get("center"):
            geometry = Point(element["center"]["lon"], element["center"]["lat"])
        elif element["type"] == "way" and element.get("geometry"):
            coords = [(point["lon"], point["lat"]) for point in element["geometry"]]
            if len(coords) >= 4 and coords[0] == coords[-1]:
                geometry = Polygon(coords)
            elif len(coords) >= 2:
                geometry = LineString(coords)
        if geometry is None or geometry.is_empty:
            continue
        records.append(
            {
                "osm_type": element["type"],
                "osm_id": element["id"],
                "amenity": tags.get("amenity"),
                "shop": tags.get("shop"),
                "highway": tags.get("highway"),
                "public_transport": tags.get("public_transport"),
                "railway": tags.get("railway"),
                "leisure": tags.get("leisure"),
                "landuse": tags.get("landuse"),
                "natural": tags.get("natural"),
                "geometry": geometry,
            }
        )
    if not records:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def _road_gdfs(payload: dict) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes = {
        element["id"]: (element["lon"], element["lat"])
        for element in payload["elements"]
        if element["type"] == "node" and "lat" in element and "lon" in element
    }
    road_records = []
    node_use: dict[int, int] = {}
    for element in payload["elements"]:
        if element["type"] != "way" or "nodes" not in element:
            continue
        node_ids = [node_id for node_id in element["nodes"] if node_id in nodes]
        coords = [nodes[node_id] for node_id in node_ids]
        if len(coords) < 2:
            continue
        road_records.append(
            {"osm_id": element["id"], "geometry": LineString(coords)}
        )
        for node_id in set(node_ids):
            node_use[node_id] = node_use.get(node_id, 0) + 1
    roads = gpd.GeoDataFrame(road_records, geometry="geometry", crs="EPSG:4326")
    # Shared nodes used by at least two distinct road ways are intersection
    # candidates. This is a reproducible topology proxy, not a traffic survey.
    intersection_records = [
        {"osm_node_id": node_id, "geometry": Point(nodes[node_id])}
        for node_id, count in node_use.items()
        if count >= 2
    ]
    intersections = gpd.GeoDataFrame(
        intersection_records, geometry="geometry", crs="EPSG:4326"
    )
    return roads, intersections


def _prepare_point_features(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    features = features.to_crs("EPSG:27700").copy()
    features["geometry"] = features.geometry.representative_point()
    return features


def _tag_equals(frame: gpd.GeoDataFrame, column: str, values: set[str]) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].astype(str).isin(values)


def _tag_present(frame: gpd.GeoDataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].notna()


def _point_counts(
    boundaries: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    mask: pd.Series,
    name: str,
) -> pd.Series:
    selected = points.loc[mask, ["geometry"]]
    if selected.empty:
        return pd.Series(0.0, index=boundaries["lsoa21cd"], name=name)
    joined = gpd.sjoin(
        selected, boundaries[["lsoa21cd", "geometry"]], predicate="within", how="inner"
    )
    counts = joined.groupby("lsoa21cd").size()
    return boundaries["lsoa21cd"].map(counts).fillna(0.0).rename(name)


def osm_covariates(
    boundaries: gpd.GeoDataFrame,
    *,
    feature_cache: Path | None = None,
    roads_cache: Path | None = None,
    intersections_cache: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    feature_cache = feature_cache or OSM_FEATURE_CACHE
    roads_cache = roads_cache or OSM_ROADS_CACHE
    intersections_cache = intersections_cache or OSM_INTERSECTIONS_CACHE
    boundaries = boundaries.to_crs("EPSG:27700").copy()
    boundary_wgs84 = boundaries.to_crs("EPSG:4326")
    left, bottom, right, top = boundary_wgs84.total_bounds
    bbox = (float(left), float(bottom), float(right), float(top))

    if feature_cache.exists():
        features = gpd.read_file(feature_cache)
    elif len(boundaries) == 1:
        print("Fetching OSM via map API for one LSOA polygon...")
        features, api_roads, api_intersections = _load_osm_via_map_api(boundary_wgs84)
        _write_geojson(features, feature_cache)
        if not roads_cache.exists():
            _write_geojson(api_roads.to_crs("EPSG:27700"), roads_cache)
        if not intersections_cache.exists():
            _write_geojson(api_intersections.to_crs("EPSG:27700"), intersections_cache)
    else:
        filt = _overpass_spatial_filter(boundary_wgs84)
        # Keep individual queries small enough for public Overpass mirrors.
        feature_queries = [
            f"""
            [out:json][timeout:180];
            (
              node["amenity"~"^(pub|bar|nightclub)$"]{filt};
              way["amenity"~"^(pub|bar|nightclub)$"]{filt};
            );
            out tags center;
            """,
            f"""
            [out:json][timeout:180];
            (node["shop"]{filt}; way["shop"]{filt};);
            out tags center;
            """,
            f"""
            [out:json][timeout:180];
            (
              node["highway"="bus_stop"]{filt};
              way["highway"="bus_stop"]{filt};
              node["public_transport"~"^(platform|station|stop_position)$"]{filt};
              way["public_transport"~"^(platform|station|stop_position)$"]{filt};
              node["railway"~"^(station|halt|tram_stop)$"]{filt};
              way["railway"~"^(station|halt|tram_stop)$"]{filt};
            );
            out tags center;
            """,
            f"""
            [out:json][timeout:180];
            (
              way["leisure"~"^(park|recreation_ground|garden)$"]{filt};
              way["landuse"~"^(grass|forest|meadow|recreation_ground|retail)$"]{filt};
              way["natural"~"^(wood|grassland)$"]{filt};
            );
            out tags geom;
            """,
        ]
        feature_frames = [
            f for f in (_feature_gdf(_overpass_json(query)) for query in feature_queries)
            if f is not None and not f.empty
        ]
        if feature_frames:
            features = gpd.GeoDataFrame(
                pd.concat(feature_frames, ignore_index=True),
                geometry="geometry",
                crs="EPSG:4326",
            )
            features = features.drop_duplicates(["osm_type", "osm_id"])
        else:
            features = gpd.GeoDataFrame(
                {"osm_type": pd.Series(dtype=str), "osm_id": pd.Series(dtype="int64")},
                geometry=gpd.GeoSeries([], crs="EPSG:4326"),
                crs="EPSG:4326",
            )
        features.to_file(feature_cache, driver="GeoJSON")

    if roads_cache.exists() and intersections_cache.exists():
        roads = gpd.read_file(roads_cache).to_crs("EPSG:27700")
        intersections = gpd.read_file(intersections_cache).to_crs("EPSG:27700")
    else:
        filt = _overpass_spatial_filter(boundary_wgs84)
        road_query = f"""
        [out:json][timeout:180];
        way["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|living_street|service)$"]{filt};
        out body;
        >;
        out skel qt;
        """
        roads, intersections = _road_gdfs(_overpass_json(road_query))
        roads = roads.to_crs("EPSG:27700")
        intersections = intersections.to_crs("EPSG:27700")
        roads.to_file(roads_cache, driver="GeoJSON")
        intersections.reset_index(drop=True).to_file(
            intersections_cache, driver="GeoJSON"
        )

    points = _prepare_point_features(features)
    area_km2 = boundaries.geometry.area / 1_000_000.0
    result = pd.DataFrame({"lsoa21cd": boundaries["lsoa21cd"].to_numpy()})
    result["lsoa_area_km2"] = area_km2.to_numpy()

    point_specs = {
        "nightlife_count": _tag_equals(points, "amenity", {"pub", "bar", "nightclub"}),
        "retail_count": _tag_present(points, "shop"),
        "transit_stop_count": (
            _tag_equals(points, "highway", {"bus_stop"})
            | _tag_present(points, "public_transport")
            | _tag_equals(points, "railway", {"station", "halt", "tram_stop"})
        ),
    }
    for name, mask in point_specs.items():
        result[name] = _point_counts(boundaries, points, mask, name).to_numpy()
        result[name.replace("_count", "_per_km2")] = result[name] / result[
            "lsoa_area_km2"
        ]

    clipped_roads = gpd.overlay(
        roads[["geometry"]], boundaries[["lsoa21cd", "geometry"]], how="intersection"
    ) if not roads.empty else gpd.GeoDataFrame(
        columns=["lsoa21cd", "geometry"], crs=boundaries.crs
    )
    clipped_roads["length_km"] = clipped_roads.geometry.length / 1000.0
    road_length = clipped_roads.groupby("lsoa21cd")["length_km"].sum()
    result["road_length_km"] = result["lsoa21cd"].map(road_length).fillna(0.0)
    result["road_length_km_per_km2"] = (
        result["road_length_km"] / result["lsoa_area_km2"]
    )

    intersection_join = gpd.sjoin(
        intersections,
        boundaries[["lsoa21cd", "geometry"]],
        predicate="within",
        how="inner",
    )
    intersection_count = intersection_join.groupby("lsoa21cd").size()
    result["intersection_count"] = (
        result["lsoa21cd"].map(intersection_count).fillna(0.0)
    )
    result["intersection_per_km2"] = (
        result["intersection_count"] / result["lsoa_area_km2"]
    )

    polygon_features = features.to_crs("EPSG:27700").copy()
    green_mask = (
        _tag_equals(
            polygon_features,
            "leisure",
            {"park", "recreation_ground", "garden"},
        )
        | _tag_equals(
            polygon_features,
            "landuse",
            {"grass", "forest", "meadow", "recreation_ground"},
        )
        | _tag_equals(polygon_features, "natural", {"wood", "grassland"})
    )
    green = polygon_features.loc[
        green_mask
        & polygon_features.geometry.geom_type.isin(["Polygon", "MultiPolygon"]),
        ["geometry"],
    ]
    if green.empty:
        result["green_space_pct"] = 0.0
    else:
        green = green.dissolve().explode(index_parts=False).reset_index(drop=True)
        clipped_green = gpd.overlay(
            green, boundaries[["lsoa21cd", "geometry"]], how="intersection"
        )
        clipped_green["green_area"] = clipped_green.geometry.area
        green_area = clipped_green.groupby("lsoa21cd")["green_area"].sum()
        result["green_space_pct"] = (
            100.0
            * result["lsoa21cd"].map(green_area).fillna(0.0)
            / (result["lsoa_area_km2"] * 1_000_000.0)
        ).clip(0, 100)

    metadata = {
        "source": "OpenStreetMap contributors via direct Overpass API queries",
        "license": "ODbL 1.0",
        "snapshot_retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "feature_cache": str(feature_cache),
        "road_cache": str(roads_cache),
        "intersection_cache": str(intersections_cache),
        "query_bbox_wgs84": list(bbox),
        "notes": (
            "OSM is a current, community-maintained snapshot. Counts may be "
            "incomplete and should be treated as contextual proxies. Green-space "
            "coverage uses tagged OSM ways; relation-only polygons are not counted. "
            "Intersections are shared topology nodes used by at least two road ways."
        ),
    }
    return result, metadata


def validate(frame: pd.DataFrame, expected_codes: list[str]) -> dict:
    coordinate_tokens = ("coord", "easting", "northing", "latitude", "longitude")
    forbidden = [
        column
        for column in frame.columns
        if column != "lsoa21cd"
        and any(token in column.lower() for token in coordinate_tokens)
    ]
    numeric = frame.drop(columns="lsoa21cd")
    return {
        "rows": int(len(frame)),
        "expected_lsoas": len(expected_codes),
        "duplicate_lsoa_codes": int(frame["lsoa21cd"].duplicated().sum()),
        "missing_lsoa_codes": sorted(set(expected_codes) - set(frame["lsoa21cd"])),
        "extra_lsoa_codes": sorted(set(frame["lsoa21cd"]) - set(expected_codes)),
        "missing_numeric_values": int(numeric.isna().sum().sum()),
        "forbidden_coordinate_columns": forbidden,
        "valid": (
            len(frame) == len(expected_codes)
            and not frame["lsoa21cd"].duplicated().any()
            and set(frame["lsoa21cd"]) == set(expected_codes)
            and not numeric.isna().any().any()
            and not forbidden
        ),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL, usecols=["lsoa21cd"])
    expected_codes = sorted(panel["lsoa21cd"].unique())
    boundaries, boundary_meta = load_boundaries(expected_codes)

    census = census_covariates(expected_codes)
    osm, osm_meta = osm_covariates(boundaries)
    combined = census.merge(osm, on="lsoa21cd", how="inner", validate="one_to_one")
    combined = combined.sort_values("lsoa21cd").reset_index(drop=True)

    report = validate(combined, expected_codes)
    if not report["valid"]:
        raise ValueError(f"Covariate validation failed: {report}")

    combined.to_csv(OUTPUT, index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "census": {
            "source": "ONS Census 2021 topic summaries via Nomis bulk downloads",
            "tables": ["TS006", "TS007A", "TS054", "TS066", "TS067"],
            "urls": [
                f"{CENSUS_BASE}/census2021-{table}.zip"
                for table in ("ts006", "ts007a", "ts054", "ts066", "ts067")
            ],
            "license": "Open Government Licence v3.0",
        },
        "osm": osm_meta,
        "boundary_validation": boundary_meta,
        "validation": report,
        "feature_definitions": {
            "percent_features": "Percent of the stated Census denominator.",
            "density_features": "Counts or lengths divided by official LSOA area in km².",
            "green_space_pct": "OSM-tagged green polygon intersection area / LSOA area.",
        },
    }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(combined)} LSOAs, {len(combined.columns) - 1} features)")
    print(f"Wrote {METADATA}")


if __name__ == "__main__":
    main()
