"""Load and validate official ONS LSOA 2021 boundaries for Kingston upon Hull."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy.spatial import Voronoi
from shapely.geometry import MultiPoint, Point, Polygon

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OFFICIAL_CACHE = DATA_DIR / "hull_lsoa_2021_official.geojson"
VORONOI_CACHE = DATA_DIR / "hull_lsoa_boundaries.geojson"
HULL_LAD = "E06000010"

ONS_QUERY = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5/FeatureServer/0/query"
)


def _finite_voronoi_polygons(points: np.ndarray, clip: Polygon) -> list[Polygon]:
    vor = Voronoi(points)
    polygons: list[Polygon] = []
    for idx in range(len(points)):
        region_idx = vor.point_region[idx]
        region = vor.regions[region_idx]
        if not region or -1 in region:
            poly = Point(points[idx]).buffer(80)
        else:
            poly = Polygon([vor.vertices[i] for i in region]).intersection(clip)
            if poly.is_empty:
                poly = Point(points[idx]).buffer(80)
        polygons.append(poly)
    return polygons


def build_voronoi_boundaries(expected_codes: list[str], panel_path: Path) -> gpd.GeoDataFrame:
    panel = pd.read_csv(panel_path)
    locations = (
        panel[["lsoa21cd", "lsoa21nm", "centroid_easting", "centroid_northing"]]
        .drop_duplicates("lsoa21cd")
        .query("lsoa21cd in @expected_codes")
        .reset_index(drop=True)
    )
    points = locations[["centroid_easting", "centroid_northing"]].to_numpy()
    clip = MultiPoint([tuple(p) for p in points]).convex_hull.buffer(250)
    cells = _finite_voronoi_polygons(points, clip)
    gdf = gpd.GeoDataFrame(
        {"lsoa21cd": locations["lsoa21cd"], "lsoa21nm": locations["lsoa21nm"], "geometry": cells},
        crs="EPSG:27700",
    )
    return clean_geometries(gdf, clip)


def clean_geometries(gdf: gpd.GeoDataFrame, clip: Polygon) -> gpd.GeoDataFrame:
    """Clip to study area and drop tiny detached fragments."""
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.intersection(clip)
    areas = gdf.geometry.area
    floor = max(float(areas.median()) * 0.05, 5000.0)
    gdf = gdf[areas >= floor].copy()
    return gdf


def _fetch_official_from_ons(expected_codes: list[str] | None = None) -> gpd.GeoDataFrame | None:
    """Fetch official ONS LSOA 2021 BGC polygons for Kingston upon Hull."""
    attempts = [
        {"where": "LSOA21NM LIKE 'Kingston upon Hull%'", "outFields": "LSOA21CD,LSOA21NM"},
    ]
    if expected_codes:
        # Chunk IN queries to stay under URL length limits.
        for start in range(0, len(expected_codes), 40):
            chunk = expected_codes[start : start + 40]
            quoted = ",".join(f"'{c}'" for c in chunk)
            attempts.append(
                {"where": f"LSOA21CD IN ({quoted})", "outFields": "LSOA21CD,LSOA21NM"}
            )

    frames: list[gpd.GeoDataFrame] = []
    for params_base in attempts:
        params = {
            **params_base,
            "returnGeometry": "true",
            "outSR": "27700",
            "f": "geojson",
            "resultRecordCount": "2000",
        }
        try:
            response = requests.get(ONS_QUERY, params=params, timeout=180)
            if not response.ok:
                continue
            payload = response.json()
            if payload.get("error") or not payload.get("features"):
                continue
            gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:27700")
            rename = {}
            for src, dst in (("LSOA21CD", "lsoa21cd"), ("LSOA21NM", "lsoa21nm")):
                if src in gdf.columns:
                    rename[src] = dst
            gdf = gdf.rename(columns=rename)
            if "lsoa21cd" not in gdf.columns:
                continue
            frames.append(gdf[["lsoa21cd", "lsoa21nm", "geometry"]])
            # Prefer the name-based query that returns the full city set.
            if "Kingston upon Hull" in params_base["where"] and len(gdf) >= 160:
                return gdf[["lsoa21cd", "lsoa21nm", "geometry"]].drop_duplicates("lsoa21cd")
        except (requests.RequestException, ValueError, KeyError):
            continue
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(combined, crs="EPSG:27700").drop_duplicates("lsoa21cd")
    return gdf


def _validate_boundaries(gdf: gpd.GeoDataFrame, expected_codes: set[str]) -> dict:
    codes = set(gdf["lsoa21cd"].astype(str))
    crs = str(gdf.crs)
    return {
        "lad21cd_target": HULL_LAD,
        "n_polygons": int(len(gdf)),
        "expected_lsoas": len(expected_codes),
        "matched_lsoas": len(codes & expected_codes),
        "missing_lsoa_codes": sorted(expected_codes - codes),
        "extra_lsoa_codes": sorted(codes - expected_codes),
        "duplicate_lsoa_codes": int(gdf["lsoa21cd"].duplicated().sum()),
        "crs": crs,
        "valid": (
            len(gdf) == len(expected_codes)
            and not (expected_codes - codes)
            and not (codes - expected_codes)
            and gdf["lsoa21cd"].duplicated().sum() == 0
            and ("27700" in crs or crs == "EPSG:27700")
        ),
    }


def load_boundaries(expected_codes: list[str]) -> tuple[gpd.GeoDataFrame, dict]:
    expected = set(expected_codes)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if OFFICIAL_CACHE.exists():
        gdf = gpd.read_file(OFFICIAL_CACHE)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:27700")
        else:
            gdf = gdf.to_crs("EPSG:27700")
        meta = _validate_boundaries(gdf, expected)
        meta["source"] = "cached_official"
        if meta["valid"]:
            return gdf, meta

    gdf = _fetch_official_from_ons(expected_codes)
    if gdf is not None:
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:27700")
        else:
            gdf = gdf.to_crs("EPSG:27700")
        # Keep only expected codes if the download returned extras.
        gdf = gdf[gdf["lsoa21cd"].isin(expected)].copy()
        meta = _validate_boundaries(gdf, expected)
        meta["source"] = "ons_arcgis_lsoa21_bgc_v5"
        if meta["valid"]:
            gdf.to_file(OFFICIAL_CACHE, driver="GeoJSON")
            return gdf, meta
        # Still cache if close, but fall through if incomplete.
        if meta["matched_lsoas"] == len(expected) and meta["duplicate_lsoa_codes"] == 0:
            gdf.to_file(OFFICIAL_CACHE, driver="GeoJSON")
            meta["valid"] = True
            meta["crs"] = "EPSG:27700"
            return gdf, meta

    panel_path = ROOT / "outputs" / "hull_lsoa_month_panel.csv"
    gdf = build_voronoi_boundaries(expected_codes, panel_path)
    meta = _validate_boundaries(gdf, expected)
    meta["source"] = "voronoi_approximate"
    meta["warning"] = "Approximate Voronoi geography — not official ONS LSOA boundaries."
    meta["valid"] = meta["n_polygons"] == len(expected_codes) and not meta["missing_lsoa_codes"]
    gdf.to_file(VORONOI_CACHE, driver="GeoJSON")
    return gdf, meta


def hull_outline(gdf: gpd.GeoDataFrame):
    return gdf.dissolve().boundary.iloc[0]


CARDINAL_DIRECTIONS = {
    "north": np.array([0.0, 1.0]),
    "east": np.array([1.0, 0.0]),
    "south": np.array([0.0, -1.0]),
    "west": np.array([-1.0, 0.0]),
}

# Automatic cardinal pick is the 35 km² Woodmansey LSOA (north) and the
# Hessle Golf Club LSOA (south). Thesis maps use a compact Cottingham LSOA
# on Hull's north-west edge, and the Hessle LSOA on the far side of the golf.
PREFERRED_CARDINAL_LSOAS = {
    "north": "E01012953",  # East Riding of Yorkshire 023A
    "south": "E01013009",  # East Riding of Yorkshire 034B (beyond the golf)
}


def fetch_official_polygons_by_codes(codes: list[str]) -> gpd.GeoDataFrame | None:
    """Fetch official ONS LSOA 2021 polygons by code (no Hull-name shortcut)."""
    wanted = [str(c) for c in codes]
    if not wanted:
        return None
    frames: list[gpd.GeoDataFrame] = []
    for start in range(0, len(wanted), 40):
        chunk = wanted[start : start + 40]
        quoted = ",".join(f"'{c}'" for c in chunk)
        params = {
            "where": f"LSOA21CD IN ({quoted})",
            "outFields": "LSOA21CD,LSOA21NM",
            "returnGeometry": "true",
            "outSR": "27700",
            "f": "geojson",
            "resultRecordCount": "2000",
        }
        try:
            response = requests.get(ONS_QUERY, params=params, timeout=180)
            if not response.ok:
                continue
            payload = response.json()
            if payload.get("error") or not payload.get("features"):
                continue
            gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:27700")
            rename = {}
            for src, dst in (("LSOA21CD", "lsoa21cd"), ("LSOA21NM", "lsoa21nm")):
                if src in gdf.columns:
                    rename[src] = dst
            gdf = gdf.rename(columns=rename)
            if "lsoa21cd" not in gdf.columns:
                continue
            frames.append(gdf[["lsoa21cd", "lsoa21nm", "geometry"]])
        except (requests.RequestException, ValueError, KeyError):
            continue
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(combined, crs="EPSG:27700").drop_duplicates("lsoa21cd")


def fetch_lsoas_in_envelope(
    minx: float, miny: float, maxx: float, maxy: float
) -> gpd.GeoDataFrame | None:
    """Fetch official LSOA 2021 polygons intersecting a BNG envelope."""
    params = {
        "where": "1=1",
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "27700",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "LSOA21CD,LSOA21NM",
        "returnGeometry": "true",
        "outSR": "27700",
        "f": "geojson",
        "resultRecordCount": "2000",
    }
    try:
        response = requests.get(ONS_QUERY, params=params, timeout=180)
        if not response.ok:
            return None
        payload = response.json()
        if payload.get("error") or not payload.get("features"):
            return None
        gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:27700")
        rename = {}
        for src, dst in (("LSOA21CD", "lsoa21cd"), ("LSOA21NM", "lsoa21nm")):
            if src in gdf.columns:
                rename[src] = dst
        gdf = gdf.rename(columns=rename)
        if "lsoa21cd" not in gdf.columns:
            return None
        return gdf[["lsoa21cd", "lsoa21nm", "geometry"]].drop_duplicates("lsoa21cd")
    except (requests.RequestException, ValueError, KeyError):
        return None


def select_cardinal_neighbors(
    hull: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    *,
    max_gap_m: float = 12000.0,
    min_alignment: float = 0.25,
) -> gpd.GeoDataFrame:
    """Pick one non-Hull LSOA per cardinal direction around the city outline.

    `PREFERRED_CARDINAL_LSOAS` overrides the automatic pick when that code is
    in the candidate set (compact north LSOA; south LSOA beyond Hessle Golf Club).
    Otherwise: share a boundary with Hull, then high alignment with the cardinal
    axis, then proximity to the Hull centroid.
    """
    hull = hull.to_crs("EPSG:27700").copy()
    candidates = candidates.to_crs("EPSG:27700").copy()
    hull_codes = set(hull["lsoa21cd"].astype(str))
    outside = candidates[~candidates["lsoa21cd"].astype(str).isin(hull_codes)].copy()
    if outside.empty:
        raise ValueError("No candidate LSOAs outside the Hull set")

    hull_union = hull.geometry.union_all()
    hull_xy = np.array([hull_union.centroid.x, hull_union.centroid.y], dtype=float)
    outside["gap_m"] = outside.geometry.distance(hull_union)
    outside = outside[outside["gap_m"] <= max_gap_m].copy()
    if outside.empty:
        raise ValueError(f"No outside LSOAs within {max_gap_m:.0f} m of Hull")

    cents = outside.geometry.centroid
    vec = np.column_stack([cents.x.to_numpy() - hull_xy[0], cents.y.to_numpy() - hull_xy[1]])
    dist = np.linalg.norm(vec, axis=1)
    dist = np.where(dist < 1.0, 1.0, dist)
    unit = vec / dist[:, None]
    outside = outside.assign(
        centroid_easting=cents.x.to_numpy(),
        centroid_northing=cents.y.to_numpy(),
        dist_to_hull_centroid_m=dist,
    )

    outside = outside.reset_index(drop=True)
    codes = outside["lsoa21cd"].astype(str).to_numpy()
    picked_rows = []
    used: set[str] = set()
    for direction, axis in CARDINAL_DIRECTIONS.items():
        alignment = unit @ axis
        chosen = None
        preferred = PREFERRED_CARDINAL_LSOAS.get(direction)
        if preferred:
            hits = np.where(codes == preferred)[0]
            if len(hits) and preferred not in used:
                chosen = int(hits[0])
        if chosen is None:
            keys = np.lexsort((dist, -alignment, outside["gap_m"].to_numpy(dtype=float)))
            for idx in keys:
                code = str(codes[idx])
                if code in used:
                    continue
                if float(alignment[idx]) < min_alignment:
                    continue
                chosen = int(idx)
                break
        if chosen is None:
            raise ValueError(f"No unused neighbor LSOA found for {direction}")
        row = outside.iloc[chosen].copy()
        row["direction"] = direction
        row["alignment"] = float(alignment[chosen])
        used.add(str(row["lsoa21cd"]))
        picked_rows.append(row)

    picked = gpd.GeoDataFrame(picked_rows, crs="EPSG:27700").reset_index(drop=True)
    return picked[
        [
            "direction",
            "lsoa21cd",
            "lsoa21nm",
            "gap_m",
            "alignment",
            "centroid_easting",
            "centroid_northing",
            "dist_to_hull_centroid_m",
            "geometry",
        ]
    ]


def save_boundary_report(meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
