"""Select four LSOAs just outside Hull (N/E/S/W) and build matching panel rows.

Police.uk street counts, Census 2021 rates, and OSM built-environment features
use the same definitions as the Hull LSOAs. Results are cached under data/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from fetch_hull_static_covariates import (
    OSM_FEATURE_CACHE,
    OSM_INTERSECTIONS_CACHE,
    OSM_ROADS_CACHE,
    OUTPUT as HULL_COVARIATES,
    census_covariates,
    osm_covariates,
    validate,
)
from geosmile_hull_twobranch import (
    DERIVED_FEATURES,
    POLICE_COLUMNS,
    SPATIAL_COLUMNS,
    _csv_members,
    _normalise_columns,
    lonlat_to_bng,
)
from hull_boundaries import (
    CARDINAL_DIRECTIONS,
    DATA_DIR,
    PREFERRED_CARDINAL_LSOAS,
    fetch_lsoas_in_envelope,
    load_boundaries,
    select_cardinal_neighbors,
)

ROOT = Path(__file__).resolve().parent
POLICE_ZIP = ROOT / "Police.uk.zip"
REGION_CACHE = DATA_DIR / "hull_region_lsoas.geojson"
NEIGHBOR_GEOJSON = DATA_DIR / "cardinal_neighbor_lsoas.geojson"
NEIGHBOR_META = DATA_DIR / "cardinal_neighbor_lsoas.json"
NEIGHBOR_COVARIATES = DATA_DIR / "cardinal_neighbor_static_covariates.csv"
NEIGHBOR_PANEL = ROOT / "outputs" / "cardinal_neighbor_lsoa_month_panel.csv"
BUFFER_M = 12000.0


def load_crimes_for_lsoas(source: Path, lsoa_codes: list[str]) -> pd.DataFrame:
    """Load Humberside street files and retain the requested LSOA codes."""
    wanted = set(str(c) for c in lsoa_codes)
    chunks: list[pd.DataFrame] = []
    seen = 0
    for name, readable in _csv_members(source):
        low = name.lower()
        if "street" not in low and source.suffix.lower() != ".csv":
            continue
        try:
            raw = pd.read_csv(readable, low_memory=False)
            frame = _normalise_columns(raw)
        except (ValueError, UnicodeDecodeError):
            continue
        seen += len(frame)
        codes = frame["lsoa21cd"].fillna("").astype(str)
        hit = frame[codes.isin(wanted)].copy()
        if not hit.empty:
            chunks.append(hit[list(POLICE_COLUMNS.values())])
    print(f"Read {seen:,} street rows; retained {sum(len(c) for c in chunks):,} neighbor rows.")
    if not chunks:
        return pd.DataFrame(columns=list(POLICE_COLUMNS.values()))
    crimes = pd.concat(chunks, ignore_index=True)
    crimes["month"] = pd.to_datetime(crimes["month"], format="%Y-%m")
    crimes["longitude"] = pd.to_numeric(crimes["longitude"], errors="coerce")
    crimes["latitude"] = pd.to_numeric(crimes["latitude"], errors="coerce")
    return crimes


def load_or_select_neighbors(hull_codes: list[str]) -> gpd.GeoDataFrame:
    """Return the four cardinal neighbor LSOAs, using cache when valid."""
    hull, _ = load_boundaries(sorted(hull_codes))
    if NEIGHBOR_GEOJSON.exists() and NEIGHBOR_META.exists():
        picked = gpd.read_file(NEIGHBOR_GEOJSON)
        if picked.crs is None:
            picked = picked.set_crs("EPSG:27700")
        else:
            picked = picked.to_crs("EPSG:27700")
        meta = json.loads(NEIGHBOR_META.read_text(encoding="utf-8"))
        dirs = [str(d) for d in picked["direction"].tolist()] if "direction" in picked.columns else []
        by_dir = (
            dict(zip(picked["direction"].astype(str), picked["lsoa21cd"].astype(str)))
            if "direction" in picked.columns
            else {}
        )
        preferred_ok = all(by_dir.get(d) == code for d, code in PREFERRED_CARDINAL_LSOAS.items())
        if set(dirs) == set(CARDINAL_DIRECTIONS) and len(picked) == 4 and preferred_ok:
            print(f"Using cached cardinal neighbors: {meta.get('lsoa21cd', dirs)}")
            return picked

    if REGION_CACHE.exists():
        region = gpd.read_file(REGION_CACHE)
        if region.crs is None:
            region = region.set_crs("EPSG:27700")
        else:
            region = region.to_crs("EPSG:27700")
    else:
        minx, miny, maxx, maxy = hull.total_bounds
        region = fetch_lsoas_in_envelope(
            minx - BUFFER_M, miny - BUFFER_M, maxx + BUFFER_M, maxy + BUFFER_M
        )
        if region is None or region.empty:
            raise RuntimeError("Could not fetch LSOAs around Hull from the ONS boundary service.")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        region.to_file(REGION_CACHE, driver="GeoJSON")

    picked = select_cardinal_neighbors(hull, region, max_gap_m=BUFFER_M)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    picked.to_file(NEIGHBOR_GEOJSON, driver="GeoJSON")
    payload = {
        "lsoa21cd": picked["lsoa21cd"].astype(str).tolist(),
        "lsoa21nm": picked["lsoa21nm"].astype(str).tolist(),
        "direction": picked["direction"].astype(str).tolist(),
        "gap_m": [float(x) for x in picked["gap_m"]],
        "alignment": [float(x) for x in picked["alignment"]],
        "buffer_m": BUFFER_M,
    }
    NEIGHBOR_META.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Selected cardinal neighbors:")
    for _, row in picked.iterrows():
        print(
            f"  {row['direction']}: {row['lsoa21cd']} {row['lsoa21nm']} "
            f"(gap={float(row['gap_m']):.0f} m)"
        )
    return picked


def collect_neighbor_covariates(neighbors: gpd.GeoDataFrame) -> pd.DataFrame:
    """Census + OSM covariates for the four neighbor LSOAs (same schema as Hull)."""
    codes = sorted(neighbors["lsoa21cd"].astype(str).unique())
    if NEIGHBOR_COVARIATES.exists():
        extra = pd.read_csv(NEIGHBOR_COVARIATES)
        if set(extra["lsoa21cd"].astype(str)) == set(codes) and not extra.isna().any().any():
            print(f"Using cached neighbor covariates ({len(extra)} LSOAs).")
            return extra

    print("Fetching Census 2021 covariates for neighbor LSOAs...")
    census = census_covariates(codes)
    print("Fetching OSM covariates for each neighbor LSOA (same OSM schema as Hull)...")
    osm_frames = []
    for _, row in neighbors.iterrows():
        code = str(row["lsoa21cd"])
        print(f"  OSM for {code} {row['lsoa21nm']}...")
        one = gpd.GeoDataFrame(
            [{"lsoa21cd": code, "lsoa21nm": row["lsoa21nm"], "geometry": row.geometry}],
            crs=neighbors.crs,
        )
        prefix = DATA_DIR / f"cardinal_neighbor_{code}"
        try:
            osm_one, _ = osm_covariates(
                one,
                feature_cache=Path(str(prefix) + "_osm_features.geojson"),
                roads_cache=Path(str(prefix) + "_osm_drive_edges.geojson"),
                intersections_cache=Path(str(prefix) + "_osm_intersections.geojson"),
            )
        except RuntimeError as exc:
            print(f"  Overpass failed for {code} ({exc}); joining Hull OSM cache onto this polygon.")
            osm_one, _ = osm_covariates(
                one,
                feature_cache=OSM_FEATURE_CACHE,
                roads_cache=OSM_ROADS_CACHE,
                intersections_cache=OSM_INTERSECTIONS_CACHE,
            )
            osm_one = _fill_osm_from_nearest_hull(osm_one, one)
        osm_frames.append(osm_one)
    osm = pd.concat(osm_frames, ignore_index=True)
    combined = census.merge(osm, on="lsoa21cd", how="inner", validate="one_to_one")
    combined = combined.sort_values("lsoa21cd").reset_index(drop=True)
    report = validate(combined, codes)
    if not report["valid"]:
        raise ValueError(f"Neighbor covariate validation failed: {report}")
    combined.to_csv(NEIGHBOR_COVARIATES, index=False)
    print(f"Wrote {NEIGHBOR_COVARIATES} ({len(combined)} LSOAs).")
    return combined


def _fill_osm_from_nearest_hull(
    osm: pd.DataFrame, neighbors: gpd.GeoDataFrame
) -> pd.DataFrame:
    """If a neighbor has no OSM roads in the Hull cache, copy densities from the nearest Hull LSOA."""
    hull_cov = pd.read_csv(HULL_COVARIATES)
    density_cols = [
        c
        for c in hull_cov.columns
        if c != "lsoa21cd"
        and (
            c.endswith("_per_km2")
            or c.endswith("_pct")
            or c == "green_space_pct"
        )
    ]
    count_cols = [
        c
        for c in hull_cov.columns
        if c.endswith("_count") or c in {"road_length_km", "lsoa_area_km2"}
    ]
    hull_gdf, _ = load_boundaries(sorted(hull_cov["lsoa21cd"].astype(str)))
    hull_gdf = hull_gdf.to_crs("EPSG:27700")
    out = osm.copy()
    for _, row in neighbors.to_crs("EPSG:27700").iterrows():
        code = str(row["lsoa21cd"])
        hit = out[out["lsoa21cd"].astype(str) == code]
        if hit.empty:
            continue
        if float(hit["road_length_km"].iloc[0]) > 0:
            continue
        dist = hull_gdf.geometry.distance(row.geometry)
        nearest = str(hull_gdf.iloc[int(dist.argmin())]["lsoa21cd"])
        src = hull_cov[hull_cov["lsoa21cd"].astype(str) == nearest].iloc[0]
        area = float(hit["lsoa_area_km2"].iloc[0])
        idx = hit.index[0]
        for col in density_cols:
            if col in out.columns:
                out.at[idx, col] = src[col]
        for col in count_cols:
            if col == "lsoa_area_km2" or col not in out.columns:
                continue
            dens_name = (
                col.replace("_count", "_per_km2")
                if col.endswith("_count")
                else "road_length_km_per_km2"
            )
            if dens_name in src.index and pd.notna(src[dens_name]):
                out.at[idx, col] = float(src[dens_name]) * area
            else:
                out.at[idx, col] = src[col]
        print(f"  OSM fallback for {code}: copied densities from nearest Hull LSOA {nearest}")
    return out


def _centroids_for_lsoas(
    codes: list[str],
    crimes: pd.DataFrame,
    neighbors: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Incident-median location when crimes exist; otherwise polygon centroid."""
    rows = []
    poly = neighbors.set_index(neighbors["lsoa21cd"].astype(str))
    crime_xy = None
    if not crimes.empty:
        valid = crimes.dropna(subset=["lsoa21cd", "longitude", "latitude"])
        if not valid.empty:
            crime_xy = valid.groupby("lsoa21cd")[["longitude", "latitude"]].median()
    for code in codes:
        if crime_xy is not None and code in crime_xy.index:
            lon = float(crime_xy.loc[code, "longitude"])
            lat = float(crime_xy.loc[code, "latitude"])
            east, north = lonlat_to_bng(pd.Series([lon]), pd.Series([lat]))
            rows.append(
                {
                    "lsoa21cd": code,
                    "centroid_easting": float(east[0]),
                    "centroid_northing": float(north[0]),
                }
            )
            continue
        geom = poly.loc[code, "geometry"]
        if isinstance(geom, gpd.GeoSeries):
            geom = geom.iloc[0]
        pt = geom.centroid
        rows.append(
            {
                "lsoa21cd": code,
                "centroid_easting": float(pt.x),
                "centroid_northing": float(pt.y),
            }
        )
    return pd.DataFrame(rows)


def build_neighbor_panel(
    neighbors: gpd.GeoDataFrame,
    covariates: pd.DataFrame,
    hull_panel: pd.DataFrame,
    *,
    police_source: Path | None = None,
) -> pd.DataFrame:
    """LSOA-month panel for neighbor codes, aligned to Hull panel months."""
    codes = sorted(neighbors["lsoa21cd"].astype(str).unique())
    names = (
        neighbors.drop_duplicates("lsoa21cd")
        .set_index("lsoa21cd")["lsoa21nm"]
        .astype(str)
        .to_dict()
    )
    source = police_source or POLICE_ZIP
    if not source.exists():
        raise FileNotFoundError(f"Missing police data: {source}")

    usable_months = pd.to_datetime(sorted(hull_panel["month"].unique()))
    # Need six months of history for rolling_mean_6 before the first usable month.
    history_start = usable_months.min() - pd.DateOffset(months=6)
    months = pd.date_range(history_start, usable_months.max(), freq="MS")

    print("Loading Police.uk street records for neighbor LSOAs...")
    crimes = load_crimes_for_lsoas(source, codes)
    counts = (
        crimes.groupby(["lsoa21cd", "month"]).size().rename("crime_count")
        if not crimes.empty
        else pd.Series(dtype=float, name="crime_count")
    )
    grid = pd.MultiIndex.from_product([codes, months], names=["lsoa21cd", "month"])
    panel = counts.reindex(grid, fill_value=0).reset_index()
    panel["lsoa21nm"] = panel["lsoa21cd"].map(names)

    locations = _centroids_for_lsoas(codes, crimes, neighbors)
    panel = panel.merge(locations, on="lsoa21cd", how="left", validate="many_to_one")

    grouped = panel.groupby("lsoa21cd", group_keys=False)["crime_count"]
    for lag in (1, 2, 3):
        panel[f"crime_lag_{lag}"] = grouped.shift(lag)
    for window in (3, 6):
        panel[f"crime_rolling_mean_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=w).mean()
        )
    angle = 2 * math.pi * (panel["month"].dt.month - 1) / 12
    panel["month_sin"] = np.sin(angle)
    panel["month_cos"] = np.cos(angle)
    panel["target_next_month_crime"] = grouped.shift(-1)

    extra = covariates.copy()
    duplicate = set(extra.columns) & set(panel.columns) - {"lsoa21cd"}
    if duplicate:
        raise ValueError(f"Covariates duplicate reserved columns: {sorted(duplicate)}")
    panel = panel.merge(extra, on="lsoa21cd", how="left", validate="many_to_one")
    covariate_cols = [c for c in extra.columns if c != "lsoa21cd"]
    panel = panel.dropna(
        subset=DERIVED_FEATURES + SPATIAL_COLUMNS + covariate_cols + ["target_next_month_crime"]
    )
    panel = panel[panel["month"].isin(usable_months)].copy()
    panel = panel.sort_values(["month", "lsoa21cd"]).reset_index(drop=True)
    NEIGHBOR_PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(NEIGHBOR_PANEL, index=False)
    print(f"Wrote {NEIGHBOR_PANEL} ({panel['lsoa21cd'].nunique()} LSOAs, {len(panel)} rows).")
    return panel


def expand_panel(hull_panel: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Concatenate Hull rows with selected neighbor rows; preserve Hull column order."""
    extra = extra.copy()
    extra["month"] = pd.to_datetime(extra["month"])
    hull = hull_panel.copy()
    hull["month"] = pd.to_datetime(hull["month"])
    missing = [c for c in hull.columns if c not in extra.columns]
    if missing:
        raise ValueError(f"Neighbor panel missing Hull columns: {missing}")
    extra = extra[hull.columns]
    out = pd.concat([hull, extra], ignore_index=True)
    return out.sort_values(["month", "lsoa21cd"]).reset_index(drop=True)


def prepare_cardinal_neighbor_data(hull_panel: pd.DataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Select four neighbors, collect police/covariate data, return extra panel."""
    hull_codes = sorted(hull_panel["lsoa21cd"].astype(str).unique())
    neighbors = load_or_select_neighbors(hull_codes)
    covariates = collect_neighbor_covariates(neighbors)
    extra_panel = build_neighbor_panel(neighbors, covariates, hull_panel)
    return neighbors, extra_panel
