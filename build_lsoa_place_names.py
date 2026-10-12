"""Orientation labels for every Hull LSOA: nearest locality + main named streets.

One Overpass query fetches OSM place nodes (suburb / quarter / neighbourhood)
and named highways for the Hull bounding box. Each LSOA gets:

  locality_label    nearest suburb/quarter node to the LSOA centroid
  nearby_localities up to 3 place nodes (any of the three types) inside or
                    within 300 m of the polygon, nearest first
  major_streets     up to 6 named streets with the longest length inside it

Labels are orientation only (the dashboard prompt forbids inferring anything
else from them). Writes data/hull_lsoa_place_names.csv; the dashboard reads
that file and never calls the network.

    python build_lsoa_place_names.py
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parent
BOUNDARIES = ROOT / "data" / "hull_lsoa_2021_official.geojson"
OUT = ROOT / "data" / "hull_lsoa_place_names.csv"
OVERPASS = "https://overpass-api.de/api/interpreter"
MAX_STREETS = 6
MIN_STREET_M = 60.0
SKIP_STREET_PREFIXES = ("Footpath", "Access to")


def fetch(bounds) -> list[dict]:
    w, s, e, n = bounds
    bbox = f"{s - 0.01},{w - 0.01},{n + 0.01},{e + 0.01}"
    query = (
        f'[out:json][timeout:180];('
        f'node["place"~"^(suburb|quarter|neighbourhood)$"]["name"]({bbox});'
        f'way["highway"~"^(primary|secondary|tertiary|unclassified|residential|pedestrian|trunk|living_street)$"]'
        f'["name"]({bbox}););out geom;'
    )
    resp = requests.post(OVERPASS, data={"data": query}, timeout=240,
                         headers={"User-Agent": "hull-geosmile-thesis"})
    resp.raise_for_status()
    return resp.json()["elements"]


def main() -> None:
    lsoa = gpd.read_file(BOUNDARIES).to_crs(4326)
    elements = fetch(lsoa.total_bounds)

    places = [
        {"name": el["tags"]["name"], "place": el["tags"]["place"],
         "geometry": Point(el["lon"], el["lat"])}
        for el in elements if el["type"] == "node"
    ]
    roads = [
        {"name": el["tags"]["name"],
         "geometry": LineString([(p["lon"], p["lat"]) for p in el["geometry"]])}
        for el in elements
        if el["type"] == "way" and len(el.get("geometry", [])) > 1
    ]
    places = gpd.GeoDataFrame(places, crs=4326).to_crs(27700)
    roads = gpd.GeoDataFrame(roads, crs=4326).to_crs(27700)
    lsoa = lsoa.to_crs(27700)
    major = places[places["place"].isin(["suburb", "quarter"])]

    rows = []
    for rec in lsoa.itertuples():
        poly = rec.geometry
        centroid = poly.centroid
        locality = major.loc[major.distance(centroid).idxmin(), "name"]

        near = places[places.distance(poly) <= 300].copy()
        near["d"] = near.distance(centroid)
        nearby = list(dict.fromkeys(near.sort_values("d")["name"]))[:3]

        hit = roads[roads.intersects(poly)].copy()
        hit["len_m"] = hit.geometry.intersection(poly).length
        lengths = hit.groupby("name")["len_m"].sum().sort_values(ascending=False)
        streets = [
            name for name, length in lengths.items()
            if length >= MIN_STREET_M and not name.startswith(SKIP_STREET_PREFIXES)
        ][:MAX_STREETS]

        rows.append({
            "lsoa21cd": rec.lsoa21cd,
            "lsoa21nm": rec.lsoa21nm,
            "locality_label": locality,
            "nearby_localities": "; ".join(nearby),
            "major_streets": "; ".join(streets),
        })

    out = pd.DataFrame(rows).sort_values("lsoa21cd")
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(out)} LSOAs, "
          f"{(out['major_streets'] == '').sum()} without streets)")


if __name__ == "__main__":
    main()
