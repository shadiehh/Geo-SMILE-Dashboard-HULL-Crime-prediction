"""Evidence for the stakeholder figure panels (Chapter 6 design), for ANY selected LSOA.

Loads the published random-donor Geo results and the published Feature branch once, and
exposes per-LSOA facts used by the figure captions. Every number shown next to a
stakeholder narrative comes from here, so the dashboard and the thesis agree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from dashboard.data.schema import BOUNDARIES_GEOJSON, GEO_DESIGN_DIR, GEO_DESIGN_N, LISA_CSV, LISA_JSON, PLACE_NAMES_CSV

HULL_ROOT = Path(__file__).resolve().parents[2]
GEO_RANKED_CSV = f"{GEO_DESIGN_DIR}/geo_coefficients_N{GEO_DESIGN_N}.csv"  # signed beta + exact single-LSOA W1
GEO_CONSISTENCY_CSV = f"{GEO_DESIGN_DIR}/n_{GEO_DESIGN_N}/consistency_per_lsoa.csv"
GEO_METRICS_JSON = f"{GEO_DESIGN_DIR}/n_{GEO_DESIGN_N}/metrics.json"
STABILITY_FIGURE = f"{GEO_DESIGN_DIR}/stability_four_directions_random_donor.png"
FEATURE_LOCAL_CSV = "outputs/feature_local_coefficients.csv"  # published Feature branch (no coordinates)
FEATURE_GLOBAL_CSV = "outputs/feature_global_importance.csv"
PANEL_CSV = "outputs/hull_lsoa_month_panel_with_covariates.csv"
REGIONS_CSV = "outputs/geo_area_data_models/top50_maps/highlighted_regions_lsoas.csv"
TEST_START = pd.Timestamp("2026-01-01")
TOP_K = 20

# Hand-checked labels for the thesis case studies; others use the OSM locality label.
PLACE_OVERRIDES = {
    "E01033104": "city centre / Old Town",
    "E01012877": "Orchard Park (around Danepark Road)",
}


@dataclass
class StakeholderData:
    geometry: gpd.GeoDataFrame          # BNG polygons with every per-LSOA column below
    geo: pd.DataFrame                   # lsoa21cd, lsoa21nm, geo_coefficient, geo_rank, exact_w1, lisa, ...
    local: pd.DataFrame                 # signed local Feature coefficients, index lsoa21cd
    consistency: pd.DataFrame           # per-LSOA seed runs, index lsoa21cd
    crime: pd.DataFrame                 # lsoa21cd, month, crime_count, crime_rolling_mean_6
    regions: set
    places: pd.DataFrame                # index lsoa21cd
    metrics: dict
    moran: dict
    clusters: dict                      # lsoa -> (kind, cluster_id, members)
    stability_png: Path

    # ------------------------------------------------------------------ helpers
    @property
    def n(self) -> int:
        return len(self.geo)

    def row(self, lsoa: str) -> pd.Series:
        return self.geo.loc[lsoa]

    def short(self, lsoa: str) -> str:
        return str(self.geo.loc[lsoa, "lsoa21nm"]).replace("Kingston upon Hull ", "")

    def place(self, lsoa: str) -> str:
        if lsoa in PLACE_OVERRIDES:
            return PLACE_OVERRIDES[lsoa]
        if lsoa in self.places.index:
            label = str(self.places.loc[lsoa, "locality_label"] or "")
            if label:
                return label
        return ""

    def streets(self, lsoa: str, k: int = 4) -> list[str]:
        if lsoa not in self.places.index:
            return []
        raw = str(self.places.loc[lsoa, "major_streets"] or "")
        return [s.strip() for s in raw.split(";") if s.strip()][:k]

    def abs_local(self, lsoa: str) -> pd.Series:
        return self.local.loc[lsoa].abs().sort_values(ascending=False)

    def crime_share(self, lsoa: str) -> float:
        a = self.abs_local(lsoa)
        return float(a[[c for c in a.index if c.startswith("crime_")]].sum() / a.sum())

    def default_contrast(self, lsoa: str) -> str:
        """Planner contrast: a low-sensitivity case for a high one, else the top case."""
        g = self.geo
        if g.loc[lsoa, "geo_coefficient"] > g["geo_coefficient"].median():
            ll = [m for kind, _, m in self.clusters.values() if kind == "LL"]
            members = max(ll, key=len) if ll else list(g.sort_values("geo_rank").index[-10:])
            return min(members, key=lambda c: g.loc[c, "geo_coefficient"])
        return str(g.sort_values("geo_rank").index[0])


def _clusters(pattern: dict) -> dict:
    out = {}
    local = pattern.get("LOCAL_SPATIAL_PATTERN", {})
    for kind in ("HH", "LL"):
        for c in local.get(f"contiguous_{kind}_clusters", []):
            members = [str(m) for m in c.get("lsoa21cd", [])]
            for m in members:
                out[m] = (kind, int(c.get("cluster_id", 0)), members)
    return out


def load_stakeholder_data(root: Path | None = None) -> StakeholderData:
    root = Path(root) if root is not None else HULL_ROOT
    geo = pd.read_csv(root / GEO_RANKED_CSV, dtype={"lsoa21cd": str})
    geo = geo.sort_values(["geo_coefficient", "lsoa21cd"], ascending=[False, True]).reset_index(drop=True)
    geo["geo_rank"] = np.arange(1, len(geo) + 1)
    lisa = pd.read_csv(root / LISA_CSV, dtype={"lsoa21cd": str})[["lsoa21cd", "lisa", "local_I", "p_value"]]
    geo = geo.merge(lisa, on="lsoa21cd", how="left")
    regions = set(pd.read_csv(root / REGIONS_CSV, dtype={"lsoa21cd": str})["lsoa21cd"])
    geo["expert_region"] = geo["lsoa21cd"].isin(regions)
    geo = geo.set_index("lsoa21cd", drop=False)

    local = pd.read_csv(root / FEATURE_LOCAL_CSV, dtype={"lsoa21cd": str}).set_index("lsoa21cd").drop(columns="lsoa21nm")
    cons = pd.read_csv(root / GEO_CONSISTENCY_CSV, dtype={"lsoa21cd": str}).set_index("lsoa21cd")
    crime = pd.read_csv(root / PANEL_CSV, usecols=["lsoa21cd", "month", "crime_count", "crime_rolling_mean_6"],
                        dtype={"lsoa21cd": str}, parse_dates=["month"])
    places = pd.read_csv(root / PLACE_NAMES_CSV, dtype={"lsoa21cd": str}).fillna("").set_index("lsoa21cd")
    pattern = json.loads((root / LISA_JSON).read_text(encoding="utf-8"))
    metrics = json.loads((root / GEO_METRICS_JSON).read_text(encoding="utf-8"))

    shapes = gpd.read_file(root / BOUNDARIES_GEOJSON)[["lsoa21cd", "geometry"]]
    shapes["lsoa21cd"] = shapes["lsoa21cd"].astype(str)
    if shapes.crs is None:
        shapes = shapes.set_crs("EPSG:27700")
    geometry = shapes.to_crs("EPSG:27700").merge(geo.reset_index(drop=True), on="lsoa21cd", how="inner")

    return StakeholderData(
        geometry=geometry, geo=geo, local=local, consistency=cons, crime=crime, regions=regions,
        places=places, metrics=metrics, moran=pattern.get("GLOBAL_SPATIAL_PATTERN", {}),
        clusters=_clusters(pattern), stability_png=root / STABILITY_FIGURE,
    )
