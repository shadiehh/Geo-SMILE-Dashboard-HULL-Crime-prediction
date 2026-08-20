"""Canonical field map for the Hull stakeholder dashboard.

Maps repository column names onto dashboard fields in one place.
Only fields that exist in the published Hull artefacts are included.
"""

from __future__ import annotations

from geosmile_hull_twobranch import FEATURE_LABELS as _FEATURE_LABELS

# Join key used by every Hull GeoSMILE table and the ONS LSOA 2021 polygons.
LSOA_ID = "lsoa21cd"
LSOA_NAME = "lsoa21nm"

# Published Geo branch (location-masking SMILE).
GEO_CSV = "outputs/geo_coefficients.csv"
GEO_SIGNED = "geo_coefficient"
GEO_ABS = "geo_abs_coefficient"

# Feature SMILE with lon/lat as attribution players (distinct from Geo branch).
LOCATION_LOCAL_CSV = "outputs/feature_with_location/location_local_coefficients.csv"
FEATURE_LOCAL_WITH_LOCATION_CSV = (
    "outputs/feature_with_location/feature_with_location_local_coefficients.csv"
)
FEATURE_GLOBAL_WITH_LOCATION_CSV = (
    "outputs/feature_with_location/feature_with_location_global_importance.csv"
)
LONGITUDE_EFFECT = "centroid_longitude"
LATITUDE_EFFECT = "centroid_latitude"
LOCATION_FEATURE_NAMES = (LONGITUDE_EFFECT, LATITUDE_EFFECT)

# Published Feature branch (coordinates excluded from attribution).
FEATURE_LOCAL_PUBLISHED_CSV = "outputs/feature_local_coefficients.csv"
FEATURE_GLOBAL_PUBLISHED_CSV = "outputs/feature_global_importance.csv"

# LISA on |Geo SMILE coefficient|. The source column named geo_sensitivity
# is a HIGH/MID/LOW tertile, not the numeric GeoSensitivity value.
LISA_CSV = "outputs/spatial_patterns/lisa_lsoa.csv"
LISA_JSON = "outputs/spatial_patterns/llm_spatial_pattern.json"
LISA_CLASS = "lisa"
LOCAL_MORAN_I = "local_I"
LOCAL_P = "p_value"
GEO_TERTILE = "geo_sensitivity"  # HIGH / MID / LOW in lisa_lsoa.csv

BOUNDARIES_GEOJSON = "data/hull_lsoa_2021_official.geojson"
AREA_CSV = "data/hull_static_covariates.csv"
AREA_FIELD = "lsoa_area_km2"
EVAL_V3_JSON = "outputs/geosmile_hull_evaluation_summary_v3.json"
ATT_STABILITY_JSON = "outputs/geosmile_hull_att_stability.json"
TWOBRANCH_SUMMARY_JSON = "outputs/run_summary_twobranch.json"

LOCATION_TOKENS = (
    "coord",
    "easting",
    "northing",
    "latitude",
    "longitude",
    "location",
    "centroid",
)

FEATURE_LABELS = {
    **_FEATURE_LABELS,
    LONGITUDE_EFFECT: "Longitude",
    LATITUDE_EFFECT: "Latitude",
}

LISA_LABELS = {
    "HH": "High–High",
    "LL": "Low–Low",
    "HL": "High–Low",
    "LH": "Low–High",
    "NS": "Not significant",
    "ISLAND": "Island (no neighbours)",
}

MAP_VIEWS = {
    "geo_sensitivity": "GeoSensitivity",
    "latitude_effect": "Latitude feature effect",
    "longitude_effect": "Longitude feature effect",
    "spatial_pattern": "Spatial pattern",
}

STAKEHOLDERS = {
    "technical": "Technical / Model Analyst",
    "domain": "Domain / Operational Expert",
    "decision": "Decision Maker / Manager",
    "governance": "Governance / Assurance",
    "public": "Public / Affected Stakeholder",
}

STAKEHOLDER_HELP = {
    "technical": "Exact values, ranks, signed effects, spatial statistics, and methodological limits.",
    "domain": "What stands out, where, local non-location features, and what to investigate. Not causal.",
    "decision": "A few findings, practical magnitude, evidence strength, and a clear limitation.",
    "governance": "Provenance, geographic dependence, stability, uncertainty, and unsupported-inference risk.",
    "public": "Plain language. What the result means, and especially what it does not mean.",
}

PROVIDERS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "compare": "Compare both",
}


def human_label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    return str(feature).replace("_", " ")


def is_location_feature(name: str) -> bool:
    low = str(name).lower()
    return any(tok in low for tok in LOCATION_TOKENS)
