"""Canonical field map for the Hull stakeholder dashboard.

Maps repository column names onto dashboard fields in one place.
Only fields that exist in the published Hull artefacts are included.
"""

from __future__ import annotations

from geosmile_hull_twobranch import FEATURE_LABELS as _FEATURE_LABELS

# Join key used by every Hull GeoSMILE table and the ONS LSOA 2021 polygons.
LSOA_ID = "lsoa21cd"
LSOA_NAME = "lsoa21nm"

# Geo branch, random-donor design (Chapter 4): the Geo branch explains XGBoost on the 30
# predictors + BNG centroid easting/northing; masking an LSOA moves the coordinates of its
# rows to the centroid of a random other Hull LSOA (K = 50 draws), all other inputs kept.
# LSOAs are ranked by the SIGNED coefficient. Files written in the dashboard layout by
# build_random_donor_dashboard_files.py. The earlier area-data design
# (outputs/geo_area_data_eval) and coordinates-only design are no longer shown.
GEO_DESIGN = "random_donor"
GEO_DESIGN_DIR = "outputs/geo_random_donor_eval"
GEO_DESIGN_N = 4000
GEO_CSV = f"{GEO_DESIGN_DIR}/n_{GEO_DESIGN_N}/geo_coefficients.csv"
GEO_METRICS_JSON = f"{GEO_DESIGN_DIR}/n_{GEO_DESIGN_N}/metrics.json"
GEO_SIGNED = "geo_coefficient"
GEO_ABS = "geo_abs_coefficient"
GEO_RANK_FIELD = GEO_SIGNED

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

# LISA on the signed random-donor Geo coefficient (compute_geo_spatial_patterns.py
# --signed). The source column named geo_sensitivity is a HIGH/MID/LOW
# tertile, not the numeric GeoSensitivity value.
LISA_CSV = f"{GEO_DESIGN_DIR}/spatial_patterns/lisa_lsoa.csv"
LISA_JSON = f"{GEO_DESIGN_DIR}/spatial_patterns/llm_spatial_pattern.json"

# Orientation-only place labels per LSOA (build_lsoa_place_names.py, OSM).
PLACE_NAMES_CSV = "data/hull_lsoa_place_names.csv"
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
FEATURE_CONSISTENCY_CSV = "outputs/geosmile_hull_feat_consistency.csv"
GEO_CONSISTENCY_CSV = "outputs/geosmile_hull_geo_consistency.csv"

# Robustness sweep (Model Diagnostics tab): eval_by_n_perturb.py output.
EVAL_SWEEP_DIR = "outputs/eval_by_n_perturb"
EVAL_SWEEP_N_GRID = (1000, 2000, 3000, 4000, 5000, 7500, 10000)
PUBLISHED_N = {"feature": 1024, "geo": GEO_DESIGN_N}
EXPERT_ALIGNMENT_FEATURE_CSV = "outputs/geosmile_hull_expert_alignment_merged_feature.csv"
EXPERT_ALIGNMENT_GEO_CSV = "outputs/geosmile_hull_expert_alignment_merged_geo.csv"
PUBLISHED_FEATURE_FIDELITY_CSV = "outputs/geosmile_hull_feat_fidelity.csv"
PUBLISHED_GEO_FIDELITY_CSV = "outputs/geosmile_hull_geo_fidelity.csv"

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
    "overlay": "Lat / lon / GeoSensitivity overlay",
    "top50_rank": "GeoSensitivity top 50 + lat/lon overlay",
    "spatial_pattern": "Spatial pattern",
}

STAKEHOLDERS = {
    "public": "Resident / Public",
    "planner": "City Planner / Local Authority",
    "police": "Police / Community Safety",
    "manager": "Manager / Strategic Decision Maker",
    "researcher": "GeoAI Researcher / Model Analyst",
    "governance": "Governance / Assurance",
}

STAKEHOLDER_HELP = {
    "public": "Which features matter most and least to the model, in plain words. Not why crime occurs.",
    "planner": "Where GeoSensitivity is highest and lowest across Hull, and where a place fits in that pattern.",
    "police": "Crime-history signals and areas that stand out for closer contextual review. No deployment advice.",
    "manager": "The most decision-relevant findings, their practical magnitude, and an evidence next step.",
    "researcher": "Exact values, ranks, signed effects, spatial statistics, and explanation-quality diagnostics.",
    "governance": "Provenance, robustness, stability, unusual behaviour, and unsupported-inference risk.",
}

OUTPUT_MODES = ("dashboard_mode", "comparison_table_mode")

PROVIDERS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
}


def human_label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    return str(feature).replace("_", " ")


def is_location_feature(name: str) -> bool:
    low = str(name).lower()
    return any(tok in low for tok in LOCATION_TOKENS)
