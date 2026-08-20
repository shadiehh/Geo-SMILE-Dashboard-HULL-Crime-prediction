"""Load and validate published Hull GeoSMILE artefacts for the dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from dashboard.data.schema import (
    AREA_CSV,
    AREA_FIELD,
    ATT_STABILITY_JSON,
    BOUNDARIES_GEOJSON,
    EVAL_V3_JSON,
    FEATURE_CONSISTENCY_CSV,
    FEATURE_GLOBAL_PUBLISHED_CSV,
    FEATURE_GLOBAL_WITH_LOCATION_CSV,
    FEATURE_LOCAL_PUBLISHED_CSV,
    FEATURE_LOCAL_WITH_LOCATION_CSV,
    GEO_ABS,
    GEO_CONSISTENCY_CSV,
    GEO_CSV,
    GEO_SIGNED,
    GEO_TERTILE,
    LISA_CLASS,
    LISA_CSV,
    LISA_JSON,
    LOCAL_MORAN_I,
    LOCAL_P,
    LOCATION_FEATURE_NAMES,
    LOCATION_LOCAL_CSV,
    LSOA_ID,
    LSOA_NAME,
    TWOBRANCH_SUMMARY_JSON,
    human_label,
    is_location_feature,
)

HULL_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class HullBundle:
    lsoa: pd.DataFrame
    geometry: gpd.GeoDataFrame
    local_features: pd.DataFrame
    global_features: pd.DataFrame
    global_features_with_location: pd.DataFrame
    spatial_pattern: dict
    evaluation: dict
    diagnostics: dict
    feature_names: list[str] = field(default_factory=list)


def _read_csv(rel: str) -> pd.DataFrame:
    path = HULL_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing required artefact: {path}")
    return pd.read_csv(path)


def _read_json(rel: str) -> dict:
    path = HULL_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing required artefact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    return value


def top_features_for_lsoa(
    local_features: pd.DataFrame,
    lsoa_id: str,
    k: int = 10,
) -> list[dict]:
    """Rank non-location local SMILE coefficients by absolute magnitude."""
    row = local_features.loc[local_features[LSOA_ID].astype(str) == str(lsoa_id)]
    if row.empty:
        return []
    row = row.iloc[0]
    skip = {LSOA_ID, LSOA_NAME}
    records = []
    for name, value in row.items():
        if name in skip or is_location_feature(str(name)):
            continue
        if not np.isscalar(value) or pd.isna(value):
            continue
        val = float(value)
        records.append(
            {
                "feature": str(name),
                "feature_label": human_label(str(name)),
                "feature_value": val,
                "abs_value": abs(val),
            }
        )
    records.sort(key=lambda r: (-r["abs_value"], r["feature"]))
    out = []
    for i, rec in enumerate(records[:k], start=1):
        rec["feature_rank"] = i
        out.append(rec)
    return out


def _cluster_lookup(pattern: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    local = pattern.get("LOCAL_SPATIAL_PATTERN") or {}
    for kind_key, kind in (
        ("contiguous_HH_clusters", "HH"),
        ("contiguous_LL_clusters", "LL"),
    ):
        for cluster in local.get(kind_key) or []:
            cid = cluster.get("cluster_id")
            codes = [str(c) for c in cluster.get("lsoa21cd") or []]
            n = int(cluster.get("n") or len(codes))
            for code in codes:
                out[code] = {
                    "cluster_id": f"{kind}-{cid}",
                    "cluster_kind": kind,
                    "cluster_size": n,
                    "cluster_members": codes,
                }
    return out


def validate_bundle(lsoa: pd.DataFrame, geometry: gpd.GeoDataFrame) -> dict:
    ids = lsoa[LSOA_ID].astype(str)
    diag = {
        "n_lsoas": int(len(lsoa)),
        "duplicate_lsoa_rows": int(ids.duplicated().sum()),
        "duplicate_polygons": int(geometry[LSOA_ID].astype(str).duplicated().sum()),
        "unmatched_results": sorted(set(ids) - set(geometry[LSOA_ID].astype(str))),
        "unmatched_polygons": sorted(set(geometry[LSOA_ID].astype(str)) - set(ids)),
        "geo_sensitivity_na": int(lsoa["geo_sensitivity"].isna().sum()),
        "latitude_effect_na": int(lsoa["latitude_effect"].isna().sum()),
        "longitude_effect_na": int(lsoa["longitude_effect"].isna().sum()),
        "lisa_class_counts": lsoa["lisa_class"].value_counts(dropna=False).to_dict(),
        "negative_geo_preserved": bool((lsoa["geo_sensitivity"] < 0).any()),
        "local_p_adjusted": "not available in published LISA table",
        "pattern_scale": "not established",
    }
    if diag["duplicate_lsoa_rows"] or diag["duplicate_polygons"]:
        raise ValueError("Join would duplicate polygons; check LSOA identifiers")
    if diag["unmatched_results"] or diag["unmatched_polygons"]:
        raise ValueError(
            f"LSOA join is not one-to-one: missing results={diag['unmatched_results'][:8]} "
            f"missing polygons={diag['unmatched_polygons'][:8]}"
        )
    return diag


def load_hull_bundle(root: Path | None = None) -> HullBundle:
    root = Path(root) if root is not None else HULL_ROOT
    geo = _read_csv(GEO_CSV) if root == HULL_ROOT else pd.read_csv(root / GEO_CSV)
    loc = pd.read_csv(root / LOCATION_LOCAL_CSV)
    lisa = pd.read_csv(root / LISA_CSV)
    local_feat = pd.read_csv(root / FEATURE_LOCAL_WITH_LOCATION_CSV)
    published_local = pd.read_csv(root / FEATURE_LOCAL_PUBLISHED_CSV)
    global_feat = pd.read_csv(root / FEATURE_GLOBAL_PUBLISHED_CSV)
    global_loc = pd.read_csv(root / FEATURE_GLOBAL_WITH_LOCATION_CSV)
    pattern = json.loads((root / LISA_JSON).read_text(encoding="utf-8"))
    area = pd.read_csv(root / AREA_CSV)[[LSOA_ID, AREA_FIELD]]
    boundaries = gpd.read_file(root / BOUNDARIES_GEOJSON)

    for frame in (geo, loc, lisa, local_feat, published_local, area):
        frame[LSOA_ID] = frame[LSOA_ID].astype(str)
    boundaries[LSOA_ID] = boundaries[LSOA_ID].astype(str)

    geo = geo.sort_values(GEO_ABS, ascending=False, kind="mergesort").reset_index(drop=True)
    geo["geo_rank"] = np.arange(1, len(geo) + 1)

    clusters = _cluster_lookup(pattern)
    lsoa = geo[[LSOA_ID, LSOA_NAME, GEO_SIGNED, GEO_ABS, "geo_rank"]].rename(
        columns={
            LSOA_ID: LSOA_ID,
            LSOA_NAME: LSOA_NAME,
            GEO_SIGNED: "geo_sensitivity",
            GEO_ABS: "geo_sensitivity_abs",
        }
    )
    lsoa = lsoa.merge(
        loc[[LSOA_ID, LOCATION_FEATURE_NAMES[0], LOCATION_FEATURE_NAMES[1]]].rename(
            columns={
                LOCATION_FEATURE_NAMES[0]: "longitude_effect",
                LOCATION_FEATURE_NAMES[1]: "latitude_effect",
            }
        ),
        on=LSOA_ID,
        how="inner",
        validate="one_to_one",
    )
    lsoa = lsoa.merge(
        lisa[[LSOA_ID, LISA_CLASS, LOCAL_MORAN_I, LOCAL_P, GEO_TERTILE]].rename(
            columns={
                LISA_CLASS: "lisa_class",
                LOCAL_MORAN_I: "local_moran_i",
                LOCAL_P: "local_p",
                GEO_TERTILE: "geo_tertile",
            }
        ),
        on=LSOA_ID,
        how="inner",
        validate="one_to_one",
    )
    lsoa = lsoa.merge(area, on=LSOA_ID, how="left", validate="one_to_one")
    lsoa["cluster_id"] = lsoa[LSOA_ID].map(lambda c: (clusters.get(c) or {}).get("cluster_id"))
    lsoa["cluster_kind"] = lsoa[LSOA_ID].map(lambda c: (clusters.get(c) or {}).get("cluster_kind"))
    lsoa["cluster_size"] = lsoa[LSOA_ID].map(lambda c: (clusters.get(c) or {}).get("cluster_size"))
    lsoa["cluster_members"] = lsoa[LSOA_ID].map(
        lambda c: (clusters.get(c) or {}).get("cluster_members")
    )

    if boundaries.crs is None:
        boundaries = boundaries.set_crs("EPSG:27700")
    geometry = boundaries[[LSOA_ID, "geometry"]].drop_duplicates(LSOA_ID)
    geometry = geometry.to_crs("EPSG:4326")
    geometry["geometry"] = geometry.geometry.simplify(0.00015, preserve_topology=True)
    geometry = geometry.merge(
        lsoa,
        on=LSOA_ID,
        how="inner",
        validate="one_to_one",
    )

    diagnostics = validate_bundle(lsoa, geometry.drop(columns="geometry"))
    diagnostics["n_polygons"] = int(len(geometry))
    diagnostics["sources"] = {
        "geo_sensitivity": str(GEO_CSV),
        "latitude_effect": str(LOCATION_LOCAL_CSV),
        "longitude_effect": str(LOCATION_LOCAL_CSV),
        "local_top_features": str(FEATURE_LOCAL_WITH_LOCATION_CSV),
        "spatial_clustering": str(LISA_CSV),
        "boundaries": str(BOUNDARIES_GEOJSON),
        "feature_consistency": str(FEATURE_CONSISTENCY_CSV),
        "geo_consistency": str(GEO_CONSISTENCY_CSV),
    }

    evaluation = {
        "v3": json.loads((root / EVAL_V3_JSON).read_text(encoding="utf-8"))
        if (root / EVAL_V3_JSON).exists()
        else {},
        "att_stability": json.loads((root / ATT_STABILITY_JSON).read_text(encoding="utf-8"))
        if (root / ATT_STABILITY_JSON).exists()
        else {},
        "twobranch": json.loads((root / TWOBRANCH_SUMMARY_JSON).read_text(encoding="utf-8"))
        if (root / TWOBRANCH_SUMMARY_JSON).exists()
        else {},
        "consistency": {
            "feature": _consistency_from_csv(root / FEATURE_CONSISTENCY_CSV),
            "geo": _consistency_from_csv(root / GEO_CONSISTENCY_CSV),
        },
    }

    feature_names = [
        c
        for c in local_feat.columns
        if c not in {LSOA_ID, LSOA_NAME} and not is_location_feature(c)
    ]
    # published local is unused in the UI but kept for diagnostics / tests
    _ = published_local

    return HullBundle(
        lsoa=lsoa,
        geometry=geometry,
        local_features=local_feat,
        global_features=global_feat,
        global_features_with_location=global_loc,
        spatial_pattern=pattern,
        evaluation=evaluation,
        diagnostics=diagnostics,
        feature_names=feature_names,
    )


def _consistency_from_csv(path: Path) -> dict:
    """Parse the one-row published multi-run consistency table."""
    if not path.exists():
        return {}
    row = pd.read_csv(path).iloc[0].to_dict()
    keys = {
        "Rank Spearman Mean": "rank_spearman_mean",
        "Rank Spearman Std": "rank_spearman_std",
        "Score Pearson Mean": "score_pearson_mean",
        "Score Pearson Std": "score_pearson_std",
        "Top-20% Jaccard Mean": "top20pct_jaccard_mean",
        "Top-20% Jaccard Std": "top20pct_jaccard_std",
        "Fidelity R2 Mean": "fidelity_r2_mean",
        "Fidelity R2 Std": "fidelity_r2_std",
    }
    out = {}
    for raw, name in keys.items():
        if raw in row and pd.notna(row[raw]):
            out[name] = _json_safe(row[raw])
    if "branch" in row and pd.notna(row["branch"]):
        out["branch"] = str(row["branch"])
    return out


def compact_evaluation(evaluation: dict) -> dict:
    v3 = evaluation.get("v3") or {}
    att = evaluation.get("att_stability") or {}
    two = evaluation.get("twobranch") or {}
    out = {
        "not_causal": True,
        "feature_holdout_r2": _json_safe((two.get("model_metrics") or {}).get("feature", {}).get("r2")),
        "geo_holdout_r2": _json_safe((two.get("model_metrics") or {}).get("geo", {}).get("r2")),
        "feature_surrogate_test_r2": _json_safe(
            (two.get("feature_surrogate") or {}).get("test_r2")
        ),
        "geo_surrogate_test_r2": _json_safe((two.get("geo_surrogate") or {}).get("test_r2")),
        "att_stability_feature_jaccard": 1.0 if att.get("status") == "Evaluated" else None,
        "att_stability_geo_jaccard": 1.0 if att.get("status") == "Evaluated" else None,
        "att_stability_status": att.get("status"),
        "att_stability_note": (
            "Controlled-input ATT stability. Distinct from multi-run Consistency."
        ),
    }
    consist = evaluation.get("consistency") or {}
    feat_c = consist.get("feature") or {}
    geo_c = consist.get("geo") or {}
    if feat_c or geo_c:
        out["consistency"] = {
            "definition": (
                "Multi-run explanation consistency across seeds "
                "(consistency_seed_base=100). Pairwise Rank Spearman, "
                "Score Pearson, and Top-20% Jaccard of |coefficients|. "
                "Not controlled-input ATT stability."
            ),
            "feature": feat_c or None,
            "geo": geo_c or None,
        }
    faith = v3.get("model_based_attribution_faithfulness") or {}
    if faith.get("feature"):
        out["feature_faithfulness_top_minus_random"] = _json_safe(
            faith["feature"].get("top_minus_random")
        )
        out["feature_faithfulness_empirical_p"] = _json_safe(
            faith["feature"].get("empirical_p_top_vs_random")
        )
    if faith.get("geo"):
        out["geo_faithfulness_top_minus_random"] = _json_safe(
            faith["geo"].get("top_minus_random")
        )
        out["geo_faithfulness_empirical_p"] = _json_safe(
            faith["geo"].get("empirical_p_top_vs_random")
        )
    return {k: v for k, v in out.items() if v is not None}


def magnitude_class(abs_value: float, median: float, p90: float) -> str:
    if abs_value >= max(p90, 5.0):
        return "large"
    if abs_value >= max(median, 1.0):
        return "moderate"
    return "modest"


def run_diagnostics(root: Path | None = None) -> dict:
    bundle = load_hull_bundle(root)
    sample = top_features_for_lsoa(bundle.local_features, str(bundle.lsoa.iloc[0][LSOA_ID]))
    loc_in_top = [r["feature"] for r in sample if is_location_feature(r["feature"])]
    bundle.diagnostics["sample_top10_location_leak"] = loc_in_top
    bundle.diagnostics["sample_top10_n"] = len(sample)
    return bundle.diagnostics
