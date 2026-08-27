"""New derived results from GeoSensitivity + per-LSOA local feature attribution.

Pure pandas/numpy/sklearn: dominant local feature per LSOA, local-attribution
concentration (entropy), and k-means "explanation archetypes". No Streamlit
dependency, so these stay directly unit-testable. GeoSensitivity (Geo branch)
and local feature attribution (Feature branch) stay conceptually separate
objects here too: nothing in this module blends them into one score, it only
juxtaposes their magnitudes (see dashboard/views/attribution_patterns.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from dashboard.data.schema import LSOA_ID, human_label


def dominant_local_feature(local_features: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Per-LSOA argmax |coefficient| among the given local feature columns."""
    ids = local_features[LSOA_ID].astype(str).to_numpy()
    values = local_features[feature_names].to_numpy(dtype=float)
    abs_values = np.abs(values)
    idx = np.argmax(abs_values, axis=1)
    names = np.asarray(feature_names)
    dominant = names[idx]
    signed = values[np.arange(len(values)), idx]
    return pd.DataFrame(
        {
            LSOA_ID: ids,
            "dominant_feature": dominant,
            "dominant_feature_label": [human_label(f) for f in dominant],
            "dominant_signed_value": signed,
            "dominant_abs_value": np.abs(signed),
        }
    )


def local_attribution_concentration(local_features: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Per-LSOA normalized entropy of |coefficient| shares.

    0 = one feature accounts for (almost) all of the local explanation
    (concentrated). 1 = the |coefficient| mass is spread evenly across all
    features (diffuse). Also reports the "effective number of features"
    (exp(entropy)), a perplexity-style count of how many features are
    effectively driving the explanation.
    """
    ids = local_features[LSOA_ID].astype(str).to_numpy()
    values = np.abs(local_features[feature_names].to_numpy(dtype=float))
    totals = values.sum(axis=1, keepdims=True)
    totals_safe = np.where(totals <= 0, 1.0, totals)
    shares = values / totals_safe
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(shares > 0, shares * np.log(shares), 0.0)
    entropy = -terms.sum(axis=1)
    max_entropy = np.log(len(feature_names))
    entropy_norm = entropy / max_entropy
    effective_n = np.exp(entropy)
    return pd.DataFrame(
        {
            LSOA_ID: ids,
            "entropy_norm": entropy_norm,
            "effective_n_features": effective_n,
            "n_features": len(feature_names),
        }
    )


def attribution_archetypes(
    local_features: pd.DataFrame,
    feature_names: list[str],
    *,
    k: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-means clusters on standardized signed local coefficients ("explanation archetypes").

    Groups LSOAs whose local explanations share a similar feature profile,
    independent of geographic adjacency (distinct from the existing LISA
    clustering, which groups by spatial adjacency of |GeoSensitivity|).

    Returns (labels: one row per LSOA with its cluster id, profiles: one row
    per cluster with its size and top-3 driving features by mean |coefficient|).
    """
    ids = local_features[LSOA_ID].astype(str).to_numpy()
    values = local_features[feature_names].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(values)
    k = max(2, min(int(k), len(values) - 1))
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = model.fit_predict(scaled)

    profiles = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        mean_coef = values[mask].mean(axis=0) if mask.any() else np.zeros(len(feature_names))
        order = np.argsort(-np.abs(mean_coef))[:3]
        top_features = [
            {
                "feature": feature_names[i],
                "feature_label": human_label(feature_names[i]),
                "mean_coefficient": float(mean_coef[i]),
            }
            for i in order
        ]
        profiles.append(
            {
                "cluster": int(cluster_id),
                "n_lsoas": int(mask.sum()),
                "top_features": top_features,
                "label": " / ".join(t["feature_label"] for t in top_features),
            }
        )
    labels_df = pd.DataFrame({LSOA_ID: ids, "cluster": labels.astype(int)})
    profile_df = pd.DataFrame(profiles)
    return labels_df, profile_df


def top_categories_with_other(series: pd.Series, *, max_categories: int = 8, other_label: str = "Other") -> pd.Series:
    """Fold everything past the max_categories most frequent values into other_label.

    Bounds how many identity colors a categorical choropleth needs: past
    ~3-4 categories a color-only encoding is no longer reliably distinguishable
    (see the dataviz skill's all-pairs guidance for choropleths), so the long
    tail is grouped and every fill still discloses its exact value via hover
    text rather than relying on color alone.
    """
    counts = series.value_counts()
    keep = set(counts.index[:max_categories])
    return series.where(series.isin(keep), other_label)
