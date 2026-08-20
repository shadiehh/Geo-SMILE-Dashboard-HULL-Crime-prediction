"""Two-branch GeoSMILE for Hull next-month crime prediction.

Branches
--------
* Feature: non-spatial predictors only (crime history, Census, OSM). Never includes
  coordinates or location variables.
* Geo: coordinates only. Attribution uses per-LSOA location-masking SMILE with
  Wasserstein-1 responses and a kernel-weighted Ridge surrogate.

Local feature coefficients are per-LSOA SMILE Ridge coefficients (mean over
test months), not median-ablation effects.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.stats import wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from hull_boundaries import hull_outline, load_boundaries, save_boundary_report

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install dependencies with: pip install -r requirements.txt") from exc


RANDOM_SEED = 42
SPATIAL_COLUMNS = ["centroid_easting", "centroid_northing"]
ID_COLUMNS = ["lsoa21cd", "lsoa21nm", "month"]
FORBIDDEN_FEATURE_TOKENS = (
    "coord",
    "easting",
    "northing",
    "latitude",
    "longitude",
    "location",
    "centroid",
)
DERIVED_FEATURES = [
    "crime_lag_1",
    "crime_lag_2",
    "crime_lag_3",
    "crime_rolling_mean_3",
    "crime_rolling_mean_6",
    "month_sin",
    "month_cos",
]
POLICE_COLUMNS = {
    "month": "month",
    "longitude": "longitude",
    "latitude": "latitude",
    "lsoa code": "lsoa21cd",
    "lsoa name": "lsoa21nm",
    "crime type": "crime_type",
}
FEATURE_LABELS = {
    "crime_lag_1": "Crime lag 1",
    "crime_lag_2": "Crime lag 2",
    "crime_lag_3": "Crime lag 3",
    "crime_rolling_mean_3": "Crime roll mean 3",
    "crime_rolling_mean_6": "Crime roll mean 6",
    "month_sin": "Month sin",
    "month_cos": "Month cos",
    "population_density_per_km2": "Population density",
    "young_adult_15_34_pct": "Young adult %",
    "older_adult_65_plus_pct": "Older adult %",
    "social_rented_pct": "Social rented %",
    "private_rented_pct": "Private rented %",
    "owner_occupied_pct": "Owner occupied %",
    "unemployed_pct_age16plus": "Unemployment %",
    "economically_inactive_pct": "Inactive %",
    "long_term_sick_disabled_pct": "Long-term sick %",
    "no_qualifications_pct": "No qualifications %",
    "level4_plus_qualification_pct": "Level 4+ qual %",
    "lsoa_area_km2": "LSOA area",
    "nightlife_count": "Nightlife count",
    "nightlife_per_km2": "Nightlife density",
    "retail_count": "Retail count",
    "retail_per_km2": "Retail density",
    "transit_stop_count": "Transit stops",
    "transit_stop_per_km2": "Transit density",
    "road_length_km": "Road length",
    "road_length_km_per_km2": "Road density",
    "intersection_count": "Intersections",
    "intersection_per_km2": "Intersection density",
    "green_space_pct": "Green space %",
}
BLUE, PINK = "#1E88E5", "#FF0051"
INK, INK_MUTED = "#12171C", "#57646D"
DIVERGING = plt.cm.colors.LinearSegmentedColormap.from_list(
    "geosmile_diverging", [BLUE, "#FFFFFF", PINK]
)
SEQUENTIAL = plt.cm.colors.LinearSegmentedColormap.from_list(
    "geosmile_abs", ["#FFFFFF", PINK, "#8A0033"]
)


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {str(c).strip().lower(): c for c in frame.columns}
    missing = [c for c in POLICE_COLUMNS if c not in aliases]
    if missing:
        raise ValueError(f"Police CSV is missing columns: {missing}")
    return frame.rename(columns={aliases[k]: v for k, v in POLICE_COLUMNS.items()})


def _csv_members(source: Path):
    if source.is_dir():
        for path in source.rglob("*.csv"):
            yield str(path), path
    elif source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".csv"):
                    with archive.open(name) as stream:
                        yield name, stream
    elif source.suffix.lower() == ".csv":
        yield str(source), source
    else:
        raise ValueError("--police-data must be a Police.uk ZIP, CSV, or directory")


def load_hull_crimes(source: Path) -> pd.DataFrame:
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
        names = frame["lsoa21nm"].fillna("").astype(str)
        hull = frame[names.str.startswith("Kingston upon Hull", na=False)].copy()
        if not hull.empty:
            chunks.append(hull[list(POLICE_COLUMNS.values())])
    if not chunks:
        raise ValueError("No Hull street-crime records found.")
    crimes = pd.concat(chunks, ignore_index=True)
    crimes["month"] = pd.to_datetime(crimes["month"], format="%Y-%m")
    crimes["longitude"] = pd.to_numeric(crimes["longitude"], errors="coerce")
    crimes["latitude"] = pd.to_numeric(crimes["latitude"], errors="coerce")
    print(f"Read {seen:,} street rows; retained {len(crimes):,} Hull rows.")
    return crimes


def lonlat_to_bng(longitude: pd.Series, latitude: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    return transformer.transform(longitude.to_numpy(), latitude.to_numpy())


def build_panel(crimes: pd.DataFrame, covariates: Path) -> pd.DataFrame:
    valid = crimes.dropna(subset=["lsoa21cd", "month"]).copy()
    names = (
        valid[["lsoa21cd", "lsoa21nm"]]
        .dropna()
        .drop_duplicates("lsoa21cd")
        .set_index("lsoa21cd")["lsoa21nm"]
    )
    lsoas = np.sort(valid["lsoa21cd"].unique())
    months = pd.date_range(valid["month"].min(), valid["month"].max(), freq="MS")
    grid = pd.MultiIndex.from_product([lsoas, months], names=["lsoa21cd", "month"])
    counts = valid.groupby(["lsoa21cd", "month"]).size().rename("crime_count")
    panel = counts.reindex(grid, fill_value=0).reset_index()
    panel["lsoa21nm"] = panel["lsoa21cd"].map(names)

    locations = valid.groupby("lsoa21cd")[["longitude", "latitude"]].median().dropna()
    east, north = lonlat_to_bng(locations["longitude"], locations["latitude"])
    locations["centroid_easting"] = east
    locations["centroid_northing"] = north
    panel = panel.merge(locations[SPATIAL_COLUMNS], on="lsoa21cd", how="left")

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

    extra = pd.read_csv(covariates)
    if "lsoa21cd" not in extra.columns:
        raise ValueError("Covariate CSV must contain lsoa21cd")
    forbidden = [
        c
        for c in extra.columns
        if c != "lsoa21cd" and any(tok in c.lower() for tok in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"Covariates contain forbidden location columns: {forbidden}")
    if extra["lsoa21cd"].duplicated().any():
        raise ValueError("Covariate CSV must have exactly one row per lsoa21cd")
    duplicate = set(extra.columns) & set(panel.columns) - {"lsoa21cd"}
    if duplicate:
        raise ValueError(f"Covariates duplicate reserved columns: {sorted(duplicate)}")
    panel = panel.merge(extra, on="lsoa21cd", how="left", validate="many_to_one")

    covariate_cols = [c for c in extra.columns if c != "lsoa21cd"]
    panel = panel.dropna(
        subset=DERIVED_FEATURES + SPATIAL_COLUMNS + covariate_cols + ["target_next_month_crime"]
    )
    return panel.sort_values(["month", "lsoa21cd"]).reset_index(drop=True)


def choose_feature_players(panel: pd.DataFrame) -> list[str]:
    reserved = set(ID_COLUMNS + SPATIAL_COLUMNS + ["crime_count", "target_next_month_crime"])
    numeric = panel.select_dtypes(include=[np.number]).columns
    players = []
    for column in numeric:
        if column in reserved:
            continue
        low = column.lower()
        if any(tok in low for tok in FORBIDDEN_FEATURE_TOKENS):
            raise ValueError(f"Feature player looks spatial and is forbidden: {column}")
        players.append(column)
    if not players:
        raise ValueError("No non-spatial numeric features are available")
    return players


def temporal_split(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    months = np.array(sorted(panel["month"].unique()))
    if len(months) < 10:
        raise ValueError("At least 10 usable months are required after lag construction")
    study_train = panel["month"] <= pd.Timestamp("2025-06-01")
    study_val = panel["month"].between(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-01"))
    study_test = panel["month"] >= pd.Timestamp("2026-01-01")
    if study_train.any() and study_val.any() and study_test.any():
        return study_train, study_val, study_test
    train_end = max(1, int(len(months) * 0.60))
    val_end = max(train_end + 1, int(len(months) * 0.80))
    train_months, val_months, test_months = np.split(months, [train_end, val_end])
    return (
        panel["month"].isin(train_months),
        panel["month"].isin(val_months),
        panel["month"].isin(test_months),
    )


def make_model(seed: int) -> XGBRegressor:
    return XGBRegressor(
        objective="count:poisson",
        n_estimators=500,
        learning_rate=0.04,
        max_depth=5,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
    )


def metrics(y, pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
    }


def apply_kernel(distance: np.ndarray, kind: str, sigma: float) -> np.ndarray:
    d = np.asarray(distance, float)
    if kind == "rbf":
        return np.exp(-((d / sigma) ** 2))
    if kind == "laplace":
        return np.exp(-(d / sigma))
    if kind == "inverse":
        return 1 / (1 + d / sigma)
    raise ValueError(kind)


def mask_pool(n_masks: int, n_features: int, rng: np.random.Generator) -> np.ndarray:
    masks = rng.integers(0, 2, size=(n_masks, n_features), dtype=np.int8)
    masks[0] = 1
    masks[1] = 0
    return np.unique(masks, axis=0)


def smile_global_feature(
    model,
    reference: pd.DataFrame,
    players: list[str],
    baselines: pd.Series,
    n_masks: int,
    seed: int,
) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray, np.ndarray]:
    """Global Feature SMILE: W1 on prediction distribution; Hamming kernel distance."""
    rng = np.random.default_rng(seed)
    masks = mask_pool(n_masks, len(players), rng)
    original = np.asarray(model.predict(reference[players]), float)
    responses = np.zeros(len(masks), dtype=float)
    all_pert = np.zeros((len(masks), len(reference)), dtype=np.float32)
    for i, mask in enumerate(masks):
        perturbed = reference[players].copy()
        for j in np.flatnonzero(mask == 0):
            perturbed[players[j]] = baselines[players[j]]
        pred = np.asarray(model.predict(perturbed), float)
        all_pert[i] = pred
        responses[i] = wasserstein_distance(original, pred)
    distances = 1.0 - masks.mean(axis=1)
    order = rng.permutation(len(masks))
    a, b = int(0.6 * len(order)), int(0.8 * len(order))
    tr, va, te = order[:a], order[a:b], order[b:]
    sigma = max(float(np.median(distances[tr])), 1 / len(players))
    scores = {}
    fitted = {}
    for kernel in ("rbf", "laplace", "inverse"):
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(
            masks[tr],
            responses[tr],
            sample_weight=apply_kernel(distances[tr], kernel, sigma),
        )
        scores[kernel] = (
            r2_score(responses[va], surrogate.predict(masks[va])) if len(va) > 1 else np.nan
        )
        fitted[kernel] = surrogate
    best = max({k: v for k, v in scores.items() if np.isfinite(v)}, key=scores.get, default="rbf")
    surrogate = fitted[best]
    test_r2 = r2_score(responses[te], surrogate.predict(masks[te])) if len(te) > 1 else np.nan
    importance = pd.DataFrame(
        {
            "feature": players,
            "coefficient": surrogate.coef_,
            "abs_coefficient": np.abs(surrogate.coef_),
        }
    ).sort_values("abs_coefficient", ascending=False)
    diagnostics = {
        "best_kernel": best,
        "validation_r2_by_kernel": scores,
        "test_r2": float(test_r2),
        "n_unique_masks": int(len(masks)),
        "response": "wasserstein_1_prediction_distribution",
        "kernel_distance": "hamming_fraction_features_masked",
    }
    return importance, diagnostics, masks, all_pert, original.astype(np.float32)


def local_feature_coefficients(
    masks: np.ndarray,
    all_pert: np.ndarray,
    original: np.ndarray,
    players: list[str],
    lsoa_codes: np.ndarray,
    kernel: str,
    seed: int,
) -> pd.DataFrame:
    """Per-observation local Ridge coefficients, then mean by LSOA."""
    distances = 1.0 - masks.mean(axis=1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(masks))
    a = int(0.6 * len(order))
    tr = order[:a]
    sigma = max(float(np.median(distances[tr])), 1 / len(players))
    weights = np.clip(apply_kernel(distances[tr], kernel, sigma), 1e-8, None)
    local = np.zeros((len(original), len(players)), dtype=np.float32)
    for i in range(len(original)):
        y_i = original[i] - all_pert[tr, i]
        ridge = Ridge(alpha=1.0)
        ridge.fit(masks[tr], y_i, sample_weight=weights)
        local[i] = ridge.coef_
    frame = pd.DataFrame(local, columns=players)
    frame["lsoa21cd"] = lsoa_codes
    return frame.groupby("lsoa21cd", as_index=False).mean(numeric_only=True)


def geo_smile_coefficients(
    model,
    panel: pd.DataFrame,
    mask: pd.Series,
    baselines: pd.Series,
    n_perturb: int,
    seed: int,
    min_sel: int = 5,
    max_sel: int = 10,
    sample_lsoas: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Per-LSOA geo SMILE: mask LSOA coordinates to training median; W1 response.

    If `sample_lsoas` is set, Hull (core) columns keep the baseline order and the
    same mask draws (same seed). Extra LSOAs in `panel` are appended as additional
    players and masked with an independent RNG, inclusion probability `n_sel / n_core`
    so a typical extra unit is selected about as often as one Hull LSOA.
    """
    frame = panel.loc[mask, ID_COLUMNS + SPATIAL_COLUMNS].copy().reset_index(drop=True)
    x = frame[SPATIAL_COLUMNS].copy()
    original = np.asarray(model.predict(x), float)
    lsoa_ids = frame["lsoa21cd"].astype(str).to_numpy()
    present = sorted(set(lsoa_ids.tolist()))
    if sample_lsoas is None:
        unique_lsoas = np.array(present, dtype=object)
        n_sample = len(unique_lsoas)
        extras: list[str] = []
    else:
        core = [str(c) for c in sample_lsoas]
        missing = [c for c in core if c not in set(present)]
        if missing:
            raise ValueError(f"sample_lsoas missing from panel: {missing[:8]}")
        extras = [c for c in present if c not in set(core)]
        unique_lsoas = np.array(core + extras, dtype=object)
        n_sample = len(core)
    n_lsoas = len(unique_lsoas)
    index_by_lsoa = {code: np.flatnonzero(lsoa_ids == code) for code in unique_lsoas}
    rng = np.random.default_rng(seed)
    extra_rng = np.random.default_rng(seed + 1_000_003)
    M = np.zeros((n_perturb, n_lsoas), dtype=np.int8)
    responses = np.zeros(n_perturb, dtype=float)
    baseline_xy = baselines[SPATIAL_COLUMNS].to_numpy(dtype=float)
    extra_hits = 0

    for j in range(n_perturb):
        n_sel = int(rng.integers(min_sel, max_sel + 1))
        sel = rng.choice(n_sample, size=n_sel, replace=False)
        M[j, sel] = 1
        chosen = list(sel)
        if extras:
            p_extra = n_sel / n_sample
            for e, _code in enumerate(extras):
                if extra_rng.random() < p_extra:
                    extra_col = n_sample + e
                    M[j, extra_col] = 1
                    chosen.append(extra_col)
                    extra_hits += 1
        perturbed = x.copy()
        affected = []
        for idx in chosen:
            rows = index_by_lsoa[unique_lsoas[idx]]
            perturbed.iloc[rows, :] = baseline_xy
            affected.extend(rows.tolist())
        affected = np.asarray(affected, dtype=int)
        pert_pred = np.asarray(model.predict(perturbed), float)
        responses[j] = wasserstein_distance(original[affected], pert_pred[affected])

    distances = M[:, :n_sample].sum(axis=1).astype(float) / n_sample
    order = rng.permutation(n_perturb)
    a, b = int(0.6 * len(order)), int(0.8 * len(order))
    tr, va, te = order[:a], order[a:b], order[b:]
    sigma = max(float(np.median(distances[tr])), 1 / n_sample)
    scores = {}
    fitted = {}
    for kernel in ("rbf", "laplace", "inverse"):
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(M[tr], responses[tr], sample_weight=apply_kernel(distances[tr], kernel, sigma))
        scores[kernel] = r2_score(responses[va], surrogate.predict(M[va])) if len(va) > 1 else np.nan
        fitted[kernel] = surrogate
    best = max({k: v for k, v in scores.items() if np.isfinite(v)}, key=scores.get, default="rbf")
    trva = np.concatenate([tr, va])
    final = Ridge(alpha=1.0)
    final.fit(
        M[trva],
        responses[trva],
        sample_weight=apply_kernel(distances[trva], best, sigma),
    )
    test_r2 = r2_score(responses[te], final.predict(M[te])) if len(te) > 1 else np.nan
    coef = pd.DataFrame(
        {
            "lsoa21cd": unique_lsoas.astype(str),
            "geo_coefficient": final.coef_,
            "geo_abs_coefficient": np.abs(final.coef_),
        }
    )
    diagnostics = {
        "method": "per_lsoa_location_masking_no_kmeans",
        "n_spatial_units": int(n_lsoas),
        "n_perturbations": int(n_perturb),
        "n_sampled_lsoas": int(n_sample),
        "n_extra_lsoas": int(len(extras)),
        "extra_inclusion_rate": (
            float(extra_hits / (n_perturb * max(len(extras), 1))) if extras else 0.0
        ),
        "min_lsoas_per_perturbation": min_sel,
        "max_lsoas_per_perturbation": max_sel,
        "best_kernel": best,
        "validation_r2_by_kernel": scores,
        "test_r2": float(test_r2),
        "response": "wasserstein_1_on_affected_lsoa_month_predictions",
        "kernel_distance": "fraction of frozen Hull LSOAs masked",
        "perturbation": "replace easting/northing of selected LSOAs with training median",
        "extra_perturbation": (
            "independent Bernoulli(n_sel/n_core) on extra LSOAs; separate RNG"
            if extras
            else "none"
        ),
    }
    return coef, diagnostics


def shared_limit(values: np.ndarray, pct: float = 98) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return max(float(np.percentile(np.abs(v), pct)), 1e-9)


def plot_choropleth(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    path: Path,
    *,
    signed: bool,
    vmax: float,
    test_period: str,
    cbar_label: str,
) -> None:
    plot_gdf = gdf.to_crs("EPSG:27700")
    fig, ax = plt.subplots(figsize=(8.5, 9.0))
    if signed:
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = DIVERGING
    else:
        norm = Normalize(vmin=0.0, vmax=vmax)
        cmap = SEQUENTIAL
    plot_gdf.plot(
        column=column,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.12,
        ax=ax,
    )
    gpd.GeoSeries([hull_outline(plot_gdf)], crs=plot_gdf.crs).plot(
        ax=ax, color="none", edgecolor=INK, linewidth=1.0
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.68, pad=0.015)
    cbar.set_label(cbar_label, fontsize=9, color=INK_MUTED)
    footer = (
        f"Test period: {test_period}. Source: ONS LSOA 2021 BGC. "
        "SMILE Ridge coefficients from Wasserstein prediction shifts. Not causal effects."
    )
    fig.text(0.5, 0.012, footer, ha="center", fontsize=7.0, color=INK_MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def write_maps(
    boundaries: gpd.GeoDataFrame,
    local_feat: pd.DataFrame,
    geo_coef: pd.DataFrame,
    players: list[str],
    test_period: str,
    out_dir: Path,
) -> dict:
    map_dir = out_dir / "maps_twobranch"
    signed_dir = map_dir / "signed"
    abs_dir = map_dir / "absolute"
    signed_dir.mkdir(parents=True, exist_ok=True)
    abs_dir.mkdir(parents=True, exist_ok=True)

    geo_map = boundaries.merge(geo_coef, on="lsoa21cd", how="inner", validate="one_to_one")
    geo_vmax = shared_limit(geo_map["geo_coefficient"].to_numpy())
    abs_geo_vmax = shared_limit(geo_map["geo_abs_coefficient"].to_numpy())
    plot_choropleth(
        geo_map,
        "geo_coefficient",
        "Hull — Geo SMILE coefficient (signed)",
        signed_dir / "geo_coefficient_signed.png",
        signed=True,
        vmax=geo_vmax,
        test_period=test_period,
        cbar_label="Signed geo SMILE coefficient",
    )
    plot_choropleth(
        geo_map,
        "geo_abs_coefficient",
        "Hull — Geo SMILE coefficient (absolute)",
        abs_dir / "geo_coefficient_abs.png",
        signed=False,
        vmax=abs_geo_vmax,
        test_period=test_period,
        cbar_label="|Geo SMILE coefficient|",
    )

    manifest = {"geo_signed": ["signed/geo_coefficient_signed.png"], "geo_abs": ["absolute/geo_coefficient_abs.png"], "feature_signed": [], "feature_abs": []}
    for player in players:
        label = FEATURE_LABELS.get(player, player)
        merged = boundaries.merge(
            local_feat[["lsoa21cd", player]], on="lsoa21cd", how="inner", validate="one_to_one"
        )
        merged["abs_value"] = merged[player].abs()
        vmax = shared_limit(merged[player].to_numpy())
        abs_vmax = shared_limit(merged["abs_value"].to_numpy())
        s_name = f"feature_{player}_signed.png"
        a_name = f"feature_{player}_abs.png"
        plot_choropleth(
            merged,
            player,
            f"Hull — {label}\nFeature SMILE coefficient (signed)",
            signed_dir / s_name,
            signed=True,
            vmax=vmax,
            test_period=test_period,
            cbar_label="Signed feature SMILE coefficient",
        )
        plot_choropleth(
            merged,
            "abs_value",
            f"Hull — {label}\nFeature SMILE coefficient (absolute)",
            abs_dir / a_name,
            signed=False,
            vmax=abs_vmax,
            test_period=test_period,
            cbar_label="|Feature SMILE coefficient|",
        )
        manifest["feature_signed"].append(f"signed/{s_name}")
        manifest["feature_abs"].append(f"absolute/{a_name}")
    (map_dir / "map_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    covariates = Path(args.covariates)
    if not covariates.exists():
        raise SystemExit(f"Missing covariates: {covariates}")

    crimes = load_hull_crimes(Path(args.police_data))
    panel = build_panel(crimes, covariates)
    players = choose_feature_players(panel)
    assert not any(any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS) for p in players)
    assert not any(c in players for c in SPATIAL_COLUMNS)

    train, validation, test = temporal_split(panel)
    y_col = "target_next_month_crime"
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()

    print(f"Feature players ({len(players)}): {players}")
    model_feat = make_model(args.seed)
    model_geo = make_model(args.seed)
    model_feat.fit(
        panel.loc[train, players],
        panel.loc[train, y_col],
        eval_set=[(panel.loc[validation, players], panel.loc[validation, y_col])],
        verbose=False,
    )
    model_geo.fit(
        panel.loc[train, SPATIAL_COLUMNS],
        panel.loc[train, y_col],
        eval_set=[(panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, y_col])],
        verbose=False,
    )

    predictions = panel.loc[test, ID_COLUMNS + [y_col]].copy()
    predictions["prediction_feature"] = model_feat.predict(panel.loc[test, players])
    predictions["prediction_geo"] = model_geo.predict(panel.loc[test, SPATIAL_COLUMNS])
    predictions.to_csv(output / "test_predictions_twobranch.csv", index=False)

    importance, feat_diag, masks, all_pert, original = smile_global_feature(
        model_feat,
        panel.loc[test, players],
        players,
        baseline,
        args.masks,
        args.seed,
    )
    importance.to_csv(output / "feature_global_importance.csv", index=False)

    local_feat = local_feature_coefficients(
        masks,
        all_pert,
        original,
        players,
        panel.loc[test, "lsoa21cd"].to_numpy(),
        feat_diag["best_kernel"],
        args.seed,
    )
    # Attach names for mapping convenience.
    names = panel[["lsoa21cd", "lsoa21nm"]].drop_duplicates("lsoa21cd")
    local_feat = names.merge(local_feat, on="lsoa21cd", how="right")
    local_feat.to_csv(output / "feature_local_coefficients.csv", index=False)
    abs_local = local_feat.copy()
    for player in players:
        abs_local[player] = abs_local[player].abs()
    abs_local.to_csv(output / "feature_local_coefficients_abs.csv", index=False)

    print(f"Running geo SMILE ({args.geo_perturbations} perturbations)...")
    geo_coef, geo_diag = geo_smile_coefficients(
        model_geo,
        panel,
        test,
        baseline,
        args.geo_perturbations,
        args.seed,
    )
    geo_coef = names.merge(geo_coef, on="lsoa21cd", how="right")
    geo_coef.to_csv(output / "geo_coefficients.csv", index=False)

    panel.to_csv(output / "hull_lsoa_month_panel_with_covariates.csv", index=False)

    boundaries, boundary_meta = load_boundaries(sorted(panel["lsoa21cd"].unique()))
    data_dir = Path(__file__).resolve().parent / "data"
    save_boundary_report(boundary_meta, data_dir / "boundary_validation.json")
    test_period = f"{predictions['month'].min().date()} to {predictions['month'].max().date()}"
    manifest = write_maps(
        boundaries, local_feat, geo_coef, players, test_period, output
    )

    report = {
        "design": "two_branch_geosmile",
        "rows": int(len(panel)),
        "lsoas": int(panel["lsoa21cd"].nunique()),
        "usable_predictor_months": [
            str(panel["month"].min().date()),
            str(panel["month"].max().date()),
        ],
        "test_period": [
            str(predictions["month"].min().date()),
            str(predictions["month"].max().date()),
        ],
        "target": y_col,
        "feature_players": players,
        "feature_players_exclude_coordinates": True,
        "spatial_columns": SPATIAL_COLUMNS,
        "covariates_file": str(covariates.resolve()),
        "model_metrics": {
            "feature": metrics(predictions[y_col], predictions["prediction_feature"]),
            "geo": metrics(predictions[y_col], predictions["prediction_geo"]),
        },
        "feature_surrogate": feat_diag,
        "geo_surrogate": geo_diag,
        "map_manifest": manifest,
        "boundary_source": boundary_meta.get("source"),
        "notes": [
            "Feature branch contains no coordinates/location variables.",
            "Feature maps show local SMILE Ridge coefficients averaged to LSOA.",
            "Geo maps show per-LSOA SMILE coefficients from location masking.",
            "Wasserstein-1 is the surrogate response; kernel distance is Hamming/mask fraction.",
        ],
    }
    (output / "run_summary_twobranch.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Outputs written to {output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--police-data", required=True)
    parser.add_argument(
        "--covariates",
        default="data/hull_static_covariates.csv",
        help="Static one-row-per-LSOA covariate CSV",
    )
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--masks", type=int, default=1024)
    parser.add_argument("--geo-perturbations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
