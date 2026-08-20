"""Fit the three-branch GeoSMILE design to Police.uk Hull crime data.

The observational unit is LSOA-month and the target is next-month total crime.
Feature-A excludes coordinates from both prediction and attribution. Feature-B
uses coordinates for prediction, but coordinates are never attribution players.
The Geo branch isolates location sensitivity.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install dependencies with: pip install -r requirements.txt") from exc


RANDOM_SEED = 42
SPATIAL_COLUMNS = ["centroid_easting", "centroid_northing"]
ID_COLUMNS = ["lsoa21cd", "lsoa21nm", "month"]
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


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {str(c).strip().lower(): c for c in frame.columns}
    missing = [c for c in POLICE_COLUMNS if c not in aliases]
    if missing:
        raise ValueError(f"Police CSV is missing columns: {missing}")
    return frame.rename(columns={aliases[k]: v for k, v in POLICE_COLUMNS.items()})


def _csv_members(source: Path):
    """Yield (display name, readable object) for CSVs in a file/dir/zip."""
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
    """Load Humberside street files and retain Kingston upon Hull LSOAs."""
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
        raise ValueError(
            "No Hull street-crime records found. Use a Police.uk crime download "
            "containing Humberside Police and Kingston upon Hull LSOAs."
        )
    # Do not deduplicate on the public columns: multiple genuine offences can
    # share the same anonymisation point, month, and category.
    crimes = pd.concat(chunks, ignore_index=True)
    crimes["month"] = pd.to_datetime(crimes["month"], format="%Y-%m")
    crimes["longitude"] = pd.to_numeric(crimes["longitude"], errors="coerce")
    crimes["latitude"] = pd.to_numeric(crimes["latitude"], errors="coerce")
    print(f"Read {seen:,} street rows; retained {len(crimes):,} Hull rows.")
    return crimes


def lonlat_to_bng(longitude: pd.Series, latitude: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Convert WGS84 to British National Grid, or fail with a clear message."""
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pyproj is required; install requirements.txt") from exc
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    return transformer.transform(longitude.to_numpy(), latitude.to_numpy())


def build_panel(crimes: pd.DataFrame, covariates: Path | None = None) -> pd.DataFrame:
    """Create a complete LSOA-month panel with leakage-safe lags and target."""
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

    # Stable incident-median location is a fallback when an official centroid
    # file is not supplied. Police.uk coordinates are anonymised; never model
    # them as exact event locations.
    locations = (
        valid.groupby("lsoa21cd")[["longitude", "latitude"]]
        .median()
        .dropna()
    )
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

    if covariates:
        extra = pd.read_csv(covariates)
        if "lsoa21cd" not in extra:
            raise ValueError("Covariate CSV must contain lsoa21cd")
        duplicate = set(extra.columns) & set(panel.columns) - {"lsoa21cd"}
        if duplicate:
            raise ValueError(f"Covariates duplicate reserved columns: {sorted(duplicate)}")
        if extra["lsoa21cd"].duplicated().any():
            raise ValueError("Covariate CSV must have exactly one row per lsoa21cd")
        panel = panel.merge(extra, on="lsoa21cd", how="left", validate="many_to_one")

    panel = panel.dropna(subset=DERIVED_FEATURES + SPATIAL_COLUMNS + ["target_next_month_crime"])
    return panel.sort_values(["month", "lsoa21cd"]).reset_index(drop=True)


def choose_features(panel: pd.DataFrame) -> list[str]:
    reserved = set(ID_COLUMNS + SPATIAL_COLUMNS + ["crime_count", "target_next_month_crime"])
    numeric = panel.select_dtypes(include=[np.number]).columns
    features = [c for c in numeric if c not in reserved]
    if not features:
        raise ValueError("No non-spatial numeric features are available")
    return features


def temporal_split(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Use the study split when covered, otherwise chronological 60/20/20."""
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


def metrics(y: pd.Series, pred: np.ndarray) -> dict[str, float]:
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


def smile_global(
    model,
    reference: pd.DataFrame,
    feature_players: list[str],
    baselines: pd.Series,
    n_masks: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Fit a global SMILE surrogate to W1 prediction-distribution shifts."""
    rng = np.random.default_rng(seed)
    masks = mask_pool(n_masks, len(feature_players), rng)
    original = np.asarray(model.predict(reference), float)
    responses = []
    for mask in masks:
        perturbed = reference.copy()
        off = np.flatnonzero(mask == 0)
        for j in off:
            name = feature_players[j]
            perturbed[name] = baselines[name]
        responses.append(wasserstein_distance(original, model.predict(perturbed)))
    responses = np.asarray(responses)
    distances = 1 - masks.mean(axis=1)
    order = rng.permutation(len(masks))
    a, b = int(0.6 * len(order)), int(0.8 * len(order))
    tr, va, te = order[:a], order[a:b], order[b:]
    sigma = max(float(np.median(distances[tr])), 1 / len(feature_players))
    scores = {}
    fitted = {}
    for kernel in ("rbf", "laplace", "inverse"):
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(masks[tr], responses[tr], sample_weight=apply_kernel(distances[tr], kernel, sigma))
        scores[kernel] = r2_score(responses[va], surrogate.predict(masks[va])) if len(va) > 1 else np.nan
        fitted[kernel] = surrogate
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    best = max(finite, key=finite.get) if finite else "rbf"
    surrogate = fitted[best]
    test_r2 = r2_score(responses[te], surrogate.predict(masks[te])) if len(te) > 1 else np.nan
    importance = pd.DataFrame(
        {"feature": feature_players, "coefficient": surrogate.coef_, "abs_importance": np.abs(surrogate.coef_)}
    ).sort_values("abs_importance", ascending=False)
    diagnostics = {
        "best_kernel": best,
        "validation_r2_by_kernel": scores,
        "test_r2": float(test_r2),
        "n_unique_masks": int(len(masks)),
    }
    return importance, diagnostics


def local_ablation_effects(model, frame: pd.DataFrame, players: list[str], baselines: pd.Series) -> pd.DataFrame:
    """Signed baseline-ablation effects (not local SMILE Ridge coefficients).

    Each value is original_prediction minus prediction with that feature replaced
    by the training median. Positive = original raises prediction vs baseline;
    negative = original lowers prediction vs baseline. Not a causal effect.
    """
    model_features = list(model.get_booster().feature_names)
    original = np.asarray(model.predict(frame[model_features]), float)
    result = frame[ID_COLUMNS].copy()
    for player in players:
        changed = frame.copy()
        changed[player] = baselines[player]
        result[player] = original - np.asarray(model.predict(changed[model_features]), float)
    return result


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    crimes = load_hull_crimes(Path(args.police_data))
    panel = build_panel(crimes, Path(args.covariates) if args.covariates else None)
    players = choose_features(panel)
    train, validation, test = temporal_split(panel)
    y_col = "target_next_month_crime"
    features_a = players
    features_b = players + SPATIAL_COLUMNS
    baseline = panel.loc[train, features_b].median()

    model_a, model_b, model_geo = make_model(args.seed), make_model(args.seed), make_model(args.seed)
    model_a.fit(panel.loc[train, features_a], panel.loc[train, y_col], eval_set=[(panel.loc[validation, features_a], panel.loc[validation, y_col])], verbose=False)
    model_b.fit(panel.loc[train, features_b], panel.loc[train, y_col], eval_set=[(panel.loc[validation, features_b], panel.loc[validation, y_col])], verbose=False)
    model_geo.fit(panel.loc[train, SPATIAL_COLUMNS], panel.loc[train, y_col], eval_set=[(panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, y_col])], verbose=False)

    predictions = panel.loc[test, ID_COLUMNS + [y_col]].copy()
    predictions["prediction_feature_a"] = model_a.predict(panel.loc[test, features_a])
    predictions["prediction_feature_b"] = model_b.predict(panel.loc[test, features_b])
    predictions["prediction_geo"] = model_geo.predict(panel.loc[test, SPATIAL_COLUMNS])

    importance_a, diag_a = smile_global(model_a, panel.loc[test, features_a], players, baseline, args.masks, args.seed)
    # Corrected Feature-B: prediction sees coordinates; attribution masks only non-spatial players.
    importance_b, diag_b = smile_global(model_b, panel.loc[test, features_b], players, baseline, args.masks, args.seed)
    local_a = local_ablation_effects(model_a, panel.loc[test, ID_COLUMNS + features_a], players, baseline)
    local_b = local_ablation_effects(model_b, panel.loc[test, ID_COLUMNS + features_b], players, baseline)

    geo_frame = panel.loc[test, SPATIAL_COLUMNS].copy()
    geo_original = model_geo.predict(geo_frame)
    geo_neutral = geo_frame.copy()
    geo_neutral.loc[:, SPATIAL_COLUMNS] = baseline[SPATIAL_COLUMNS].to_numpy()
    geo_effect = panel.loc[test, ID_COLUMNS].copy()
    geo_effect["location_baseline_ablation_effect"] = geo_original - model_geo.predict(geo_neutral)
    geo_effect["location_effect"] = geo_effect["location_baseline_ablation_effect"]

    from hull_eval import (
        benchmark_rows,
        feature_b_incremental,
        fit_spatial_holdout,
        geosmile_rows,
        spatial_holdout_lsoas,
        write_feature_b_json,
        write_model_comparison,
    )

    comparison_rows = benchmark_rows(panel, train, test)
    comparison_rows.extend(geosmile_rows(predictions, "temporal_holdout"))
    write_model_comparison(output, comparison_rows)

    fb_value = feature_b_incremental(predictions, n_boot=5000, seed=args.seed)
    write_feature_b_json(output, fb_value)

    held_out = spatial_holdout_lsoas(panel, seed=args.seed)
    spatial_preds, spatial_summary = fit_spatial_holdout(panel, players, held_out, args.seed, make_model)
    spatial_preds.to_csv(output / "spatial_holdout_predictions.csv", index=False)
    spatial_rows = benchmark_rows(
        panel,
        ~panel["lsoa21cd"].isin(held_out),
        panel["lsoa21cd"].isin(held_out),
        evaluation="spatial_holdout",
    )
    spatial_rows.extend(geosmile_rows(spatial_preds, "spatial_holdout"))
    spatial_df = pd.read_csv(output / "model_comparison.csv")
    spatial_df = pd.concat([spatial_df, pd.DataFrame(spatial_rows)], ignore_index=True)
    spatial_df.to_csv(output / "model_comparison.csv", index=False)

    importance_a.to_csv(output / "feature_a_global_importance.csv", index=False)
    importance_b.to_csv(output / "feature_b_global_importance.csv", index=False)
    local_a.to_csv(output / "feature_a_local_effects.csv", index=False)
    local_b.to_csv(output / "feature_b_local_effects.csv", index=False)
    geo_effect.to_csv(output / "geo_location_effects.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    panel.to_csv(output / "hull_lsoa_month_panel.csv", index=False)

    report = {
        "rows": int(len(panel)),
        "lsoas": int(panel["lsoa21cd"].nunique()),
        "usable_predictor_months": [str(panel["month"].min().date()), str(panel["month"].max().date())],
        "target": y_col,
        "feature_players": players,
        "spatial_columns": SPATIAL_COLUMNS,
        "feature_b_coordinates_attributed": False,
        "test_period": [str(predictions["month"].min().date()), str(predictions["month"].max().date())],
        "model_metrics": {
            "feature_a": metrics(predictions[y_col], predictions["prediction_feature_a"]),
            "feature_b": metrics(predictions[y_col], predictions["prediction_feature_b"]),
            "geo": metrics(predictions[y_col], predictions["prediction_geo"]),
        },
        "surrogate_diagnostics": {"feature_a": diag_a, "feature_b": diag_b},
        "geo_mean_absolute_location_effect": float(geo_effect["location_baseline_ablation_effect"].abs().mean()),
        "feature_b_incremental_value": fb_value,
        "spatial_holdout": spatial_summary,
        "local_effect_definition": (
            "Signed baseline-ablation effect: original prediction minus prediction "
            "with the feature replaced by the training median. Not a causal effect."
        ),
        "coordinate_note": "Police.uk anonymised incident-median LSOA locations; replace with official population-weighted centroids for final analysis.",
    }
    (output / "spatial_holdout_summary.json").write_text(json.dumps(spatial_summary, indent=2), encoding="utf-8")
    (output / "run_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Outputs written to {output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--police-data", required=True, help="Police.uk ZIP, CSV, or extracted directory")
    parser.add_argument("--covariates", help="Optional one-row-per-LSOA numeric feature CSV")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--masks", type=int, default=1024, help="Requested masks per feature branch")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
