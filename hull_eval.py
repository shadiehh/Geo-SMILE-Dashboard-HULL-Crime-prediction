"""Benchmark comparisons, Feature-B incremental value, and spatial holdout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SPATIAL_COLUMNS = ["centroid_easting", "centroid_northing"]


def metrics(y: pd.Series | np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_arr, pred_arr)),
        "rmse": float(mean_squared_error(y_arr, pred_arr) ** 0.5),
        "r2": float(r2_score(y_arr, pred_arr)),
    }


def benchmark_rows(
    panel: pd.DataFrame,
    train: pd.Series,
    test: pd.Series,
    evaluation: str = "temporal_holdout",
) -> list[dict]:
    """Simple baselines evaluated on the supplied test partition."""
    test_df = panel.loc[test].copy()
    y = test_df["target_next_month_crime"].to_numpy()
    lsoa_train_mean = panel.loc[train].groupby("lsoa21cd")["crime_count"].mean()
    global_train_mean = float(panel.loc[train, "crime_count"].mean())
    # Held-out LSOAs in a spatial split have no LSOA-specific training mean;
    # fall back to the overall training mean so the baseline remains defined.
    lsoa_mean_pred = test_df["lsoa21cd"].map(lsoa_train_mean).fillna(global_train_mean).to_numpy()

    candidates = {
        "current_month_crime": test_df["crime_count"].to_numpy(),
        "one_month_lag": test_df["crime_lag_1"].to_numpy(),
        "three_month_rolling_mean": test_df["crime_rolling_mean_3"].to_numpy(),
        "six_month_rolling_mean": test_df["crime_rolling_mean_6"].to_numpy(),
        "lsoa_training_mean": lsoa_mean_pred,
    }
    rows = []
    for name, pred in candidates.items():
        row = {"model": name, "evaluation": evaluation, **metrics(y, pred)}
        rows.append(row)
    return rows


def geosmile_rows(predictions: pd.DataFrame, evaluation: str) -> list[dict]:
    y = predictions["target_next_month_crime"]
    rows = []
    for model, col in (
        ("feature_a", "prediction_feature_a"),
        ("feature_b", "prediction_feature_b"),
        ("geo", "prediction_geo"),
    ):
        rows.append({"model": model, "evaluation": evaluation, **metrics(y, predictions[col].to_numpy())})
    return rows


def feature_b_incremental(
    predictions: pd.DataFrame,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """Paired Feature-A vs Feature-B comparison with LSOA-clustered bootstrap."""
    frame = predictions.reset_index(drop=True).copy()
    frame["abs_err_a"] = (frame["target_next_month_crime"] - frame["prediction_feature_a"]).abs()
    frame["abs_err_b"] = (frame["target_next_month_crime"] - frame["prediction_feature_b"]).abs()
    y = frame["target_next_month_crime"].to_numpy()
    pred_a = frame["prediction_feature_a"].to_numpy()
    pred_b = frame["prediction_feature_b"].to_numpy()

    mae_a = mean_absolute_error(y, pred_a)
    mae_b = mean_absolute_error(y, pred_b)
    rmse_a = mean_squared_error(y, pred_a) ** 0.5
    rmse_b = mean_squared_error(y, pred_b) ** 0.5
    r2_a = r2_score(y, pred_a)
    r2_b = r2_score(y, pred_b)

    point = {
        "mae_difference_a_minus_b": float(mae_a - mae_b),
        "rmse_difference_a_minus_b": float(rmse_a - rmse_b),
        "r2_difference_b_minus_a": float(r2_b - r2_a),
        "pct_observations_b_lower_abs_error": float((frame["abs_err_b"] < frame["abs_err_a"]).mean() * 100),
        "prediction_correlation": float(np.corrcoef(pred_a, pred_b)[0, 1]),
        "feature_a_mae": float(mae_a),
        "feature_b_mae": float(mae_b),
        "feature_a_rmse": float(rmse_a),
        "feature_b_rmse": float(rmse_b),
        "feature_a_r2": float(r2_a),
        "feature_b_r2": float(r2_b),
    }

    lsoa_ids = frame["lsoa21cd"].unique()
    lsoa_index = {
        code: frame.index[frame["lsoa21cd"] == code].to_numpy() for code in lsoa_ids
    }
    rng = np.random.default_rng(seed)
    mae_diffs: list[float] = []
    rmse_diffs: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(lsoa_ids, size=len(lsoa_ids), replace=True)
        idx = np.concatenate([lsoa_index[code] for code in sampled])
        yy = y[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]
        mae_diffs.append(mean_absolute_error(yy, pa) - mean_absolute_error(yy, pb))
        rmse_diffs.append(mean_squared_error(yy, pa) ** 0.5 - mean_squared_error(yy, pb) ** 0.5)

    mae_ci = np.percentile(mae_diffs, [2.5, 97.5]).tolist()
    rmse_ci = np.percentile(rmse_diffs, [2.5, 97.5]).tolist()
    point["bootstrap"] = {
        "n_repetitions": n_boot,
        "cluster_unit": "lsoa21cd",
        "mae_improvement_a_minus_b_ci95": mae_ci,
        "rmse_improvement_a_minus_b_ci95": rmse_ci,
        "mae_improvement_significant_at_95": bool(mae_ci[0] > 0 or mae_ci[1] < 0),
        "rmse_improvement_significant_at_95": bool(rmse_ci[0] > 0 or rmse_ci[1] < 0),
    }
    point["interpretation_note"] = (
        "Positive MAE/RMSE difference (A minus B) means Feature-B has lower error. "
        "Do not claim coordinate improvement unless the 95% CI excludes zero."
    )
    return point


def spatial_holdout_lsoas(panel: pd.DataFrame, n_clusters: int = 5, holdout_clusters: int = 1, seed: int = 42) -> set[str]:
    """Select spatially clustered held-out LSOAs via KMeans on BNG centroids."""
    loc = (
        panel[["lsoa21cd", "centroid_easting", "centroid_northing"]]
        .drop_duplicates("lsoa21cd")
        .set_index("lsoa21cd")
    )
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(loc[["centroid_easting", "centroid_northing"]])
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    # Hold out the cluster with median size for balance.
    holdout_label = int(cluster_sizes.index[len(cluster_sizes) // 2])
    held_out = set(loc.index[labels == holdout_label])
    return held_out


ID_COLUMNS = ["lsoa21cd", "lsoa21nm", "month"]


def fit_spatial_holdout(
    panel: pd.DataFrame,
    players: list[str],
    held_out_lsoas: set[str],
    seed: int,
    make_model,
) -> tuple[pd.DataFrame, dict]:
    """Train on non-held-out LSOAs; evaluate on held-out LSOAs across all months."""
    y_col = "target_next_month_crime"
    features_a = players
    features_b = players + SPATIAL_COLUMNS
    spatial_train = ~panel["lsoa21cd"].isin(held_out_lsoas)
    spatial_test = panel["lsoa21cd"].isin(held_out_lsoas)
    train_df = panel.loc[spatial_train]
    test_df = panel.loc[spatial_test]
    if train_df.empty or test_df.empty:
        raise ValueError("Spatial holdout produced empty train or test partition")

    baseline = train_df[features_b].median()
    model_a = make_model(seed)
    model_b = make_model(seed)
    model_geo = make_model(seed)
    model_a.fit(train_df[features_a], train_df[y_col], verbose=False)
    model_b.fit(train_df[features_b], train_df[y_col], verbose=False)
    model_geo.fit(train_df[SPATIAL_COLUMNS], train_df[y_col], verbose=False)

    predictions = test_df[ID_COLUMNS + [y_col]].copy()
    predictions["prediction_feature_a"] = model_a.predict(test_df[features_a])
    predictions["prediction_feature_b"] = model_b.predict(test_df[features_b])
    predictions["prediction_geo"] = model_geo.predict(test_df[SPATIAL_COLUMNS])
    predictions["evaluation"] = "spatial_holdout"

    # Benchmarks on spatial test using training LSOAs only for LSOA means.
    bench_train = spatial_train
    bench_test = spatial_test
    bench_rows = benchmark_rows(panel, bench_train, bench_test, evaluation="spatial_holdout")
    geo_rows = geosmile_rows(predictions, "spatial_holdout")
    summary = {
        "held_out_lsoas": sorted(held_out_lsoas),
        "n_held_out_lsoas": len(held_out_lsoas),
        "n_train_lsoas": int(panel.loc[spatial_train, "lsoa21cd"].nunique()),
        "n_test_rows": int(len(test_df)),
        "model_metrics": {r["model"]: {k: r[k] for k in ("mae", "rmse", "r2")} for r in geo_rows},
        "benchmark_metrics": {r["model"]: {k: r[k] for k in ("mae", "rmse", "r2")} for r in bench_rows},
        "lsoa_training_mean_note": (
            "For spatially held-out LSOAs, lsoa_training_mean falls back to the overall "
            "training-period mean because those LSOAs never appear in training."
        ),
        "note": (
            "Harder spatial-transfer test: held-out LSOAs never appear in training. "
            "Same LSOAs can contribute multiple test months."
        ),
    }
    return predictions, summary


def write_model_comparison(output: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(output / "model_comparison.csv", index=False)


def write_feature_b_json(output: Path, payload: dict) -> None:
    (output / "feature_b_incremental_value.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
