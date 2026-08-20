"""Thesis-ready figures for Hull GeoSMILE evaluation and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hull_boundaries import load_boundaries

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

BLUE, PINK, TEAL = "#1E88E5", "#FF0051", "#147D74"
INK, INK_MUTED = "#12171C", "#57646D"
FEATURE_LABELS = {
    "crime_lag_1": "Lag 1",
    "crime_lag_2": "Lag 2",
    "crime_lag_3": "Lag 3",
    "crime_rolling_mean_3": "Roll mean 3",
    "crime_rolling_mean_6": "Roll mean 6",
    "month_sin": "Month sin",
    "month_cos": "Month cos",
}


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def model_performance_chart() -> None:
    comp = pd.read_csv(OUT_DIR / "model_comparison.csv")
    temporal = comp[comp["evaluation"] == "temporal_holdout"].copy()
    order = [
        "six_month_rolling_mean",
        "three_month_rolling_mean",
        "one_month_lag",
        "current_month_crime",
        "lsoa_training_mean",
        "feature_a",
        "feature_b",
        "geo",
    ]
    labels = {
        "six_month_rolling_mean": "6-month rolling mean",
        "three_month_rolling_mean": "3-month rolling mean",
        "one_month_lag": "1-month lag",
        "current_month_crime": "Current month",
        "lsoa_training_mean": "LSOA training mean",
        "feature_a": "GeoSMILE Feature-A",
        "feature_b": "GeoSMILE Feature-B",
        "geo": "GeoSMILE Geo",
    }
    temporal["model"] = pd.Categorical(temporal["model"], categories=order, ordered=True)
    temporal = temporal.sort_values("model")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(temporal))
    for ax, metric, title in zip(
        axes,
        ["mae", "rmse", "r2"],
        ["MAE (lower better)", "RMSE (lower better)", "R² (higher better)"],
    ):
        colors = [TEAL if m.startswith("feature") or m == "geo" else INK_MUTED for m in temporal["model"]]
        ax.bar(x, temporal[metric], color=colors, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in temporal["model"]], rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Temporal holdout model comparison (next-month crime)", fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, "Simple baselines compared with three GeoSMILE branches on the same test months.", ha="center", fontsize=8, color=INK_MUTED)
    _save(fig, "model_performance_comparison.png")


def feature_b_incremental_chart() -> None:
    payload = json.loads((OUT_DIR / "feature_b_incremental_value.json").read_text(encoding="utf-8"))
    boot = payload["bootstrap"]
    mae_diff = payload["mae_difference_a_minus_b"]
    rmse_diff = payload["rmse_difference_a_minus_b"]
    mae_ci = boot["mae_improvement_a_minus_b_ci95"]
    rmse_ci = boot["rmse_improvement_a_minus_b_ci95"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    metrics = ["MAE\n(A − B)", "RMSE\n(A − B)"]
    vals = [mae_diff, rmse_diff]
    err_low = [vals[0] - mae_ci[0], vals[1] - rmse_ci[0]]
    err_high = [mae_ci[1] - vals[0], rmse_ci[1] - vals[1]]
    ax.bar(metrics, vals, color=[BLUE, PINK], yerr=[err_low, err_high], capsize=6, ecolor=INK)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_ylabel("Error difference (positive favours Feature-B)")
    ax.set_title("Feature-B incremental value (LSOA-clustered bootstrap, 95% CI)", fontweight="bold")
    note = (
        f"Prediction correlation: {payload['prediction_correlation']:.4f}. "
        f"Feature-B lower |error| on {payload['pct_observations_b_lower_abs_error']:.1f}% of test rows. "
        f"MAE CI significant: {boot['mae_improvement_significant_at_95']}; "
        f"RMSE CI significant: {boot['rmse_improvement_significant_at_95']}."
    )
    fig.text(0.5, -0.05, note, ha="center", fontsize=8, color=INK_MUTED, wrap=True)
    _save(fig, "feature_b_incremental_value.png")


def global_importance_charts() -> None:
    imp_a = pd.read_csv(OUT_DIR / "feature_a_global_importance.csv")
    imp_b = pd.read_csv(OUT_DIR / "feature_b_global_importance.csv")
    merged = imp_a.merge(imp_b, on="feature", suffixes=("_a", "_b"))
    merged["feature_label"] = merged["feature"].map(FEATURE_LABELS).fillna(merged["feature"])
    merged = merged.sort_values("abs_importance_a", ascending=True)
    ymax = max(merged["abs_importance_a"].max(), merged["abs_importance_b"].max()) * 1.08

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(merged))
    h = 0.35
    ax.barh(y - h / 2, merged["abs_importance_a"], height=h, label="Feature-A", color=BLUE)
    ax.barh(y + h / 2, merged["abs_importance_b"], height=h, label="Feature-B", color=PINK)
    ax.set_yticks(y)
    ax.set_yticklabels(merged["feature_label"])
    ax.set_xlim(0, ymax)
    ax.set_xlabel("|SMILE surrogate coefficient|")
    ax.set_title("Global SMILE surrogate importances (Wasserstein prediction shifts)", fontweight="bold")
    ax.legend()
    fig.text(0.5, -0.03, "Not predictive accuracy or causal effects — attribution approximation only.", ha="center", fontsize=8, color=INK_MUTED)
    _save(fig, "global_importance_comparison.png")

    merged["diff_b_minus_a"] = merged["abs_importance_b"] - merged["abs_importance_a"]
    merged = merged.sort_values("diff_b_minus_a")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PINK if v >= 0 else BLUE for v in merged["diff_b_minus_a"]]
    ax.barh(merged["feature_label"], merged["diff_b_minus_a"], color=colors)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("Feature-B minus Feature-A |coefficient|")
    ax.set_title("Change in global SMILE importance when coordinates condition prediction", fontweight="bold")
    _save(fig, "global_importance_difference.png")


def prediction_diagnostics() -> None:
    preds = pd.read_csv(OUT_DIR / "test_predictions.csv")
    preds["month"] = pd.to_datetime(preds["month"])
    y = preds["target_next_month_crime"]
    branches = [
        ("Feature-A", "prediction_feature_a", BLUE),
        ("Feature-B", "prediction_feature_b", PINK),
        ("Geo", "prediction_geo", TEAL),
    ]
    lim = max(y.max(), preds[[c for _, c, _ in branches]].max().max()) * 1.05

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    for ax, (name, col, color) in zip(axes, branches):
        ax.scatter(y, preds[col], alpha=0.35, s=12, color=color, edgecolors="none")
        ax.plot([0, lim], [0, lim], "--", color=INK_MUTED, linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Observed next-month crime")
        if ax is axes[0]:
            ax.set_ylabel("Predicted next-month crime")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    fig.suptitle("Observed vs predicted (temporal holdout)", fontweight="bold")
    _save(fig, "observed_vs_predicted.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, col, color) in zip(axes, branches):
        resid = y - preds[col]
        ax.hist(resid, bins=30, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("Residual (observed − predicted)")
    fig.suptitle("Residual distributions (temporal holdout)", fontweight="bold")
    _save(fig, "residual_distributions.png")

    monthly_rows = []
    for name, col, _ in branches:
        for month, g in preds.groupby("month"):
            monthly_rows.append(
                {
                    "month": month,
                    "model": name,
                    "mae": float((g["target_next_month_crime"] - g[col]).abs().mean()),
                }
            )
    monthly = pd.DataFrame(monthly_rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, color in [("Feature-A", BLUE), ("Feature-B", PINK), ("Geo", TEAL)]:
        sub = monthly[monthly["model"] == name]
        ax.plot(sub["month"], sub["mae"], marker="o", label=name, color=color)
    ax.set_ylabel("MAE")
    ax.set_title("Error by test month", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "error_by_test_month.png")

    preds["volume_quintile"] = pd.qcut(preds["target_next_month_crime"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    quint_rows = []
    for name, col, _ in branches:
        for q, g in preds.groupby("volume_quintile", observed=True):
            quint_rows.append(
                {
                    "volume_quintile": q,
                    "model": name,
                    "mae": float((g["target_next_month_crime"] - g[col]).abs().mean()),
                }
            )
    quint = pd.DataFrame(quint_rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.25
    x = np.arange(5)
    for i, (name, color) in enumerate([("Feature-A", BLUE), ("Feature-B", PINK), ("Geo", TEAL)]):
        sub = quint[quint["model"] == name]
        ax.bar(x + (i - 1) * width, sub["mae"], width=width, label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"])
    ax.set_ylabel("MAE")
    ax.set_title("Error by observed crime-volume quintile", fontweight="bold")
    ax.legend()
    _save(fig, "error_by_crime_quintile.png")

    # Residual maps (mean residual per LSOA, Feature-B)
    lsoa_codes = sorted(preds["lsoa21cd"].unique())
    boundaries, meta = load_boundaries(lsoa_codes)
    preds["residual_b"] = preds["target_next_month_crime"] - preds["prediction_feature_b"]
    resid_lsoa = preds.groupby("lsoa21cd")["residual_b"].mean().reset_index()
    rmap = boundaries.merge(resid_lsoa, on="lsoa21cd", how="inner")
    vmax = float(np.percentile(np.abs(rmap["residual_b"]), 98))
    fig, ax = plt.subplots(figsize=(8.5, 9))
    rmap.to_crs("EPSG:27700").plot(
        column="residual_b",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        linewidth=0.1,
        edgecolor="white",
        ax=ax,
        legend=True,
    )
    ax.set_axis_off()
    ax.set_title("Mean Feature-B residual by LSOA (temporal holdout)", fontweight="bold")
    if meta.get("source") == "voronoi_approximate" or meta.get("warning"):
        fig.text(0.5, 0.97, "Approximate Voronoi geography", ha="center", fontsize=10, color="#B00020", fontweight="bold")
        fig.text(0.5, 0.01, meta.get("warning", "Approximate Voronoi geography"), ha="center", fontsize=8, color="#B00020")
    else:
        fig.text(0.5, 0.01, "Source: ONS LSOA 2021 BGC. Residuals = observed − predicted.", ha="center", fontsize=8, color=INK_MUTED)
    _save(fig, "residual_map_feature_b.png")


def main() -> None:
    model_performance_chart()
    feature_b_incremental_chart()
    global_importance_charts()
    prediction_diagnostics()


if __name__ == "__main__":
    main()
