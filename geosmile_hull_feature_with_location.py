"""Feature SMILE with location as attribution players.

Distinct from the published two-branch run, where Feature excludes coordinates
and Feature-B used coordinates for prediction only.

This file trains and attributes on the 30 non-spatial Feature players plus
centroid longitude and latitude (32 players). Outputs go under
outputs/feature_with_location/ and do not overwrite published rankings.

Same SMILE design as the published Feature run: 1024 masks, seed 42.
Not causal.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

from hull_boundaries import load_boundaries
from geosmile_hull_twobranch import (
    FEATURE_LABELS,
    RANDOM_SEED,
    SPATIAL_COLUMNS,
    choose_feature_players,
    local_feature_coefficients,
    make_model,
    metrics,
    plot_choropleth,
    shared_limit,
    smile_global_feature,
    temporal_split,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"
DEST = OUT / "feature_with_location"
LOCATION_PLAYERS = ["centroid_longitude", "centroid_latitude"]
N_MASKS = 1024
SEED = RANDOM_SEED
INK, INK_MUTED = "#12171C", "#57646D"
PINK, MUTED_BAR = "#FF0051", "#B7C0C8"

LOCATION_LABELS = {
    "centroid_longitude": "Longitude",
    "centroid_latitude": "Latitude",
}


def bng_to_lonlat(easting: pd.Series, northing: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    return transformer.transform(easting.to_numpy(), northing.to_numpy())


def add_lonlat(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach LSOA centroid lon/lat from existing BNG easting/northing."""
    out = panel.copy()
    loc = (
        out[["lsoa21cd", "centroid_easting", "centroid_northing"]]
        .drop_duplicates("lsoa21cd")
        .dropna()
    )
    lon, lat = bng_to_lonlat(loc["centroid_easting"], loc["centroid_northing"])
    loc = loc.assign(centroid_longitude=lon, centroid_latitude=lat)
    if "centroid_longitude" in out.columns:
        out = out.drop(columns=["centroid_longitude", "centroid_latitude"], errors="ignore")
    return out.merge(
        loc[["lsoa21cd", "centroid_longitude", "centroid_latitude"]],
        on="lsoa21cd",
        how="left",
        validate="many_to_one",
    )


def choose_feature_players_with_location(panel: pd.DataFrame) -> list[str]:
    """Published 30 Feature players plus longitude and latitude.

    Location columns are excluded from ``choose_feature_players`` (that helper
    forbids lat/lon tokens) and then appended as explicit SMILE players.
    """
    missing = [c for c in LOCATION_PLAYERS if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel missing location columns: {missing}")
    stripped = panel.drop(columns=list(LOCATION_PLAYERS), errors="ignore")
    base = choose_feature_players(stripped)
    if any(c in base for c in LOCATION_PLAYERS):
        raise ValueError("Location columns leaked into the non-spatial Feature set")
    if any(c in base for c in SPATIAL_COLUMNS):
        raise ValueError("BNG coordinates must not be Feature players here")
    return list(base) + list(LOCATION_PLAYERS)


def _label(name: str) -> str:
    return LOCATION_LABELS.get(name, FEATURE_LABELS.get(name, name.replace("_", " ")))


def save_ranking_png(importance: pd.DataFrame, path: Path) -> None:
    ranked = importance.sort_values(
        ["abs_coefficient", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    loc_set = set(LOCATION_PLAYERS)
    labels = [f"{int(r)}  {_label(f)}" for r, f in zip(ranked["rank"], ranked["feature"])]
    vals = ranked["abs_coefficient"].to_numpy(dtype=float)
    colors = [PINK if f in loc_set else MUTED_BAR for f in ranked["feature"].astype(str)]
    fig, ax = plt.subplots(figsize=(8.4, max(6.8, 0.28 * len(ranked) + 1.6)))
    y = np.arange(len(ranked))[::-1]
    ax.barh(y, vals, color=colors, edgecolor="white", height=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.set_xlabel("|SMILE coefficient|", fontsize=10, color=INK_MUTED)
    ax.set_title(
        "Feature SMILE with location players (lon/lat + 30 covariates)",
        fontsize=12,
        fontweight="bold",
        color=INK,
        pad=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.text(
        0.5,
        0.01,
        "Pink bars are centroid longitude and latitude. "
        "Same SMILE design as published Feature (1024 masks, seed 42). Not causal.",
        ha="center",
        fontsize=7.0,
        color=INK_MUTED,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    plt.close("all")


def local_tables(
    local: pd.DataFrame, players: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wide absolute, long-form (LSOA × feature), and location-only local tables."""
    missing = [c for c in players if c not in local.columns]
    if missing:
        raise ValueError(f"Local coefficients missing players: {missing}")
    wide_abs = local.copy()
    for player in players:
        wide_abs[player] = wide_abs[player].abs()
    long = local.melt(
        id_vars=["lsoa21cd", "lsoa21nm"],
        value_vars=players,
        var_name="feature",
        value_name="coefficient",
    )
    long["abs_coefficient"] = long["coefficient"].abs()
    long["is_location"] = long["feature"].isin(LOCATION_PLAYERS).astype(int)
    long = long.sort_values(
        ["lsoa21cd", "is_location", "abs_coefficient"],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    loc = local[["lsoa21cd", "lsoa21nm", *LOCATION_PLAYERS]].copy()
    loc["centroid_longitude_abs"] = loc["centroid_longitude"].abs()
    loc["centroid_latitude_abs"] = loc["centroid_latitude"].abs()
    loc["location_abs_mean"] = loc[
        ["centroid_longitude_abs", "centroid_latitude_abs"]
    ].mean(axis=1)
    return wide_abs, long, loc


def location_local_summary(loc: pd.DataFrame) -> dict:
    out = {"n_lsoas": int(len(loc))}
    for col in LOCATION_PLAYERS:
        vals = loc[col].to_numpy(dtype=float)
        out[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean_abs": float(np.mean(np.abs(vals))),
        }
    return out


def write_local_maps(
    local: pd.DataFrame,
    players: list[str],
    test_period: str,
    out_dir: Path,
) -> dict:
    """Signed and absolute choropleths for every player, including lon/lat."""
    boundaries, _ = load_boundaries(sorted(local["lsoa21cd"].astype(str).unique()))
    signed_dir = out_dir / "maps" / "signed"
    abs_dir = out_dir / "maps" / "absolute"
    signed_dir.mkdir(parents=True, exist_ok=True)
    abs_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[str]] = {"feature_signed": [], "feature_abs": []}
    for player in players:
        merged = boundaries.merge(
            local[["lsoa21cd", player]], on="lsoa21cd", how="inner", validate="one_to_one"
        )
        merged["abs_value"] = merged[player].abs()
        vmax = shared_limit(merged[player].to_numpy())
        abs_vmax = shared_limit(merged["abs_value"].to_numpy())
        s_name = f"feature_{player}_signed.png"
        a_name = f"feature_{player}_abs.png"
        plot_choropleth(
            merged,
            player,
            f"Hull — {_label(player)}\nFeature SMILE local coefficient (signed)",
            signed_dir / s_name,
            signed=True,
            vmax=vmax,
            test_period=test_period,
            cbar_label="Signed local SMILE coefficient",
        )
        plot_choropleth(
            merged,
            "abs_value",
            f"Hull — {_label(player)}\nFeature SMILE local coefficient (absolute)",
            abs_dir / a_name,
            signed=False,
            vmax=abs_vmax,
            test_period=test_period,
            cbar_label="|Local SMILE coefficient|",
        )
        manifest["feature_signed"].append(f"signed/{s_name}")
        manifest["feature_abs"].append(f"absolute/{a_name}")
    (out_dir / "maps" / "map_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    if not PANEL_PATH.exists():
        raise SystemExit(f"Missing {PANEL_PATH}")
    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    panel = add_lonlat(panel)
    if panel[LOCATION_PLAYERS].isna().any().any():
        raise SystemExit("NA in longitude/latitude")

    players = choose_feature_players_with_location(panel)
    if len(players) != 32:
        raise SystemExit(f"Expected 32 players (30 + lon/lat), got {len(players)}: {players}")
    print(f"Players ({len(players)}): {players}")

    train, validation, test = temporal_split(panel)
    y_col = "target_next_month_crime"
    baseline = panel.loc[train, players].median()

    model = make_model(SEED)
    model.fit(
        panel.loc[train, players],
        panel.loc[train, y_col],
        eval_set=[(panel.loc[validation, players], panel.loc[validation, y_col])],
        verbose=False,
    )

    X_test = panel.loc[test].reset_index(drop=True)
    pred = np.asarray(model.predict(X_test[players]), float)
    y_true = X_test[y_col].to_numpy(dtype=float)
    holdout = metrics(y_true, pred)

    importance, diag, masks, all_pert, original = smile_global_feature(
        model, X_test[players], players, baseline, N_MASKS, SEED
    )
    importance = importance.sort_values(
        ["abs_coefficient", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    importance["is_location"] = importance["feature"].isin(LOCATION_PLAYERS).astype(int)

    local = local_feature_coefficients(
        masks,
        all_pert,
        original,
        players,
        X_test["lsoa21cd"].to_numpy(),
        diag["best_kernel"],
        SEED,
    )
    names = panel[["lsoa21cd", "lsoa21nm"]].drop_duplicates("lsoa21cd")
    local = names.merge(local, on="lsoa21cd", how="right")

    DEST.mkdir(parents=True, exist_ok=True)
    importance.to_csv(DEST / "feature_with_location_global_importance.csv", index=False)
    wide_abs, long_local, loc_local = local_tables(local, players)
    local.to_csv(DEST / "feature_with_location_local_coefficients.csv", index=False)
    wide_abs.to_csv(DEST / "feature_with_location_local_coefficients_abs.csv", index=False)
    long_local.to_csv(DEST / "feature_with_location_local_coefficients_long.csv", index=False)
    loc_local.to_csv(DEST / "location_local_coefficients.csv", index=False)
    preds = X_test[["lsoa21cd", "lsoa21nm", "month", y_col]].copy()
    preds["prediction_feature_with_location"] = pred
    preds.to_csv(DEST / "test_predictions.csv", index=False)
    save_ranking_png(importance, DEST / "feature_with_location_ranking.png")
    test_period = f"{preds['month'].min().date()} to {preds['month'].max().date()}"
    write_local_maps(local, LOCATION_PLAYERS, test_period, DEST)

    loc_rows = importance[importance["is_location"] == 1]
    report = {
        "design": "feature_smile_with_location_players",
        "n_players": len(players),
        "players": players,
        "location_players": LOCATION_PLAYERS,
        "n_nonspatial_players": 30,
        "n_lsoas_local": int(local["lsoa21cd"].nunique()),
        "feat_masks": N_MASKS,
        "smile_seed": SEED,
        "holdout": holdout,
        "surrogate": diag,
        "location_attribution_global": [
            {
                "feature": str(r.feature),
                "rank": int(r.rank),
                "coefficient": float(r.coefficient),
                "abs_coefficient": float(r.abs_coefficient),
            }
            for r in loc_rows.itertuples()
        ],
        "location_attribution_local": location_local_summary(loc_local),
        "notes": [
            "Location (centroid longitude and latitude) is both a predictor and a SMILE player.",
            "Local coefficients are per-observation SMILE Ridge coefficients, averaged to LSOA.",
            "This is distinct from published Feature (no coordinates) and Feature-B (coordinates in prediction only).",
            "Does not overwrite outputs/feature_global_importance.csv.",
            "Not causal.",
        ],
    }
    (DEST / "run_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {DEST}")


def export_local_from_csv() -> None:
    """Write LSOA local tables and lon/lat maps from the existing local CSV."""
    local_path = DEST / "feature_with_location_local_coefficients.csv"
    if not local_path.exists():
        raise SystemExit(f"Missing {local_path}; run the full SMILE script first")
    local = pd.read_csv(local_path)
    players = [c for c in local.columns if c not in {"lsoa21cd", "lsoa21nm"}]
    if any(c not in players for c in LOCATION_PLAYERS):
        raise SystemExit("Local CSV is missing longitude/latitude coefficients")
    preds_path = DEST / "test_predictions.csv"
    if preds_path.exists():
        preds = pd.read_csv(preds_path)
        preds["month"] = pd.to_datetime(preds["month"])
        test_period = f"{preds['month'].min().date()} to {preds['month'].max().date()}"
    else:
        test_period = "hold-out test months"
    wide_abs, long_local, loc_local = local_tables(local, players)
    DEST.mkdir(parents=True, exist_ok=True)
    wide_abs.to_csv(DEST / "feature_with_location_local_coefficients_abs.csv", index=False)
    long_local.to_csv(DEST / "feature_with_location_local_coefficients_long.csv", index=False)
    loc_local.to_csv(DEST / "location_local_coefficients.csv", index=False)
    write_local_maps(local, LOCATION_PLAYERS, test_period, DEST)
    summary_path = DEST / "run_summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    report["n_lsoas_local"] = int(local["lsoa21cd"].nunique())
    report["location_attribution_local"] = location_local_summary(loc_local)
    notes = list(report.get("notes") or [])
    extra = "Local coefficients are per-observation SMILE Ridge coefficients, averaged to LSOA."
    if extra not in notes:
        notes.append(extra)
    report["notes"] = notes
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report.get("location_attribution_local"), indent=2))
    print(f"Wrote local LSOA artefacts to {DEST}")


if __name__ == "__main__":
    import sys

    if "--from-existing" in sys.argv:
        export_local_from_csv()
    else:
        main()
