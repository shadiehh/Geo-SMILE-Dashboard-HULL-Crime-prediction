"""Write the Chapter 4 results the dashboard reads into outputs/dashboard_final/.

    python build_final_dashboard_results.py

Source run: hull_aligned_design_eval.py (XGBoost on the 30 predictors plus the two centroid
coordinates, N = 4,000, seed 42), outputs/geo_aligned_design/xgboost/.

Local Feature coefficients follow the convention of Chapter 3: masks code a retained player as 1
and the response is the perturbed minus the original prediction, so a positive coefficient means
the area's own value raises its prediction relative to the masking reference. Runs stored before
that convention was fixed carry the opposite sign. The sign of each stored file is measured here,
against the data, and flipped only if needed, so rerunning this script is safe.

ATT Faithfulness is carried for the Geo branch only (the random-donor run).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SRC = OUT / "geo_aligned_design" / "xgboost"
DEST = OUT / "dashboard_final"
TEST_START = pd.Timestamp("2026-01-01")
PROBE = "crime_rolling_mean_6"
ID, NAME = "lsoa21cd", "lsoa21nm"
LOCATION_PLAYERS = ["centroid_longitude", "centroid_latitude"]


def test_means() -> pd.DataFrame:
    panel = pd.read_csv(OUT / "hull_lsoa_month_panel_with_covariates.csv", dtype={ID: str}, parse_dates=["month"])
    return panel[panel["month"] >= TEST_START].groupby(ID).mean(numeric_only=True)


def stored_sign(local: pd.DataFrame, means: pd.DataFrame) -> tuple[int, float]:
    """+1 if higher recent crime gets a higher coefficient (the Chapter 3 convention), else -1."""
    joined = local[[PROBE]].join(means[[PROBE]], rsuffix="_value").dropna()
    rho = float(spearmanr(joined.iloc[:, 0], joined.iloc[:, 1]).statistic)
    if abs(rho) < 0.9:
        raise SystemExit(f"Cannot establish the stored sign: rho = {rho:.3f}")
    return (1 if rho > 0 else -1), rho


def names() -> pd.Series:
    return pd.read_csv(OUT / "geo_random_donor_eval" / "geo_coefficients_N4000.csv", dtype={ID: str}).set_index(ID)[NAME]


def write_local(means: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(SRC / "feature_local_coefficients_N4000.csv", dtype={ID: str}).set_index(ID)
    frame = frame[[c for c in frame.columns if not c.startswith("Unnamed") and c != NAME]]
    sign, rho = stored_sign(frame, means)
    out = sign * frame
    final_rho = stored_sign(out, means)[1]
    if final_rho <= 0 or out.loc["E01033104", PROBE] <= 0:
        raise SystemExit("Corrected local coefficients fail the direction check")
    out.insert(0, NAME, names().reindex(out.index))
    if out[NAME].isna().any() or out.shape != (168, 31):
        raise SystemExit(f"Unexpected local table: {out.shape}")
    out.to_csv(DEST / "feature_local_coefficients_N4000.csv")
    print(f"local: stored rho {rho:+.3f}, multiplied by {sign:+d}, final rho {final_rho:+.3f}")
    return out


def write_location(means: pd.DataFrame) -> None:
    """The coordinate-inclusive diagnostic run, sign-aligned from its own substantive players."""
    full = pd.read_csv(OUT / "feature_with_location" / "feature_with_location_local_coefficients.csv", dtype={ID: str})
    sign, rho = stored_sign(full.set_index(ID), means)
    location = pd.read_csv(OUT / "feature_with_location" / "location_local_coefficients.csv", dtype={ID: str})
    location[LOCATION_PLAYERS] = sign * location[LOCATION_PLAYERS]
    location.to_csv(DEST / "location_local_coefficients.csv", index=False)
    print(f"location: columns {list(location.columns)}, stored rho {rho:+.3f}, multiplied by {sign:+d}")


def write_metrics() -> None:
    run = json.loads((SRC / "n_4000.json").read_text(encoding="utf-8"))
    (DEST / "feature_metrics_N4000.json").write_text(
        json.dumps({"n": run["n"], "model": run["model"], "predictive": run["predictive"], "feature": run["feature"]},
                   indent=2), encoding="utf-8")
    pd.read_csv(SRC / "feature_global_importance_N4000.csv").sort_values(
        "abs_coefficient", ascending=False, kind="mergesort").to_csv(
        DEST / "feature_global_importance_N4000.csv", index=False)


def write_faithfulness() -> None:
    donor = pd.read_csv(OUT / "faithfulness_accuracy_r2" / "att_faithfulness_random_donor.csv")
    donor[donor["branch"] == "Geo"].to_csv(DEST / "att_faithfulness.csv", index=False)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    means = test_means()
    write_local(means)
    write_location(means)
    write_metrics()
    write_faithfulness()
    print(f"Wrote {DEST}")


if __name__ == "__main__":
    main()
