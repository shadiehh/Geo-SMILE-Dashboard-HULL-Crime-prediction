"""Write thesis interpretation from verified pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def main() -> None:
    summary = json.loads((OUT / "run_summary.json").read_text(encoding="utf-8"))
    fb = json.loads((OUT / "feature_b_incremental_value.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(OUT / "model_comparison.csv")
    temporal = comparison[comparison["evaluation"] == "temporal_holdout"].set_index("model")
    spatial = comparison[comparison["evaluation"] == "spatial_holdout"].set_index("model")
    imp_a = pd.read_csv(OUT / "feature_a_global_importance.csv")
    imp_b = pd.read_csv(OUT / "feature_b_global_importance.csv")
    boundary_meta = {}
    boundary_path = ROOT / "data" / "boundary_validation.json"
    if boundary_path.exists():
        boundary_meta = json.loads(boundary_path.read_text(encoding="utf-8"))

    top_a = imp_a.iloc[0]
    top_b = imp_b.iloc[0]
    roll6 = temporal.loc["six_month_rolling_mean"]
    fa = temporal.loc["feature_a"]
    fb_row = temporal.loc["feature_b"]
    geo = temporal.loc["geo"]
    boot = fb["bootstrap"]
    mae_ci = boot["mae_improvement_a_minus_b_ci95"]
    rmse_ci = boot["rmse_improvement_a_minus_b_ci95"]
    mae_sig = boot["mae_improvement_significant_at_95"]
    rmse_sig = boot["rmse_improvement_significant_at_95"]

    if mae_sig and mae_ci[0] > 0:
        coord_claim = (
            "The LSOA-clustered bootstrap 95% CI for the MAE improvement (A − B) excludes zero "
            f"([{_fmt(mae_ci[0], 4)}, {_fmt(mae_ci[1], 4)}]), so a reduction in MAE for Feature-B "
            "is supported at the 95% level on this temporal holdout."
        )
    else:
        coord_claim = (
            "The LSOA-clustered bootstrap 95% CI for the MAE improvement (A − B) is "
            f"[{_fmt(mae_ci[0], 4)}, {_fmt(mae_ci[1], 4)}] and does **not** exclude zero. "
            "Therefore this analysis does **not** claim that coordinates improve prediction "
            "beyond Feature-A on the temporal holdout."
        )

    spatial_note = summary.get("spatial_holdout", {})
    boundary_warning = ""
    if boundary_meta.get("source") == "voronoi_approximate" or boundary_meta.get("warning"):
        boundary_warning = (
            "\n## Boundary limitation\n\n"
            "Official ONS LSOA 2021 polygons could not be obtained automatically. Maps use "
            "**approximate Voronoi geography** derived from Police.uk incident-median centroids. "
            "These shapes must not be treated as official LSOA boundaries in publication.\n"
        )

    text = f"""# Hull GeoSMILE results interpretation

Verified outputs from the Hull pipeline (`outputs/run_summary.json` and companion files).
Target: next-month total street-crime count at LSOA-month resolution.
Temporal test period: {summary['test_period'][0]} to {summary['test_period'][1]}.
Panel: {summary['lsoas']} LSOAs; {summary['rows']} usable LSOA-months after lag construction.

## Local-effect terminology

Mapped values are **signed baseline-ablation effects**, not local SMILE Ridge coefficients.

- Effect = prediction with the original feature value − prediction after replacing that feature with the **training-period median**.
- Positive: the original value **raises** the prediction relative to the median replacement.
- Negative: the original value **lowers** the prediction relative to the median replacement.
- These are association diagnostics for the fitted model, **not causal effects**.

Feature-B attribution players exclude coordinates. Coordinates condition the Feature-B prediction but are never SMILE attribution players (`feature_b_coordinates_attributed: false`).

## Dominant predictors

Recent six-month crime history dominates both global SMILE surrogate rankings:

| Branch | Top feature | |surrogate coefficient| |
|---|---|---|
| Feature-A | `{top_a['feature']}` | {_fmt(float(top_a['abs_importance']), 4)} |
| Feature-B | `{top_b['feature']}` | {_fmt(float(top_b['abs_importance']), 4)} |

Global importances are SMILE surrogate coefficients fitted to Wasserstein shifts of the prediction distribution under feature masking. High surrogate fidelity measures **attribution approximation quality**, not predictive accuracy or causal validity.

Surrogate test R²: Feature-A {_fmt(summary['surrogate_diagnostics']['feature_a']['test_r2'], 4)}; Feature-B {_fmt(summary['surrogate_diagnostics']['feature_b']['test_r2'], 4)}.

## Temporal holdout prediction performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Current-month crime | {_fmt(temporal.loc['current_month_crime','mae'])} | {_fmt(temporal.loc['current_month_crime','rmse'])} | {_fmt(temporal.loc['current_month_crime','r2'])} |
| One-month lag | {_fmt(temporal.loc['one_month_lag','mae'])} | {_fmt(temporal.loc['one_month_lag','rmse'])} | {_fmt(temporal.loc['one_month_lag','r2'])} |
| Three-month rolling mean | {_fmt(temporal.loc['three_month_rolling_mean','mae'])} | {_fmt(temporal.loc['three_month_rolling_mean','rmse'])} | {_fmt(temporal.loc['three_month_rolling_mean','r2'])} |
| **Six-month rolling mean** | **{_fmt(roll6['mae'])}** | **{_fmt(roll6['rmse'])}** | **{_fmt(roll6['r2'])}** |
| LSOA training mean | {_fmt(temporal.loc['lsoa_training_mean','mae'])} | {_fmt(temporal.loc['lsoa_training_mean','rmse'])} | {_fmt(temporal.loc['lsoa_training_mean','r2'])} |
| GeoSMILE Feature-A | {_fmt(fa['mae'])} | {_fmt(fa['rmse'])} | {_fmt(fa['r2'])} |
| GeoSMILE Feature-B | {_fmt(fb_row['mae'])} | {_fmt(fb_row['rmse'])} | {_fmt(fb_row['r2'])} |
| GeoSMILE Geo | {_fmt(geo['mae'])} | {_fmt(geo['rmse'])} | {_fmt(geo['r2'])} |

### Comparison with the six-month rolling-mean baseline

The six-month rolling mean is a strong leakage-safe baseline (uses only past crime). Relative to that baseline:

- Feature-A MAE difference (baseline − Feature-A): {_fmt(float(roll6['mae'] - fa['mae']))}
- Feature-B MAE difference (baseline − Feature-B): {_fmt(float(roll6['mae'] - fb_row['mae']))}
- Feature-A R² − baseline R²: {_fmt(float(fa['r2'] - roll6['r2']))}
- Feature-B R² − baseline R²: {_fmt(float(fb_row['r2'] - roll6['r2']))}

Positive MAE differences favour GeoSMILE (lower error than the baseline). On this temporal holdout the six-month rolling mean remains competitive: Feature-A/B show only small R² gains and do not clearly dominate MAE. Gains should be interpreted alongside residual diagnostics and bootstrap uncertainty on the Feature-A vs Feature-B contrast.

## Feature-B incremental value (temporal holdout)

Paired Feature-A vs Feature-B differences:

| Quantity | Value |
|---|---:|
| MAE (A − B) | {_fmt(fb['mae_difference_a_minus_b'])} |
| RMSE (A − B) | {_fmt(fb['rmse_difference_a_minus_b'])} |
| R² (B − A) | {_fmt(fb['r2_difference_b_minus_a'])} |
| % test rows where Feature-B has lower absolute error | {_fmt(fb['pct_observations_b_lower_abs_error'], 2)}% |
| Correlation of Feature-A and Feature-B predictions | {_fmt(fb['prediction_correlation'], 6)} |

LSOA-clustered bootstrap ({boot['n_repetitions']} repetitions):

- MAE improvement CI95 (A − B): [{_fmt(mae_ci[0])}, {_fmt(mae_ci[1])}]
- RMSE improvement CI95 (A − B): [{_fmt(rmse_ci[0])}, {_fmt(rmse_ci[1])}]
- MAE CI excludes zero: {mae_sig}
- RMSE CI excludes zero: {rmse_sig}

{coord_claim}

Feature-B improves only **marginally** over Feature-A in point estimates, and Feature-A / Feature-B predictions are **almost identical** (correlation {_fmt(fb['prediction_correlation'], 4)}).

## Geo-only branch

Geo MAE {_fmt(geo['mae'])}, RMSE {_fmt(geo['rmse'])}, R² {_fmt(geo['r2'])} on the temporal holdout.
Mean absolute location baseline-ablation effect: {_fmt(summary['geo_mean_absolute_location_effect'], 3)}.

Geo-only performance partly reflects **persistent between-LSOA crime differences**: the same LSOAs appear in training and testing under the temporal split, so location can act as a stable area identifier. That is useful for isolating spatial sensitivity, but it is **not** a test of transfer to unseen places.

## Temporal holdout vs geographic transfer

The temporal holdout tests **forecasting future months for known LSOAs**. It does not evaluate spatial transfer.

A geographically blocked evaluation holds out spatially clustered LSOAs so that those areas never appear in training. That is a harder spatial-transfer test and must be reported separately.

### Spatial holdout (verified)

- Held-out LSOAs: {spatial_note.get('n_held_out_lsoas', 'n/a')}
- Train LSOAs: {spatial_note.get('n_train_lsoas', 'n/a')}
- Test rows: {spatial_note.get('n_test_rows', 'n/a')}

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Six-month rolling mean | {_fmt(spatial.loc['six_month_rolling_mean','mae']) if 'six_month_rolling_mean' in spatial.index else float('nan')} | {_fmt(spatial.loc['six_month_rolling_mean','rmse']) if 'six_month_rolling_mean' in spatial.index else float('nan')} | {_fmt(spatial.loc['six_month_rolling_mean','r2']) if 'six_month_rolling_mean' in spatial.index else float('nan')} |
| Feature-A | {_fmt(spatial.loc['feature_a','mae']) if 'feature_a' in spatial.index else float('nan')} | {_fmt(spatial.loc['feature_a','rmse']) if 'feature_a' in spatial.index else float('nan')} | {_fmt(spatial.loc['feature_a','r2']) if 'feature_a' in spatial.index else float('nan')} |
| Feature-B | {_fmt(spatial.loc['feature_b','mae']) if 'feature_b' in spatial.index else float('nan')} | {_fmt(spatial.loc['feature_b','rmse']) if 'feature_b' in spatial.index else float('nan')} | {_fmt(spatial.loc['feature_b','r2']) if 'feature_b' in spatial.index else float('nan')} |
| Geo | {_fmt(spatial.loc['geo','mae']) if 'geo' in spatial.index else float('nan')} | {_fmt(spatial.loc['geo','rmse']) if 'geo' in spatial.index else float('nan')} | {_fmt(spatial.loc['geo','r2']) if 'geo' in spatial.index else float('nan')} |

{boundary_warning}
## Remaining limitations

1. Police.uk coordinates are anonymised; LSOA centroids used for modelling are incident medians unless replaced with official population-weighted centroids.
2. Baseline-ablation maps are model-sensitivity diagnostics, not causal maps.
3. Temporal holdout reuses LSOAs across train/test; spatial holdout is required for transfer claims.
4. Surrogate fidelity ≠ predictive skill ≠ causal validity.
5. If Voronoi geography is used, map shapes are approximate.
"""
    path = OUT / "hull_results_interpretation.md"
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
