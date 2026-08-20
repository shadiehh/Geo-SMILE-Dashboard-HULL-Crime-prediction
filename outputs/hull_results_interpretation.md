# Hull GeoSMILE results interpretation

Verified outputs from the Hull pipeline (`outputs/run_summary.json` and companion files).
Target: next-month total street-crime count at LSOA-month resolution.
Temporal test period: 2026-01-01 to 2026-04-01.
Panel: 168 LSOAs; 4704 usable LSOA-months after lag construction.

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
| Feature-A | `crime_rolling_mean_6` | 5.3030 |
| Feature-B | `crime_rolling_mean_6` | 5.7215 |

Global importances are SMILE surrogate coefficients fitted to Wasserstein shifts of the prediction distribution under feature masking. High surrogate fidelity measures **attribution approximation quality**, not predictive accuracy or causal validity.

Surrogate test R²: Feature-A 0.9577; Feature-B 0.9702.

## Temporal holdout prediction performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Current-month crime | 5.5536 | 7.5982 | 0.8732 |
| One-month lag | 5.9301 | 8.1322 | 0.8547 |
| Three-month rolling mean | 5.2302 | 7.4515 | 0.8780 |
| **Six-month rolling mean** | **4.8683** | **7.0679** | **0.8902** |
| LSOA training mean | 5.2555 | 7.2837 | 0.8834 |
| GeoSMILE Feature-A | 4.9549 | 7.0379 | 0.8912 |
| GeoSMILE Feature-B | 4.8882 | 6.9796 | 0.8930 |
| GeoSMILE Geo | 5.5109 | 7.5376 | 0.8752 |

### Comparison with the six-month rolling-mean baseline

The six-month rolling mean is a strong leakage-safe baseline (uses only past crime). Relative to that baseline:

- Feature-A MAE difference (baseline − Feature-A): -0.0866
- Feature-B MAE difference (baseline − Feature-B): -0.0199
- Feature-A R² − baseline R²: 0.0009
- Feature-B R² − baseline R²: 0.0027

Positive MAE differences favour GeoSMILE (lower error than the baseline). On this temporal holdout the six-month rolling mean remains competitive: Feature-A/B show only small R² gains and do not clearly dominate MAE. Gains should be interpreted alongside residual diagnostics and bootstrap uncertainty on the Feature-A vs Feature-B contrast.

## Feature-B incremental value (temporal holdout)

Paired Feature-A vs Feature-B differences:

| Quantity | Value |
|---|---:|
| MAE (A − B) | 0.0668 |
| RMSE (A − B) | 0.0583 |
| R² (B − A) | 0.0018 |
| % test rows where Feature-B has lower absolute error | 50.15% |
| Correlation of Feature-A and Feature-B predictions | 0.997492 |

LSOA-clustered bootstrap (5000 repetitions):

- MAE improvement CI95 (A − B): [-0.0415, 0.1727]
- RMSE improvement CI95 (A − B): [-0.2068, 0.2677]
- MAE CI excludes zero: False
- RMSE CI excludes zero: False

The LSOA-clustered bootstrap 95% CI for the MAE improvement (A − B) is [-0.0415, 0.1727] and does **not** exclude zero. Therefore this analysis does **not** claim that coordinates improve prediction beyond Feature-A on the temporal holdout.

Feature-B improves only **marginally** over Feature-A in point estimates, and Feature-A / Feature-B predictions are **almost identical** (correlation 0.9975).

## Geo-only branch

Geo MAE 5.5109, RMSE 7.5376, R² 0.8752 on the temporal holdout.
Mean absolute location baseline-ablation effect: 11.776.

Geo-only performance partly reflects **persistent between-LSOA crime differences**: the same LSOAs appear in training and testing under the temporal split, so location can act as a stable area identifier. That is useful for isolating spatial sensitivity, but it is **not** a test of transfer to unseen places.

## Temporal holdout vs geographic transfer

The temporal holdout tests **forecasting future months for known LSOAs**. It does not evaluate spatial transfer.

A geographically blocked evaluation holds out spatially clustered LSOAs so that those areas never appear in training. That is a harder spatial-transfer test and must be reported separately.

### Spatial holdout (verified)

- Held-out LSOAs: 36
- Train LSOAs: 132
- Test rows: 1008

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Six-month rolling mean | 4.7257 | 7.2864 | 0.7696 |
| Feature-A | 4.8148 | 8.0327 | 0.7200 |
| Feature-B | 5.6213 | 9.3535 | 0.6204 |
| Geo | 13.0473 | 17.8985 | -0.3901 |


## Remaining limitations

1. Police.uk coordinates are anonymised; LSOA centroids used for modelling are incident medians unless replaced with official population-weighted centroids.
2. Baseline-ablation maps are model-sensitivity diagnostics, not causal maps.
3. Temporal holdout reuses LSOAs across train/test; spatial holdout is required for transfer claims.
4. Surrogate fidelity ≠ predictive skill ≠ causal validity.
5. If Voronoi geography is used, map shapes are approximate.
