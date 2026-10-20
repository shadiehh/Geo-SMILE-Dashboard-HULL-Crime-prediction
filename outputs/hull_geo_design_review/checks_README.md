# Hull Geo-branch design checks

Checks run after the area-data Geo design was published in Chapter 4. None of them replaces or
changes a published output (`outputs/geo_area_data_eval/` is untouched). Not causal.

All three use XGBoost, seed 42, the published temporal split, N = 4,000 coalitions of 5–10 LSOAs on
test-period rows, the W1 response on affected rows and a kernel-weighted Ridge (RBF / Laplace /
inverse chosen on validation), exactly as in `geo_area_data.area_geo_coefficients`.

## 1. `crime_history_check/` — does area-data GeoSensitivity track crime level?

Script: `geo_crime_history_check.py`. Same 30-predictor model and coalitions as the published run.

| Ranking | Top-20 inside the 58 expert-region LSOAs (chance ≈ 6.9) | ρ with published Geo |
|---|---|---|
| Published area-data Geo (all 30 inputs masked) | 18/20 | 1.00 (reproduced) |
| Mask only the 5 crime-history inputs | 18/20 | 0.958 |
| Mask only the 25 non-crime inputs | 12/20 | 0.699 |
| Rank LSOAs by test-period crime level | 16/20 | — |

Spearman of signed Geo with each LSOA's distance from the training median: crime-history inputs
0.92, all inputs 0.59, non-crime inputs 0.39; with mean crime count 0.11, although 17 of the Geo
top 20 are among the 20 highest-crime LSOAs. The expert-region rescoring (table above) was computed
from `variant_*.csv`; `geo_crime_check.json` scores against the separate 20-LSOA expert top-20 list
(Geo 9/20, crime level 10/20).

## 2. `coordinate_masking_check/` — Seattle/Chicago-matched design vs area-data ranking

Script: `geo_coordinate_design_check.py`. Model: the 32-player XGBoost of
`geosmile_hull_feature_with_location.py` (30 predictors + centroid longitude/latitude; holdout
R² 0.895). Masking an LSOA moves **only** its rows' longitude/latitude to the single global
training-median coordinate (−0.349, 53.764); all other inputs are kept — the Seattle/Chicago Geo
operator.

- Signed coordinate Geo vs published area-data Geo: Spearman 0.03, Kendall 0.01, Pearson 0.47
  (0.27 without Old Town); top-20 overlap 7/20 (Jaccard 0.21). Only 029E Old Town is rank 1 in both.
- Top 20 inside the expert regions: 7/20 (chance 6.9).

## 3. `coordinate_masking_eval/` — Chapter-4 Geo metrics for the matched design

Script: `geo_coordinate_design_eval.py`. Computes every Chapter-4 Geo number except ATT accuracy and
ATT Faithfulness, with the published functions (`hull_geo_area_data_eval`, `eval_by_n_perturb`,
`compute_geo_spatial_patterns`); only the masking operator differs. `n_<N>.json` per budget,
`summary_by_n.csv`, `geo_coefficients_N4000.csv` (ranked, with single-LSOA shift).

At N = 4,000 (published area-data value in brackets):

| Quantity | Coordinate-only | (Area-data) |
|---|---|---|
| Fidelity R² / weighted R² | 0.9238 / 0.9238 | (0.9484 / 0.9461) |
| MAE / WMAE / RMSE / WRMSE | 0.0253 / 0.0256 / 0.0343 / 0.0347 | (0.785 / 0.801 / 1.355 / 1.400) |
| Mean W1 coalition response, crimes/month | 0.32 | (9.67) |
| Signed vs single-LSOA shift ρ_s; \|β\| | 0.984; −0.14 | (0.982; −0.29) |
| Single-LSOA shift median / P90 | 0.24 / 0.78 | (6.4 / 22.0) |
| Rank 1 | 029E Old Town, β = 0.228, shift 2.07 | (029E, 22.33, 176.7) |
| Global Moran's I (signed, queen, 999 perm.) | 0.29, p = 0.001; LISA 16 HH, 19 LL, 3 HL, 7 LH | (0.22, p = 0.001) |
| Concentration (shift): Hoyer / Gini / entropy | 0.322 / 0.493 / 4.712 | (0.516 / 0.529 / 4.544) |
| Intervention top / bottom / random mean | 1.050 / 0.037 / 0.323 (95% 0.200–0.480) | (37.84 / 1.42 / 11.61) |
| Top − random; p | 0.727; 0.001 | (26.22; 0.001) |
| Stability Jaccard / mean ρ; added ranks N/E/S/W | 1.000 / 0.9998; 82 / 44 / 30 / 83 | (1.000 / 0.9998; 82 / 57 / 101 / 53) |
| Fixed-kernel west | J 1.000, ρ 0.9999, rank 83 | (J 1.000, ρ 0.9998, rank 53) |
| Consistency mean / median variance | 9.1e-6 / 6.9e-6 | (0.0154 / 0.0109) |
| Mean / median / max SD | 0.0028 / 0.0026 / 0.0067 (002C) | (0.110 / 0.104 / 0.651, 029E) |
| Rank ρ / score Pearson / top-20% Jaccard | 0.980 / 0.996 / 0.932 | (0.963 / 0.997 / 0.773) |
| ρ_s with published area-data ranking | 0.03 | — |

Across N = 1,000 … 10,000 (`summary_by_n.csv`): fidelity 0.899 → 0.924 (highest at 4,000);
intervention top − random 0.715–0.730, p = 0.001 throughout; stability Jaccard 1.000 except 0.905 at
N = 2,000 (19/20 retained); mean variance falls from 3.4e-5 to 3.3e-6 and rank ρ rises from 0.926 to
0.991.

Re-run (from the repo root, ~1.5 h for the full sweep):

    python geo_crime_history_check.py
    python geo_coordinate_design_check.py
    python geo_coordinate_design_eval.py            # or: python geo_coordinate_design_eval.py 4000

## 4. `coordinate_reference_check/` — does the coordinate design depend on the reference point?

Script: `geo_coordinate_reference_check.py`. As check 2, but masked LSOAs' coordinates are moved to
different points: the training-median point, the four extreme points of the Hull boundary, 5 km beyond
the west edge, and a far point (−1.55, 53.80). All non-median points lie outside the range of the
training centroids.

| Reference | Fidelity R² | Mean W1 | Old Town rank | Top 20 in expert regions | ρ with median-point ranking | ρ with area-data Geo |
|---|---|---|---|---|---|---|
| Median point | 0.917 | 0.32 | 1 | 7 | 1.00 | 0.03 |
| Border north | 0.944 | 1.45 | 1 | 17 | 0.53 | 0.40 |
| Border south | 0.940 | 0.72 | 1 | 15 | 0.28 | 0.29 |
| Border east | 0.943 | 1.11 | 1 | 14 | 0.27 | 0.18 |
| Border west | 0.942 | 0.70 | 1 | 17 | 0.25 | 0.13 |
| 5 km outside west | 0.942 | 0.70 | 1 | 17 | 0.25 | 0.12 |
| Far outside (Leeds) | 0.945 | 1.42 | 1 | 17 | 0.42 | 0.28 |

West border vs 5 km outside: ρ = 0.998 (trees treat any point beyond the edge like the edge).
Between border references ρ = 0.57–0.89. Old Town is rank 1 under every reference point.
