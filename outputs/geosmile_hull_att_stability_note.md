# ATT / controlled-input stability (computed)

ATT stability compares SMILE attributions on the Hull-only test reference with
attributions after adding one real LSOA from each cardinal direction just
outside the city. Police.uk street counts and the same Census 2021 / OSM
covariates used for Hull LSOAs are collected for those neighbors. Quantified by:

- Jaccard overlap of top-k attribution sets (Feature k=10, Geo k=20)
- Spearman correlation of absolute coefficients over the full attribution vector

The predictive models stay Hull-trained. Hull-row prediction Spearman is a
sanity check that original test rows are unchanged. This is distinct from
multi-run Consistency. Baseline Feature and Geo rankings are the published
Hull-only GeoSMILE attributions (1024 Feature masks, 2000
Geo perturbations, seed 42); expansions reuse that design.

## Neighbors

- **North**: `E01012953` East Riding of Yorkshire 023A — Feature Jaccard 1.000, Geo Jaccard 1.000
- **East**: `E01013094` East Riding of Yorkshire 029A — Feature Jaccard 1.000, Geo Jaccard 1.000
- **South**: `E01013009` East Riding of Yorkshire 034B — Feature Jaccard 1.000, Geo Jaccard 1.000
- **West**: `E01013112` East Riding of Yorkshire 025D — Feature Jaccard 1.000, Geo Jaccard 1.000

## Results (four one-at-a-time additions)

Accepted if Hull-row prediction Spearman ≥ 0.99. Feature k=10, Geo k=20.

| Addition | Branch | Top-k Jaccard | abs coef Spearman | Pred. Spearman | Accepted |
|----------|--------|---------------|-------------------|----------------|----------|
| North | Feature | 1.000 | 1.000 | 1.000 | Yes |
| North | Geo | 1.000 | 0.999 | 1.000 | Yes |
| East | Feature | 1.000 | 0.999 | 1.000 | Yes |
| East | Geo | 1.000 | 0.999 | 1.000 | Yes |
| South | Feature | 1.000 | 1.000 | 1.000 | Yes |
| South | Geo | 1.000 | 1.000 | 1.000 | Yes |
| West | Feature | 1.000 | 0.999 | 1.000 | Yes |
| West | Geo | 1.000 | 0.999 | 1.000 | Yes |
| Mean | Feature | 1.000 | 0.999 | 1.000 | 4/4 |
| Mean | Geo | 1.000 | 0.999 | 1.000 | 4/4 |

Each run adds **one** neighbor LSOA as a real Geo player (fitted coefficient, not forced to 0). Hull core samples stay synchronized with the baseline. SMILE budgets: Feature masks=1024, Geo perturbations=2000.

## Extra LSOA Geo rank

The added LSOA is out of the Geo top-20 in every direction:

- **North**: extra rank 120, |coef| 0.317, in top-20: No
- **East**: extra rank 110, |coef| 0.433, in top-20: No
- **South**: extra rank 112, |coef| 0.396, in top-20: No
- **West**: extra rank 46, |coef| 0.945, in top-20: No

## Visuals

Per-snapshot folders under `D:/GEO SMILE CODE/hull/outputs/att_stability`:

- `baseline/`, `north/`, `east/`, `south/`, `west/`
- `cardinal_neighbors_map.png` — locator for the four added LSOAs
- `feature_importance_ranking.png` — Feature |coefficient| ranking (top-10 highlighted)
- `geo_top20_map.png` — Geo top-20 LSOA importance map
- matching `feature_importance.csv` / `geo_coefficients.csv`
