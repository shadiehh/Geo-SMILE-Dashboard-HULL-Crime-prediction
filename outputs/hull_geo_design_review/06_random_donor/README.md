# Random-donor coordinate masking (aligned design)

Script: `geo_random_donor_check.py`; map: `build_geo_design_review.py` → `random_donor_map_N4000.png`.
Same model, coalitions and surrogate as the aligned design, but a masked LSOA's rows get the
centroid of a random other Hull LSOA (10 draws per perturbation, mean W1); single-LSOA check = exact
mean over all 167 donors. Not causal.

| | Random donor | Median point (aligned) |
|---|---|---|
| Fidelity R² / wR² | 0.871 / 0.871 | 0.908 / 0.910 |
| Mean coalition W1 | 0.51 | 0.30 |
| Signed vs single-LSOA shift ρ (\|β\|) | 0.914 (−0.22) | 0.982 (−0.14) |
| Top 20 in expert regions | 16/20, lift 2.32, p = 1.2e-5, AUROC 0.61 | 10/20, p = 0.098 |
| Intervention top − random | 1.50 (p = 0.001) | 0.58 (p = 0.001) |
| Consistency rank ρ / mean SD | 0.868 / 0.0085 | 0.983 / 0.0023 |
| Moran's I | 0.247 (p = 0.001), 11 HH | 0.236 (p = 0.001) |
| Old Town rank | 1 | 2 |
| ρ with distance from median point | −0.01 | 0.22 |
| ρ with crime level (test period) | 0.78 (15/20 top-20 shared) | 0.63 (10/20) |
| ρ with median-point ranking / top-20 overlap | 0.74 / 9 | — |
| ρ with area-data ranking / top-20 overlap | 0.31 / 18 | 0.06 / — |

Top 10: 029E Old Town, 030B, 015D, 024B, 029C, 029D, 018C, 026A, 029G, 021B.

Scale check: the XGBoost model uses a Poisson (log-link) objective, so a location effect is
multiplicative and the absolute W1 shift grows with an area's prediction level. Dividing the
single-LSOA shift by mean crime removes the crime-level link (ρ −0.15) and leaves 6/20 inside the
expert regions (chance ≈ 7) with Old Town at rank 34. Ranking LSOAs by crime level alone gives 16/20.
