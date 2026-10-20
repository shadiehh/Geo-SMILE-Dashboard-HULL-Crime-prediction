# Random-donor coordinate masking: number of donor draws K (log-scale response)

Script: `geo_random_donor_ksweep.py`. Aligned design (XGBoost on 30 predictors + BNG easting/northing),
N = 4,000, seed 42; response = mean over K donor draws of W1 on log(1 + prediction), which removes the
Poisson scaling of absolute shifts with crime level. Not causal.

| K | Fidelity R² | Consistency rank ρ (5 seeds) | ρ with exact (167 donors) | ρ with K=100 | Top-20 Jaccard with K=100 | Minutes |
|---|---|---|---|---|---|---|
| 1 | 0.208 | 0.794 | 0.867 | 0.885 | 0.82 | 1.6 |
| 2 | 0.358 | 0.859 | 0.904 | 0.936 | 0.74 | 1.8 |
| 5 | 0.565 | 0.929 | 0.943 | 0.970 | 0.90 | 2.0 |
| 10 | 0.662 | 0.956 | 0.945 | 0.985 | 0.90 | 2.6 |
| 20 | 0.742 | 0.967 | 0.959 | 0.989 | 0.82 | 3.3 |
| 30 | 0.761 | 0.969 | 0.961 | 0.993 | 0.90 | 4.2 |
| **50** | **0.798** | **0.975** | **0.964** | **0.994** | **1.00** | 6.0 |
| 100 | 0.804 | 0.975 | 0.965 | 1.000 | 1.00 | 10.3 |

Recommended K = 50: consistency and the ranking have converged (ρ 0.994 and identical top 20 vs
K = 100); K = 100 adds only +0.006 fidelity at twice the cost (the strict rule in the script, fidelity
within 0.005, picks 100). Fidelity plateaus near 0.80, below the other designs.

At K = 100: ρ with crime level 0.20 (absolute-scale K=10 version: 0.78); ρ with distance from the
median point 0.06; Moran's I 0.137 (p = 0.004); Old Town rank 6; top 5 = 021C, 015D, 034E, 009B,
030B. Agreement: area-data ρ 0.55 (top-20 overlap 9), median point 0.46 (9), absolute random-donor
0.62 (11). Map: `random_donor_log_K100_map.png`.
