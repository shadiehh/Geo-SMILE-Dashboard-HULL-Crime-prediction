# Hull, Seattle/Chicago-aligned design — Chapter-4 numbers

Script: `hull_aligned_design_eval.py` (run log `run.log`). Replaces nothing; published results stay in
`outputs/geo_area_data_eval/` and `outputs/geo_area_data_models/`. Not causal.

**Design.** One model on 32 inputs: the 30 predictors + BNG centroid easting/northing (model_factory,
seed 42). Feature branch: 30 players; masked predictors → training median, coordinates kept true.
Geo branch: 168 LSOAs; masking moves only the LSOA's coordinates to the training-median point; all 30
predictors kept. Every metric uses the published Chapter-4 functions. XGBoost holdout: R² 0.8952,
MAE 4.831, RMSE 6.907 (Chapter-4 Stage-1 "+ coordinates" column lists 0.9000 — a different run).

Files: `xgboost/n_<N>.json` (all metrics per N), `xgboost/feature_global_importance_N4000.csv`,
`xgboost/feature_local_coefficients_N4000.csv`, `xgboost/geo_coefficients_N4000.csv`,
`models/<model>.json`, `models/model_summary.csv`, `models/{feature,geo}_surrogate_menu_r2.csv`,
`models/att_faithfulness.csv`.

## XGBoost, N = 4,000 (published value in brackets)

| | Feature | Geo |
|---|---|---|
| Fidelity R² / wR² | 0.9850 / 0.9850 (0.9770 / 0.9771) | 0.9083 / 0.9096 (0.9484 / 0.9461) |
| MAE / WMAE / RMSE / WRMSE | 0.296 / 0.296 / 0.340 / 0.341 | 0.0229 / 0.0231 / 0.0305 / 0.0307 |
| ATT accuracy | 7/10 (6/10); AUROC 0.845; Jaccard 0.538 | 10/20 in regions (18/20); lift 1.45, p = 0.098; AUROC 0.521 |
| Intervention top − random (p) | 4.499 (0.010) (4.246, 0.018) | 0.580 (0.001) (26.22, 0.001) |
| Stability J / ρ | 1.000 / 0.9991 | 1.000 / 0.9998; added ranks 99/56/32/91 |
| Consistency mean var / mean SD | 1.97e-4 / 0.0130 | 6.4e-6 / 0.00234 |
| Rank ρ / top-20% Jaccard | 0.915 / 1.000 | 0.983 / 0.916 |

Feature top 10 (coef): rolling mean 6 −5.320, rolling mean 3 −1.330, unemployment −0.396, long-term sick
−0.341, lag 1 −0.225, owner occupied −0.200, lag 2 −0.180, social rented −0.106, road length −0.100,
private rented −0.087. Mapped six: the first six. Concentration Hoyer 0.858, Gini 0.814, entropy 1.696.
Shared with reference: lag 1, lag 2, rolling 3, rolling 6, unemployment, owner occupied, social rented.

Geo: top 10 = 034E (0.183), 029E Old Town (0.135), 002D, 034B, 030B, 028D, 029D, 031E, 029C, 030C.
Signed vs single-LSOA shift ρ = 0.982 (|β| −0.137); shift median 0.22, P90 0.69 crimes/month; mean
coalition W1 0.30; 114 negative coefficients; Moran's I 0.236 (p = 0.001), LISA 12 HH / 17 LL;
concentration Hoyer 0.277, Gini 0.444, entropy 4.794; ρ with published area-data ranking 0.06.
Geo accuracy k = 42: 19 (lift 1.31, p = 0.068); k = 50: 20 (p = 0.21). Rule robustness (ref size,
top-20 inside, p): 63/10/0.16, 65/9/0.35, 41/6/0.35, 28/5/0.22.

## Sweep (Feature R², accuracy, ΔW1 | Geo R², accuracy, ΔW1)

| N | Feature | Geo |
|---|---|---|
| 1,000 | 0.9837, 7/10, 4.505 | 0.8641, 9/20, 0.577 |
| 2,000 | 0.9851, 7/10, 4.513 | 0.9030, 10/20, 0.576 |
| 3,000 | 0.9845, 7/10, 4.509 | 0.9028, 10/20, 0.576 |
| 4,000 | 0.9850, 7/10, 4.499 | 0.9083, 10/20, 0.580 |
| 5,000 | 0.9850, 7/10, 4.508 | 0.9080, 10/20, 0.573 |
| 7,500 | 0.9852, 7/10, 4.510 | 0.9088, 10/20, 0.574 |
| 10,000 | 0.9842, 7/10, 4.503 | 0.9183, 10/20, 0.577 |

## Ten models (N = 4,000; `models/model_summary.csv`)

| Model | R² | Feature Ridge R² | Geo Ridge R² | Feature top-10 | Geo top-20 in regions |
|---|---|---|---|---|---|
| XGBoost | 0.895 | 0.985 | 0.908 | 7 | 10 |
| LightGBM | 0.893 | 0.986 | 0.884 | 5 | 6 |
| CatBoost | 0.894 | 0.971 | 0.948 | 6 | 8 |
| GP | 0.899 | 0.996 | 0.870 | 5 | 7 |
| SVR | 0.868 | 0.965 | 0.871 | 6 | 7 |
| ANN | 0.878 | 0.943 | 0.864 | 5 | 9 |
| CNN | 0.861 | 0.961 | 0.885 | 6 | 11 |
| MLP | 0.527 | 0.229 | 0.906 | 6 | 11 |
| Transformer | 0.753 | 0.986 | 0.911 | 6 | 2 |
| Keras (deep) | 0.788 | 0.240 | 0.903 | 3 | 8 |

ATT Faithfulness (Pearson r across 10 models, two-sided p): Feature accuracy −0.01 (0.98), ATT-AUC
0.17 (0.63); Geo accuracy −0.15 (0.68), ATT-AUC 0.13 (0.71). Without dl/cnn (M = 8): −0.19, 0.12,
−0.21, 0.11, all p > 0.6.
