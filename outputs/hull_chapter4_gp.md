# Predictive and explanation models (Gaussian Process application)

This chapter repeats the Hull GeoSMILE application of the preceding chapter with Gaussian Process (GP) regression in place of XGBoost as the predictive model for both branches. GP was selected for this second application because, of every alternative screened against XGBoost (LightGBM, CatBoost, SVR, GP, a scikit-learn MLP, a deeper Keras network, plus an ANN/CNN/Transformer follow-up round), it was the strongest candidate on the criteria this thesis treats as most informative about whether GeoSMILE can be trusted: held-out predictive fit and surrogate fidelity, on both branches, with no fidelity collapse of the kind found for the two neural-network candidates. The trade-off, discussed at the end of this chapter, is computational cost.

Separate GP count models were fitted for the Feature and Geo branches, each a `ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1.0)` kernel with `normalize_y=True`, wrapped in a `StandardScaler` plus a `log1p`/`expm1` target transform (floored at zero) so the symmetric-loss GP is appropriate for skewed, non-negative crime counts — the same wrapping applied to every non-boosting candidate screened in this project, not a GP-specific accommodation. Random seed 42 throughout. As with the XGBoost application, the Feature model received the 30 non-spatial predictors and the Geo model received only centroid easting and northing; the two branches remain parallel explanations of separate predictive models, not additive components of one combined prediction. GP was not hyperparameter-tuned beyond this: the purpose of carrying it forward was to test GeoSMILE against the strongest available alternative under comparable effort, not to optimise GP itself.

The masking design, response definitions, surrogate-fitting procedure (kernel-weighted Ridge, α = 1, RBF/Laplace/inverse kernels compared by validation R², 60/20/20 mask-pool split), and importance scoring (absolute surrogate coefficient) are all unchanged from the XGBoost application and are not repeated here. One design parameter does differ: the Geo perturbation budget used throughout this GP application is **N = 4,000**, not the N = 2,000 used in the XGBoost chapter. That change was adopted project-wide (for every candidate model, not re-derived per model) after a dedicated XGBoost sweep found materially higher Geo fidelity at N = 4,000; §*Surrogate fidelity and perturbation budget* below reports the GP-specific sweep that confirms the same plateau holds for GP.

**Table 1. Research design for the Feature and Geo branches (GP application).**

| Design element | Feature branch | Geo branch |
|---|---|---|
| Purpose | Explain non-spatial predictive factors | Explain LSOA-specific geographic sensitivity |
| Predictive inputs | 30 non-coordinate variables | Centroid easting and northing |
| Explanation players | 30 features | 168 LSOAs |
| Predictive model | Gaussian Process regression (RBF + White kernel) | Gaussian Process regression (RBF + White kernel) |
| Selected perturbations | 1,024 binary masks | 4,000 LSOA perturbations |
| Perturbed units | Any binary subset of features | Random sets of 5–10 LSOAs |
| Masking baseline | Training median of each feature | Training median easting and northing |
| Perturbation response | Global W₁ between original and perturbed prediction distributions | W₁ over predictions for affected LSOA-month observations |
| Kernel distance | Fraction of features masked | Fraction of LSOAs masked |
| Selected kernel (production ranking) | Inverse | RBF |
| Surrogate | Weighted Ridge, α = 1 | Weighted Ridge, α = 1 |
| Surrogate mask split | 60% training, 20% validation, 20% testing | 60% training, 20% validation, 20% testing |
| Importance score | Absolute feature coefficient | Absolute LSOA coefficient |
| Expert-reference comparison | *Pending — awaiting shared expert reference* | *Pending* |
| Faithfulness experiment | *Pending — awaiting shared expert reference* | *Pending* |

Note on the Geo kernel: the production ranking (the one actually reported as GP's Geo attribution) selected RBF. A second, independent kernel-selection pass computed purely for the ATT-fidelity diagnostic — which uses a different train/validation/test partition of the same mask pool — selected inverse instead (validation R² 0.929 RBF vs. 0.941 inverse: close enough to flip under a different split). This is the same kernel-selection sensitivity documented for XGBoost's Geo branch elsewhere in this project; it changes which kernel label is reported, not the resulting coefficients, which are reproducible to floating-point precision regardless.

---

# Results

## Global Feature importance

> **Figure 1 (GP).** `outputs/top_maps/figures/feature_top10_gp.png` — Global Feature-SMILE ranking for the GP Hull crime-prediction model, all 30 predictors, top-10 highlighted. N = 1,024 masks, seed 42.

> **Figure 1b (GP, zoomed).** `outputs/hull_chapter4_gp_assets/feature_top10_gp_zoomed.png` — The same ranking with rank 1 (`crime_rolling_mean_6`) excluded and the axis rescaled, so ranks 2–30 are legible. Highlighting still marks the true top-10 from the full ranking (Figure 1), not a re-ranked top-10 within this subset — note `road_length` at position 10 here is correctly left grey, since it did not make the original top-10.

The single most influential predictor's coefficient is disproportionately larger than the rest of the ranking, to the point that Figure 1 alone cannot show relative differences among ranks 2–30: on the 0–5 axis needed to fit the six-month rolling crime mean's coefficient (−4.853), every other bar is compressed into a sliver a few pixels wide. The six-month rolling mean is roughly **9.3 times** the magnitude of the second-ranked predictor (`crime_lag_1`, −0.522) — a far steeper drop-off than XGBoost's corresponding ratio of about 4.1 (crime_rolling_mean_6 at −5.088 vs. crime_rolling_mean_3 at −1.285). Figure 1b removes rank 1 and rescales the axis to show that ranks 2–10 are not, in fact, similarly compressed relative to each other: they span a smooth, gradually-declining range (0.52 down to 0.14), and the top-10/rest-of-ranking boundary (dashed line) falls at a natural elbow rather than an arbitrary cutoff. The practical reading is that GP's Feature explanation has one overwhelmingly dominant driver (recent crime persistence) sitting on top of a second tier of nine comparably-scaled contributing predictors, rather than a smoothly decaying ranking throughout — consistent with the higher Feature-coefficient concentration reported in the diagnostics below (Hoyer 0.831) despite every one of the 30 predictors carrying a non-zero coefficient.

As with XGBoost, the GP Feature explanation is dominated by recent crime history, though the internal ordering differs. The six-month rolling crime mean has by far the largest coefficient magnitude (−4.853), an order of magnitude above every other predictor. The one-month crime lag ranks second (−0.522), ahead of unemployment (−0.389), the three-month rolling crime mean (−0.369), transit-stop count (−0.366), and the two-month crime lag (−0.310). Long-term sickness or disability (−0.267), the three-month crime lag (−0.232), lack of qualifications (−0.170), and the 65-plus population share (−0.144) complete the ten highest-ranked predictors.

**Table G1. GP global Feature-SMILE top-10 (signed coefficient).**

| Rank | Predictor | Coefficient | \|Coefficient\| |
|---:|---|---:|---:|
| 1 | crime_rolling_mean_6 | −4.853 | 4.853 |
| 2 | crime_lag_1 | −0.522 | 0.522 |
| 3 | unemployed_pct_age16plus | −0.389 | 0.389 |
| 4 | crime_rolling_mean_3 | −0.369 | 0.369 |
| 5 | transit_stop_count | −0.366 | 0.366 |
| 6 | crime_lag_2 | −0.310 | 0.310 |
| 7 | long_term_sick_disabled_pct | −0.267 | 0.267 |
| 8 | crime_lag_3 | −0.232 | 0.232 |
| 9 | no_qualifications_pct | −0.170 | 0.170 |
| 10 | older_adult_65_plus_pct | −0.144 | 0.144 |

Two differences from the XGBoost ranking stand out. First, GP's ranking is far more concentrated on the single leading predictor: the six-month rolling mean's coefficient is roughly 9.3 times the runner-up's, against roughly 4.1 times for XGBoost — consistent with GP's own smoother, more globally-averaged response to the single most temporally-persistent signal. Second, GP surfaces `transit_stop_count` and `no_qualifications_pct` in its top ten where XGBoost surfaced owner occupation and social renting instead; both are plausible predictive correlates of crime exposure, and the substitution illustrates that "globally important to the model" is model-specific, not a fixed property of the data. As with XGBoost, the negative signs describe the direction of the surrogate response under median ablation, not a claim that higher values of these predictors reduce crime; importance is read from the absolute coefficient.

## Geographic sensitivity

> **Figure 2 (GP).** `outputs/hull_chapter4_gp_assets/geo_signed_full_map_gp.png` — Signed geographic baseline-ablation effect for the GP Geo branch, all 168 Hull LSOAs. Red indicates a positive model-response shift when geographic position is perturbed relative to the training-location baseline; blue indicates a negative shift. N = 4,000 perturbations, seed 42.

As with XGBoost, GP's geographic sensitivity is not uniform across Hull: the signed map shows both LSOAs whose location systematically raises the model's prediction relative to the city-wide training baseline and LSOAs where it systematically lowers it, with the strongest responses (both directions) again concentrated in the central and inner-urban area rather than spread evenly across the city. This heterogeneity is the same qualitative finding as the XGBoost application and supports the same conclusion: an LSOA-specific Geo branch captures structure a single city-wide location coefficient would erase. The sign is, as before, a description of the fitted model's response under the baseline-ablation experiment, not a causal claim about location and crime. GP's underlying concentration statistics (below) show its Geo coefficients are somewhat *less* concentrated than XGBoost's (Hoyer 0.527 vs. 0.631), consistent with a visibly smoother, more gradually-varying signed surface than a tree ensemble's sharper LSOA-to-LSOA contrasts.

## Local Feature importance

> **Figure 3 (GP).** `outputs/hull_chapter4_gp_assets/local_top6_drivers_gp.png` — Spatial expression of the six highest-ranked global Feature-SMILE predictors for GP: six-month rolling crime mean, one-month crime lag, unemployment, three-month rolling crime mean, transit-stop count, two-month crime lag. Signed local coefficients, shared colour scale.

Local coefficients were computed as per-observation Ridge fits on the same Feature mask pool used for the global ranking (`local_feature_coefficients`, reusing the pool rather than a fresh perturbation run), then averaged to LSOA level — the identical procedure used for XGBoost's Figure 3, applied here to GP for the first time. As with XGBoost, a globally top-ranked predictor is not spatially uniform: the two crime-history terms (the six-month rolling mean and the one-month lag) show their strongest local effects concentrated in the central and southern parts of Hull, consistent with where recorded crime itself is most persistent, while unemployment and the three-month rolling mean show more dispersed contrasts across northern, central, and southern LSOAs. Transit-stop count — a predictor that does not appear in XGBoost's global top-10 at all — shows a locally concentrated pattern around Hull's main transport corridors, illustrating that GP's choice to weight this predictor globally is traceable to a spatially coherent, not diffuse, local signal. As with the XGBoost application, this confirms that global importance can conceal considerable neighbourhood-level heterogeneity in a predictor's actual contribution to the fitted model.

## Surrogate fidelity and perturbation budget

Only an independent-runs view was computed for GP (no nested-subset convergence check, unlike the XGBoost chapter) — a scope reduction made for this application given GP's substantially higher per-perturbation cost (see the closing discussion). Independent SMILE pools were rebuilt from scratch at Feature N ∈ {512, 1024, 2048, 4096} and Geo N ∈ {2000, 4000, 6000, 8000}, seed 42, Wasserstein-1 response, kernel-weighted Ridge surrogate.

**Table G2. Held-out GeoSMILE surrogate fidelity by perturbation count N (GP).**

| N | Feature R² | Feature weighted R² | Geo R² | Geo weighted R² |
|---:|---:|---:|---:|---:|
| 512 | 0.9959 | 0.9960 | — | — |
| **1,024 (selected)** | **0.9964** | **0.9964** | — | — |
| 2,048 | 0.9960 | 0.9958 | — | — |
| 4,096 | 0.9955 | 0.9954 | — | — |
| 2,000 | — | — | 0.9172 | 0.9356 |
| **4,000 (selected)** | — | — | **0.9461** | **0.9441** |
| 6,000 | — | — | 0.9482 | 0.9483 |
| 8,000 | — | — | 0.9479 | 0.9486 |

Feature fidelity is essentially saturated across the whole grid tested (R² 0.9955–0.9964); the selected N = 1,024 run is, if anything, marginally above the rest rather than below, so there is no under-sampling concern on this branch at any budget from 512 to 4,096. Geo fidelity rises sharply from N = 2,000 (0.917) to N = 4,000 (0.946) and is essentially flat beyond that (0.948 at 6,000, 0.948 at 8,000) — the same plateau-by-4,000 pattern found for XGBoost and, in a separate check spanning six further candidate architectures, found to hold generally across every model tested in this project regardless of predictive family. This is why N = 4,000 (not the originally-drafted N = 2,000) is the Geo budget used throughout this GP application: it sits at the start of the plateau rather than materially below it, and pushing to 6,000–8,000 buys no further fidelity for roughly double the perturbation cost.

## ATT accuracy

*Pending — awaiting the shared expert reference. Once available: reference overlap, ATT ACC/F1/AUROC, and the comparison map (top-20 GeoSMILE vs. top-20 expert) at the same k = 10 Feature / k = 20 Geo cardinalities used for XGBoost.*

## Faithfulness

*Pending — awaiting the shared expert reference, per the same scope decision as ATT accuracy above.*

## ATT stability

Controlled-input stability used the identical protocol as the XGBoost application: the same four real LSOAs just outside Hull (Table 7 of the preceding chapter — `E01012953` north, `E01013094` east, `E01013009` south, `E01013112` west), added one at a time to the SMILE explanation reference, Hull-only predictive models unchanged, same seed (42) and the same GP-specific budget (Feature 1,024 masks; Geo 4,000 perturbations).

**Table G3. Controlled-input (ATT) stability after adding one real LSOA per cardinal direction (GP).**

| Addition | Branch | Top-k Jaccard | \|coef\| Spearman |
|---|---|---:|---:|
| North | Feature | 1.000 | 0.9996 |
| North | Geo | 1.000 | 0.9998 |
| East | Feature | 1.000 | 0.9978 |
| East | Geo | 1.000 | 0.9998 |
| South | Feature | 1.000 | 0.9991 |
| South | Geo | 1.000 | 0.9998 |
| West | Feature | 1.000 | 0.9978 |
| West | Geo | 1.000 | 0.9998 |
| **Mean** | **Feature** | **1.000** | **0.9986** |
| **Mean** | **Geo** | **1.000** | **0.9998** |

> **Figure G4.** `outputs/top_maps/figures/gp_stability_directions.png` — Per-direction cardinal-neighbour stability, Feature and Geo branches, top-k Jaccard and \|coefficient\| Spearman.

GP's stability is, if anything, tighter than XGBoost's: Feature top-10 membership and Geo top-20 membership are both perfectly preserved in every direction (Jaccard = 1.000), and the full-vector rank correlations are higher than XGBoost's corresponding Geo figure (0.9998 vs. 0.999) and comparable on Feature. As with XGBoost, the added neighbour never enters the Geo top-20 in any direction: extra ranks range from 161st to 162nd of 169 units (west, the closest unit geographically, ranks 162nd — notably *not* the closest-to-entry direction, unlike XGBoost's west-closest pattern, since GP's smoother spatial response does not concentrate sensitivity near the city boundary the way XGBoost's tree splits can), so none of the four additions come close to displacing a genuine Hull LSOA from the ranking.

## ATT consistency

Consistency was assessed by repeating the explanation ten times (R = 10, seeds 42–51) rather than the five repeats used for XGBoost, matching the R = 10 convention used for every model in this project's later screening round.

**Table G4. Consistency of the GP explanations across ten independent perturbation runs.**

| Metric | Feature | Geo |
|---|---:|---:|
| Mean rank Spearman | 0.9555 | 0.9610 |
| SD rank Spearman | 0.0159 | 0.0043 |
| Mean score Pearson | 0.9998 | 0.9965 |
| SD score Pearson | 0.0001 | 0.0004 |
| Mean rerun fidelity R² | 0.9948 | 0.9383 |
| SD rerun fidelity R² | — | — |
| Mean top-20% Jaccard | 1.000 | 0.6934 |
| SD top-20% Jaccard | 0.000 | 0.0563 |

> **Figure G5.** `outputs/top_maps/figures/gp_consistency_r10.png` — Rank Spearman and top-20% Jaccard (mean ± SD) across the ten reseeds, Feature and Geo branches.

GP's Feature consistency is markedly higher than XGBoost's on every metric reported for both models: rank Spearman 0.956 vs. 0.694, and — most strikingly — top-20% Jaccard is a perfect 1.000 (SD 0.000) across all ten reseeds, against 0.829 for XGBoost. The same top-10 Feature predictors are selected on every single reseed; only their fine-grained rank order among themselves shows any reseed-to-reseed movement (hence rank Spearman below 1). On Geo, GP's rank Spearman (0.961) exceeds XGBoost's (0.743), while top-20% Jaccard (0.693) is lower than XGBoost's Feature figure but comparable to XGBoost's own Geo figure (0.623) — the same qualitative pattern as XGBoost, where exact top-set membership is less reseed-stable on Geo than the overall coefficient pattern (score Pearson 0.997).

## ATT fidelity

**Table G5. ATT fidelity at the selected design (GP; Feature 1,024 masks; Geo 4,000 perturbations; seed 42).**

| Metric | Feature | Geo |
|---|---:|---:|
| Held-out R² | 0.9964 | 0.9461 |
| Held-out weighted R² | 0.9964 | 0.9441 |
| MAE | 0.1219 | 0.6660 |
| Weighted MAE | 0.1213 | 0.6721 |
| RMSE | 0.1533 | 1.1968 |
| Weighted RMSE | 0.1532 | 1.2339 |
| Selected kernel (fidelity diagnostic) | Inverse | Inverse |
| Best validation R² | 0.9951 | 0.9412 |
| N_p (test perturbations) | 205 | 800 |
| N_s (surrogate parameters) | 30 | 168 |
| Weighted-adjusted R² | 0.9957 | 0.9292 |

Both branches exceed XGBoost's corresponding fidelity (Feature 0.9964 vs. 0.9853; Geo 0.9461 vs. 0.9308, at N = 4,000 for both models). Feature fidelity in particular is close to ceiling: a weighted-adjusted R² of 0.9957 after penalising for the ratio of test perturbations to surrogate parameters (N_p = 205, N_s = 30) confirms this is not an artefact of a generous test split. Geo fidelity's weighted-adjusted R² (0.9292) sits closer to its raw value, consistent with the larger, more comfortable N_p:N_s ratio on that branch (800:168).

---

# Additional diagnostics

**Concentration and perturbation-cardinality response.**

**Table G6. GP coefficient concentration, Feature vs. Geo.**

| Statistic | Feature (n=30) | Geo (n=168) |
|---|---:|---:|
| Hoyer sparsity | 0.8313 | 0.5266 |
| Gini coefficient | 0.7497 | 0.4378 |
| Shannon entropy | 1.9709 | 4.6483 |
| Non-zero coefficients | 30 of 30 | 168 of 168 |

GP's Feature coefficients are concentrated (Hoyer 0.831, comparable in spirit to XGBoost's 0.848, though slightly lower) even though every one of the 30 predictors carries a non-zero coefficient — a small subset, led overwhelmingly by the six-month rolling crime mean, dominates the explanation while the rest make detectable but much smaller contributions. Geo coefficients are, as with XGBoost, distributed far more evenly across the 168 LSOAs (Hoyer 0.527 vs. XGBoost's 0.631) — GP's geographic sensitivity is real everywhere but concentrated nowhere near as sharply as its Feature sensitivity, and slightly less sharply than XGBoost's own Geo branch. This is consistent with the visibly smoother signed Geo map in Figure 2 above.

Perturbation-cardinality response (the correlation between how many features/LSOAs a given perturbation touches and the resulting Wasserstein response) was not recomputed for GP in this pass; the diagnostic requires access to the raw per-perturbation mask/response pool, which the Feature run above retained but the Geo run did not (the Geo N-sweep in *Surrogate fidelity* discarded pools after scoring, to bound the cost of the already-expensive Geo protocol). This can be added as a follow-up if wanted.

Graded-ranking, factor-group alignment, and correlation-with-expert-ranking diagnostics are pending the shared expert reference, as they were derived from it in the XGBoost chapter.

---

# Discussion (partial — pending expert-reference sections)

The GP application confirms that GeoSMILE transfers cleanly to a second predictive-model family with no loss of the internal, model-agnostic evaluation criteria that carry the weight of the XGBoost chapter's conclusions. On every internally-computable metric — predictive fit, fidelity, stability, and consistency — GP matches or exceeds XGBoost: higher Feature and Geo predictive R² (0.899/0.878 vs. XGBoost's corresponding figures), higher fidelity on both branches, equal or better cardinal-neighbour stability, and markedly better reseed consistency, especially the perfect Feature top-set stability under reseeding (Jaccard 1.000 across R = 10) that XGBoost does not achieve (0.829).

This did not come for free. Across the protocol run for this chapter (predictive fit, the four-point fidelity sweep on each branch, four cardinal-neighbour stability directions, and ten consistency reseeds), GP took on the order of hours to explain — concretely, roughly 25–35 minutes for a single Feature-branch fidelity sweep and over two hours for the equivalent Geo-branch sweep in this project's own timing, against single-digit minutes for XGBoost's entire equivalent protocol. The cause is structural rather than an implementation inefficiency: Gaussian Process prediction cost scales with training-set size (each `.predict()` call is roughly O(n_train) per point against the training kernel), and GeoSMILE's masking loop issues thousands of individual `.predict()` calls per branch per pass. Every number in this chapter is therefore genuine — nothing here is approximated to save compute — but reproducing it, or extending it (a finer N-sweep, more reseeds, additional stability directions), costs an order of magnitude more machine time than the equivalent XGBoost run. Whether that trade-off is worth making for a given deployment is a decision this chapter's quality metrics inform but do not resolve on their own.

The remainder of this discussion — how GP's Feature and Geo rankings agree with the expert reference, whether GP's top-ranked sets are more or less faithful to the fitted model than XGBoost's, and what the factor-group and graded-ranking diagnostics show — is deferred until the expert reference for this application is shared, per the scope agreed for this chapter.

---

## Note on outstanding items

The background asset-generation run (`generate_gp_chapter_assets.py`) has completed: the Geo signed map, local top-6 driver map, and concentration/entropy diagnostics are filled in above. As a sanity check, that run also re-derived GP's global Feature coefficients from a fresh SMILE pass (same seed, same N) and confirmed they reproduce the already-saved `outputs/verify_metrics_all_models/gp/feature_importance.csv` values to within floating-point tolerance.

Still outstanding: the ATT accuracy, faithfulness, and expert-correlation/factor-group-alignment sections, all pending the shared expert reference. Once that is shared, the remaining tables (Table 5/6/12 equivalents), Figure 6's comparison map, and the corresponding paragraphs of the Discussion can be completed following the exact same procedure used for XGBoost.

All figures and tables above that are *not* marked pending are drawn from already-computed, saved results in this project (`outputs/verify_metrics_all_models/gp/`, `outputs/eval_by_model/predictive_performance.csv`, `outputs/response_metric_comparison/response_metric_comparison_summary.csv`, `outputs/top_maps/`, `outputs/hull_chapter4_gp_assets/`) — no numbers in this file were invented or approximated.
