# Hull GeoSMILE expert-reference attribution accuracy

## Status of the labels

The workbook `geosmile_hull_single_expert_annotations_v1.xlsx` is a **provisional single-expert reference**.
It is **not** expert consensus and **not** objective ground truth.
Annotators were intended to remain **blinded to GeoSMILE outputs**.

## What this evaluation measures

**Expert-reference attribution accuracy** (also called domain-expert alignment) quantifies agreement between model attribution magnitudes and expert relevance ratings.

It is **not**:

- **Model faithfulness** — agreement with actual model behaviour under intervention
- **Perturbation response monotonicity** — correlation of mask cardinality with Wasserstein distance (formerly mislabelled “Faithfulness” in some Hull tables)
- **Consistency** — repeated execution of the same explanation
- **Controlled-input stability** — prediction-preserving input changes (not evaluated here)

## Reference-rating construction

Ratings are reconstructed in Python (Excel formula caches are not trusted):

```
reference_rating = human_override_1to5
    if human_override_1to5 is non-null
    else expert_relevance_1to5
```

`expert_positive = 1` if `reference_rating >= 4`, else `0`.

For LSOAs, a separate sensitivity uses `no_crime_rating_1to5` (**no_crime_domain_reference**), which excludes the external IoD Crime Domain from the annotation rubric. This is **not** averaged with the primary **external_crime_informed_reference**.

## Why absolute coefficients

SMILE surrogates here respond to a **non-negative distributional distance** (Wasserstein-1). Coefficient sign is therefore **not** interpreted as increasing vs decreasing predicted crime. Directional/sign agreement is **not** calculated. Fields such as `expected_association` / `expected_crime_context` remain descriptive only.

## Joins and coverage

- Features join on exact `feature` IDs to global importance exports (`feature_global_importance.csv`; Feature-A/B subset exports labelled as restricted).
- Geo joins on `lsoa21cd` to `geo_coefficients.csv` using `geo_abs_coefficient` (not signed coefficients).
- Local-effect files are not used as global attribution scores.
- Primary Feature expects 30/30 coverage; primary Geo expects 168/168. Incomplete coverage fails loudly.

## Uncertainty

- Deterministic seed `42`, 5000 bootstrap and 5000 permutation repetitions (configurable).
- Geo primary geographic uncertainty uses an **MSOA-block bootstrap** derived from the numeric token embedded in ONS LSOA names. Ordinary LSOA bootstrap intervals are exploratory and **do not** account for spatial dependence.
- No ward/neighbourhood graph is invented when adjacency is unavailable.

## Limitations

- Single provisional expert; subjectivity and rubric dependence remain.
- Correlated temporal features (lags / rolling means) complicate unique attribution.
- Spatial dependence among LSOAs.
- Expert agreement ≠ causal importance ≠ model faithfulness.
