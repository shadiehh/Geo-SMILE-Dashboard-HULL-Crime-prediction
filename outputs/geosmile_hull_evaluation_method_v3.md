# Hull GeoSMILE evaluation method (v3)

## Active branches

1. **Feature (location excluded)**  -  non-spatial predictors only; global attribution from `feature_global_importance.csv`. Excludes coordinates, lon/lat, centroids, LSOA IDs, and spatial-location variables.
2. **Geo**  -  per-LSOA location masking; attribution from `geo_coefficients.csv`.

Feature-A / Feature-B are **not** active evaluation branches (legacy files retained on disk only).

## Expert reference

Workbook: `hull_expert_top20.xlsx`  
Status: `expert_reference`

Expert reference.

```
final_expert_rating = human_override_1to5 if not null else expert_relevance_1to5
```

Primary binary reference is **fixed expert top sets**, not rating ≥ 4:
- Feature: top 10 of 30
- Geo: top 20 of 168

## Attribution accuracy vs faithfulness

- **Expert-reference attribution accuracy**  -  agreement with the expert top-10/20 sets (ATT metrics, set overlap, graded Spearman/Kendall/nDCG).
- **Model-based attribution faithfulness** - top vs random vs bottom interventions at fixed cardinality (Feature=10, Geo=20; WD response), matching the expert reference sets.
- **Dehghani-style expert faithfulness**  -  correlation of predictive TEST performance with ATT across ≥5 independent predictive-model runs.
  - Current status: **Not estimable**  -  Insufficient independent predictive-model runs with both test performance and expert-reference attribution metrics.
- **Perturbation response monotonicity**  -  cardinality vs WD (restored); not faithfulness.
- **Consistency**  -  multi-run explanation agreement (restored); **not** controlled-input stability.
- **Controlled-input stability (ATT stability)** - **Evaluated** - Feature top-10 Jaccard 1.000; Geo top-20 Jaccard 1.000 (mean over four cardinal neighbor-LSOA expansions just outside Hull). Distinct from multi-run Consistency.

## Provenance

Historical validated metrics are restored from dedicated per-metric files first, then `geosmile_hull_evaluation_full.csv`. Conflicts are recorded in `geosmile_hull_result_conflicts.csv`. Consistency values are not overwritten by a new random seed run.

## Maps

Expert **top 20** LSOAs use official Hull LSOA 2021 geometry via `hull_boundaries`. Accuracy and maps both use Geo k=20.

## Limitations

- Expert reference ≠ causal importance ≠ model faithfulness.
- Absolute coefficients; sign is not validated as crime increase/decrease.
- No invented ward/adjacency graphs.
