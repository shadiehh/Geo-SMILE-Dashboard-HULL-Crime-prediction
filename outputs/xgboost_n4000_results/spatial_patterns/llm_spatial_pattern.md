# GeoSMILE spatial pattern (LLM input)

Use these statistics. Do not infer clustering from map colour alone.

## GLOBAL_SPATIAL_PATTERN

- Moran_I: 0.0444
- expected_I: -0.0060
- p_value: 0.0290
- n_permutations: 999
- weights: queen_contiguity_row_standardised
- value_field: geo_abs_coefficient
- interpretation: positive spatial clustering of geosensitivity

Allowed claim: geosensitivity is spatially clustered if interpretation says so.
Forbidden claim: "the map appears clustered."

## LOCAL_SPATIAL_PATTERN (LISA)

Counts at p<0.05: HH=7, LL=7, HL=2, LH=5, NS=147.

HH = high |coefficient| surrounded by high (regional concentration).
HL = high surrounded by low (localised anomaly, not a regional pattern).

### Contiguous HH clusters (n>=2)

- cluster 1: n=3; E01012891, E01012892, E01012893
- cluster 2: n=2; E01012854, E01035466

### HIGH geosensitivity + HL (localised anomalies)

- E01012801 Kingston upon Hull 019E |coef|=1.138
- E01012841 Kingston upon Hull 017A |coef|=1.043

Association of |Geo SMILE coefficient| in space. HH = high geosensitivity surrounded by high; HL = high surrounded by low (localised anomaly). Not a causal crime effect.
