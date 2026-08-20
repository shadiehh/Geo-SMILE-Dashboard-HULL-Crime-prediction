# GeoSMILE spatial pattern (LLM input)

Use these statistics. Do not infer clustering from map colour alone.

## GLOBAL_SPATIAL_PATTERN

- Moran_I: 0.0438
- expected_I: -0.0060
- p_value: 0.0270
- n_permutations: 999
- weights: queen_contiguity_row_standardised
- value_field: geo_abs_coefficient
- interpretation: positive spatial clustering of geosensitivity

Allowed claim: geosensitivity is spatially clustered if interpretation says so.
Forbidden claim: "the map appears clustered."

## LOCAL_SPATIAL_PATTERN (LISA)

Counts at p<0.05: HH=7, LL=6, HL=1, LH=6, NS=148.

HH = high |coefficient| surrounded by high (regional concentration).
HL = high surrounded by low (localised anomaly, not a regional pattern).

### Contiguous HH clusters (n>=2)

- cluster 1: n=3; E01012891, E01012892, E01012893
- cluster 2: n=2; E01012854, E01035466
- cluster 3: n=2; E01012858, E01012888

### HIGH geosensitivity + HL (localised anomalies)

- E01012821 Kingston upon Hull 011A |coef|=0.861

Association of |Geo SMILE coefficient| in space. HH = high geosensitivity surrounded by high; HL = high surrounded by low (localised anomaly). Not a causal crime effect.
