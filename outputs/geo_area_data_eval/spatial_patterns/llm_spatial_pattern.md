# GeoSMILE spatial pattern (LLM input)

Use these statistics. Do not infer clustering from map colour alone.

## GLOBAL_SPATIAL_PATTERN

- Moran_I: 0.2191
- expected_I: -0.0060
- p_value: 0.0010
- n_permutations: 999
- weights: queen_contiguity_row_standardised
- value_field: geo_coefficient
- interpretation: positive spatial clustering of geosensitivity

Allowed claim: geosensitivity is spatially clustered if interpretation says so.
Forbidden claim: "the map appears clustered."

## LOCAL_SPATIAL_PATTERN (LISA)

Counts at p<0.05: HH=9, LL=9, HL=3, LH=5, NS=142.

HH = high |coefficient| surrounded by high (regional concentration).
HL = high surrounded by low (localised anomaly, not a regional pattern).

### Contiguous HH clusters (n>=2)

- cluster 1: n=9; E01012850, E01012852, E01012854, E01012855, E01012891, E01033104, E01033109, E01035465, E01035466

### HIGH geosensitivity + HL (localised anomalies)

- E01012787 Kingston upon Hull 002D geo_coefficient=2.811
- E01012827 Kingston upon Hull 011D geo_coefficient=0.613
- E01012773 Kingston upon Hull 027A geo_coefficient=0.073

Association of geo_coefficient in space. HH = high geosensitivity surrounded by high; HL = high surrounded by low (localised anomaly). Not a causal crime effect.
