# Hull Geo-design review — all checks, results and maps

Built by `build_geo_design_review.py` (copies the outputs below and draws the new maps). The
originals stay where their scripts write them; nothing published was changed. Not causal.

| Folder | Question | Script | Key maps |
|---|---|---|---|
| `01_crime_history_check/` | Does area-data GeoSensitivity track crime level? | `geo_crime_history_check.py` | — |
| `02_coordinate_lonlat_check/` | Coordinate-only masking (lon/lat model) vs the published ranking | `geo_coordinate_design_check.py` | — |
| `03_coordinate_lonlat_eval/` | Chapter-4 Geo metrics, coordinate-only, lon/lat model, N sweep | `geo_coordinate_design_eval.py`, `plot_geo_coordinate_design_map.py` | `geo_coordinate_map_N4000.png`, `geo_coordinate_vs_area_data_N4000.png` |
| `04_reference_point_check/` | Does the ranking depend on where coordinates are moved? | `geo_coordinate_reference_check.py` | `reference_point_maps.png` |
| `05_aligned_design/` | Full Seattle/Chicago-aligned design (30 + easting/northing model; Feature and Geo branches), XGBoost N sweep + ten models | `hull_aligned_design_eval.py` | `geo_map_N4000.png`, `geo_aligned_vs_area_data_N4000.png`, `feature_local_top6_N4000.png` |

Detailed numbers: `05_aligned_design/README.md` (aligned design) and, for checks 01–04,
`outputs/geo_design_checks/README.md` (copied below as `checks_README.md`).

Headline: the aligned design passes fidelity (Feature 0.985, Geo 0.908), stability (J 1.000),
consistency and intervention tests; Feature accuracy 7/10; Geo accuracy 10/20 in the expert regions
(p = 0.098, not significant); mean Geo response 0.30 crimes/month; Geo ranking unrelated to the
published area-data ranking (ρ = 0.06) and dependent on the reference point and coordinate encoding;
ATT Faithfulness not demonstrated in either branch.
