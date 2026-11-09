# GeoSMILE for Hull crime prediction

Stakeholder-adaptive Streamlit dashboard for the Hull next-month crime
GeoSMILE experiment.

**Repository:** https://github.com/shadiehh/Geo-SMILE-Dashboard-HULL-Crime-prediction

## Dashboard

From this folder:

```powershell
python -m pip install -r requirements.txt
streamlit run dashboard/app.py
```

Or: `streamlit run streamlit_app.py`

Copy `.env.example` to `.env` and add `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` to
generate stakeholder explanations. The maps and LSOA evidence work without keys.

The dashboard reads published artefacts only. It does not recompute GeoSMILE.

## Published design (what the dashboard shows)

One XGBoost model predicts next-month crime per LSOA from 30 predictors plus the LSOA
centroid easting and northing (hold-out R² 0.895, seed 42). Geo-SMILE explains that one
model through two branches, each with N = 4,000 perturbations:

| Branch | What is masked | What it attributes |
|---|---|---|
| Feature | The 30 predictors (coordinates kept at their true values, not players) | Global and local importance of each predictor |
| Geo (random donor) | An LSOA's coordinates, moved to the centroid of a random other Hull LSOA (K = 50 draws), all other inputs kept | GeoSensitivity: how far predictions depend on knowing the area's location |

Local Feature coefficients are signed: masks code a retained player as 1 and the response is
the perturbed minus the original prediction, so a positive value means the area's own value
raises its prediction relative to the masking reference. GeoSensitivity's sign is a relative
contribution, not a direction of crime. Neither is a causal effect.

Results come from `hull_aligned_design_eval.py` (outputs/geo_aligned_design/xgboost/) and are
copied into `outputs/dashboard_final/` by `build_final_dashboard_results.py`. Area predictions
come from `build_dashboard_predictions.py`, which refits the model and stops unless it
reproduces the published R². Stakeholder cards and narratives come from
`generate_card_narrative.py` and `generate_detailed_narrative.py`.

## Earlier experiment: three-branch design (not shown in the dashboard)

`geosmile_hull.py` and `run_pipeline.py` fit an earlier three-branch design (Geo on coordinates
only; Feature-A on non-spatial variables; Feature-B on non-spatial variables plus coordinates).
Its maps used signed baseline-ablation effects rather than SMILE coefficients. The sections
below describe that pipeline; its outputs are kept for comparison and are not read by the
dashboard.

## Get the input

1. Open <https://data.police.uk/data/>.
2. Select a date range, **Humberside Police**, and **crime data**.
3. Generate and download the ZIP. Do not select outcomes or stop-and-search.

Police.uk coordinates are anonymised. The pipeline aggregates them to LSOA-month
and never treats them as exact offence locations.

## Run

From this folder:

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py
```

Or step by step:

```powershell
python geosmile_hull.py --police-data ".\Police.uk.zip" --output outputs
python make_visuals_hull.py
python make_figures_hull.py
python write_interpretation.py
```

Optional covariates (one row per `lsoa21cd`):

```powershell
python geosmile_hull.py --police-data ".\Police.uk.zip" `
  --covariates hull_covariates.csv --output outputs
```

Do not include overall IMD or its crime domain; they leak crime information.

## Outputs

Core tables:

- `outputs/hull_lsoa_month_panel.csv`
- `outputs/test_predictions.csv`
- `outputs/model_comparison.csv` — GeoSMILE branches plus simple baselines
- `outputs/feature_b_incremental_value.json` — paired A vs B with clustered bootstrap
- `outputs/spatial_holdout_predictions.csv` / `spatial_holdout_summary.json`
- `outputs/feature_*_global_importance.csv` — SMILE surrogate importances
- `outputs/feature_*_local_effects.csv` — signed baseline-ablation effects
- `outputs/geo_location_effects.csv`
- `outputs/run_summary.json`
- `outputs/hull_results_interpretation.md`

Maps and figures:

- `outputs/maps/` — Geo sensitivity + Feature-A/B maps (shared colour limits)
- `outputs/maps/differences/` — Feature-B minus Feature-A effect maps
- `outputs/figures/` — model comparison, incremental value, importances, diagnostics

Boundaries:

- `data/hull_lsoa_2021_official.geojson` — official ONS LSOA 2021 BGC for Hull
- `data/boundary_validation.json`

## Evaluation design

Temporal holdout (primary): train through June 2025, validate July–December 2025,
test from January 2026 onward when the download covers that window.

Spatial holdout (harder transfer test): contiguous/clustered LSOAs are held out so
those areas never appear in training. Report separately from the temporal results.

## Caveats

- Modelling centroids are Police.uk incident medians unless replaced with official
  population-weighted centroids.
- High SMILE surrogate fidelity measures attribution approximation quality, not
  predictive accuracy or causal validity.
- Do not claim Feature-B coordinate gains unless the clustered-bootstrap CI excludes zero.
