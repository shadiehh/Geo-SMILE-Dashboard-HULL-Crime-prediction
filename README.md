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

## Branch definitions

This folder turns Police.uk street-crime files into an LSOA-month next-month
crime prediction experiment and fits the corrected three-branch GeoSMILE design.

## Branch definitions

| Branch | Predictive inputs | Attributed variables |
|---|---|---|
| Geo | LSOA easting and northing | Location only |
| Feature-A | Non-spatial variables | Non-spatial variables |
| Feature-B | Non-spatial variables plus easting and northing | Non-spatial variables only |

Coordinates are deliberately **not** Feature-B attribution players. They condition
the prediction and therefore may change the attribution of the substantive
variables, while location remains isolated in the Geo branch.

Mapped local values are **signed baseline-ablation effects** (prediction with the
original feature minus prediction after median replacement). They are not local
SMILE Ridge coefficients and are not causal effects.

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
