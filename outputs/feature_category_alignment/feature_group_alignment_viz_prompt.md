# Prompt: Feature-group alignment visual (Hull GeoSMILE Feature branch)

Create a clean, thesis-ready figure comparing **expert** vs **GeoSMILE** feature importance, organised by **feature group (category)**.

## Data

Use:

- `outputs/feature_category_alignment/feature_group_alignment_for_viz.csv` (feature-level)
- `outputs/feature_category_alignment/feature_group_summary_for_viz.csv` (group-level)

Key fields:

- `category` / `feature_group`
- `expert_rank`, `geosmile_rank` (1 = most important; 30 features)
- `expert_category_rank`, `geosmile_category_rank` (1 = highest group importance)
- `in_expert_top10`, `in_geosmile_top10`, `in_both_top10`
- `geosmile_mass_share` (group share of absolute coefficient mass)

Design cuts: Feature **top 10 of 30**. Current mutual top-10 overlap = **7/10**. Category Spearman ≈ **0.96**; feature Spearman ≈ **0.46**.

## Recommended composition (one figure, two panels)

**Panel A — Group alignment**

- Show the 7 feature groups.
- Compare expert group rank vs GeoSMILE group rank, or paired bars of `geosmile_mass_share` ordered by expert group rank.
- Emphasize that the top-3 groups agree: Crime/temporal, Economic/labour & tenure, Built environment/activity.

**Panel B — Within-group feature ranks**

- Faceted or grouped dumbbell/slope chart: for each feature, connect `expert_rank` to `geosmile_rank`.
- Colour by top-10 membership: mutual / expert-only / GeoSMILE-only / neither.
- Keep group labels as section headers so group structure reads first, features second.

## Visual rules

- One message: groups align strongly; individual feature ranks align moderately (top-10 overlap 7/10).
- No dashboard clutter: no KPI cards, no purple gradients, no decorative glow.
- Restrained two-colour system (expert vs model), plus a third colour only for mutual top-10.
- Label only ranks/features that matter (top-10 and large rank gaps).
- Print-legible thesis figure; white background; thin grid if needed.
- Caption note: expert feature ranking versus GeoSMILE absolute global-coefficient ranks; Feature branch, location excluded.

## Do not

- Treat ranks as causal effects.
- Invent extra groups or features.
- Hide the category axis; group structure is the point of the figure.

## Deliverable

One publication-quality figure (PNG/PDF) plus a short caption stating group Spearman, feature Spearman, and top-10 overlap.
