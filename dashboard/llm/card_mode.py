"""City Planner narratives in two lengths: the short card and the comprehensive briefing.

Both modes send the master prompt (GEOSMILE_SYSTEM_PROMPT, unchanged) followed by a mode
addendum that replaces only its section 14 length/form rules. The model marks the figures it
cites as [[n:...]] and the key terms as [[t:...]]; the dashboard and the thesis card render
both as one pink highlight.

  short          JSON card: 4 stat tiles, section_01 (map 01), section_02 (map 02), footer.
                 Plain language: no test statistics. Same text as the Chapter 6 card
                 (generate_planner_card_narrative.py imports this module).
  comprehensive  three short paragraphs: where sensitivity concentrates and how it clusters
                 (Moran / LISA allowed), what the model relies on by feature category and
                 built feature, and the focus vs contrast LSOA. Shown with feature charts
                 and clustering maps.

Every cited figure is checked against the evidence package; failures are shown, not retried.
"""

from __future__ import annotations

import html
import json
import re

import numpy as np
import pandas as pd

from dashboard.data.schema import human_label
from dashboard.data.stakeholder_data import StakeholderData

# Plain-language labels for the crime-history and season players (others use human_label).
PLAIN_LABELS = {
    "crime_rolling_mean_6": "Crime, 6-month average", "crime_rolling_mean_3": "Crime, 3-month average",
    "crime_lag_1": "Crime, last month", "crime_lag_2": "Crime, 2 months ago", "crime_lag_3": "Crime, 3 months ago",
    "month_sin": "Time of year", "month_cos": "Time of year (phase)",
}


def plain_label(feature: str) -> str:
    return PLAIN_LABELS.get(feature, human_label(feature))

PLANNER_QUESTION = ("Which areas matter most to the model, and where does it rely on built-environment "
                    "features, and which ones?")
CENTRE = "E01033104"  # Kingston upon Hull 029E, rank 1 GeoSensitivity
N_TOP = 5
MODES = {"short": "Short card", "comprehensive": "Comprehensive"}
MODE_VERSION = {"short": "card-v1", "comprehensive": "comprehensive-v2"}

# The 12 built-environment Feature players, count and density variants pooled.
BUILT = {
    "lsoa_area_km2": "area",
    "nightlife_count": "nightlife", "nightlife_per_km2": "nightlife",
    "retail_count": "retail", "retail_per_km2": "retail",
    "transit_stop_count": "transit stops", "transit_stop_per_km2": "transit stops",
    "road_length_km": "roads", "road_length_km_per_km2": "roads",
    "intersection_count": "intersections", "intersection_per_km2": "intersections",
    "green_space_pct": "green space",
}
CATEGORIES = {
    "crime": "crime history (lags and rolling means)",
    "census": "census population characteristics (employment, health, tenure, age, qualifications, density)",
    "built": "built environment (12 players)",
    "season": "seasonality (month of year)",
}

LIMITS = {
    "short": {"section_01": 60, "section_02": 65, "footer": 15},
    "comprehensive": {"text": 260},
}
TERM_RANGE = {
    "short": {"section_01": (2, 4), "section_02": (2, 4)},
    "comprehensive": {"text": (4, 10)},
}
# Test-statistic vocabulary the short planner card should not need.
JARGON = re.compile(r"Moran|LISA|p\s*=|p-value|High-High|\bHH\b|coefficient|SMILE", re.I)

_MARKUP_RULES = """
Highlighting markup (you choose what to mark):

- wrap every figure you cite as [[n:...]], e.g. [[n:14%]];
- wrap the key terms a planner should catch when scanning as [[t:...]]. This is required,
  not optional: keep to the counts given above, each at most 4 words (the finding phrase
  itself, the leading feature names, or the key place names). Before answering, count the
  [[t:...]] spans and fix any count outside the range;
- no other markup and no markdown bold.

Every figure must be copied or rounded from EVIDENCE_JSON (shares as percentages:
0.139 -> 14%). Do not compute new figures and do not cite a figure that is not supplied.
"""

CARD_MODE_ADDENDUM = """
---

## 17. CARD MODE ADDENDUM

output_mode is card_mode: the text fills a short card for a City Planner / Local Authority
reader, with two maps (map 01: signed GeoSensitivity, rank-1 LSOA outlined; map 02:
built-environment share of each LSOA's local importance). This section replaces section 14
"Length and form" and its bolding rule. Every other section, including prohibited claims and
the self-check, still applies.

Plain language for this reader overrides the section 9 priority on clustering statistics:
do not cite Moran's I, p-values, LISA class names, coefficients or other test statistics.
Say what the model relies on, and where, in words a planner uses. Shares, percentages and
ranks are fine.

Return ONLY one JSON object with exactly these keys:

- "headline": one sentence, at most 20 words: the single most decision-relevant finding
  that answers user_question.
- "tiles": exactly 4 objects {"value": ..., "label": ...}. value is one figure from the
  evidence (at most 8 characters, e.g. "1/168", "14%"); label is at most 6 words saying what
  the figure is and where. Choose the four figures a planner should remember.
- "section_01": at most 60 words, about map 01: which areas matter most to the model (the
  rank-1 GeoSensitivity LSOA and the next highest, by locality name), and what the model
  relies on in the rank-1 LSOA (its crime-history and built-environment shares, and the
  leading built feature).
- "section_02": at most 65 words, about map 02: what the model relies on across Hull by
  feature category (hull_feature_category_shares), which built-environment features carry
  that category's share (hull_built_feature_shares), and the highest-share LSOAs with their
  leading built feature.
- "footer": one limitation, at most 15 words.

Key-term counts: 1-2 in the headline and 2-4 in EACH section. Tiles and footer carry no
markup.
""" + _MARKUP_RULES + """
ILLUSTRATIVE_NARRATIVE in the user message shows the expected structure, order, tone and
markup density. It is an example of wording, not evidence: take every figure from
EVIDENCE_JSON.
"""
CARD_MODE_ADDENDUM = CARD_MODE_ADDENDUM.rstrip()

COMPREHENSIVE_MODE_ADDENDUM = """
---

## 17. COMPREHENSIVE MODE ADDENDUM

output_mode is comprehensive_mode: a City Planner / Local Authority reader asked for the
full briefing, shown next to feature charts and clustering maps. This section replaces
section 14 "Length and form" and its bolding rule. Every other section, including prohibited
claims and the self-check, still applies.

Write THREE short paragraphs separated by a blank line, with no headers or bullet lists.
Hard limit: at most 260 words in total. Count the words before answering and cut detail
(street names first) until the count is within the limit.

1. Where the model is most and least sensitive to an area's own data (the highest and lowest
   GeoSensitivity LSOAs by locality name, at most three of each, no street names) and how that
   pattern clusters: cite Moran's I and its p-value and the LISA class counts from
   global_spatial_pattern. Each LISA count is a number of LSOAs in that class, never a number
   of clusters: write "9 LSOAs are classed High-High", not "9 high clusters". Do not describe
   a cluster's shape, centre or membership unless a supplied field states it. Clustering is
   not magnitude and not a crime hotspot. GeoSensitivity values are signed, not absolute.
2. What the model relies on across Hull by feature category
   (built_environment_share.hull_feature_category_shares), which built-environment features
   carry that share (hull_built_feature_shares), and where the built-environment share is
   highest (highest_built_share_lsoas, each with exactly its dominant_built_feature).
3. The focus and contrast LSOAs in dashboard_figures: their GeoSensitivity ranks and what
   leads their local explanation, from the figure captions supplied. End with one limitation.

Key terms: 4-10 [[t:...]] spans in the whole text, not more. Mark only the findings and the
two or three most important places or features; leave other place names unmarked. Every
figure, including percentages in parentheses, is wrapped in [[n:...]].
""" + _MARKUP_RULES
COMPREHENSIVE_MODE_ADDENDUM = COMPREHENSIVE_MODE_ADDENDUM.rstrip()

ADDENDA = {"short": CARD_MODE_ADDENDUM, "comprehensive": COMPREHENSIVE_MODE_ADDENDUM}

ILLUSTRATIVE_NARRATIVE = {
    "headline": "The model depends most on [[t:Old Town]], but leans on built-environment "
                "features in [[t:residential areas]] instead.",
    "tiles": [
        {"value": "1/168", "label": "Old Town, GeoSensitivity rank"},
        {"value": "61%", "label": "Crime history, across Hull"},
        {"value": "14%", "label": "Built environment, across Hull"},
        {"value": "45%", "label": "Highest built share, Longhill"},
    ],
    "section_01": "The model depends most on [[t:Old Town]] (029E, rank [[n:1/168]]), well ahead "
                  "of Dairycoates and Sculcoates. There it leans on [[t:recent crime history]] "
                  "([[n:83%]] of its local importance); built-environment features carry only "
                  "[[n:6%]], mostly [[t:retail]].",
    "section_02": "Across Hull the model relies mainly on [[t:crime history]] ([[n:61%]]), then "
                  "census population characteristics ([[n:20%]]); built-environment features carry "
                  "[[n:14%]], led by retail, green space and roads. They matter most in "
                  "[[t:residential areas]] ([[n:41-45%]]): Longhill, Avenues and Sculcoates on "
                  "[[t:green space]], Bilton Grange on intersections, Orchard Park on transit stops.",
    "footer": "Model behaviour, not planning targets or causes of crime.",
}


# ---------------------------------------------------------------- evidence
def category(feature: str) -> str:
    """Category key of one of the 30 Feature players."""
    if feature.startswith("crime_"):
        return "crime"
    if feature.startswith("month_"):
        return "season"
    if feature in BUILT:
        return "built"
    return "census"


def built_share_table(d: StakeholderData) -> pd.DataFrame:
    """Per-LSOA built-environment share of local importance, as in plot_planner_built_share.py."""
    mag = d.local.abs()
    assert mag.shape[1] == 30 and set(BUILT) <= set(mag.columns)
    total = mag.sum(axis=1)
    by_feature = mag[list(BUILT)].T.groupby(BUILT).sum().T
    out = pd.DataFrame({
        "lsoa21nm": d.geo["lsoa21nm"].reindex(mag.index),
        "locality": d.places["locality_label"].reindex(mag.index),
        "total_local_importance": total,
        "built_importance": mag[list(BUILT)].sum(axis=1),
        "dominant_built_feature": by_feature.idxmax(axis=1),
    })
    out["built_share"] = out["built_importance"] / out["total_local_importance"]
    out = out.sort_values("built_share", ascending=False)
    out["rank"] = range(1, len(out) + 1)
    return out


def category_shares(d: StakeholderData, lsoa: str | None = None) -> pd.Series:
    """Share of local importance by category, Hull-wide (pooled) or for one LSOA."""
    mag = d.local.abs()
    values = mag.sum() if lsoa is None else mag.loc[lsoa]
    return (values.groupby(category).sum() / float(values.sum())).reindex(list(CATEGORIES))


def built_feature_shares(d: StakeholderData, lsoa: str | None = None) -> pd.Series:
    """Share of total local importance from each built-environment feature."""
    mag = d.local.abs()
    values = mag.sum() if lsoa is None else mag.loc[lsoa]
    return (values[list(BUILT)].groupby(BUILT).sum() / float(values.sum())).sort_values(ascending=False)


def built_environment_block(d: StakeholderData) -> dict:
    share = built_share_table(d)

    def rec(code: str) -> dict:
        row = share.loc[code]
        return {
            "id": code,
            "name": row["lsoa21nm"],
            "locality_label": row["locality"],
            "built_share": round(float(row["built_share"]), 3),
            "built_share_rank": int(row["rank"]),
            "dominant_built_feature": row["dominant_built_feature"],
        }

    mag = d.local.abs().loc[CENTRE]
    crime = [c for c in mag.index if c.startswith("crime_")]
    top_local = [{"feature": f, "share_of_local_importance": round(float(v / mag.sum()), 3)}
                 for f, v in mag.sort_values(ascending=False).head(5).items()]
    return {
        "definition": (
            "Share of each LSOA's total local Feature importance (|local coefficient| summed over "
            "all 30 Feature players) that comes from the 12 built-environment players: area, "
            "nightlife, retail, transit stops, roads, intersections, green space (count and "
            "density variants pooled). It is a share of model attribution, not an effect on crime."
        ),
        "hull_feature_category_shares": {CATEGORIES[k]: round(float(v), 3)
                                         for k, v in category_shares(d).sort_values(ascending=False).items()},
        "hull_built_feature_shares": {k: round(float(v), 3) for k, v in built_feature_shares(d).items()},
        "hull_share_note": "Hull-wide shares pool |local coefficient| over all 168 LSOAs; categories sum to 1.",
        "hull_share": round(float(share["built_importance"].sum() / share["total_local_importance"].sum()), 3),
        "median_lsoa_share": round(float(share["built_share"].median()), 3),
        "highest_built_share_lsoas": [rec(c) for c in share.index[:N_TOP]],
        "dominant_feature_counts_top20": share.head(20)["dominant_built_feature"].value_counts().to_dict(),
        "rank1_geo_sensitivity_lsoa": {
            **rec(CENTRE), "geo_rank": 1, "geo_rank_of": 168,
            "crime_history_share": round(float(mag[crime].sum() / mag.sum()), 3),
            "crime_history_features": crime,
            "top_local_features": top_local,
        },
        "limitation": "Attribution shares describe model behaviour; they are not planning targets.",
    }


def add_planner_evidence(evidence: dict, d: StakeholderData, mode: str) -> dict:
    """Attach the planner blocks and the output mode to a build_evidence package (in place)."""
    return add_evidence(evidence, d, "planner", mode)


# ---------------------------------------------------------------- police
POLICE_QUESTION = ("Where does the model lean on recent months rather than the long-run crime level, "
                   "and where is recorded crime rising or falling?")
RECENT = ("crime_lag_1", "crime_lag_2", "crime_lag_3", "crime_rolling_mean_3")
LONG_RUN = "crime_rolling_mean_6"
TREND_MIN_MEAN = 5      # areas below this many crimes a month in the earlier window are not assessed
TREND_THRESHOLD = 0.25  # "rising" / "falling" = a change of more than a quarter
TREND_LABEL_MIN = 10    # largest rises/falls are listed only for areas with at least this mean

POLICE_CARD_ADDENDUM = """
---

## 17. CARD MODE ADDENDUM

output_mode is card_mode: the text fills a short card for a Police / Community Safety reader,
shown with two dashboard panels (01: share of each LSOA's local explanation from recent-month
crime signals, Old Town outlined; 02: recorded change in crime, last three months against the
six before). This section
replaces section 14 "Length and form" and its bolding rule. Every other section, including
prohibited claims, the no-deployment rule and the self-check, still applies.

Plain language for this reader: do not cite Moran's I, p-values, LISA class names,
coefficients, GeoSensitivity values or other test statistics. Shares, percentages, counts of
areas and ranks (including the GeoSensitivity rank) are fine. GeoSensitivity means the model is
sensitive to an area's own data; never write that the model "relies on" or "depends on" an area.
Refer to the two windows as "the last three months" and "the six months before", without dates.

Return ONLY one JSON object with exactly these keys:

- "headline": one sentence, at most 20 words: the single most relevant finding that answers
  user_question.
- "tiles": exactly 4 objects {"value": ..., "label": ...}. value is one figure from the
  evidence (at most 8 characters, e.g. "51%", "25"); label is at most 6 words saying what the
  figure is and where. Choose the four figures this reader should remember.
- "section_01": at most 70 words, about panel 01: open with how sensitive the model is to
  the focus area's own data (focus_lsoa.geo_rank of geo_rank_of), then how much the model leans
  on recent months against the long-run (six-month) level across Hull (hull medians), and the
  areas where recent months carry the most (highest_recent_share_lsoas, by locality name).
- "section_02": at most 70 words, about panel 02: the Hull-wide recorded change, how many
  assessed areas rose or fell by more than a quarter, the largest changes by locality name,
  whether the areas the model reads through recent months are rising (use only the supplied
  fields; do not infer a link), and whether the focus area's recorded crime is stable.
- "footer": one limitation, at most 15 words, ruling out deployment or targeting.

Key-term counts: 1-2 in the headline and 2-4 in EACH section. Tiles and footer carry no
markup.
""" + _MARKUP_RULES + """
ILLUSTRATIVE_NARRATIVE in the user message shows the expected structure, order, tone and
markup density. It is an example of wording, not evidence: take every figure from
EVIDENCE_JSON.
"""
POLICE_CARD_ADDENDUM = POLICE_CARD_ADDENDUM.rstrip()

POLICE_COMPREHENSIVE_ADDENDUM = """
---

## 17. COMPREHENSIVE MODE ADDENDUM

output_mode is comprehensive_mode: a Police / Community Safety reader asked for the full
briefing, shown next to feature charts, maps and a crime timeline. This section replaces
section 14 "Length and form" and its bolding rule. Every other section, including prohibited
claims, the no-deployment rule and the self-check, still applies.

Write THREE short paragraphs separated by a blank line, with no headers or bullet lists.
Hard limit: at most 260 words in total, about 85 per paragraph. Count the words before
answering and cut detail (street names and LSOA codes first) until the count is within the
limit. Refer to the windows as "the last three months" and "the six months before", without
dates. Name areas by locality; LSOA codes such as 029E are names, never [[n:...]] figures.

1. Which signals lead local explanations across Hull (crime_signals.leading_feature_counts,
   hull medians of crime-history, recent-month and long-run shares) and where recent months
   carry the most (highest_recent_share_lsoas). A spatial statistic from
   local_attribution_patterns describes only the quantity it names (for example reliance on
   the leading feature); never attach it to the recent-month share. No GeoSensitivity or LISA
   class in this briefing.
2. Recorded change (recorded_trend): Hull-wide change, how many assessed areas rose or fell by
   more than a quarter, the largest rises and falls by locality name, and the supplied overlap
   between recent-month reliance and recorded rises. Do not infer a link the fields do not state.
3. The focus LSOA in dashboard_figures, from the figure captions supplied: what leads its local
   explanation and whether its recorded crime is persistent or changing. End by saying the area
   is worth closer contextual review only if the evidence supports it, and that this is not a
   basis for deployment.

Key terms: 4-10 [[t:...]] spans in the whole text, not more. Mark only the findings and the
two or three most important places or signals; leave other place names unmarked. Every
figure, including percentages in parentheses, is wrapped in [[n:...]].
""" + _MARKUP_RULES
POLICE_COMPREHENSIVE_ADDENDUM = POLICE_COMPREHENSIVE_ADDENDUM.rstrip()

POLICE_ILLUSTRATIVE_NARRATIVE = {
    "headline": "The model reads most areas through their [[t:long-run crime level]]; "
                "[[t:Old Town]] is the exception.",
    "tiles": [
        {"value": "1/168", "label": "Old Town, GeoSensitivity rank"},
        {"value": "51%", "label": "Old Town, recent-month reliance"},
        {"value": "25", "label": "Areas with crime up >25%"},
        {"value": "+38%", "label": "Dairycoates, recorded change"},
    ],
    "section_01": "The model is [[t:most sensitive]] to [[t:Old Town]]'s own data (rank [[n:1/168]]). "
                  "It reads most areas through their [[t:long-run crime level]]: the six-month "
                  "average is a median [[n:41%]] of local explanations, recent months [[n:12%]]. "
                  "Old Town is the exception, with [[n:51%]] from [[t:recent months]], ahead of "
                  "Dairycoates ([[n:32%]]) and Sculcoates ([[n:31%]]).",
    "section_02": "Across Hull, recorded crime rose [[n:3.6%]] in the last three months against the "
                  "six before. Of [[n:148]] assessed areas, [[n:25]] [[t:rose by over a quarter]] and "
                  "[[n:25]] fell by as much. [[t:Dairycoates]] rose [[n:38%]] where the model also "
                  "leans on recent months; Old Town stayed [[t:stable]] at about [[n:217]] crimes a "
                  "month ([[n:2%]] up).",
    "footer": "Model reliance and recorded counts are not a basis for deployment.",
}


def crime_signal_shares(d: StakeholderData) -> pd.DataFrame:
    """Per LSOA: share of the whole local explanation from recent-month and long-run crime signals."""
    mag = d.local.abs()
    total = mag.sum(axis=1)
    crime = [c for c in mag.columns if c.startswith("crime_")]
    out = pd.DataFrame({
        "recent_share": mag[list(RECENT)].sum(axis=1) / total,
        "long_run_share": mag[LONG_RUN] / total,
        "crime_history_share": mag[crime].sum(axis=1) / total,
        "leading_feature": mag.idxmax(axis=1),
    })
    return out.sort_values("recent_share", ascending=False)


def recorded_trend(d: StakeholderData) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Mean monthly crime in the last 3 months against the 6 before; change only where assessed."""
    m = d.crime.pivot_table(index="lsoa21cd", columns="month", values="crime_count")
    recent, previous = list(m.columns[-3:]), list(m.columns[-9:-3])
    out = pd.DataFrame({"recent_mean": m[recent].mean(axis=1), "previous_mean": m[previous].mean(axis=1)})
    out["assessed"] = out["previous_mean"] >= TREND_MIN_MEAN
    out["change"] = (out["recent_mean"] / out["previous_mean"] - 1).where(out["assessed"])
    months = lambda cols: [c.strftime("%Y-%m") for c in cols]
    return out, months(recent), months(previous)


def police_block(d: StakeholderData) -> dict:
    sig = crime_signal_shares(d)
    trend, recent_months, previous_months = recorded_trend(d)
    sig = sig.join(trend)

    def rec(code: str) -> dict:
        r = sig.loc[code]
        return {
            "id": code,
            "name": str(d.geo.loc[code, "lsoa21nm"]),
            "locality_label": str(d.places["locality_label"].get(code, "") or "") or None,
            "recent_share": round(float(r["recent_share"]), 3),
            "long_run_share": round(float(r["long_run_share"]), 3),
            "crime_history_share": round(float(r["crime_history_share"]), 3),
            "leading_feature": plain_label(str(r["leading_feature"])),
            "previous_mean_monthly_crime": round(float(r["previous_mean"]), 1),
            "recent_mean_monthly_crime": round(float(r["recent_mean"]), 1),
            "recorded_change": None if pd.isna(r["change"]) else round(float(r["change"]), 3),
        }

    assessed = sig[sig["assessed"]]
    big = assessed[assessed["previous_mean"] >= TREND_LABEL_MIN]
    top_recent = list(sig.index[:10])
    rising = assessed["change"] > TREND_THRESHOLD
    return {
        "definitions": {
            "recent_share": ("share of the whole local explanation (|local coefficient| over all 30 Feature "
                             "players) from crime in the last 1, 2 and 3 months and the 3-month average"),
            "long_run_share": "share of the whole local explanation from the 6-month crime average",
            "recorded_change": (f"mean monthly recorded crime in the last three months ({', '.join(recent_months)}) "
                                f"against the six months before ({', '.join(previous_months)}); assessed only where "
                                f"the earlier mean is at least {TREND_MIN_MEAN} crimes a month"),
            "note": "Shares describe what the model relies on, not why crime occurs.",
        },
        "crime_signals": {
            "hull_median_crime_history_share": round(float(sig["crime_history_share"].median()), 3),
            "hull_median_recent_share": round(float(sig["recent_share"].median()), 3),
            "hull_median_long_run_share": round(float(sig["long_run_share"].median()), 3),
            "leading_feature_counts": {plain_label(k): int(v) for k, v in
                                       sig["leading_feature"].value_counts().head(5).items()},
            "spearman_recent_share_vs_crime_level": round(float(sig["recent_share"].corr(
                sig["previous_mean"], method="spearman")), 2),
            "highest_recent_share_lsoas": [rec(c) for c in sig.index[:N_TOP]],
        },
        "recorded_trend": {
            "hull_total_change": round(float(trend["recent_mean"].sum() / trend["previous_mean"].sum() - 1), 3),
            "n_assessed": int(len(assessed)),
            "n_rising_over_quarter": int(rising.sum()),
            "n_falling_over_quarter": int((assessed["change"] < -TREND_THRESHOLD).sum()),
            "largest_rises": [rec(c) for c in big["change"].nlargest(3).index],
            "largest_falls": [rec(c) for c in big["change"].nsmallest(3).index],
            "top10_recent_share_rising_over_quarter": int(rising.reindex(top_recent, fill_value=False).sum()),
        },
        "focus_lsoa": {**rec(CENTRE), "geo_rank": int(d.geo.loc[CENTRE, "geo_rank"]), "geo_rank_of": d.n},
        "limitation": "Model reliance and recorded counts are not a basis for deployment, enforcement or targeting.",
    }


# ---------------------------------------------------------------- manager
MANAGER_QUESTION = ("What drives the model's predictions across Hull, what is exceptional, what can be set aside, "
                    "and how far can the results be trusted?")
TOP_K = 20
MANAGER_CONTRAST = "E01012916"  # Newland 012D: most sensitive of the areas not led by crime history
SET_ASIDE_SHARE = 0.01          # a feature is a set-aside candidate below 1% of pooled local importance
ATT_FAITHFULNESS_CSV = "outputs/faithfulness_accuracy_r2/att_faithfulness_area_data.csv"

MANAGER_CARD_ADDENDUM = """
---

## 17. CARD MODE ADDENDUM

output_mode is card_mode: the text fills a short card for a Manager / Strategic Decision Maker
(for example in local government or insurance), shown with four stat tiles and two dashboard
panels (01: map of the 20 most geographically sensitive LSOAs with the expert-highlighted
regions; 02: what the model relies on by feature category, Hull-wide against the contrast area
in manager_overview.contrast_area). This section replaces section 14 "Length and form" and its
bolding rule. Every other section, including prohibited claims, the evidence-review "next step"
rule and the self-check, still applies.

The card combines both branches: the Feature branch says WHAT the model relies on, the Geo
branch says WHERE it is most sensitive to an area's own data. Keep the two apart in wording.

Plain language for this reader: do not cite Moran's I, p-values, LISA class names, coefficients,
correlation values or other test statistics. GeoSensitivity values, ranks, shares, counts and
"N times" ratios are fine. Never write that the model "relies on" or "depends on" an area; it is
sensitive to an area's own data. Name areas by locality; no street names.

Return ONLY one JSON object with exactly these keys:

- "headline": one sentence, at most 20 words: the single most decision-relevant finding that
  answers user_question.
- "tiles": exactly 4 objects {"value": ..., "label": ...}. value is one figure from the
  evidence (at most 8 characters, e.g. "61%", "18/20"); label is at most 10 plain words that
  make the figure meaningful on its own to a reader who has not read the text: say what it
  measures in everyday words ("of the model's reasoning", not "share of local importance") and,
  where the evidence allows, what it is compared with (e.g. "against 20% Hull-wide"). A count
  is written as a fraction of its whole ("10/30", not "10"). One tile each for: what drives
  the model, the exception, what can be set aside, and confidence.
- "section_01": at most 70 words, drivers and exceptions (panel 02): what the model relies on
  across Hull (manager_overview.drivers), then the contrast area by locality name: what leads it
  and how it differs from Hull, and where such areas sit in GeoSensitivity
  (drivers.not_crime_led). Mention Old Town only as the GeoSensitivity exception, in one clause.
  The tiles carry the counts; do not restate how many areas the six-month average leads.
- "section_02": at most 60 words, what can be set aside and how far to trust it: name three of
  the set-aside candidates in plain words (manager_overview.set_aside.features) and their
  combined share, without restating how many there are (the tile gives it); add the caveat in
  plain words; then end with one
  clause saying the most sensitive areas agree with the expert's regions, pointing to panel 01
  for the count. No chance comparison, no stability or across-model remarks and no next-step
  sentence: this replaces the section 9 next-step item for this card.
- "footer": one limitation, at most 15 words, ruling out causal or resourcing readings.

No repetition: every figure appears ONCE on the card. The panels already print the figures
in manager_overview.shown_on_panels (the expert agreement in panel 01's title and the category
shares in panel 02), so neither tiles nor sections may cite them; refer to the panel in words
instead ("as panel 02 shows"). A tile figure is not cited again in the sections, not even its first number
("144/168" on a tile rules out "144 areas" in the text).

Key-term counts: 1-2 in the headline and 2-4 in EACH section. Tiles and footer carry no
markup.
""" + _MARKUP_RULES + """
ILLUSTRATIVE_NARRATIVE in the user message shows the expected structure, order, tone and
markup density. It is an example of wording, not evidence: take every figure from
EVIDENCE_JSON, and write your own sentences.
"""
MANAGER_CARD_ADDENDUM = MANAGER_CARD_ADDENDUM.rstrip()

MANAGER_COMPREHENSIVE_ADDENDUM = """
---

## 17. COMPREHENSIVE MODE ADDENDUM

output_mode is comprehensive_mode: a Manager / Strategic Decision Maker asked for the full
briefing, shown next to the top-20 map, the ranked GeoSensitivity chart and the feature-category
chart. This section replaces section 14 "Length and form" and its bolding rule. Every other
section, including prohibited claims and the evidence-review "next step" rule, still applies.

Write THREE short paragraphs separated by a blank line, with no headers or bullet lists.
Hard limit: at most 230 words in total. No test statistics, correlation values or p-values.

1. Drivers and exceptions (manager_overview.drivers and .concentration): what the model relies
   on across Hull, the contrast area, where non-crime-led areas sit in GeoSensitivity, and the
   GeoSensitivity exception. If an LSOA is selected, say whether it is exceptional or ordinary.
2. What can be set aside (manager_overview.set_aside) with its caveat, and how far to trust
   the results (manager_overview.confidence).
3. One evidence-review next step and one limitation.

Key terms: 4-8 [[t:...]] spans in the whole text. Every figure is wrapped in [[n:...]].
""" + _MARKUP_RULES
MANAGER_COMPREHENSIVE_ADDENDUM = MANAGER_COMPREHENSIVE_ADDENDUM.rstrip()

MANAGER_ILLUSTRATIVE_NARRATIVE = {
    "headline": "[[t:Recent crime history]] drives the model almost everywhere; a handful of features "
                "can be [[t:set aside]].",
    "tiles": [
        {"value": "144/168", "label": "areas where the six-month crime average leads"},
        {"value": "3.1x", "label": "Old Town's sensitivity against the next area"},
        {"value": "10/30", "label": "features the model barely uses"},
        {"value": "20/20", "label": "top areas kept when neighbouring areas are added"},
    ],
    "section_01": "As panel 02 shows, the model reads Hull mainly through [[t:crime history]], then "
                  "census characteristics. [[t:Newland]] is the reverse: census characteristics lead, "
                  "with long-term sickness alone [[n:21%]] of its reasoning. It is one of [[n:24]] "
                  "areas not led by crime history, and these are [[t:not where the model is most "
                  "sensitive]]: none ranks above [[n:116]]. Old Town remains the sensitivity exception.",
    "section_02": "Road density, intersection density and nightlife count are among the "
                  "[[t:candidates to set aside]]: each carries under [[n:1%]], all of them together "
                  "[[n:7%]] of local importance. Related count and "
                  "density measures can share weight, so judge them as groups. The areas where the "
                  "model is most sensitive agree with [[t:local expertise]] (panel 01).",
    "footer": "Describes what the model uses, not causes of crime or where to put resources.",
}


def manager_block(d: StakeholderData) -> dict:
    from pathlib import Path

    g = d.geo.sort_values("geo_rank")
    first, second = g.iloc[0], g.iloc[1]
    top = g.head(TOP_K)
    mag = d.local.abs()
    lead = mag.idxmax(axis=1)
    pooled = (mag.sum() / float(mag.values.sum())).sort_values()
    low = pooled[pooled < SET_ASIDE_SHARE]
    not_crime = lead[~lead.str.startswith("crime_")].index
    stab = d.metrics["stability"]
    att = pd.read_csv(Path(__file__).resolve().parents[2] / ATT_FAITHFULNESS_CSV)
    feat_acc = att[(att["branch"] == "Feature") & (att["measure"] == "att_accuracy")].iloc[0]
    geo_acc = att[(att["branch"] == "Geo") & (att["measure"] == "att_accuracy")].iloc[0]
    c = MANAGER_CONTRAST
    local = mag.loc[c] / mag.loc[c].sum()

    def area(code: str) -> dict:
        return {"id": code, "name": str(d.geo.loc[code, "lsoa21nm"]),
                "locality_label": str(d.places["locality_label"].get(code, "") or "") or None,
                "geo_sensitivity": round(float(d.geo.loc[code, "geo_coefficient"]), 2),
                "geo_rank": int(d.geo.loc[code, "geo_rank"]), "geo_rank_of": d.n}

    def group(f: str) -> str:
        return BUILT.get(f, f)

    return {
        "drivers": {
            "hull_feature_category_shares": {CATEGORIES[k]: round(float(v), 3)
                                             for k, v in category_shares(d).sort_values(ascending=False).items()},
            "leading_feature_counts": {plain_label(k): int(v) for k, v in lead.value_counts().head(3).items()},
            "n_lsoas": d.n,
            "not_crime_led": {"n_areas": int(len(not_crime)),
                              "highest_geo_rank": int(d.geo.loc[not_crime, "geo_rank"].min()),
                              "meaning": "areas whose leading local feature is not crime history; the "
                                         "highest-placed of them is this rank in GeoSensitivity"},
        },
        "contrast_area": {
            **area(c),
            "why_chosen": "most geographically sensitive of the areas not led by crime history",
            "category_shares": {CATEGORIES[k]: round(float(v), 3)
                                for k, v in category_shares(d, c).sort_values(ascending=False).items()},
            "top_local_features": [{"feature": plain_label(f), "share_of_local_importance": round(float(v), 3)}
                                   for f, v in local.sort_values(ascending=False).head(3).items()],
        },
        "concentration": {
            "rank1": area(first["lsoa21cd"]),
            "rank2": area(second["lsoa21cd"]),
            "rank1_to_rank2_ratio": round(float(first["geo_coefficient"] / second["geo_coefficient"]), 1),
            "top20_share_of_total_single_area_shift": round(float(top["exact_w1"].sum() / g["exact_w1"].sum()), 3),
        },
        "set_aside": {
            "threshold_share": SET_ASIDE_SHARE,
            "n_features_below": int(len(low)),
            "n_features_total": int(mag.shape[1]),
            "combined_share": round(float(low.sum()), 3),
            "features": [{"feature": plain_label(f), "share": round(float(v), 4),
                          "related_group_share": round(float(pooled[[x for x in pooled.index if group(x) == group(f)]].sum()), 4)}
                         for f, v in low.items()],
            "seasonality_share": round(float(category_shares(d)["season"]), 3),
            "caveat": ("low weight in this model, not irrelevance in reality; count and density versions of the "
                       "same measure can split weight, so judge them as groups (related_group_share)"),
        },
        "confidence": {
            "expert_agreement": {"top20_in_expert_regions": int(top["expert_region"].sum()), "k": TOP_K,
                                 "share": round(float(top["expert_region"].mean()), 2),
                                 "chance_share": round(len(d.regions) / d.n, 3)},
            "neighbour_stability": {"directions_tested": len(stab["directions"]),
                                    "top20_retained_min": min(v["top20_retained"] for v in stab["directions"].values())},
            "seeds": {"plain_reading": "the area ranking is very stable across runs; the order of features below "
                                       "the top few is less stable (evaluation.consistency)"},
            "across_models": {"n_models": int(geo_acc["n_models"]),
                              "plain_reading": ("across the candidate models, higher accuracy did not reliably give "
                                                "closer agreement with expert judgement, for areas or for features"
                                                if max(geo_acc["pearson_r"], feat_acc["pearson_r"]) < 0.3 else
                                                "more accurate models gave closer agreement with expert judgement")},
        },
        "shown_on_panels": {
            "note": "figures the two panels already print; tiles and text must not repeat them",
            "figures": ([f"{int(top['expert_region'].sum())}/{TOP_K}"]
                        + sorted({f"{100 * v:.0f}%" for v in [*category_shares(d), *category_shares(d, c)]})),
        },
        "limitations": [
            "Attribution describes what the model uses, not causes of crime or a basis for resourcing.",
            "Census characteristics (for example unemployment, long-term sickness) can act as proxies for "
            "protected groups; any use beyond explanation needs a fairness review.",
        ],
    }



# ---------------------------------------------------------------- researcher
RESEARCHER_QUESTION = ("How is Geo-SMILE behaving technically across Hull, and how faithful, spatially coherent "
                       "and reproducible are its explanations?")
LOCATION_GLOBAL_CSV = "outputs/feature_with_location/feature_with_location_global_importance.csv"

RESEARCHER_CARD_ADDENDUM = """
---

## 17. CARD MODE ADDENDUM

output_mode is card_mode: the text fills a short card for a GeoAI Researcher / Model Analyst,
shown with four stat tiles and two Hull-wide dashboard panels (01, Geo branch: signed
Geo-SMILE coefficient against the exact single-LSOA W1 shift for all 168 LSOAs; 02, Feature
branch: map of the local effect magnitude, |local coefficient|, of the most important feature).
The card is CITY-WIDE: report the main Hull-wide results of both branches and the
explanation-quality evaluation. Do not build the narrative around a single
LSOA; an extreme value may appear only as a property of the distribution. This section replaces
section 14 "Length and form" and its bolding rule. Every other section, including prohibited
claims and the self-check, still applies.

Technical language is expected: exact values, test statistics, correlation values, sign
conventions are allowed. Keep the sign convention exact: the signed Geo coefficient is a
relative contribution to the coalition W1, not a direction of crime change. Do not discuss
spatial clustering (Moran's I, LISA) or ATT Faithfulness on this card; the intervention result
(top 20 against random masking) is intervention impact, not faithfulness.

No repetition: every figure appears ONCE on the card. The panels already print the figures in
researcher_diagnostics.shown_on_panels, so neither tiles nor sections cite them; refer to the
panel instead ("panel 01"). A tile figure is not cited again in the sections, not even its
first number.

Return ONLY one JSON object with exactly these keys:

- "headline": one sentence, at most 20 words.
- "tiles": exactly 4 objects {"value": ..., "label": ...}. value is one figure from the
  evidence (at most 8 characters); label is at most 8 words naming the quantity exactly
  (e.g. "Geo surrogate held-out R2"). Use exactly these four, in this order: Geo surrogate R2,
  Geo top-20 expert-region accuracy, neighbour-addition Jaccard, Geo seed rank Spearman.
- "section_01": at most 80 words, how the explanation behaves across Hull: Geo branch (panel 01)
  that the signed ranking tracks the exact shift (value is in panel 01; do not cite it) while
  the absolute ranking does not, what the sign means, and how skewed the single-LSOA shift
  distribution is; Feature branch (panel 02) which feature matters most, in how many LSOAs it
  leads, how its local effect varies across Hull (median against the highest, and where by
  locality), and the location players' global ranks.
- "section_02": at most 80 words, explanation quality NOT on the tiles: Feature surrogate
  fidelity, intervention impact (top 20 against random masking, with its p-value), and Feature
  seed rank Spearman against the Geo branch's; then the limitation that matters most. Do not
  cite any tile value. Mark at least two key terms here (e.g. the intervention result and the
  reproducibility contrast).
- "footer": one limitation, at most 15 words, from researcher_diagnostics.limitations; not about
  spatial clustering or LISA.

Key-term counts: 1-2 in the headline and 2-4 in EACH section. Tiles and footer carry no
markup.
""" + _MARKUP_RULES + """
ILLUSTRATIVE_NARRATIVE in the user message shows the expected structure, order, tone and
markup density. It is an example of wording, not evidence: take every figure from
EVIDENCE_JSON, and write your own sentences.
"""
RESEARCHER_CARD_ADDENDUM = RESEARCHER_CARD_ADDENDUM.rstrip()

RESEARCHER_COMPREHENSIVE_ADDENDUM = """
---

## 17. COMPREHENSIVE MODE ADDENDUM

output_mode is comprehensive_mode: a GeoAI Researcher / Model Analyst asked for the full
technical briefing, shown next to the surrogate check, the LISA map and the seed-consistency
chart. This section replaces section 14 "Length and form" and its bolding rule. Every other
section, including prohibited claims and the self-check, still applies. Technical language,
exact values and test statistics are expected. ATT Faithfulness and intervention impact are
different measures: never conclude one from the other.

Write FOUR short paragraphs separated by a blank line, with no headers or bullet lists.
Hard limit: at most 300 words in total; count before answering.

1. Geo branch across Hull (researcher_diagnostics.geo): signed against absolute ranking with
   the sign convention, and the single-LSOA shift distribution. If an LSOA is selected, place it
   in that distribution.
2. Feature branch (researcher_diagnostics.feature): the most important feature, where its local
   effect is strongest and weakest, category shares and the location-player contrast.
3. Spatial structure of GeoSensitivity: global Moran's I with its p-value and permutations, and
   the LISA class counts (each a number of LSOAs, not clusters).
4. Explanation quality (researcher_diagnostics.quality and evaluation), ending with one
   limitation.

Key terms: 4-10 [[t:...]] spans in the whole text. Every figure is wrapped in [[n:...]].
""" + _MARKUP_RULES
RESEARCHER_COMPREHENSIVE_ADDENDUM = RESEARCHER_COMPREHENSIVE_ADDENDUM.rstrip()

RESEARCHER_ILLUSTRATIVE_NARRATIVE = {
    "headline": "Geo-SMILE's [[t:signed ranking]] tracks exact masking; locally, [[t:six-month crime]] "
                "dominates almost everywhere.",
    "tiles": [
        {"value": "0.948", "label": "Geo surrogate held-out R2"},
        {"value": "18/20", "label": "Geo top-20 in expert regions"},
        {"value": "1.0", "label": "Top-20 Jaccard, neighbour additions"},
        {"value": "0.963", "label": "Geo rank Spearman across seeds"},
    ],
    "section_01": "The [[t:signed coefficient]] tracks the exact shift (panel 01); its absolute value does not "
                  "([[n:-0.29]]), so the sign marks a relative contribution to the coalition W1. Shifts are "
                  "skewed: median [[n:6.4]], maximum [[n:176.7]]. In the Feature branch, the "
                  "[[t:six-month crime average]] leads in [[n:144]] LSOAs; its local effect (panel 02) has a "
                  "median of [[n:3.9]] but reaches [[n:48.5]] in Old Town. Location players rank only "
                  "[[n:20]] and [[n:23]] of [[n:32]].",
    "section_02": "Feature surrogate fidelity is [[n:0.983]]. Masking the Geo top 20 shifts predictions "
                  "[[n:26.22]] more than random sets ([[n:0.001]]), an [[t:intervention impact]]. Feature ranks "
                  "are [[t:less reproducible]] across seeds ([[n:0.69]]) than Geo ranks. Attributions remain "
                  "associational.",
    "footer": "Attributions are associational; local LISA p-values are unadjusted.",
}


def researcher_block(d: StakeholderData) -> dict:
    from pathlib import Path

    from scipy.stats import spearmanr

    root = Path(__file__).resolve().parents[2]
    g = d.geo
    mag = d.local.abs()
    pooled = (mag.sum() / float(mag.values.sum())).sort_values(ascending=False)
    loc_global = pd.read_csv(root / LOCATION_GLOBAL_CSV)
    m = d.metrics
    top_feature = mag.sum().idxmax()
    eff = mag[top_feature]
    rho_signed = spearmanr(g["geo_coefficient"], g["exact_w1"]).statistic
    rho_abs = spearmanr(g["geo_abs_coefficient"], g["exact_w1"]).statistic
    lisa = g["lisa"].value_counts()
    top = g.nsmallest(TOP_K, "geo_rank")
    return {
        "geo": {
            "n_lsoas": d.n, "perturbations": int(m["n"]),
            "spearman_signed_vs_exact_w1": round(float(rho_signed), 3),
            "spearman_abs_vs_exact_w1": round(float(rho_abs), 2),
            "exact_w1_median": round(float(g["exact_w1"].median()), 1),
            "exact_w1_p90": round(float(g["exact_w1"].quantile(0.9)), 1),
            "exact_w1_max": round(float(g["exact_w1"].max()), 1),
            "top20_share_of_total_exact_w1": round(float(top["exact_w1"].sum() / g["exact_w1"].sum()), 3),
            "n_negative_coefficients": int((g["geo_coefficient"] < 0).sum()),
            "moran_i": round(float(d.moran.get("Moran_I", float("nan"))), 2),
            "moran_p": d.moran.get("p_value"), "moran_permutations": d.moran.get("n_permutations"),
            "lisa_counts_lsoas": {k: int(v) for k, v in lisa.items()},
            "spatial_note": "for the comprehensive briefing only; not on the card",
        },
        "feature": {
            "category_shares": {CATEGORIES[k]: round(float(v), 3)
                                for k, v in category_shares(d).sort_values(ascending=False).items()},
            "top_pooled_local_shares": {k: round(float(v), 3) for k, v in pooled.head(3).items()},
            "total_local_abs_median": round(float(mag.sum(axis=1).median()), 1),
            "top_feature_local_effect": {
                "feature": top_feature, "label": plain_label(top_feature),
                "leads_in_lsoas": int((mag.idxmax(axis=1) == top_feature).sum()),
                "local_abs_median": round(float(eff.median()), 1), "local_abs_p90": round(float(eff.quantile(0.9)), 1),
                "local_abs_max": round(float(eff.max()), 1), "local_abs_min": round(float(eff.min()), 2),
                "highest_localities": list(dict.fromkeys(
                    str(d.places["locality_label"].get(c, "")) for c in eff.sort_values(ascending=False).index[:4])),
                "note": "magnitude of the local coefficient; the sign is not interpreted on the card",
            },
            "location_player_global_ranks": {row["feature"]: int(row["rank"])
                                             for _, row in loc_global[loc_global["is_location"] == 1].iterrows()},
            "location_player_rank_of": int(len(loc_global)),
        },
        "quality": {
            "geo_surrogate_r2": round(float(m["fidelity"]["R2"]), 3),
            "expert_region_accuracy_top20": f"{int(top['expert_region'].sum())}/{TOP_K}",
            "intervention_top20_minus_random_w1": round(float(m["faithfulness"]["top_minus_random"]), 2),
            "intervention_p": round(float(m["faithfulness"]["empirical_p_top_vs_random"]), 3),
            "intervention_note": "supplementary intervention impact, not ATT Faithfulness",
            "neighbour_addition_jaccard_mean": round(float(m["stability"]["jaccard_mean"]), 2),
            "geo_seed_rank_spearman": round(float(m["consistency"]["rank_spearman_mean"]), 3),
            "feature_fidelity_and_seed_consistency": "see evaluation (feature_surrogate_test_r2, consistency.feature)",
        },
        "shown_on_panels": {
            "note": "figures the two panels already print; tiles and text must not repeat them",
            "figures": [f"{rho_signed:.3f}", f"{float(np.percentile(eff, 98)):.1f}"],
        },
        "limitations": ["Attributions are associational, not causal.",
                        "Local coefficient magnitudes describe model attribution, not effects on crime."],
    }


# ---------------------------------------------------------------- public
PUBLIC_QUESTION = "What seems to matter to the model for crime predictions in places like mine?"
PUBLIC_FOCUS = "E01012806"  # Stoneferry: mid-ranked, total local importance close to the Hull median

PUBLIC_CARD_ADDENDUM = """
---

## 17. CARD MODE ADDENDUM

output_mode is card_mode: the text fills a short card for a Resident / Public reader, shown with
four stat tiles and two dashboard panels (01: Hull map of how strongly the model's predictions
depend on each area's own data, the example area outlined; 02: the example area's most
influential features). The example area is public_overview.example_area, chosen as a typical
place, not an extreme one. This section replaces section 14 "Length and form" and its bolding
rule. Every other section, including prohibited claims and the self-check, still applies.

Plain language only: no coefficients, test statistics, GeoSensitivity, LISA or other model
terms. Shares and "out of" counts are fine. Answer what the model pays attention to, never why
crime happens, and never describe residents or communities as risky.

Return ONLY one JSON object with exactly these keys:

- "headline": one sentence, at most 20 words.
- "tiles": exactly 4 objects {"value": ..., "label": ...}. value is one figure from the
  evidence (at most 8 characters); label is at most 10 everyday words that make the figure
  meaningful on its own. A count is written as a fraction of its whole. Use exactly these
  four, in this order, each from the named field: public_overview.tile_figures.crime_history_share,
  .built_environment_share, .areas_led_by_six_month_average, and .least_influential_feature
  (its share, naming that same feature).
- "section_01": at most 65 words, across Hull (panel 01): what the model mostly pays attention
  to, what matters comparatively little, and the places (by locality) where predictions depend
  most on the area's own data. Do not cite any tile value.
- "section_02": at most 65 words, the example area (panel 02) by locality and its main streets:
  its most influential features and whether its pattern is like Hull's or different. Do not
  cite any tile value or panel figure.
- "footer": one non-causal limitation, at most 15 words.

No repetition: every figure appears ONCE on the card. The panels already print the example
area's rank and its feature shares (public_overview.shown_on_panels), so neither tiles nor
sections cite them; say "panel 02 shows" instead. A tile figure is not cited again in the
sections. Key-term counts: 1-2 in the headline and 2-4 in EACH section. Tiles and footer carry
no markup.
""" + _MARKUP_RULES + """
ILLUSTRATIVE_NARRATIVE in the user message shows the expected structure, order, tone and
markup density. It is an example of wording, not evidence: take every figure from
EVIDENCE_JSON, and write your own sentences.
"""
PUBLIC_CARD_ADDENDUM = PUBLIC_CARD_ADDENDUM.rstrip()

PUBLIC_COMPREHENSIVE_ADDENDUM = """
---

## 17. COMPREHENSIVE MODE ADDENDUM

output_mode is comprehensive_mode: a Resident / Public reader asked for the fuller
explanation, shown next to the Hull map and the feature charts. This section replaces section
14 "Length and form" and its bolding rule. Every other section still applies.

Write THREE short paragraphs separated by a blank line, with no headers or bullet lists.
Hard limit: at most 230 words in total. Plain language only (section 9, Resident / Public).

1. What the model mostly pays attention to across Hull, and what matters comparatively little.
2. The selected area (or public_overview.example_area if none), by locality and streets: its
   most and least influential features and whether its pattern is like Hull's.
3. Where predictions depend most on an area's own data (by locality), said plainly, and one
   non-causal limitation.

Key terms: 4-8 [[t:...]] spans in the whole text. Every figure is wrapped in [[n:...]].
""" + _MARKUP_RULES
PUBLIC_COMPREHENSIVE_ADDENDUM = PUBLIC_COMPREHENSIVE_ADDENDUM.rstrip()

PUBLIC_ILLUSTRATIVE_NARRATIVE = {
    "headline": "The model mostly looks at [[t:recent crime levels]]; local features add a smaller part.",
    "tiles": [
        {"value": "61%", "label": "of what the model weighs is past crime"},
        {"value": "14%", "label": "of what it weighs is streets, shops and green space"},
        {"value": "144/168", "label": "areas where the six-month crime average matters most"},
        {"value": "0.5%", "label": "of what it weighs is intersection density"},
    ],
    "section_01": "Across Hull the model pays most attention to [[t:recent crime levels]]. Census "
                  "characteristics such as unemployment come next ([[n:20%]]), then streets, shops "
                  "and green space. [[t:Road and junction density]] matter very little. "
                  "Predictions depend most on an area's own data in [[t:Old Town]], Dairycoates and "
                  "Sculcoates.",
    "section_02": "In [[t:Stoneferry]], around Stoneferry Road, the model also leans on the six-month "
                  "crime average, then the three-month average and private renting. Its pattern is "
                  "[[t:much like Hull's]], and its predictions depend on its own data about as much as "
                  "a [[t:typical area]].",
    "footer": "Shows what the model pays attention to, not why crime happens.",
}


def public_block(d: StakeholderData) -> dict:
    mag = d.local.abs()
    pooled = (mag.sum() / float(mag.values.sum())).sort_values(ascending=False)
    f = PUBLIC_FOCUS
    local = (mag.loc[f] / mag.loc[f].sum()).sort_values(ascending=False)
    g = d.geo.sort_values("geo_rank")

    def place(code: str) -> str:
        return str(d.places["locality_label"].get(code, "") or "")

    return {
        "hull": {
            "feature_category_shares": {CATEGORIES[k]: round(float(v), 3)
                                        for k, v in category_shares(d).sort_values(ascending=False).items()},
            "most_influential_features": [{"feature": plain_label(k), "share": round(float(v), 3)}
                                          for k, v in pooled.head(5).items()],
            "least_influential_features": [{"feature": plain_label(k), "share": round(float(v), 4)}
                                           for k, v in pooled.tail(3).items()],
            "leading_feature_counts": {plain_label(k): int(v) for k, v in
                                       mag.idxmax(axis=1).value_counts().head(3).items()},
            "n_areas": d.n,
            "places_where_own_data_matters_most": list(dict.fromkeys(place(c) for c in g.index[:5])),
        },
        "tile_figures": {
            "crime_history_share": f"{100 * category_shares(d)['crime']:.0f}%",
            "built_environment_share": f"{100 * category_shares(d)['built']:.0f}%",
            "areas_led_by_six_month_average": f"{int((mag.idxmax(axis=1) == 'crime_rolling_mean_6').sum())}/{d.n}",
            "least_influential_feature": {"feature": plain_label(pooled.index[-1]),
                                          "share": f"{100 * pooled.iloc[-1]:.1f}%"},
        },
        "example_area": {
            "id": f, "name": str(d.geo.loc[f, "lsoa21nm"]), "locality_label": place(f),
            "major_streets": str(d.places.loc[f, "major_streets"]).split("; ")[:3],
            "why_chosen": "a typical area: mid-ranked, total importance close to the Hull median",
            "own_data_rank": int(d.geo.loc[f, "geo_rank"]), "own_data_rank_of": d.n,
            "most_influential_features": [{"feature": plain_label(k), "share": round(float(v), 3)}
                                          for k, v in local.head(3).items()],
            "least_influential_features": [plain_label(k) for k in local.tail(3).index],
            "category_shares": {CATEGORIES[k]: round(float(v), 3)
                                for k, v in category_shares(d, f).sort_values(ascending=False).items()},
            "total_importance_vs_hull_median": round(float(mag.loc[f].sum() / mag.sum(axis=1).median()), 2),
        },
        "shown_on_panels": {
            "note": "figures the two panels already print; tiles and text must not repeat them",
            "figures": [f"{int(d.geo.loc[f, 'geo_rank'])}/{d.n}"] + [f"{100 * v:.0f}%" for v in local.head(2)],
        },
        "limitation": "The model shows what it pays attention to, not why crime happens or who is at risk.",
    }


# ---------------------------------------------------------------- governance
GOVERNANCE_QUESTION = ("Is the explanation traceable and reproducible, and which limitations require explicit "
                       "review before the evidence is relied upon?")

GOVERNANCE_CARD_ADDENDUM = """
---

## 17. CARD MODE ADDENDUM

output_mode is card_mode: the text fills a short card for a Governance / Assurance reader,
shown with four stat tiles and two Hull-wide dashboard panels (01: GeoSensitivity of the top 20
areas across five perturbation seeds; 02: the top 20 under a neighbouring-area addition in each
of four directions, titled with the model, design, perturbation count and seed). The card is
CITY-WIDE: system-level provenance, reproducibility and limitations, not one LSOA's story; an
unusual area appears only as a case for review. This section replaces section 14 "Length and
form" and its bolding rule. Every other section, including prohibited claims and the
self-check, still applies.

Assurance language: exact metric names and values are allowed; keep each metric's definition
exact. ATT Faithfulness (across-model r between predictive R2 and ATT accuracy) and
intervention impact (top 20 against random masking) are different measures: never conclude one
from the other.

No repetition: every figure appears ONCE on the card. The panels already print the figures in
governance_assurance.shown_on_panels (perturbation count, seed, "20/20 retained"), so neither
tiles nor sections cite them; refer to the panel instead ("panel 02").

Return ONLY one JSON object with exactly these keys:

- "headline": one sentence, at most 20 words.
- "tiles": exactly 4 objects {"value": ..., "label": ...}. value is one figure from the
  evidence (at most 8 characters); label is at most 8 words naming the measure exactly. Use
  exactly these four, in this order: Geo rank Spearman across seeds, Geo top-fifth Jaccard
  across seeds, rank-1 to rank-2 ratio, Geo ATT Faithfulness r.
- "section_01": at most 75 words, provenance and reproducibility (panels 01-02): which model and
  design produced the explanation (point to panel 02 for count and seed), how many areas each
  perturbation masks, the expert reference's status, and seed reproducibility of the most
  variable area and of the Feature branch. Do not cite any tile value.
- "section_02": at most 75 words, limitations requiring review: the sign is not a direction of
  crime; the single provisional expert; unadjusted local LISA p-values; ATT Faithfulness not
  demonstrated (describe in words, its value is on a tile); the rank-1 area's dominance as a
  case for separate model review. Do not cite any tile value.
- "footer": one limitation, at most 15 words.

Key-term counts: 1-2 in the headline and 2-4 in EACH section. Tiles and footer carry no
markup.
""" + _MARKUP_RULES + """
ILLUSTRATIVE_NARRATIVE in the user message shows the expected structure, order, tone and
markup density. It is an example of wording, not evidence: take every figure from
EVIDENCE_JSON, and write your own sentences.
"""
GOVERNANCE_CARD_ADDENDUM = GOVERNANCE_CARD_ADDENDUM.rstrip()

GOVERNANCE_COMPREHENSIVE_ADDENDUM = """
---

## 17. COMPREHENSIVE MODE ADDENDUM

output_mode is comprehensive_mode: a Governance / Assurance reader asked for the full
assurance briefing, shown next to the seed-consistency chart and the stability figure. This
section replaces section 14 "Length and form" and its bolding rule. Every other section still
applies. Exact metric names and values are expected; ATT Faithfulness and intervention impact
are different measures.

Write THREE short paragraphs separated by a blank line, with no headers or bullet lists.
Hard limit: at most 260 words in total.

1. Provenance: what each branch explains, model, design, perturbations, seeds, the expert
   reference's status and the prompt version.
2. Robustness: fidelity, neighbour stability, seed consistency for both branches, intervention
   impact (supplementary) and ATT Faithfulness across models. If an LSOA is selected, whether
   its explanation is unusual or unstable relative to Hull.
3. Unusual cases, missing evidence and risks of unsupported inference, ending with the
   boundary of appropriate use.

Key terms: 4-10 [[t:...]] spans in the whole text. Every figure is wrapped in [[n:...]].
""" + _MARKUP_RULES
GOVERNANCE_COMPREHENSIVE_ADDENDUM = GOVERNANCE_COMPREHENSIVE_ADDENDUM.rstrip()

GOVERNANCE_ILLUSTRATIVE_NARRATIVE = {
    "headline": "The explanation is [[t:traceable and reproducible]]; four limitations need review before "
                "reliance.",
    "tiles": [
        {"value": "0.963", "label": "Geo rank Spearman across seeds"},
        {"value": "0.77", "label": "Geo top-fifth Jaccard across seeds"},
        {"value": "3.1x", "label": "Rank-1 to rank-2 GeoSensitivity ratio"},
        {"value": "0.15", "label": "Geo ATT Faithfulness r, ten models"},
    ],
    "section_01": "GeoSensitivity comes from one XGBoost model under [[t:area-data masking]] (panel 02 gives "
                  "the count and seed); each perturbation masks [[n:5]] to [[n:10]] areas. The expert "
                  "reference is a [[t:provisional single expert]], intended to be blind to the outputs. Old "
                  "Town varies most across seeds ([[n:2.8%]] of its mean); Feature ranks are [[t:less "
                  "reproducible]] ([[n:0.69]]).",
    "section_02": "The coefficient [[t:sign is not a direction]] of crime. The expert view comes from one "
                  "person, and local LISA p-values are unadjusted. Across models, accuracy does not track "
                  "expert agreement, so [[t:ATT Faithfulness]] is not demonstrated. Old Town dominates the "
                  "geographic picture and needs [[t:separate model review]].",
    "footer": "Associational explanation; not a basis for decisions about individuals or communities.",
}


def governance_block(d: StakeholderData) -> dict:
    from pathlib import Path

    m = d.metrics
    s = m["smile_diagnostics"]
    att = pd.read_csv(Path(__file__).resolve().parents[2] / ATT_FAITHFULNESS_CSV)
    att_r = {b: round(float(att[(att["branch"] == b) & (att["measure"] == "att_accuracy")]["pearson_r"].iloc[0]), 2)
             for b in ("Feature", "Geo")}
    c = m["consistency"]
    g = d.geo.sort_values("geo_rank")
    return {
        "provenance": {
            "model": "XGBoost", "geo_design": s["perturbation"], "geo_response": s["response"],
            "geo_perturbations": int(m["n"]), "base_seed": 42, "consistency_seeds": c["seeds"],
            "lsoas_masked_per_perturbation": [int(s["min_lsoas_per_perturbation"]),
                                              int(s["max_lsoas_per_perturbation"])],
            "n_areas": d.n, "feature_branch": "30 Feature players, location excluded",
            "expert_reference": "provisional single-expert reference; annotators intended to be blinded to "
                                "GeoSMILE outputs",
        },
        "reproducibility": {
            "geo_seed_rank_spearman": round(float(c["rank_spearman_mean"]), 3),
            "geo_seed_top20pct_jaccard": round(float(c["top20pct_jaccard_mean"]), 2),
            "feature_seed_rank_spearman": "see evaluation.consistency.feature",
            "neighbour_addition_jaccard_mean": round(float(m["stability"]["jaccard_mean"]), 2),
            "geo_surrogate_r2": round(float(m["fidelity"]["R2"]), 3),
            "intervention_top20_minus_random_w1": round(float(m["faithfulness"]["top_minus_random"]), 2),
            "intervention_note": "supplementary intervention impact, not ATT Faithfulness",
            "att_faithfulness_pearson_r": {**att_r, "n_models": int(att["n_models"].iloc[0]),
                                           "definition": "across-model Pearson r between predictive R2 and ATT accuracy"},
        },
        "unusual_cases": {
            "most_variable_area": {"id": c["max_sd_lsoa"], "name": c["max_sd_lsoa_name"],
                                   "locality_label": str(d.places["locality_label"].get(c["max_sd_lsoa"], "")),
                                   "sd": round(float(c["max_sd"]), 2),
                                   "sd_pct_of_mean": round(float(c["max_sd_pct_of_mean"]), 1)},
            "rank1_to_rank2_ratio": round(float(g["geo_coefficient"].iloc[0] / g["geo_coefficient"].iloc[1]), 1),
        },
        "risks": ["The signed coefficient is a relative contribution to the coalition W1, not a direction of crime.",
                  "The expert reference is a single provisional rater.",
                  "Local LISA p-values are unadjusted for multiple testing.",
                  "ATT Faithfulness was not demonstrated across models.",
                  "Census characteristics can act as proxies for protected groups.",
                  "Attributions are associational, not causal."],
        "shown_on_panels": {
            "note": "figures the two panels already print; tiles and text must not repeat them",
            "figures": ["4,000", "4000", "42", "20/20"],
        },
    }


# ---------------------------------------------------------------- registry
SPECS = {
    "planner": {
        "question": PLANNER_QUESTION,
        "addenda": {"short": CARD_MODE_ADDENDUM, "comprehensive": COMPREHENSIVE_MODE_ADDENDUM},
        "example": ILLUSTRATIVE_NARRATIVE,
        "block": ("built_environment_share", built_environment_block),
        "titles": ("01 · Most important areas", "02 · Built-environment share"),
        "version": MODE_VERSION,
    },
    "police": {
        "question": POLICE_QUESTION,
        "addenda": {"short": POLICE_CARD_ADDENDUM, "comprehensive": POLICE_COMPREHENSIVE_ADDENDUM},
        "example": POLICE_ILLUSTRATIVE_NARRATIVE,
        "block": ("police_signals", police_block),
        "titles": ("01 · Recent or long-run signal", "02 · Recorded change"),
        "version": {"short": "police-card-v3", "comprehensive": "police-comprehensive-v2"},
        "limits": {"section_01": 70, "section_02": 70, "footer": 15},
    },
    "manager": {
        "question": MANAGER_QUESTION,
        "addenda": {"short": MANAGER_CARD_ADDENDUM, "comprehensive": MANAGER_COMPREHENSIVE_ADDENDUM},
        "example": MANAGER_ILLUSTRATIVE_NARRATIVE,
        "block": ("manager_overview", manager_block),
        "titles": ("Drivers and exceptions", "Set aside and confidence"),
        "version": {"short": "manager-card-v6", "comprehensive": "manager-comprehensive-v2"},
        "limits": {"section_01": 70, "section_02": 60, "footer": 15},
        "no_repeat": True,
        "comprehensive_limits": {"text": 230},
    },
    "researcher": {
        "question": RESEARCHER_QUESTION,
        "addenda": {"short": RESEARCHER_CARD_ADDENDUM, "comprehensive": RESEARCHER_COMPREHENSIVE_ADDENDUM},
        "example": RESEARCHER_ILLUSTRATIVE_NARRATIVE,
        "block": ("researcher_diagnostics", researcher_block),
        "titles": ("01 · Explanation behaviour across Hull", "02 · Explanation quality"),
        "version": {"short": "researcher-card-v9", "comprehensive": "researcher-comprehensive-v3"},
        "limits": {"section_01": 80, "section_02": 80, "footer": 15},
        "short_drop_limitations": ("LISA", "pattern_scale"),
        "technical": True,
        "no_repeat": True,
        "comprehensive_limits": {"text": 300},
    },
    "public": {
        "question": PUBLIC_QUESTION,
        "addenda": {"short": PUBLIC_CARD_ADDENDUM, "comprehensive": PUBLIC_COMPREHENSIVE_ADDENDUM},
        "example": PUBLIC_ILLUSTRATIVE_NARRATIVE,
        "block": ("public_overview", public_block),
        "titles": ("01 · Across Hull", "02 · A typical area"),
        "version": {"short": "public-card-v5", "comprehensive": "public-comprehensive-v1"},
        "limits": {"section_01": 65, "section_02": 65, "footer": 15},
        "no_repeat": True,
        "comprehensive_limits": {"text": 230},
    },
    "governance": {
        "question": GOVERNANCE_QUESTION,
        "addenda": {"short": GOVERNANCE_CARD_ADDENDUM, "comprehensive": GOVERNANCE_COMPREHENSIVE_ADDENDUM},
        "example": GOVERNANCE_ILLUSTRATIVE_NARRATIVE,
        "block": ("governance_assurance", governance_block),
        "titles": ("01 · Provenance and reproducibility", "02 · Limitations requiring review"),
        "version": {"short": "governance-card-v4", "comprehensive": "governance-comprehensive-v2"},
        "limits": {"section_01": 75, "section_02": 75, "footer": 15},
        "technical": True,
        "no_repeat": True,
        "comprehensive_limits": {"text": 260},
    },
}


def add_evidence(evidence: dict, d: StakeholderData, stakeholder: str, mode: str) -> dict:
    """Attach the stakeholder's card block and the output mode to a build_evidence package (in place)."""
    key, block = SPECS[stakeholder]["block"]
    evidence["output_mode"] = {"short": "card_mode", "comprehensive": "comprehensive_mode"}[mode]
    evidence[key] = block(d)
    evidence["card_technical"] = bool(SPECS[stakeholder].get("technical"))
    drop = SPECS[stakeholder].get("short_drop_limitations", ()) if mode == "short" else ()
    if drop:  # limitations about evidence the short card does not show
        evidence["limitations"] = [x for x in evidence.get("limitations", []) if not any(s in x for s in drop)]
    if mode == "short" and SPECS[stakeholder].get("no_repeat"):
        evidence["card_panel_figures"] = evidence[key].get("shown_on_panels", {}).get("figures", [])
    evidence["global_spatial_pattern"]["lisa_counts_unit"] = (
        "number of LSOAs in each LISA class, not number of clusters")
    return evidence


def user_suffix(mode: str, stakeholder: str = "planner") -> str | None:
    if mode != "short":
        return None
    return ("\n\nILLUSTRATIVE_NARRATIVE (example of structure only, not evidence):\n"
            + json.dumps(SPECS[stakeholder]["example"], ensure_ascii=True, indent=1))


# ---------------------------------------------------------------- parsing and checks
def parse_card(text: str) -> dict:
    return json.loads(text[text.find("{"): text.rfind("}") + 1])


FIG = re.compile(r"-?\d+(?:\.\d+)?")


def _evidence_numbers(obj) -> list[float]:
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in _evidence_numbers(x)]
    if isinstance(obj, (list, tuple)):
        return [v for x in obj for v in _evidence_numbers(x)]
    if isinstance(obj, bool) or obj is None:
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    return [float(m) for m in FIG.findall(str(obj))]


def _supported(token: str, pool: list[float]) -> bool:
    v = float(token)
    tol = 0.5 * 10 ** -(len(token.split(".")[1]) if "." in token else 0) + 1e-9
    # Sign-free: a fall cited as "44%" matches a change of -0.44.
    return any(abs(abs(e) - v) <= tol or abs(abs(100 * e) - v) <= tol for e in pool)


def _bare(text: str) -> str:
    return re.sub(r"\[\[[nt]:(.+?)\]\]", r"\1", text)


def _variants(fig: str) -> list[str]:
    """A figure and its spelled-out forms: "18/20" also matches "18 of 20" and "18 of the 20"."""
    m = re.fullmatch(r"(\d+)/(\d+)", fig.strip())
    return [fig] + ([f"{m[1]} of {m[2]}", f"{m[1]} of the {m[2]}"] if m else [])


def _repeats(parts: dict, panel_figures: list[str]) -> list[str]:
    """Figures printed twice: a panel figure on a tile or in the text, or a tile figure in the text."""
    def found(fig: str, text: str) -> bool:
        return any(re.search(rf"(?<![\d.]){re.escape(x)}(?!\d)", text) for x in _variants(fig))

    tiles = [str(t.get("value", "")) for t in parts.get("tiles", [])]
    text = " ".join(_bare(parts.get(k, "")) for k in ("section_01", "section_02"))
    return ([f"tile repeats panel figure {v!r}" for v in tiles if any(found(p, v) for p in panel_figures)]
            + [f"text repeats panel figure {p!r}" for p in panel_figures if found(p, text)]
            + [f"text repeats tile figure {v!r}" for v in tiles
               if found(v, text) or found(v.split("/")[0], text) and "/" in v])


def check(content: dict | str, evidence: dict, mode: str = "short", limits: dict | None = None) -> dict:
    """Figures traced to the evidence, length and key-term limits, markup and wording rules."""
    parts = {"text": content} if mode == "comprehensive" else content
    pool = _evidence_numbers(evidence)
    problems = []
    if mode == "short":
        if sorted(parts) != sorted(["headline", "tiles", "section_01", "section_02", "footer"]):
            problems.append(f"keys: {sorted(parts)}")
        if len(parts.get("tiles", [])) != 4:
            problems.append(f"tiles: {len(parts.get('tiles', []))} != 4")
    for key, limit in (limits or LIMITS[mode]).items():
        words = len(_bare(parts.get(key, "")).split())
        if words > limit:
            problems.append(f"{key}: {words} words > {limit}")
    for key, (lo, hi) in TERM_RANGE[mode].items():
        n = len(re.findall(r"\[\[t:", parts.get(key, "")))
        if not lo <= n <= hi:
            problems.append(f"{key}: {n} key terms, expected {lo}-{hi}")
    texts = [k for k in parts if k not in ("tiles", "footer", "headline")]
    cited = [m for k in texts for m in re.findall(r"\[\[n:(.+?)\]\]", parts.get(k, ""))]
    cited += [t.get("value", "") for t in parts.get("tiles", [])]
    for fig in cited:
        for token in FIG.findall(fig.replace("–", "-").replace("−", "-")):
            if not _supported(token.lstrip("-"), pool):
                problems.append(f"figure not in evidence: {token!r} (in {fig!r})")
    for k in texts:
        text = parts.get(k, "")
        bare = re.sub(r"\[\[[nt]:.+?\]\]", "", text)
        # Names, not figures: LSOA suffixes (029E), panel references, W1 / R2, "top-20", ordinals.
        names = r"\b\d{3}[A-Z]\b|\b[Pp]anel \d+|\b(?:W1|R2|W_1)\b|[Tt]op[- ]?\d+|\b\d+(?:st|nd|rd|th)\b"
        if re.search(r"\d", re.sub(names, "", bare)):
            problems.append(f"{k}: unmarked figure")
        if "**" in text:
            problems.append(f"{k}: markdown bold used")
        if mode == "short" and not evidence.get("card_technical") and JARGON.search(text):
            problems.append(f"{k}: test-statistic wording {JARGON.search(text).group(0)!r}")
    if mode == "short" and "card_panel_figures" in evidence:
        problems += _repeats(parts, evidence["card_panel_figures"])
    return {"ok": not problems, "problems": problems}


# ---------------------------------------------------------------- rendering
PINK, WINE = "#FCE4EC", "#8A0033"


def marked_html(text: str) -> str:
    """Escape, then render [[n:]] and [[t:]] as one pink highlight; blank lines split paragraphs."""
    out = html.escape(text)
    out = re.sub(r"\[\[[nt]:(.+?)\]\]",
                 rf"<span style='background:{PINK}; color:{WINE}; padding:0 2px; border-radius:2px;'>\1</span>",
                 out)
    return "".join(f"<p style='margin:0 0 0.6rem 0; line-height:1.55;'>{p.strip()}</p>"
                   for p in re.split(r"\n\s*\n", out) if p.strip())


def tiles_html(tiles: list[dict]) -> str:
    cells = "".join(
        f"<div style='flex:1; background:{PINK}; border-radius:6px; padding:8px 10px;'>"
        f"<div style='color:{WINE}; font-weight:700; font-size:1.15rem;'>{html.escape(str(t.get('value', '')))}</div>"
        f"<div style='opacity:0.75; font-size:0.8rem;'>{html.escape(str(t.get('label', '')))}</div></div>"
        for t in tiles[:4])
    return f"<div style='display:flex; gap:8px; margin:0.4rem 0 0.8rem 0;'>{cells}</div>"


UNICODE_TEX = {"–": "--", "—": "---", "−": "$-$", "’": "'", "‘": "`", "“": "``", "”": "''",
               "β": r"$\beta$", "≈": r"$\approx$", "×": r"$\times$", "·": r"\textperiodcentered{}"}


def tex_escape(text: str) -> str:
    for ch in "&#_$%":
        text = text.replace(ch, "\\" + ch)
    text = text.replace("<", "$<$").replace(">", "$>$")  # bare < > print as ¡ ¿ in OT1
    for k, v in UNICODE_TEX.items():
        text = text.replace(k, v)
    return re.sub(r"(?<=\d)-(?=\d)", "--", text)


def marked_tex(text: str) -> str:
    text = tex_escape(text)
    text = re.sub(r"\[\[n:(.+?)\]\]", r"\\pv{\1}", text)
    return re.sub(r"\[\[t:(.+?)\]\]", r"\\kt{\1}", text)
