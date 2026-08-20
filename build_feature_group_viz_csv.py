"""Build visualization CSVs for Feature-group (category) alignment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs" / "feature_category_alignment"
EXPERT = Path(__file__).resolve().parent / "outputs" / "feature_expert_ranking_v2.csv"


def main() -> None:
    feat = pd.read_csv(OUT / "feature_with_categories.csv")
    expert = pd.read_csv(EXPERT)
    cat = pd.read_csv(OUT / "category_alignment_thesis.csv")

    m = feat.merge(expert[["feature", "expert_category"]], on="feature", how="left")
    m["category"] = m["expert_category"].fillna(m["thesis_category"])
    m = m.merge(
        cat[
            [
                "category",
                "expert_category_rank",
                "geosmile_category_rank",
                "geosmile_mass_share",
                "expert_mean_feature_rank",
                "n_expert_top10",
                "n_geosmile_top10",
            ]
        ],
        on="category",
        how="left",
    )
    m["in_expert_top10"] = (m["expert_rank"] <= 10).astype(int)
    m["in_geosmile_top10"] = (m["geosmile_rank"] <= 10).astype(int)
    m["in_both_top10"] = (
        (m["in_expert_top10"] == 1) & (m["in_geosmile_top10"] == 1)
    ).astype(int)
    m["rank_gap"] = (m["geosmile_rank"] - m["expert_rank"]).astype(int)
    m = m.sort_values(
        ["expert_category_rank", "expert_rank", "feature"], kind="mergesort"
    ).reset_index(drop=True)

    feature_out = m[
        [
            "category",
            "expert_category_rank",
            "geosmile_category_rank",
            "geosmile_mass_share",
            "expert_mean_feature_rank",
            "feature",
            "expert_rank",
            "geosmile_rank",
            "rank_gap",
            "abs_coefficient",
            "coefficient",
            "in_expert_top10",
            "in_geosmile_top10",
            "in_both_top10",
            "n_expert_top10",
            "n_geosmile_top10",
        ]
    ].rename(
        columns={
            "abs_coefficient": "geosmile_abs_coefficient",
            "coefficient": "geosmile_coefficient",
            "n_expert_top10": "category_n_expert_top10",
            "n_geosmile_top10": "category_n_geosmile_top10",
        }
    )
    feature_path = OUT / "feature_group_alignment_for_viz.csv"
    feature_out.to_csv(feature_path, index=False)

    cat_out = (
        cat.sort_values("expert_category_rank")
        .rename(columns={"category": "feature_group"})[
            [
                "feature_group",
                "expert_category_rank",
                "geosmile_category_rank",
                "n_features",
                "geosmile_mass_share",
                "geosmile_abs_mass",
                "expert_mean_feature_rank",
                "expert_median_feature_rank",
                "n_expert_top10",
                "n_geosmile_top10",
                "features",
            ]
        ]
    )
    cat_path = OUT / "feature_group_summary_for_viz.csv"
    cat_out.to_csv(cat_path, index=False)

    print(f"Wrote {feature_path} rows={len(feature_out)}")
    print(f"Wrote {cat_path} rows={len(cat_out)}")


if __name__ == "__main__":
    main()
