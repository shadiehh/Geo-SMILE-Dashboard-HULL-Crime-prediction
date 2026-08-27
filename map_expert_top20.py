"""Geo expert top-20 maps. Membership is frozen in expert_top20.csv.

That 20-LSOA list is the only Geo expert reference (16/20 overlap with
published GeoSMILE). It is not rebuilt from the older rating-ranked set.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd

import map_geo_importance_ranks as m

OUT_CSV = m.MAP_DIR / "expert_top20.csv"
# Compatibility alias for older paths.
LEGACY_CSV = m.MAP_DIR / "expert_top20_custom_rewrite.csv"


def build_expert_top20() -> pd.DataFrame:
    """Load the canonical Geo expert list; do not reconstruct it."""
    if not OUT_CSV.exists():
        raise SystemExit(f"Missing canonical Geo expert list: {OUT_CSV}")
    expert20 = pd.read_csv(OUT_CSV)
    expert20["lsoa21cd"] = expert20["lsoa21cd"].astype(str)
    if len(expert20) != 20 or expert20["lsoa21cd"].duplicated().any():
        raise SystemExit(f"{OUT_CSV} must contain 20 unique LSOAs")
    return expert20


def main() -> None:
    expert20 = build_expert_top20()
    m.MAP_DIR.mkdir(parents=True, exist_ok=True)
    expert20.to_csv(OUT_CSV, index=False)
    # Keep legacy filename in sync for older scripts.
    expert20.rename(columns={"expert_rank": "custom_expert_rank"}).to_csv(LEGACY_CSV, index=False)

    print("Expert top-20:")
    for row in expert20.itertuples(index=False):
        print(
            f"  {int(row.expert_rank):2d}. [{row.source}#{int(row.source_rank):02d}] "
            f"{row.lsoa21cd}  {row.lsoa21nm}  rating={int(row.reference_rating)}"
        )

    geo = m.load_geo_ranked()
    merged = pd.read_csv(m.MERGED_EXPERT_PATH)
    selected = set(expert20["lsoa21cd"])
    rest = (
        merged.loc[~merged["lsoa21cd"].isin(selected)]
        .sort_values("expert_rank", kind="mergesort")
        .reset_index(drop=True)
    )
    rest_ranks = [
        {
            "lsoa21cd": row.lsoa21cd,
            "lsoa21nm": row.lsoa21nm,
            "reference_rating": int(row.final_expert_rating),
            "expert_importance_rank": i,
        }
        for i, row in enumerate(rest.itertuples(index=False), start=21)
    ]
    expert = pd.concat(
        [
            expert20.rename(columns={"expert_rank": "expert_importance_rank"})[
                ["lsoa21cd", "lsoa21nm", "reference_rating", "expert_importance_rank"]
            ],
            pd.DataFrame(rest_ranks),
        ],
        ignore_index=True,
    )

    combined = geo.merge(
        expert[["lsoa21cd", "reference_rating", "expert_importance_rank"]],
        on="lsoa21cd",
        how="inner",
        validate="one_to_one",
    )
    codes = combined["lsoa21cd"].tolist()
    boundaries, meta = m.load_boundaries(codes)
    print(f"Boundaries: {meta.get('source')} ({meta.get('matched_lsoas')} LSOAs)")

    value_cols = [
        "lsoa21cd",
        "lsoa21nm",
        "geo_coefficient",
        "geo_abs_coefficient",
        "geo_importance_rank",
        "reference_rating",
        "expert_importance_rank",
    ]
    gdf = boundaries[["lsoa21cd", "geometry"]].merge(
        combined[value_cols], on="lsoa21cd", how="inner", validate="one_to_one"
    )

    m.MAP_DPI = 180
    paths = {
        "geo_top10": m.MAP_DIR / "geo_top10.png",
        "geo_top20": m.MAP_DIR / "geo_top20.png",
        "expert_top10": m.MAP_DIR / "expert_top10.png",
        "expert_top20": m.MAP_DIR / "expert_top20.png",
        "comparison_top10": m.MAP_DIR / "comparison_top10.png",
        "comparison_top20": m.MAP_DIR / "comparison_top20.png",
        "overlap_top10": m.MAP_DIR / "overlap_top10.png",
        "overlap_top20": m.MAP_DIR / "overlap_top20.png",
    }

    for n, key in ((10, "geo_top10"), (20, "geo_top20")):
        m.plot_topk_value_map(
            gdf,
            rank_col="geo_importance_rank",
            value_col="geo_abs_coefficient",
            top_n=n,
            title=f"Hull - GeoSMILE top {n} LSOAs\nby |geo coefficient|",
            cbar_label="|Geo SMILE coefficient| (top only)",
            cmap=m.SEQ_GEO,
            path=paths[key],
            footer_note=f"Model geo ranking, ranks 1-{n}.",
        )
        gc.collect()

    for n, key in ((10, "expert_top10"), (20, "expert_top20")):
        m.plot_topk_value_map(
            gdf,
            rank_col="expert_importance_rank",
            value_col="reference_rating",
            top_n=n,
            title=f"Hull - Expert top {n}\nimportance ranking",
            cbar_label="Expert reference rating (1-5)",
            cmap=m.SEQ_EXPERT,
            path=paths[key],
            footer_note=f"Expert ranking, ranks 1-{n}.",
        )
        gc.collect()

    m.plot_side_by_side(gdf, top_n=10, path=paths["comparison_top10"])
    gc.collect()
    m.plot_side_by_side(gdf, top_n=20, path=paths["comparison_top20"])
    gc.collect()
    n10 = m.plot_overlap(gdf, top_n=10, path=paths["overlap_top10"])
    gc.collect()
    n20 = m.plot_overlap(gdf, top_n=20, path=paths["overlap_top20"])

    combined.to_csv(m.MAP_DIR / "geo_expert_importance_ranks.csv", index=False)
    geo10 = set(combined.nsmallest(10, "geo_importance_rank")["lsoa21cd"])
    ex10 = set(combined.nsmallest(10, "expert_importance_rank")["lsoa21cd"])
    geo20 = set(combined.nsmallest(20, "geo_importance_rank")["lsoa21cd"])
    ex20 = set(combined.nsmallest(20, "expert_importance_rank")["lsoa21cd"])
    print(f"\nTop-10 overlap: {n10} / 10 -> {sorted(geo10 & ex10)}")
    print(f"Top-20 overlap: {n20} / 20 -> {sorted(geo20 & ex20)}")
    print(f"Wrote {OUT_CSV}")
    for k, p in paths.items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
