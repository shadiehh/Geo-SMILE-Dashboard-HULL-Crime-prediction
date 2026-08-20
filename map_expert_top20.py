"""Build Geo expert top-20 ranking maps and comparison figures.

Expert top-20 membership and importance order are defined here.
Outputs under outputs/maps_twobranch/ranking/:
  expert_top20.csv, geo/expert/comparison/overlap maps for k=10 and k=20.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd

import map_geo_importance_ranks as m

# Expert top-20 construction (final reference set).
BASE_EXPERT_RANKS = [1, 2, 18, 12, 14, 3, 19, 20, 16, 9, 6, 7, 10]
BASE_GEO_RANKS_BEFORE_GAP = [8, 12, 15, 13]
BASE_GAP_FILL_LSOAS = ["E01012899"]
BASE_GEO_RANKS_AFTER_GAP = [16, 17]

# Final importance order: permutation of base labels 1..20.
EXPERT_IMPORTANCE_ORDER = [
    2, 3, 4, 5, 7, 8, 20, 15, 16, 17, 18, 11, 12, 14, 6, 1, 9, 10, 13, 19,
]

OUT_CSV = m.MAP_DIR / "expert_top20.csv"
# Compatibility alias for older paths.
LEGACY_CSV = m.MAP_DIR / "expert_top20_custom_rewrite.csv"


def build_base_top20() -> pd.DataFrame:
    merged = pd.read_csv(m.MERGED_EXPERT_PATH)
    coef = pd.read_csv(m.COEF_PATH)
    coef = coef.sort_values(
        ["geo_abs_coefficient", "lsoa21cd"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    coef["geo_importance_rank"] = range(1, len(coef) + 1)

    rows: list[dict] = []

    def append_expert(base_label: int, er: int) -> None:
        row = merged.loc[merged["expert_rank"] == er].iloc[0]
        rows.append(
            {
                "base_label": base_label,
                "source": "expert",
                "source_rank": int(er),
                "lsoa21cd": row["lsoa21cd"],
                "lsoa21nm": row["lsoa21nm"],
                "reference_rating": int(row["final_expert_rating"]),
                "expert_relevance_score": float(row.get("expert_relevance_score", float("nan"))),
            }
        )

    def append_geo(base_label: int, gr: int) -> None:
        row = coef.loc[coef["geo_importance_rank"] == gr].iloc[0]
        mrow = merged.loc[merged["lsoa21cd"] == row["lsoa21cd"]].iloc[0]
        rows.append(
            {
                "base_label": base_label,
                "source": "geo",
                "source_rank": int(gr),
                "lsoa21cd": row["lsoa21cd"],
                "lsoa21nm": row["lsoa21nm"],
                "reference_rating": int(mrow["final_expert_rating"]),
                "expert_relevance_score": float(mrow.get("expert_relevance_score", float("nan"))),
            }
        )

    def append_gap(base_label: int, code: str) -> None:
        mrow = merged.loc[merged["lsoa21cd"] == code].iloc[0]
        rows.append(
            {
                "base_label": base_label,
                "source": "gap_fill",
                "source_rank": 0,
                "lsoa21cd": mrow["lsoa21cd"],
                "lsoa21nm": mrow["lsoa21nm"],
                "reference_rating": int(mrow["final_expert_rating"]),
                "expert_relevance_score": float(mrow.get("expert_relevance_score", float("nan"))),
            }
        )

    for base_label, er in enumerate(BASE_EXPERT_RANKS, start=1):
        append_expert(base_label, er)

    rank = len(BASE_EXPERT_RANKS)
    for gr in BASE_GEO_RANKS_BEFORE_GAP:
        rank += 1
        append_geo(rank, gr)
    for code in BASE_GAP_FILL_LSOAS:
        rank += 1
        append_gap(rank, code)
    for gr in BASE_GEO_RANKS_AFTER_GAP:
        rank += 1
        append_geo(rank, gr)

    base = pd.DataFrame(rows)
    if base["lsoa21cd"].duplicated().any():
        raise SystemExit("Duplicate LSOAs in expert top-20")
    if len(base) != 20:
        raise SystemExit(f"Expected 20 LSOAs, got {len(base)}")
    if sorted(base["base_label"].tolist()) != list(range(1, 21)):
        raise SystemExit("base_label must be 1..20")
    return base


def apply_importance_order(base: pd.DataFrame) -> pd.DataFrame:
    order = list(EXPERT_IMPORTANCE_ORDER)
    if sorted(order) != list(range(1, 21)):
        raise SystemExit(f"EXPERT_IMPORTANCE_ORDER must be a permutation of 1..20; got {order}")
    by_base = base.set_index("base_label")
    rows = []
    for new_rank, base_label in enumerate(order, start=1):
        row = by_base.loc[base_label]
        rows.append(
            {
                "expert_rank": new_rank,
                "base_label": int(base_label),
                "source": row["source"],
                "source_rank": int(row["source_rank"]),
                "lsoa21cd": row["lsoa21cd"],
                "lsoa21nm": row["lsoa21nm"],
                "reference_rating": int(row["reference_rating"]),
                "expert_relevance_score": float(row["expert_relevance_score"]),
            }
        )
    out = pd.DataFrame(rows)
    if out["lsoa21cd"].duplicated().any() or len(out) != 20:
        raise SystemExit("Invalid expert top-20 after reordering")
    return out


def build_expert_top20() -> pd.DataFrame:
    return apply_importance_order(build_base_top20())


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
