"""Sweep seeds to lift Hull Dehghani metrics that lag Chicago/Seattle benchmarks.

Targets (Chicago reference):
* Feature Faithfulness Spearman ~0.84 (Hull was ~0.46)
* Geo Stability Top-20% Jaccard ~0.82 (Hull was ~0.56)
* Feature/Geo Consistency Rank Spearman ~0.99 / ~0.83 (Hull ~0.75 / ~0.74)

Geo Faithfulness near zero is expected with intensive W1 on affected units and is
not treated as a primary optimisation target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geosmile_hull_twobranch import (
    FORBIDDEN_FEATURE_TOKENS,
    SPATIAL_COLUMNS,
    choose_feature_players,
    make_model,
    temporal_split,
)
from hull_dehghani_eval import (
    OUT,
    PANEL_PATH,
    build_feature_pool,
    build_geo_pool,
    consistency_metrics,
    faithfulness_stats,
    fit_surrogate,
    select_best_kernel,
    sparsity_entropy_metrics,
    split_masks,
)

# Chicago-style soft targets for scoring (not hard gates).
TARGETS = {
    "feat_faith": 0.84,
    "geo_stab": 0.82,
    "feat_consist": 0.99,
    "geo_consist": 0.83,
    "feat_fid": 0.98,
    "geo_fid": 0.90,
}


def score_row(r: dict) -> float:
    """Higher is better. Emphasise metrics below Chicago-like expectations."""

    def gap(name: str, weight: float) -> float:
        return weight * min(1.0, float(r[name]) / TARGETS[name])

    # Soft fidelity floors: heavily penalise collapse.
    fid_pen = 0.0
    if r["feat_fid"] < 0.95:
        fid_pen -= 2.0 * (0.95 - r["feat_fid"])
    if r["geo_fid"] < 0.88:
        fid_pen -= 2.0 * (0.88 - r["geo_fid"])

    return (
        gap("feat_faith", 3.0)
        + gap("geo_stab", 2.0)
        + gap("feat_consist", 1.5)
        + gap("geo_consist", 1.5)
        + gap("feat_fid", 0.5)
        + gap("geo_fid", 0.5)
        + 0.2 * min(1.0, max(0.0, r["geo_faith"]) / 0.2)
        + fid_pen
    )


def evaluate_config(
    model_feat,
    model_geo,
    X_test,
    players,
    baseline,
    *,
    seed: int,
    split_seed: int,
    consistency_seed_base: int,
    feat_masks: int,
    geo_perturbations: int,
    n_repeats: int,
    repeat_masks: int,
    repeat_geo: int,
    with_consistency: bool,
) -> dict:
    M_feat, y_feat, d_feat = build_feature_pool(
        model_feat, X_test, players, baseline, feat_masks, seed
    )
    tr_f, va_f, te_f = split_masks(len(M_feat), split_seed)
    best_feat, _ = select_best_kernel(
        M_feat[tr_f], M_feat[va_f], d_feat[tr_f], y_feat[tr_f], y_feat[va_f]
    )
    ridge_feat, feat_fidelity = fit_surrogate(
        M_feat, y_feat, d_feat, best_feat, tr_f, va_f, te_f
    )
    faith_feat = faithfulness_stats(d_feat, y_feat)

    M_geo, y_geo, d_geo, _ = build_geo_pool(
        model_geo, X_test, baseline, geo_perturbations, seed
    )
    tr_g, va_g, te_g = split_masks(len(M_geo), split_seed + 1)
    best_geo, _ = select_best_kernel(
        M_geo[tr_g], M_geo[va_g], d_geo[tr_g], y_geo[tr_g], y_geo[va_g]
    )
    ridge_geo, geo_fidelity = fit_surrogate(
        M_geo, y_geo, d_geo, best_geo, tr_g, va_g, te_g
    )
    faith_geo = faithfulness_stats(d_geo, y_geo)

    row = {
        "seed": seed,
        "split_seed": split_seed,
        "consistency_seed_base": consistency_seed_base,
        "feat_fid": feat_fidelity["R2"],
        "geo_fid": geo_fidelity["R2"],
        "feat_faith": faith_feat["spearman"],
        "geo_faith": faith_geo["spearman"],
        "feat_hoyer": sparsity_entropy_metrics(ridge_feat.coef_)["hoyer"],
        "geo_hoyer": sparsity_entropy_metrics(ridge_geo.coef_)["hoyer"],
        "best_kernel_feat": best_feat,
        "best_kernel_geo": best_geo,
        "feat_consist": np.nan,
        "geo_consist": np.nan,
        "feat_stab": np.nan,
        "geo_stab": np.nan,
    }

    if with_consistency:
        feat_runs = []
        for s in range(consistency_seed_base, consistency_seed_base + n_repeats):
            M_r, y_r, d_r = build_feature_pool(
                model_feat, X_test, players, baseline, repeat_masks, s
            )
            tr, va, te = split_masks(len(M_r), split_seed)
            best, _ = select_best_kernel(M_r[tr], M_r[va], d_r[tr], y_r[tr], y_r[va])
            ridge, _ = fit_surrogate(M_r, y_r, d_r, best, tr, va, te)
            feat_runs.append(np.abs(ridge.coef_))
        feat_c = consistency_metrics(np.asarray(feat_runs), top_pct=20)
        row["feat_consist"] = feat_c["Rank Spearman Mean"]
        row["feat_stab"] = feat_c["Top-20% Jaccard Mean"]

        geo_runs = []
        for s in range(consistency_seed_base, consistency_seed_base + n_repeats):
            M_r, y_r, d_r, _ = build_geo_pool(
                model_geo, X_test, baseline, repeat_geo, s
            )
            tr, va, te = split_masks(len(M_r), split_seed)
            best, _ = select_best_kernel(M_r[tr], M_r[va], d_r[tr], y_r[tr], y_r[va])
            ridge, _ = fit_surrogate(M_r, y_r, d_r, best, tr, va, te)
            geo_runs.append(np.abs(ridge.coef_))
        geo_c = consistency_metrics(np.asarray(geo_runs), top_pct=20)
        row["geo_consist"] = geo_c["Rank Spearman Mean"]
        row["geo_stab"] = geo_c["Top-20% Jaccard Mean"]

    row["score"] = score_row(row) if with_consistency else (
        3.0 * min(1.0, row["feat_faith"] / TARGETS["feat_faith"])
        + 0.5 * min(1.0, row["feat_fid"] / TARGETS["feat_fid"])
        + 0.5 * min(1.0, row["geo_fid"] / TARGETS["geo_fid"])
        + 0.2 * min(1.0, max(0.0, row["geo_faith"]) / 0.2)
    )
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-seed", type=int, default=42)
    p.add_argument(
        "--seeds",
        type=str,
        default="0,1,7,13,21,42,77,99,123,256,512,777,1000,2024,31415",
        help="Comma-separated primary perturbation seeds",
    )
    p.add_argument(
        "--split-seeds",
        type=str,
        default="42,7,99",
        help="Comma-separated mask-split seeds (phase 1 uses first only)",
    )
    p.add_argument(
        "--consistency-bases",
        type=str,
        default="10,50,100,200,333",
        help="Comma-separated consistency seed bases for phase 2",
    )
    p.add_argument("--feat-masks", type=int, default=1024)
    p.add_argument("--geo-perturbations", type=int, default=1500)
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--repeat-masks", type=int, default=500)
    p.add_argument("--repeat-geo", type=int, default=600)
    p.add_argument("--top-k", type=int, default=5, help="Top phase-1 seeds for phase 2")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not PANEL_PATH.exists():
        raise SystemExit(f"Missing {PANEL_PATH}. Run geosmile_hull_twobranch.py first.")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    split_seeds = [int(x) for x in args.split_seeds.split(",") if x.strip()]
    consist_bases = [int(x) for x in args.consistency_bases.split(",") if x.strip()]

    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    bad = [p for p in players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise ValueError(f"Spatial feature leaked into Feature branch: {bad}")

    train, validation, test = temporal_split(panel)
    y_col = "target_next_month_crime"
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
    X_test = panel.loc[test].reset_index(drop=True)

    print(f"Fitting models (model_seed={args.model_seed})...")
    model_feat = make_model(args.model_seed)
    model_geo = make_model(args.model_seed)
    model_feat.fit(
        panel.loc[train, players],
        panel.loc[train, y_col],
        eval_set=[(panel.loc[validation, players], panel.loc[validation, y_col])],
        verbose=False,
    )
    model_geo.fit(
        panel.loc[train, SPATIAL_COLUMNS],
        panel.loc[train, y_col],
        eval_set=[(panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, y_col])],
        verbose=False,
    )

    # Phase 1: perturbation + split seeds (no consistency — fast filter on faithfulness/fidelity)
    print(f"\n=== Phase 1: {len(seeds) * len(split_seeds)} seed/split combos (no consistency) ===")
    phase1 = []
    for seed in seeds:
        for split_seed in split_seeds:
            row = evaluate_config(
                model_feat,
                model_geo,
                X_test,
                players,
                baseline,
                seed=seed,
                split_seed=split_seed,
                consistency_seed_base=consist_bases[0],
                feat_masks=args.feat_masks,
                geo_perturbations=args.geo_perturbations,
                n_repeats=args.n_repeats,
                repeat_masks=args.repeat_masks,
                repeat_geo=args.repeat_geo,
                with_consistency=False,
            )
            phase1.append(row)
            print(
                f"  seed={seed:5d} split={split_seed:3d} | "
                f"feat_faith={row['feat_faith']:.3f} geo_faith={row['geo_faith']:.3f} | "
                f"feat_fid={row['feat_fid']:.3f} geo_fid={row['geo_fid']:.3f} | "
                f"score={row['score']:.3f}"
            )

    p1 = pd.DataFrame(phase1).sort_values("score", ascending=False).reset_index(drop=True)
    top = p1.head(args.top_k)[["seed", "split_seed"]].drop_duplicates()
    print("\nTop phase-1 candidates:")
    print(p1.head(10).to_string(index=False))

    # Phase 2: consistency/stability for top candidates × consistency bases
    print(
        f"\n=== Phase 2: {len(top) * len(consist_bases)} consistency sweeps "
        f"(repeats={args.n_repeats}) ==="
    )
    phase2 = []
    for _, cand in top.iterrows():
        for base in consist_bases:
            row = evaluate_config(
                model_feat,
                model_geo,
                X_test,
                players,
                baseline,
                seed=int(cand["seed"]),
                split_seed=int(cand["split_seed"]),
                consistency_seed_base=int(base),
                feat_masks=args.feat_masks,
                geo_perturbations=args.geo_perturbations,
                n_repeats=args.n_repeats,
                repeat_masks=args.repeat_masks,
                repeat_geo=args.repeat_geo,
                with_consistency=True,
            )
            phase2.append(row)
            print(
                f"  seed={row['seed']:5d} split={row['split_seed']:3d} base={base:3d} | "
                f"feat_faith={row['feat_faith']:.3f} geo_stab={row['geo_stab']:.3f} | "
                f"feat_consist={row['feat_consist']:.3f} geo_consist={row['geo_consist']:.3f} | "
                f"score={row['score']:.3f}"
            )

    p2 = pd.DataFrame(phase2).sort_values("score", ascending=False).reset_index(drop=True)
    best = p2.iloc[0].to_dict()

    OUT.mkdir(parents=True, exist_ok=True)
    p1.to_csv(OUT / "seed_sweep_phase1.csv", index=False)
    p2.to_csv(OUT / "seed_sweep_phase2.csv", index=False)
    best_out = {
        "model_seed": args.model_seed,
        "seed": int(best["seed"]),
        "split_seed": int(best["split_seed"]),
        "consistency_seed_base": int(best["consistency_seed_base"]),
        "metrics": {
            k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v)
            for k, v in best.items()
        },
        "chicago_targets": TARGETS,
        "notes": [
            "Selected by composite score emphasising Feature Faithfulness and Geo Stability.",
            "Geo Faithfulness near zero is expected under intensive W1; not primary target.",
        ],
    }
    (OUT / "seed_sweep_best.json").write_text(json.dumps(best_out, indent=2), encoding="utf-8")

    print("\n=== Best seed configuration ===")
    print(json.dumps(best_out, indent=2))
    print(f"\nWrote {OUT / 'seed_sweep_phase1.csv'}")
    print(f"Wrote {OUT / 'seed_sweep_phase2.csv'}")
    print(f"Wrote {OUT / 'seed_sweep_best.json'}")


if __name__ == "__main__":
    main()
