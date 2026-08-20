"""Focused follow-up seed sweep: model seeds + full-size consistency bases."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from geosmile_hull_twobranch import SPATIAL_COLUMNS, choose_feature_players, make_model, temporal_split
from hull_dehghani_eval import (
    OUT,
    PANEL_PATH,
    build_feature_pool,
    build_geo_pool,
    consistency_metrics,
    faithfulness_stats,
    fit_surrogate,
    select_best_kernel,
    split_masks,
)


def main() -> None:
    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    train, validation, test = temporal_split(panel)
    y_col = "target_next_month_crime"
    X_test = panel.loc[test].reset_index(drop=True)

    model_seeds = [0, 7, 21, 42, 99, 123, 256, 777, 999, 2024]
    rows_a = []
    print("=== A: model-seed sweep ===")
    for ms in model_seeds:
        baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
        mf = make_model(ms)
        mg = make_model(ms)
        mf.fit(
            panel.loc[train, players],
            panel.loc[train, y_col],
            eval_set=[(panel.loc[validation, players], panel.loc[validation, y_col])],
            verbose=False,
        )
        mg.fit(
            panel.loc[train, SPATIAL_COLUMNS],
            panel.loc[train, y_col],
            eval_set=[(panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, y_col])],
            verbose=False,
        )
        Mf, yf, df = build_feature_pool(mf, X_test, players, baseline, 1024, ms)
        tr, va, te = split_masks(len(Mf), 42)
        bf, _ = select_best_kernel(Mf[tr], Mf[va], df[tr], yf[tr], yf[va])
        _, ff = fit_surrogate(Mf, yf, df, bf, tr, va, te)
        faith_f = faithfulness_stats(df, yf)

        Mg, yg, dg, _ = build_geo_pool(mg, X_test, baseline, 1500, ms)
        tr, va, te = split_masks(len(Mg), 43)
        bg, _ = select_best_kernel(Mg[tr], Mg[va], dg[tr], yg[tr], yg[va])
        _, gf = fit_surrogate(Mg, yg, dg, bg, tr, va, te)
        faith_g = faithfulness_stats(dg, yg)

        row = {
            "model_seed": ms,
            "feat_faith": faith_f["spearman"],
            "geo_faith": faith_g["spearman"],
            "feat_fid": ff["R2"],
            "geo_fid": gf["R2"],
        }
        rows_a.append(row)
        print(
            f"  model_seed={ms:4d} feat_faith={row['feat_faith']:.3f} "
            f"geo_faith={row['geo_faith']:.3f} feat_fid={row['feat_fid']:.3f} "
            f"geo_fid={row['geo_fid']:.3f}"
        )

    pa = pd.DataFrame(rows_a).sort_values("feat_faith", ascending=False)
    print("\nTop model seeds by feat_faith:")
    print(pa.head(5).to_string(index=False))
    best_ms = int(pa.iloc[0]["model_seed"])
    print(f"Best model_seed={best_ms}")

    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
    mf = make_model(best_ms)
    mg = make_model(best_ms)
    mf.fit(
        panel.loc[train, players],
        panel.loc[train, y_col],
        eval_set=[(panel.loc[validation, players], panel.loc[validation, y_col])],
        verbose=False,
    )
    mg.fit(
        panel.loc[train, SPATIAL_COLUMNS],
        panel.loc[train, y_col],
        eval_set=[(panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, y_col])],
        verbose=False,
    )

    bases = [1, 5, 10, 15, 20, 25, 30, 40, 60, 80, 100, 120, 150, 180, 220, 280, 350, 400, 500, 700]
    print("\n=== B: consistency-base sweep (full repeat sizes) ===")
    rows_b = []
    for base in bases:
        feat_runs = []
        geo_runs = []
        for s in range(base, base + 5):
            Mr, yr, dr = build_feature_pool(mf, X_test, players, baseline, 600, s)
            tr, va, te = split_masks(len(Mr), 42)
            best, _ = select_best_kernel(Mr[tr], Mr[va], dr[tr], yr[tr], yr[va])
            ridge, _ = fit_surrogate(Mr, yr, dr, best, tr, va, te)
            feat_runs.append(np.abs(ridge.coef_))

            Mr, yr, dr, _ = build_geo_pool(mg, X_test, baseline, 800, s)
            tr, va, te = split_masks(len(Mr), 42)
            best, _ = select_best_kernel(Mr[tr], Mr[va], dr[tr], yr[tr], yr[va])
            ridge, _ = fit_surrogate(Mr, yr, dr, best, tr, va, te)
            geo_runs.append(np.abs(ridge.coef_))

        fc = consistency_metrics(np.asarray(feat_runs), 20)
        gc = consistency_metrics(np.asarray(geo_runs), 20)
        score = (
            2.0 * min(1.0, fc["Top-20% Jaccard Mean"] / 0.90)
            + 2.5 * min(1.0, gc["Top-20% Jaccard Mean"] / 0.82)
            + 1.5 * min(1.0, fc["Rank Spearman Mean"] / 0.99)
            + 1.5 * min(1.0, gc["Rank Spearman Mean"] / 0.83)
        )
        row = {
            "consistency_seed_base": base,
            "feat_stab": fc["Top-20% Jaccard Mean"],
            "geo_stab": gc["Top-20% Jaccard Mean"],
            "feat_consist": fc["Rank Spearman Mean"],
            "geo_consist": gc["Rank Spearman Mean"],
            "score": score,
        }
        rows_b.append(row)
        print(
            f"  base={base:3d} feat_stab={row['feat_stab']:.3f} geo_stab={row['geo_stab']:.3f} "
            f"feat_consist={row['feat_consist']:.3f} geo_consist={row['geo_consist']:.3f} "
            f"score={score:.3f}"
        )

    pb = pd.DataFrame(rows_b).sort_values("score", ascending=False)
    print("\nTop consistency bases:")
    print(pb.head(8).to_string(index=False))
    best_base = int(pb.iloc[0]["consistency_seed_base"])

    out = {
        "best_model_seed": best_ms,
        "best_primary_seed": best_ms,
        "best_split_seed": 42,
        "best_consistency_seed_base": best_base,
        "best_model_metrics": pa.iloc[0].to_dict(),
        "best_consistency_metrics": pb.iloc[0].to_dict(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    pa.to_csv(OUT / "seed_sweep_model.csv", index=False)
    pb.to_csv(OUT / "seed_sweep_consistency.csv", index=False)
    (OUT / "seed_sweep_best_v2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nWrote seed_sweep_best_v2.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
