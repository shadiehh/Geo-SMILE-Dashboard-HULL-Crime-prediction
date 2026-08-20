"""
Build a consolidated Feature + Geo evaluation CSV for GeoSMILE Hull.

Merges:
  1) full_v3 long-format metrics (Feature + Geo branches)
  2) Feature expert top-10 alignment metrics
  3) Geo expert top-20 overlap / ranking metrics
  4) Two-branch summary predictive metrics (if present)

Outputs:
  outputs/geosmile_hull_feature_geo_evaluation.csv
  outputs/geosmile_hull_feature_geo_evaluation_key.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

SCHEMA = [
    "branch",
    "evaluation_set",
    "category",
    "metric",
    "value",
    "ci95_low",
    "ci95_high",
    "p_value",
    "n",
    "k",
    "source_file",
    "result_status",
    "notes",
]

FEATURE_BRANCH = "Feature (location excluded)"
GEO_BRANCH = "Geo"
EXPERT_CAT = "Expert-reference attribution accuracy"

KEY_V3_CATEGORIES = {
    "Fidelity",
    "Consistency",
    "Perturbation response monotonicity",
    "Sparsity",
    "Sparsity / concentration",
    "Controlled-input stability",
    "Model-based attribution faithfulness",
}

MISSING: list[str] = []


def _empty_row(**kwargs) -> dict:
    row = {c: pd.NA for c in SCHEMA}
    row.update(kwargs)
    return row


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _require(path: Path) -> Path | None:
    if path.exists():
        return path
    MISSING.append(_rel(path))
    return None


def _to_float(x):
    try:
        if pd.isna(x):
            return pd.NA
        return float(x)
    except (TypeError, ValueError):
        return pd.NA


def load_full_v3() -> pd.DataFrame:
    path = _require(OUT_DIR / "geosmile_hull_evaluation_full_v3.csv")
    if path is None:
        return pd.DataFrame(columns=SCHEMA)

    df = pd.read_csv(path)
    keep = df["branch"].isin([FEATURE_BRANCH, GEO_BRANCH])
    df = df.loc[keep].copy()
    # Feature expert-reference in full_v3 used fixed top-6; drop those so
    # Feature expert metrics come only from ranking_v2 top-10.
    drop_feat_expert = (df["branch"] == FEATURE_BRANCH) & (
        df["category"].astype(str) == EXPERT_CAT
    )
    df = df.loc[~drop_feat_expert].copy()
    df["evaluation_set"] = "full_v3"
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = pd.NA
    return df[SCHEMA]


def load_feature_top10_att() -> pd.DataFrame:
    """Compute Feature expert ATT metrics at k=10 from ranking v2 table."""
    feat_path = _require(OUT_DIR / "feature_category_alignment" / "feature_with_categories.csv")
    if feat_path is None:
        return pd.DataFrame(columns=SCHEMA)

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    merged = pd.read_csv(feat_path)
    need = {"feature", "expert_rank", "geosmile_rank", "abs_coefficient"}
    if not need.issubset(merged.columns):
        MISSING.append(f"{_rel(feat_path)} missing {sorted(need - set(merged.columns))}")
        return pd.DataFrame(columns=SCHEMA)

    k = 10
    g = set(merged.loc[merged["geosmile_rank"] <= k, "feature"].astype(str))
    e = set(merged.loc[merged["expert_rank"] <= k, "feature"].astype(str))
    both = sorted(g & e)
    y_true = (merged["expert_rank"] <= k).astype(int).to_numpy()
    y_pred = (merged["geosmile_rank"] <= k).astype(int).to_numpy()
    y_score = pd.to_numeric(merged["abs_coefficient"], errors="coerce").to_numpy()
    n = int(len(merged))
    src = _rel(feat_path)
    note = f"feature_ranking_v2 top-10; both={both}"

    metrics = {
        "Top-10 overlap": float(len(both)),
        "Top-10 overlap ratio": float(len(both)) / float(k),
        "ATT accuracy": float(accuracy_score(y_true, y_pred)),
        "ATT precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "ATT recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "ATT F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Jaccard@k": float(len(both) / len(g | e)) if (g | e) else pd.NA,
        "overlap coefficient": float(len(both) / min(len(g), len(e))) if min(len(g), len(e)) else pd.NA,
        "lift": float((len(both) / k) / (k / n)),
    }
    if y_true.min() != y_true.max():
        metrics["ATT AUROC"] = float(roc_auc_score(y_true, y_score))
        metrics["ATT AUPRC"] = float(average_precision_score(y_true, y_score))
        metrics["balanced accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["MCC"] = float(matthews_corrcoef(y_true, y_pred))

    rows = []
    for metric, value in metrics.items():
        rows.append(
            _empty_row(
                branch=FEATURE_BRANCH,
                evaluation_set="feature_ranking_v2",
                category=EXPERT_CAT,
                metric=f"feature: {metric}",
                value=_to_float(value),
                n=n,
                k=k,
                source_file=src,
                result_status="ok",
                notes=note,
            )
        )
    return pd.DataFrame(rows, columns=SCHEMA)


def load_feature_ranking_v2() -> pd.DataFrame:
    rows: list[dict] = []
    eval_path = _require(
        OUT_DIR / "feature_category_alignment" / "feature_expert_alignment_eval_v2.csv"
    )
    if eval_path is not None:
        df = pd.read_csv(eval_path)
        for _, r in df.iterrows():
            level = str(r.get("level", "")).strip()
            metric_raw = str(r.get("metric", "")).strip()
            metric = f"{level}: {metric_raw}" if level else metric_raw
            notes = r.get("notes", pd.NA)
            p_value = pd.NA
            # notes often encode p=...
            if isinstance(notes, str) and "p=" in notes.lower():
                try:
                    p_value = float(notes.lower().split("p=")[-1].split()[0].strip(",;"))
                except ValueError:
                    p_value = pd.NA
            rows.append(
                _empty_row(
                    branch=r.get("branch", FEATURE_BRANCH),
                    evaluation_set="feature_ranking_v2",
                    category=EXPERT_CAT,
                    metric=metric,
                    value=_to_float(r.get("value")),
                    p_value=p_value,
                    source_file=_rel(eval_path),
                    result_status="ok",
                    notes=notes,
                )
            )

    optional = [
        OUT_DIR / "feature_category_alignment" / "feature_top10_comparison_v2.csv",
        OUT_DIR
        / "feature_category_alignment"
        / "feature_category_importance_comparison.csv",
    ]
    for opt in optional:
        if not opt.exists():
            MISSING.append(_rel(opt))
            continue
        df = pd.read_csv(opt)
        # Best-effort long-format / wide conversion
        if {"metric", "value"}.issubset(df.columns):
            for _, r in df.iterrows():
                level = str(r.get("level", "feature")).strip() or "feature"
                metric_raw = str(r.get("metric", "")).strip()
                rows.append(
                    _empty_row(
                        branch=r.get("branch", FEATURE_BRANCH),
                        evaluation_set="feature_ranking_v2",
                        category=EXPERT_CAT,
                        metric=f"{level}: {metric_raw}",
                        value=_to_float(r.get("value")),
                        source_file=_rel(opt),
                        result_status="ok",
                        notes=r.get("notes", pd.NA),
                    )
                )
        else:
            # store numeric columns as metrics
            id_cols = {
                c
                for c in df.columns
                if c.lower() in {"feature", "category", "branch", "notes", "level"}
            }
            for _, r in df.iterrows():
                for col in df.columns:
                    if col in id_cols:
                        continue
                    val = _to_float(r[col])
                    if pd.isna(val):
                        continue
                    level = str(r.get("level", "feature")).strip() or "feature"
                    rows.append(
                        _empty_row(
                            branch=r.get("branch", FEATURE_BRANCH),
                            evaluation_set="feature_ranking_v2",
                            category=EXPERT_CAT,
                            metric=f"{level}: {col}",
                            value=val,
                            source_file=_rel(opt),
                            result_status="ok",
                            notes=r.get("notes", pd.NA),
                        )
                    )

    # Derive Top-10 overlap from ranked feature table if available and Top-10 not already present
    feat_path = OUT_DIR / "feature_category_alignment" / "feature_with_categories.csv"
    has_top10 = any(
        isinstance(r.get("metric"), str) and "Top-10" in r["metric"] for r in rows
    )
    if feat_path.exists() and not has_top10:
        fdf = pd.read_csv(feat_path)
        if {"expert_rank", "geosmile_rank"}.issubset(fdf.columns):
            k = 10
            expert_top = set(fdf.loc[fdf["expert_rank"] <= k, "feature"].astype(str))
            model_top = set(fdf.loc[fdf["geosmile_rank"] <= k, "feature"].astype(str))
            overlap = len(expert_top & model_top)
            both = sorted(expert_top & model_top)
            rows.append(
                _empty_row(
                    branch=FEATURE_BRANCH,
                    evaluation_set="feature_ranking_v2",
                    category=EXPERT_CAT,
                    metric="feature: Top-10 overlap",
                    value=float(overlap),
                    n=int(len(fdf)),
                    k=k,
                    source_file=_rel(feat_path),
                    result_status="derived",
                    notes=f"both={both}",
                )
            )
            rows.append(
                _empty_row(
                    branch=FEATURE_BRANCH,
                    evaluation_set="feature_ranking_v2",
                    category=EXPERT_CAT,
                    metric="feature: Top-10 overlap ratio",
                    value=float(overlap) / float(k),
                    n=int(len(fdf)),
                    k=k,
                    source_file=_rel(feat_path),
                    result_status="derived",
                    notes=f"overlap/{k}",
                )
            )
    elif not feat_path.exists():
        # only note missing if we needed it and Top-10 was absent from eval
        if not has_top10:
            MISSING.append(_rel(feat_path))

    out = pd.DataFrame(rows, columns=SCHEMA) if rows else pd.DataFrame(columns=SCHEMA)
    # Feature cut is top-10 only — drop any residual Top-6 rows
    if not out.empty:
        out = out[
            ~out["metric"].astype(str).str.contains(r"Top-6|top6|top-6", case=False, regex=True)
        ].copy()
    return out


def _topk_overlap_metrics(
    expert_rank: pd.Series,
    model_rank: pd.Series,
    ks: list[int],
    *,
    branch: str,
    evaluation_set: str,
    source_file: str,
    id_series: pd.Series | None = None,
) -> list[dict]:
    rows: list[dict] = []
    n = int(len(expert_rank))
    for k in ks:
        if id_series is not None:
            expert_set = set(id_series.loc[expert_rank <= k].astype(str))
            model_set = set(id_series.loc[model_rank <= k].astype(str))
            overlap = len(expert_set & model_set)
            note = f"both={sorted(expert_set & model_set)[:12]}"
            if overlap > 12:
                note += f" ... (+{overlap - 12} more)"
        else:
            overlap = int(((expert_rank <= k) & (model_rank <= k)).sum())
            note = "count of items with expert_rank<=k and model_rank<=k"
        rows.append(
            _empty_row(
                branch=branch,
                evaluation_set=evaluation_set,
                category=EXPERT_CAT,
                metric=f"Top-{k} overlap",
                value=float(overlap),
                n=n,
                k=k,
                source_file=source_file,
                result_status="ok",
                notes=note,
            )
        )
        rows.append(
            _empty_row(
                branch=branch,
                evaluation_set=evaluation_set,
                category=EXPERT_CAT,
                metric=f"Top-{k} overlap ratio",
                value=float(overlap) / float(k),
                n=n,
                k=k,
                source_file=source_file,
                result_status="ok",
                notes=f"overlap/{k}",
            )
        )
    return rows


def load_geo_stored_expert_rank() -> pd.DataFrame:
    path = _require(OUT_DIR / "geosmile_hull_expert_alignment_merged_geo.csv")
    if path is None:
        return pd.DataFrame(columns=SCHEMA)

    df = pd.read_csv(path)
    # Prefer existing geosmile_rank; otherwise rank by |importance|
    if "geosmile_rank" in df.columns:
        model_rank = pd.to_numeric(df["geosmile_rank"], errors="coerce")
        rank_note = "model_rank=geosmile_rank"
    elif "geosmile_abs_attribution" in df.columns:
        model_rank = (
            pd.to_numeric(df["geosmile_abs_attribution"], errors="coerce")
            .abs()
            .rank(ascending=False, method="average")
        )
        rank_note = "model_rank from |geosmile_abs_attribution|"
    elif "geo_coefficient" in df.columns:
        model_rank = (
            pd.to_numeric(df["geo_coefficient"], errors="coerce")
            .abs()
            .rank(ascending=False, method="average")
        )
        rank_note = "model_rank from |geo_coefficient|"
    else:
        raise ValueError(f"No model importance/rank columns in {path}")

    expert_rank = pd.to_numeric(df["expert_rank"], errors="coerce")
    valid = expert_rank.notna() & model_rank.notna()
    expert_rank = expert_rank.loc[valid]
    model_rank = model_rank.loc[valid]
    ids = df.loc[valid, "lsoa21cd"] if "lsoa21cd" in df.columns else None

    src = _rel(path)
    rows = _topk_overlap_metrics(
        expert_rank,
        model_rank,
        [10, 17, 20],
        branch=GEO_BRANCH,
        evaluation_set="geo_stored_expert_rank",
        source_file=src,
        id_series=ids,
    )

    rho, p = spearmanr(expert_rank, model_rank)
    rows.append(
        _empty_row(
            branch=GEO_BRANCH,
            evaluation_set="geo_stored_expert_rank",
            category=EXPERT_CAT,
            metric="Spearman rho (expert_rank vs model_rank)",
            value=float(rho),
            p_value=float(p),
            n=int(valid.sum()),
            source_file=src,
            result_status="ok",
            notes=rank_note,
        )
    )
    return pd.DataFrame(rows, columns=SCHEMA)


def _pick_geo_rank_col(df: pd.DataFrame) -> str:
    for cand in ("geo_rank", "model_rank", "geo_importance_rank", "geosmile_rank"):
        if cand in df.columns:
            return cand
    raise ValueError(f"No geo/model rank column found. Columns={list(df.columns)}")


def load_geo_expert_top20() -> pd.DataFrame:
    expert_path = None
    for name in ("expert_top20.csv", "expert_top20_custom_rewrite.csv"):
        cand = OUT_DIR / "maps_twobranch" / "ranking" / name
        if cand.exists():
            expert_path = cand
            break
    if expert_path is None:
        MISSING.append("outputs/maps_twobranch/ranking/expert_top20.csv")
        return pd.DataFrame(columns=SCHEMA)

    geo_path = _require(
        OUT_DIR / "maps_twobranch" / "ranking" / "geo_expert_importance_ranks.csv"
    )
    if geo_path is None:
        return pd.DataFrame(columns=SCHEMA)

    expert = pd.read_csv(expert_path)
    geo = pd.read_csv(geo_path)
    rank_col = _pick_geo_rank_col(geo)
    expert_rank_col = (
        "expert_rank"
        if "expert_rank" in expert.columns
        else ("custom_expert_rank" if "custom_expert_rank" in expert.columns else None)
    )

    expert_ids = set(expert["lsoa21cd"].astype(str))
    geo_top20 = set(
        geo.loc[pd.to_numeric(geo[rank_col], errors="coerce") <= 20, "lsoa21cd"].astype(
            str
        )
    )
    overlap_ids = sorted(expert_ids & geo_top20)
    overlap = len(overlap_ids)
    k = 20
    src = f"{_rel(expert_path)} vs {_rel(geo_path)}"
    eset = "geo_expert_top20"

    rows = [
        _empty_row(
            branch=GEO_BRANCH,
            evaluation_set=eset,
            category=EXPERT_CAT,
            metric="Top-20 overlap",
            value=float(overlap),
            n=int(len(geo)),
            k=k,
            source_file=src,
            result_status="ok",
            notes=f"rank_col={rank_col}; both={overlap_ids}",
        ),
        _empty_row(
            branch=GEO_BRANCH,
            evaluation_set=eset,
            category=EXPERT_CAT,
            metric="Top-20 overlap ratio",
            value=float(overlap) / float(k),
            n=int(len(geo)),
            k=k,
            source_file=src,
            result_status="ok",
            notes=f"overlap/{k}; rank_col={rank_col}",
        ),
    ]

    if expert_rank_col is not None:
        merged = expert[["lsoa21cd", expert_rank_col]].merge(
            geo[["lsoa21cd", rank_col]],
            on="lsoa21cd",
            how="inner",
        )
        if len(merged) >= 3:
            rho, p = spearmanr(
                pd.to_numeric(merged[expert_rank_col], errors="coerce"),
                pd.to_numeric(merged[rank_col], errors="coerce"),
            )
            rows.append(
                _empty_row(
                    branch=GEO_BRANCH,
                    evaluation_set=eset,
                    category=EXPERT_CAT,
                    metric="Spearman rho (expert_rank vs geo_rank on top-20)",
                    value=float(rho) if pd.notna(rho) else pd.NA,
                    p_value=float(p) if pd.notna(p) else pd.NA,
                    n=int(len(merged)),
                    k=k,
                    source_file=src,
                    result_status="ok",
                    notes=f"inner-join on expert top-20 LSOAs; rank_col={rank_col}",
                )
            )

    return pd.DataFrame(rows, columns=SCHEMA)


def load_two_branch_summary() -> pd.DataFrame:
    path = _require(OUT_DIR / "geosmile_hull_two_branch_summary.csv")
    if path is None:
        return pd.DataFrame(columns=SCHEMA)

    df = pd.read_csv(path)
    # Predictive metrics: fidelity / test-score style rows
    pred = df[
        df["category"].astype(str).str.lower().isin({"fidelity", "predictive", "predictive performance"})
        | df["metric"].astype(str).str.contains(r"Test\s|R2|RMSE|MAE|WMAE", case=False, regex=True)
    ].copy()
    # Prefer Fidelity-category rows if present to avoid duplicating faithfulness etc.
    if (df["category"].astype(str) == "Fidelity").any():
        pred = df[df["category"].astype(str) == "Fidelity"].copy()

    rows: list[dict] = []
    branch_map = {"Feature": FEATURE_BRANCH, "Geo": GEO_BRANCH}
    for _, r in pred.iterrows():
        branch = branch_map.get(str(r.get("branch", "")).strip(), r.get("branch"))
        if branch not in {FEATURE_BRANCH, GEO_BRANCH}:
            continue
        rows.append(
            _empty_row(
                branch=branch,
                evaluation_set="two_branch_summary",
                category="Predictive performance",
                metric=r.get("metric"),
                value=_to_float(r.get("value")),
                source_file=_rel(path),
                result_status="ok",
                notes=f"original_category={r.get('category')}",
            )
        )
    return pd.DataFrame(rows, columns=SCHEMA) if rows else pd.DataFrame(columns=SCHEMA)


def build_key_subset(full: pd.DataFrame) -> pd.DataFrame:
    rank_sets = {"feature_ranking_v2", "geo_expert_top20"}
    mask = full["evaluation_set"].isin(rank_sets) | (
        (full["evaluation_set"] == "full_v3")
        & (full["category"].isin(KEY_V3_CATEGORIES))
    )
    return full.loc[mask].copy()


def _metric_lookup(df: pd.DataFrame, evaluation_set: str, metric_substr: str) -> str:
    sub = df[
        (df["evaluation_set"] == evaluation_set)
        & (df["metric"].astype(str).str.contains(metric_substr, case=False, regex=False))
    ]
    if sub.empty:
        return "n/a"
    # Prefer exact-ish overlap count metrics over ratio when both match
    prefer = sub[~sub["metric"].astype(str).str.contains("ratio", case=False)]
    use = prefer if not prefer.empty else sub
    row = use.iloc[0]
    val = row["value"]
    return f"{row['metric']}={val}" if pd.notna(val) else "n/a"


def main() -> None:
    parts = [
        load_full_v3(),
        load_feature_ranking_v2(),
        load_feature_top10_att(),
        # Keep Geo expert top-20 as the Geo expert reference.
        load_geo_expert_top20(),
        load_two_branch_summary(),
    ]
    full = pd.concat([p for p in parts if len(p)], ignore_index=True)
    # Deduplicate Feature top-10 overlap if both ranking eval and ATT loader wrote it
    if not full.empty:
        full = full.sort_values(
            ["branch", "evaluation_set", "category", "metric", "result_status"],
            kind="mergesort",
        )
        full = full.drop_duplicates(
            subset=["branch", "evaluation_set", "category", "metric"],
            keep="last",
        ).reset_index(drop=True)
    full = full[SCHEMA]

    out_full = OUT_DIR / "geosmile_hull_feature_geo_evaluation.csv"
    out_key = OUT_DIR / "geosmile_hull_feature_geo_evaluation_key.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _write_csv(df: pd.DataFrame, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp, index=False)
        try:
            tmp.replace(path)
        except PermissionError:
            alt = path.with_name(path.stem + "_updated" + path.suffix)
            tmp.replace(alt)
            print(f"WARNING: locked {path.name}; wrote {alt.name} instead")

    _write_csv(full, out_full)

    key = build_key_subset(full)
    _write_csv(key, out_key)

    print("=== OUTPUTS ===")
    print(f"full: {_rel(out_full)}  rows={len(full)}")
    print(f"key:  {_rel(out_key)}  rows={len(key)}")
    print()
    print("=== BY evaluation_set ===")
    print(full.groupby(["evaluation_set", "branch"]).size().to_string())
    print()
    print("=== LATEST RANKING METRICS ===")
    fr = full[full["evaluation_set"] == "feature_ranking_v2"]
    for needle in ("Top-10 overlap", "ATT F1", "ATT AUROC", "Spearman rho (ranks)"):
        hit = fr[fr["metric"].astype(str).str.contains(needle, regex=False)]
        if "Spearman" in needle:
            feat = hit[hit["metric"].astype(str).str.startswith("feature:")]
            hit = feat if not feat.empty else hit
        if "Top-10" in needle:
            hit = hit[~hit["metric"].astype(str).str.contains("ratio", case=False)]
        if hit.empty:
            print(f"Feature {needle}: n/a")
        else:
            r = hit.iloc[0]
            print(f"Feature {r['metric']}: {r['value']}")

    geo_custom = full[full["evaluation_set"] == "geo_expert_top20"]
    hit = geo_custom[
        geo_custom["metric"].astype(str).str.contains("Top-20 overlap", regex=False)
        & ~geo_custom["metric"].astype(str).str.contains("ratio", case=False)
    ]
    if hit.empty:
        print("Geo expert Top-20 overlap: n/a")
    else:
        r = hit.iloc[0]
        print(f"Geo expert {r['metric']}: {r['value']}")

    print()
    print("=== MISSING / SKIPPED INPUTS ===")
    if MISSING:
        for m in MISSING:
            print(f"  - {m}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
