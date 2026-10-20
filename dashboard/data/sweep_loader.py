"""Load the robustness sweep for the dashboard diagnostics tab.

Feature branch: eval_by_n_perturb. Geo branch: the random-donor design sweep
(geo_random_donor_eval.py via build_random_donor_dashboard_files.py, outputs/geo_random_donor_eval/n_*/metrics.json),
ranked by the signed coefficient; its published N is GEO_DESIGN_N.

Expert-reference overlap is always computed here from the raw exported
rankings (feature_importance.csv / geo_coefficients.csv). This module never
opens the per-N accuracy summary file and never reads its intersection /
accuracy / precision / recall / f1 / overlap_ratio / jaccard_at_k fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dashboard.data.schema import (
    ATT_STABILITY_JSON,
    EVAL_SWEEP_DIR,
    EVAL_SWEEP_N_GRID,
    EVAL_V3_JSON,
    EXPERT_ALIGNMENT_FEATURE_CSV,
    EXPERT_ALIGNMENT_GEO_CSV,
    FEATURE_CONSISTENCY_CSV,
    FEATURE_GLOBAL_PUBLISHED_CSV,
    GEO_CONSISTENCY_CSV,
    GEO_CSV,
    GEO_DESIGN_DIR,
    GEO_DESIGN_N,
    PUBLISHED_FEATURE_FIDELITY_CSV,
    PUBLISHED_GEO_FIDELITY_CSV,
    PUBLISHED_N,
    TWOBRANCH_SUMMARY_JSON,
)

HULL_ROOT = Path(__file__).resolve().parents[2]

FEATURE_TOP_K = 10
GEO_TOP_K = 20
BRANCHES = ("feature", "geo")


def _branch_label(branch: str) -> str:
    return "Feature" if branch == "feature" else "Geo"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_one_row_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    return pd.read_csv(path).iloc[0].to_dict()


def rank_by_abs_magnitude(frame: pd.DataFrame, *, id_col: str, mag_col: str) -> pd.DataFrame:
    """Deterministic rank (1 = largest |coefficient|), ties broken by id ascending.

    Computed directly from the raw exported ranking CSV. Independent of any
    field in the per-N accuracy summary file.
    """
    out = frame.sort_values([mag_col, id_col], ascending=[False, True], kind="mergesort").reset_index(
        drop=True
    )
    out["rank"] = range(1, len(out) + 1)
    return out


def flag_expert_reference(frame: pd.DataFrame, *, id_col: str, expert_ids: set[str]) -> pd.DataFrame:
    out = frame.copy()
    out["in_expert_reference"] = out[id_col].astype(str).isin(expert_ids)
    return out


def expert_overlap_count(ranking: pd.DataFrame, *, k: int) -> int:
    """Count of the run's own top-k (by |coefficient|) that are in the expert reference set.

    This is the manual-verification method: rank the raw CSV by abs(coefficient),
    take the top k, and intersect with the expert reference set. Independent of
    the per-N accuracy summary file.
    """
    top = ranking[ranking["rank"] <= k]
    return int(top["in_expert_reference"].sum())


@dataclass
class SweepBundle:
    n_grid: list[int]
    metrics: pd.DataFrame
    rankings: dict[tuple[int, str], pd.DataFrame]
    published_rankings: dict[str, pd.DataFrame]
    lsoa_names: dict[str, str]
    expert_ids: dict[str, set[str]]
    published_n: dict[str, int]


def _load_expert_ids(root: Path) -> dict[str, set[str]]:
    feat_path = root / EXPERT_ALIGNMENT_FEATURE_CSV
    geo_path = root / EXPERT_ALIGNMENT_GEO_CSV
    if not feat_path.exists() or not geo_path.exists():
        raise FileNotFoundError(
            f"Missing expert-reference alignment tables: {feat_path} / {geo_path}"
        )
    feat = pd.read_csv(feat_path)
    geo = pd.read_csv(geo_path)
    return {
        "feature": set(feat.loc[feat["expert_top10_reference"] == 1, "feature"].astype(str)),
        "geo": set(geo.loc[geo["expert_top10_reference"] == 1, "lsoa21cd"].astype(str)),
    }


def _lsoa_names(root: Path) -> dict[str, str]:
    geo_path = root / EXPERT_ALIGNMENT_GEO_CSV
    if not geo_path.exists():
        return {}
    geo = pd.read_csv(geo_path)
    return dict(zip(geo["lsoa21cd"].astype(str), geo["lsoa21nm"].astype(str)))


def _ranking_for(path: Path, branch: str, expert_ids: set[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if branch == "feature":
        id_col, mag_col = "feature", "abs_coefficient"
    else:
        # Area-data design ranks by the signed coefficient (descending).
        id_col, mag_col = "lsoa21cd", "geo_coefficient"
    frame = pd.read_csv(path)
    frame[id_col] = frame[id_col].astype(str)
    ranked = rank_by_abs_magnitude(frame, id_col=id_col, mag_col=mag_col)
    return flag_expert_reference(ranked, id_col=id_col, expert_ids=expert_ids)


def _area_data_geo_row(n: int, metrics_path: Path, ranking: pd.DataFrame | None, *, published: bool) -> dict:
    m = _read_json(metrics_path)
    fid = m.get("fidelity") or {}
    faith = m.get("faithfulness") or {}
    stab = m.get("stability") or {}
    cons = m.get("consistency") or {}
    overlap = expert_overlap_count(ranking, k=GEO_TOP_K) if ranking is not None else None
    return {
        "n_perturb": n,
        "branch": _branch_label("geo"),
        "is_published": published,
        "surrogate_fidelity_r2": fid.get("R2"),
        "kernel_diagnostic_r2": (m.get("smile_diagnostics") or {}).get("test_r2"),
        "consistency_fidelity_r2_mean": None,
        "consistency_fidelity_r2_std": None,
        "faithfulness_top_minus_random": faith.get("top_minus_random"),
        "faithfulness_empirical_p": faith.get("empirical_p_top_vs_random"),
        "att_stability_jaccard": stab.get("jaccard_mean"),
        "att_consistency_rank_spearman_mean": cons.get("rank_spearman_mean"),
        "att_consistency_rank_spearman_std": None,
        "expert_overlap_count": overlap,
        "expert_overlap_k": GEO_TOP_K,
        "expert_overlap_fraction": (overlap / GEO_TOP_K) if overlap is not None else None,
    }


def _sweep_metric_row(n: int, branch: str, n_dir: Path, ranking: pd.DataFrame | None) -> dict:
    if branch == "geo":
        return _area_data_geo_row(n, n_dir / "metrics.json", ranking, published=False)
    fid = _read_json(n_dir / "fidelity.json").get(branch, {}) or {}
    faith = _read_json(n_dir / "faithfulness.json").get(branch, {}) or {}
    stab = _read_json(n_dir / "att_stability.json").get(branch, {}) or {}
    cons = _read_json(n_dir / "att_consistency.json").get(branch, {}) or {}
    k = FEATURE_TOP_K if branch == "feature" else GEO_TOP_K
    overlap = expert_overlap_count(ranking, k=k) if ranking is not None else None
    return {
        "n_perturb": n,
        "branch": _branch_label(branch),
        "is_published": False,
        "surrogate_fidelity_r2": fid.get("R2"),
        "kernel_diagnostic_r2": fid.get("smile_diagnostics_test_r2"),
        "consistency_fidelity_r2_mean": cons.get("Fidelity R2 Mean"),
        "consistency_fidelity_r2_std": cons.get("Fidelity R2 Std"),
        "faithfulness_top_minus_random": faith.get("top_minus_random"),
        "faithfulness_empirical_p": faith.get("empirical_p_top_vs_random"),
        "att_stability_jaccard": stab.get("top_k_jaccard_mean"),
        "att_consistency_rank_spearman_mean": cons.get("Rank Spearman Mean"),
        "att_consistency_rank_spearman_std": cons.get("Rank Spearman Std"),
        "expert_overlap_count": overlap,
        "expert_overlap_k": k,
        "expert_overlap_fraction": (overlap / k) if overlap is not None else None,
    }


def _published_metric_row(root: Path, branch: str, ranking: pd.DataFrame | None) -> dict:
    n = PUBLISHED_N[branch]
    if branch == "geo":
        return _area_data_geo_row(
            n, root / GEO_DESIGN_DIR / f"n_{GEO_DESIGN_N}" / "metrics.json", ranking, published=True
        )
    fid_rel = PUBLISHED_FEATURE_FIDELITY_CSV if branch == "feature" else PUBLISHED_GEO_FIDELITY_CSV
    fid = _read_one_row_csv(root / fid_rel)

    two = _read_json(root / TWOBRANCH_SUMMARY_JSON)
    surrogate_key = "feature_surrogate" if branch == "feature" else "geo_surrogate"
    kernel_diag_r2 = (two.get(surrogate_key) or {}).get("test_r2")

    v3 = _read_json(root / EVAL_V3_JSON)
    faith = (v3.get("model_based_attribution_faithfulness") or {}).get(branch, {}) or {}

    stab = _read_json(root / ATT_STABILITY_JSON).get(branch, {}) or {}

    cons_rel = FEATURE_CONSISTENCY_CSV if branch == "feature" else GEO_CONSISTENCY_CSV
    cons = _read_one_row_csv(root / cons_rel)

    k = FEATURE_TOP_K if branch == "feature" else GEO_TOP_K
    overlap = expert_overlap_count(ranking, k=k) if ranking is not None else None

    return {
        "n_perturb": n,
        "branch": _branch_label(branch),
        "is_published": True,
        "surrogate_fidelity_r2": fid.get("R2"),
        "kernel_diagnostic_r2": kernel_diag_r2,
        "consistency_fidelity_r2_mean": cons.get("Fidelity R2 Mean"),
        "consistency_fidelity_r2_std": cons.get("Fidelity R2 Std"),
        "faithfulness_top_minus_random": faith.get("top_minus_random"),
        "faithfulness_empirical_p": faith.get("empirical_p_top_vs_random"),
        "att_stability_jaccard": stab.get("top_k_jaccard_mean"),
        "att_consistency_rank_spearman_mean": cons.get("Rank Spearman Mean"),
        "att_consistency_rank_spearman_std": cons.get("Rank Spearman Std"),
        "expert_overlap_count": overlap,
        "expert_overlap_k": k,
        "expert_overlap_fraction": (overlap / k) if overlap is not None else None,
    }


def load_sweep_bundle(root: Path | None = None) -> SweepBundle:
    """Uncached loader. The Streamlit call site wraps this with st.cache_data."""
    root = Path(root) if root is not None else HULL_ROOT
    sweep_dir = root / EVAL_SWEEP_DIR
    expert_ids = _load_expert_ids(root)
    lsoa_names = _lsoa_names(root)

    rankings: dict[tuple[int, str], pd.DataFrame] = {}
    rows: list[dict] = []
    n_grid = sorted(EVAL_SWEEP_N_GRID)
    for n in n_grid:
        for branch in BRANCHES:
            n_dir = (sweep_dir if branch == "feature" else root / GEO_DESIGN_DIR) / f"n_{n}"
            fname = "feature_importance.csv" if branch == "feature" else "geo_coefficients.csv"
            ranking = _ranking_for(n_dir / fname, branch, expert_ids[branch])
            if ranking is not None:
                rankings[(n, branch)] = ranking
            rows.append(_sweep_metric_row(n, branch, n_dir, ranking))

    published_rankings: dict[str, pd.DataFrame] = {}
    for branch in BRANCHES:
        path = root / (FEATURE_GLOBAL_PUBLISHED_CSV if branch == "feature" else GEO_CSV)
        ranking = _ranking_for(path, branch, expert_ids[branch])
        if ranking is not None:
            published_rankings[branch] = ranking
        rows.append(_published_metric_row(root, branch, ranking))

    return SweepBundle(
        n_grid=n_grid,
        metrics=pd.DataFrame(rows),
        rankings=rankings,
        published_rankings=published_rankings,
        lsoa_names=lsoa_names,
        expert_ids=expert_ids,
        published_n=dict(PUBLISHED_N),
    )


def topk_ranking(bundle: SweepBundle, *, branch: str, n_perturb: int | None) -> pd.DataFrame:
    """Top-k ranked rows (k=10 Feature / k=20 Geo) for N, or the published ranking if N is None."""
    k = FEATURE_TOP_K if branch == "feature" else GEO_TOP_K
    frame = bundle.published_rankings.get(branch) if n_perturb is None else bundle.rankings.get(
        (n_perturb, branch)
    )
    if frame is None or frame.empty:
        return pd.DataFrame()
    top = frame[frame["rank"] <= k].copy()
    if branch == "geo":
        top["label"] = top["lsoa21cd"].map(bundle.lsoa_names).fillna(top["lsoa21cd"])
        top["signed_coefficient"] = top["geo_coefficient"]
        top["abs_coefficient"] = top["geo_abs_coefficient"]
    else:
        top["label"] = top["feature"]
        top["signed_coefficient"] = top["coefficient"]
    return top.sort_values("rank")
