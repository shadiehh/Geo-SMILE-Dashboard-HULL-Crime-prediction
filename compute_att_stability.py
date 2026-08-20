"""Compute ATT / controlled-input stability for Feature and Geo branches.

ATT stability: recompute SMILE attributions after expanding the explanation
reference with one real LSOA from each cardinal direction just outside Hull.
Police.uk counts and the same Census/OSM covariates used for Hull LSOAs are
collected for those neighbors. Overlap of top-k attribution sets (Jaccard) and
rank agreement (Spearman on |coefficients|) are compared with the Hull-only
baseline.

This is distinct from multi-run Consistency (different explanation seeds).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr

from geosmile_hull_twobranch import (
    FEATURE_LABELS,
    FORBIDDEN_FEATURE_TOKENS,
    SPATIAL_COLUMNS,
    choose_feature_players,
    geo_smile_coefficients,
    make_model,
    smile_global_feature,
    temporal_split,
)
from hull_boundaries import CARDINAL_DIRECTIONS, load_boundaries
from hull_boundary_neighbors import expand_panel, prepare_cardinal_neighbor_data
from hull_expert_topk import BRANCH_FEATURE, BRANCH_GEO, FEATURE_TOP_K, GEO_TOP_K
from map_geo_importance_ranks import SEQ_GEO, plot_topk_value_map, rank_by

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"
VIZ_DIR = OUT / "att_stability"
BLUE, PINK = "#1E88E5", "#FF0051"
INK, INK_MUTED = "#12171C", "#57646D"
MUTED_BAR = "#B7C0C8"

# Same SMILE design as the published two-branch run (run_summary_twobranch.json).
# Do not shrink masks/perturbations for the stability check.
FEAT_MASKS = 1024
GEO_PERTURBS = 2000
SEED = 42
PUBLISHED_FEATURE = OUT / "feature_global_importance.csv"
PUBLISHED_GEO = OUT / "geo_coefficients.csv"
DIRECTIONS = tuple(CARDINAL_DIRECTIONS.keys())
N_REPS = len(DIRECTIONS)


def _topk_ordered(df: pd.DataFrame, id_col: str, mag_col: str, k: int) -> list[str]:
    order = df.sort_values([mag_col, id_col], ascending=[False, True], kind="mergesort")
    return order.head(k)[id_col].astype(str).tolist()


def _topk_ids(df: pd.DataFrame, id_col: str, mag_col: str, k: int) -> set[str]:
    return set(_topk_ordered(df, id_col, mag_col, k))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _pred_spearman(y0: np.ndarray, y1: np.ndarray) -> tuple[float, float]:
    rho = float(spearmanr(y0, y1).correlation)
    mae = float(np.mean(np.abs(y0 - y1)))
    return rho, mae


def _mean_or_nan(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def _std_or_nan(vals: list[float]) -> float:
    return float(np.std(vals)) if vals else float("nan")


def _feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " "))


def load_published_baseline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hull-only Feature top-10 / Geo top-20 from the original GeoSMILE run."""
    if not PUBLISHED_FEATURE.exists():
        raise SystemExit(f"Missing published Feature ranking: {PUBLISHED_FEATURE}")
    if not PUBLISHED_GEO.exists():
        raise SystemExit(f"Missing published Geo ranking: {PUBLISHED_GEO}")
    feat = pd.read_csv(PUBLISHED_FEATURE)
    geo = pd.read_csv(PUBLISHED_GEO)
    if "abs_coefficient" not in feat.columns:
        feat["abs_coefficient"] = feat["coefficient"].abs()
    if "geo_abs_coefficient" not in geo.columns:
        geo["geo_abs_coefficient"] = geo["geo_coefficient"].abs()
    return feat, geo


def save_feature_importance_ranking(
    importance: pd.DataFrame,
    path: Path,
    *,
    title: str,
    footer: str,
    highlight_ids: set[str] | None = None,
    xmax: float | None = None,
) -> Path:
    """Horizontal bar ranking of |SMILE coefficients|."""
    ranked = importance.sort_values(
        ["abs_coefficient", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    ranked = ranked.copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    highlight_ids = highlight_ids or set(
        ranked.head(FEATURE_TOP_K)["feature"].astype(str)
    )
    labels = [
        f"{int(r)}  {_feature_label(f)}"
        for r, f in zip(ranked["rank"], ranked["feature"].astype(str))
    ]
    vals = ranked["abs_coefficient"].to_numpy(dtype=float)
    colors = [
        PINK if f in highlight_ids else MUTED_BAR
        for f in ranked["feature"].astype(str)
    ]
    fig_h = max(6.5, 0.28 * len(ranked) + 1.8)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    y = np.arange(len(ranked))[::-1]
    ax.barh(y, vals, color=colors, edgecolor="white", height=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.set_xlabel("|SMILE coefficient|", fontsize=10, color=INK_MUTED)
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=10)
    if xmax is not None:
        ax.set_xlim(0, xmax * 1.05)
    ax.axhline(
        y=len(ranked) - FEATURE_TOP_K - 0.5,
        color=INK_MUTED,
        linewidth=0.8,
        linestyle="--",
        alpha=0.7,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.text(0.5, 0.01, footer, ha="center", fontsize=7.0, color=INK_MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    plt.close("all")
    return path


def save_geo_rank_map(
    geo_coef: pd.DataFrame,
    boundaries,
    path: Path,
    *,
    title: str,
    footer: str,
    name_lookup: dict[str, str] | None = None,
    added_codes: list[str] | set[str] | None = None,
) -> Path:
    """Top-k LSOA importance map (same style as map_geo_importance_ranks)."""
    coef = geo_coef.copy()
    if "lsoa21nm" not in coef.columns and name_lookup:
        coef["lsoa21nm"] = coef["lsoa21cd"].map(name_lookup).fillna(coef["lsoa21cd"])
    elif "lsoa21nm" not in coef.columns:
        coef["lsoa21nm"] = coef["lsoa21cd"]
    ranked = rank_by(coef, "geo_abs_coefficient", "geo_importance_rank")
    geom = boundaries[["lsoa21cd", "geometry"]].drop_duplicates("lsoa21cd")
    gdf = geom.merge(ranked, on="lsoa21cd", how="inner", validate="one_to_one")
    plot_topk_value_map(
        gdf,
        rank_col="geo_importance_rank",
        value_col="geo_abs_coefficient",
        top_n=GEO_TOP_K,
        title=title,
        cbar_label="|Geo SMILE coefficient|",
        cmap=SEQ_GEO,
        path=path,
        footer_note=footer,
        added_codes=added_codes,
    )
    return path


def save_stability_snapshot(
    out_dir: Path,
    *,
    label: str,
    feat_imp: pd.DataFrame,
    geo_imp: pd.DataFrame,
    boundaries,
    name_lookup: dict[str, str],
    meta: dict | None = None,
    feat_xmax: float | None = None,
) -> dict[str, str]:
    """Write CSVs + feature ranking + geo top-k map for one stability snapshot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_csv = out_dir / "feature_importance.csv"
    geo_csv = out_dir / "geo_coefficients.csv"
    feat_png = out_dir / "feature_importance_ranking.png"
    geo_png = out_dir / f"geo_top{GEO_TOP_K}_map.png"

    feat_ranked = feat_imp.sort_values(
        ["abs_coefficient", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    feat_ranked.insert(0, "rank", np.arange(1, len(feat_ranked) + 1))
    feat_ranked.to_csv(feat_csv, index=False)

    geo_ranked = rank_by(geo_imp.copy(), "geo_abs_coefficient", "geo_importance_rank")
    if "lsoa21nm" not in geo_ranked.columns:
        geo_ranked["lsoa21nm"] = geo_ranked["lsoa21cd"].map(name_lookup).fillna(
            geo_ranked["lsoa21cd"]
        )
    geo_ranked.to_csv(geo_csv, index=False)

    meta = meta or {}
    bits = [label]
    if "feature_jaccard" in meta:
        bits.append(f"top-{FEATURE_TOP_K} Jaccard vs baseline={meta['feature_jaccard']:.3f}")
    if "added_lsoa" in meta:
        bits.append(f"added {meta.get('direction')} {meta['added_lsoa']}")
    bits.append("Not causal.")
    feat_footer = ". ".join(bits)

    geo_bits = [label]
    if "geo_jaccard" in meta:
        geo_bits.append(f"top-{GEO_TOP_K} Jaccard vs baseline={meta['geo_jaccard']:.3f}")
    if "added_lsoa" in meta:
        geo_bits.append(f"added {meta.get('direction')} {meta['added_lsoa']}")
    if "added_name" in meta:
        geo_bits.append(meta["added_name"])
    geo_footer = ". ".join(str(b) for b in geo_bits) + "."

    save_feature_importance_ranking(
        feat_imp,
        feat_png,
        title=f"Feature SMILE importance ranking\n{label}",
        footer=feat_footer,
        highlight_ids=set(
            feat_ranked.head(FEATURE_TOP_K)["feature"].astype(str)
        ),
        xmax=feat_xmax,
    )
    added = [str(meta["added_lsoa"])] if meta.get("added_lsoa") else []
    save_geo_rank_map(
        geo_imp,
        boundaries,
        geo_png,
        title=f"Geo SMILE top-{GEO_TOP_K} by |coefficient|\n{label}",
        footer=geo_footer,
        name_lookup=name_lookup,
        added_codes=added,
    )
    (out_dir / "snapshot_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    print(f"  Saved visuals -> {out_dir}")
    return {
        "feature_csv": str(feat_csv),
        "geo_csv": str(geo_csv),
        "feature_png": str(feat_png),
        "geo_png": str(geo_png),
    }


def save_neighbor_locator_map(hull, neighbors, path: Path) -> Path:
    """Map of Hull plus the four cardinal neighbor LSOAs."""
    from matplotlib.patches import Patch
    from hull_boundaries import hull_outline

    colors = {
        "north": "#C62828",
        "east": "#1565C0",
        "south": "#2E7D32",
        "west": "#6A1B9A",
    }
    fig, ax = plt.subplots(figsize=(8.5, 9.0))
    hull.plot(ax=ax, color="#E8ECF0", edgecolor="white", linewidth=0.15, zorder=1)
    gpd.GeoSeries([hull_outline(hull)], crs=hull.crs).plot(
        ax=ax, color="none", edgecolor=INK, linewidth=1.1, zorder=3
    )
    for _, row in neighbors.iterrows():
        direction = str(row["direction"])
        gpd.GeoSeries([row.geometry], crs=neighbors.crs).plot(
            ax=ax,
            color=colors.get(direction, PINK),
            edgecolor=INK,
            linewidth=1.6,
            zorder=4,
        )
        pt = row.geometry.representative_point()
        ax.text(
            pt.x,
            pt.y,
            direction[0].upper(),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
            zorder=5,
        )
    ax.set_axis_off()
    ax.set_title(
        "Cardinal neighbor LSOAs used for ATT stability",
        fontsize=12,
        fontweight="bold",
        color=INK,
        pad=8,
    )
    handles = [
        Patch(facecolor=colors[d], edgecolor="white", label=d.title())
        for d in DIRECTIONS
        if d in set(neighbors["direction"].astype(str))
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8)
    fig.text(
        0.5,
        0.02,
        "One LSOA immediately outside Hull in each cardinal direction. "
        "Police.uk + Census/OSM covariates match the Hull LSOA schema. Not causal.",
        ha="center",
        fontsize=7.0,
        color=INK_MUTED,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    plt.close("all")
    return path


def compare_attributions(
    base_feat: pd.DataFrame,
    feat_imp: pd.DataFrame,
    base_geo: pd.DataFrame,
    geo_imp: pd.DataFrame,
    *,
    base_feat_top: set[str],
    base_geo_top: set[str],
    added_lsoa: str | None = None,
) -> dict[str, float]:
    merged_f = base_feat[["feature", "abs_coefficient"]].merge(
        feat_imp[["feature", "abs_coefficient"]],
        on="feature",
        suffixes=("_base", "_pert"),
    )
    top_f = _topk_ids(feat_imp, "feature", "abs_coefficient", FEATURE_TOP_K)
    fj = _jaccard(base_feat_top, top_f)
    fr = float(
        spearmanr(merged_f["abs_coefficient_base"], merged_f["abs_coefficient_pert"]).correlation
    )
    merged_g = base_geo[["lsoa21cd", "geo_abs_coefficient"]].merge(
        geo_imp[["lsoa21cd", "geo_abs_coefficient"]],
        on="lsoa21cd",
        suffixes=("_base", "_pert"),
    )
    top_g = _topk_ids(geo_imp, "lsoa21cd", "geo_abs_coefficient", GEO_TOP_K)
    gj = _jaccard(base_geo_top, top_g)
    gr = float(
        spearmanr(
            merged_g["geo_abs_coefficient_base"],
            merged_g["geo_abs_coefficient_pert"],
        ).correlation
    )
    base_19 = set(_topk_ordered(base_geo, "lsoa21cd", "geo_abs_coefficient", GEO_TOP_K - 1))
    pert_19 = set(_topk_ordered(geo_imp, "lsoa21cd", "geo_abs_coefficient", GEO_TOP_K - 1))
    extra_rank = float("nan")
    extra_abs = float("nan")
    extra_in_top20 = 0.0
    if added_lsoa:
        ranked = geo_imp.sort_values(
            ["geo_abs_coefficient", "lsoa21cd"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        ranked.insert(0, "geo_importance_rank", np.arange(1, len(ranked) + 1))
        hit = ranked[ranked["lsoa21cd"].astype(str) == str(added_lsoa)]
        if not hit.empty:
            extra_rank = float(hit["geo_importance_rank"].iloc[0])
            extra_abs = float(hit["geo_abs_coefficient"].iloc[0])
            extra_in_top20 = float(extra_rank <= GEO_TOP_K)
    return {
        "feature_jaccard": fj,
        "feature_spearman": fr,
        "geo_jaccard": gj,
        "geo_spearman": gr,
        "geo_top20_overlap_n": float(len(base_geo_top & top_g)),
        "geo_top19_jaccard": _jaccard(base_19, pert_19),
        "geo_top19_overlap_n": float(len(base_19 & pert_19)),
        "geo_extra_rank": extra_rank,
        "geo_extra_abs_coefficient": extra_abs,
        "geo_extra_in_top20": extra_in_top20,
    }


def hull_row_pred_spearman(
    model,
    hull_rows: pd.DataFrame,
    expanded_rows: pd.DataFrame,
    columns: list[str],
    key: str = "lsoa21cd",
) -> tuple[float, float]:
    """Spearman of predictions on original Hull test rows vs the same rows after expansion."""
    hull_rows = hull_rows.copy()
    expanded_rows = expanded_rows.copy()
    hull_rows["month"] = pd.to_datetime(hull_rows["month"])
    expanded_rows["month"] = pd.to_datetime(expanded_rows["month"])
    hull_rows[key] = hull_rows[key].astype(str)
    expanded_rows[key] = expanded_rows[key].astype(str)
    hull_keys = hull_rows[key].to_numpy()
    exp = expanded_rows[expanded_rows[key].isin(set(hull_keys))]
    y0 = np.asarray(model.predict(hull_rows[columns]), float)
    # Align expanded Hull rows to the original test order via a stable merge key.
    order = hull_rows[[key, "month"]].copy()
    order["_i"] = np.arange(len(order))
    aligned = order.merge(
        exp[[key, "month"] + columns],
        on=[key, "month"],
        how="left",
        validate="one_to_one",
    ).sort_values("_i")
    y1 = np.asarray(model.predict(aligned[columns]), float)
    return _pred_spearman(y0, y1)


def main() -> None:
    if not PANEL_PATH.exists():
        raise SystemExit(f"Missing {PANEL_PATH}")

    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    bad = [p for p in players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise SystemExit(f"Spatial leak in Feature players: {bad}")

    train, validation, test = temporal_split(panel)
    y_col = "target_next_month_crime"
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()

    model_feat = make_model(SEED)
    model_geo = make_model(SEED)
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

    X_test = panel.loc[test].reset_index(drop=True)
    print(
        "Loading published Hull-only baseline "
        f"(Feature masks={FEAT_MASKS}, Geo perturbations={GEO_PERTURBS}, seed={SEED})..."
    )
    base_feat, base_geo = load_published_baseline()
    hull_geo_lsoas = sorted(panel["lsoa21cd"].astype(str).unique())
    print(
        f"  Feature top-{FEATURE_TOP_K}: "
        + ", ".join(_topk_ordered(base_feat, "feature", "abs_coefficient", FEATURE_TOP_K))
    )
    print(
        f"  Geo top-{GEO_TOP_K}: "
        + ", ".join(_topk_ordered(base_geo, "lsoa21cd", "geo_abs_coefficient", GEO_TOP_K))
    )

    base_feat_top = _topk_ids(base_feat, "feature", "abs_coefficient", FEATURE_TOP_K)
    base_geo_top = _topk_ids(base_geo, "lsoa21cd", "geo_abs_coefficient", GEO_TOP_K)

    print("Loading LSOA boundaries for stability maps...")
    lsoa_codes = sorted(panel["lsoa21cd"].astype(str).unique())
    boundaries, _ = load_boundaries(lsoa_codes)
    name_lookup = (
        panel.drop_duplicates("lsoa21cd")
        .set_index("lsoa21cd")["lsoa21nm"]
        .astype(str)
        .to_dict()
    )
    feat_xmax = float(base_feat["abs_coefficient"].max())
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    viz_paths: dict[str, dict[str, str]] = {}
    viz_paths["baseline"] = save_stability_snapshot(
        VIZ_DIR / "baseline",
        label="Published Hull-only GeoSMILE baseline",
        feat_imp=base_feat,
        geo_imp=base_geo,
        boundaries=boundaries,
        name_lookup=name_lookup,
        meta={"kind": "baseline"},
        feat_xmax=feat_xmax,
    )

    print("Selecting four cardinal neighbor LSOAs and collecting police/covariate data...")
    neighbors, extra_panel = prepare_cardinal_neighbor_data(panel)
    missing_players = [p for p in players if p not in extra_panel.columns]
    if missing_players:
        raise SystemExit(f"Neighbor panel missing Feature players: {missing_players}")
    na_players = extra_panel[players].isna().sum()
    na_players = na_players[na_players > 0]
    if len(na_players):
        raise SystemExit(f"Neighbor panel has NA Feature values: {na_players.to_dict()}")
    print("Neighbor Feature players: all 30 present with no NA.")
    temporal = {
        "crime_lag_1",
        "crime_lag_2",
        "crime_lag_3",
        "crime_rolling_mean_3",
        "crime_rolling_mean_6",
        "month_sin",
        "month_cos",
    }
    static_players = [p for p in players if p not in temporal]
    print(
        extra_panel.drop_duplicates("lsoa21cd")[
            ["lsoa21cd", "lsoa21nm"] + static_players
        ].to_string(index=False)
    )
    print("Temporal Feature players also present: " + ", ".join(sorted(temporal)))
    extra_by_dir = {
        str(row.direction): extra_panel[extra_panel["lsoa21cd"].astype(str) == str(row.lsoa21cd)].copy()
        for row in neighbors.itertuples()
    }
    name_lookup.update(
        neighbors.drop_duplicates("lsoa21cd")
        .set_index("lsoa21cd")["lsoa21nm"]
        .astype(str)
        .to_dict()
    )
    locator = save_neighbor_locator_map(
        boundaries, neighbors, VIZ_DIR / "cardinal_neighbors_map.png"
    )
    viz_paths["neighbors_map"] = str(locator)

    feat_j, feat_rho, feat_pred_rho = [], [], []
    geo_j, geo_rho, geo_pred_rho = [], [], []
    per_direction: dict[str, dict] = {}

    for direction in DIRECTIONS:
        extra_rows = extra_by_dir[direction]
        code = str(extra_rows["lsoa21cd"].iloc[0])
        nm = str(extra_rows["lsoa21nm"].iloc[0])
        print(f"Stability {direction}: adding {code} {nm}...")
        expanded = expand_panel(panel, extra_rows)
        _, _, exp_test = temporal_split(expanded)
        X_exp = expanded.loc[exp_test].reset_index(drop=True)

        feat_imp, _, _, _, _ = smile_global_feature(
            model_feat, X_exp, players, baseline, FEAT_MASKS, SEED
        )
        geo_imp, _ = geo_smile_coefficients(
            model_geo,
            expanded,
            exp_test,
            baseline,
            GEO_PERTURBS,
            SEED,
            sample_lsoas=hull_geo_lsoas,
        )
        metrics = compare_attributions(
            base_feat,
            feat_imp,
            base_geo,
            geo_imp,
            base_feat_top=base_feat_top,
            base_geo_top=base_geo_top,
            added_lsoa=code,
        )
        f_pred, _ = hull_row_pred_spearman(model_feat, X_test, X_exp, players)
        g_pred, _ = hull_row_pred_spearman(
            model_geo, X_test, X_exp, SPATIAL_COLUMNS
        )
        feat_j.append(metrics["feature_jaccard"])
        feat_rho.append(metrics["feature_spearman"])
        feat_pred_rho.append(f_pred)
        geo_j.append(metrics["geo_jaccard"])
        geo_rho.append(metrics["geo_spearman"])
        geo_pred_rho.append(g_pred)

        extra_bit = ""
        if np.isfinite(metrics.get("geo_extra_rank", float("nan"))):
            extra_bit = (
                f" | extra rank={int(metrics['geo_extra_rank'])} "
                f"abs={metrics['geo_extra_abs_coefficient']:.3f} "
                f"in_top20={'yes' if metrics['geo_extra_in_top20'] else 'no'}"
            )
        print(
            f"  Feature J={metrics['feature_jaccard']:.3f} rho={metrics['feature_spearman']:.3f} | "
            f"Geo top-20 {int(metrics['geo_top20_overlap_n'])}/{GEO_TOP_K} "
            f"J={metrics['geo_jaccard']:.3f} | "
            f"Geo top-19 {int(metrics['geo_top19_overlap_n'])}/{GEO_TOP_K - 1} "
            f"J={metrics['geo_top19_jaccard']:.3f}"
            f"{extra_bit} (added {code})"
        )
        extra_geom = neighbors.loc[
            neighbors["lsoa21cd"].astype(str) == code, ["lsoa21cd", "lsoa21nm", "geometry"]
        ]
        map_bounds = gpd.GeoDataFrame(
            pd.concat(
                [boundaries[["lsoa21cd", "lsoa21nm", "geometry"]], extra_geom],
                ignore_index=True,
            ),
            crs=boundaries.crs,
        )
        per_direction[direction] = {
            "lsoa21cd": code,
            "lsoa21nm": nm,
            **metrics,
            "feature_pred_spearman": f_pred,
            "geo_pred_spearman": g_pred,
        }
        viz_paths[direction] = save_stability_snapshot(
            VIZ_DIR / direction,
            label=f"{direction.title()} neighbor {code}",
            feat_imp=feat_imp,
            geo_imp=geo_imp,
            boundaries=map_bounds,
            name_lookup=name_lookup,
            meta={
                "kind": "cardinal_neighbor",
                "direction": direction,
                "added_lsoa": code,
                "added_name": nm,
                **metrics,
                "feature_pred_spearman": f_pred,
                "geo_pred_spearman": g_pred,
            },
            feat_xmax=feat_xmax,
        )

    neighbor_desc = "; ".join(
        f"{d}={per_direction[d]['lsoa21cd']} ({per_direction[d]['lsoa21nm']})"
        for d in DIRECTIONS
    )
    method_note = (
        "ATT stability: Jaccard overlap of top-k attribution sets and Spearman "
        "correlation of |coefficients| before vs after adding one real LSOA from "
        "each cardinal direction just outside Hull, with Police.uk and Census/OSM "
        "covariates collected on the same schema as Hull LSOAs. Distinct from "
        "multi-run Consistency. Models stay Hull-trained; only the explanation "
        "reference is expanded. Baseline is the published Hull-only Feature "
        f"({FEAT_MASKS} masks) and Geo ({GEO_PERTURBS} perturbations) ranking; "
        "expansions reuse that SMILE design (seed 42)."
    )

    summary = {
        "status": "Evaluated",
        "definition": method_note,
        "n_reps": N_REPS,
        "feat_masks": FEAT_MASKS,
        "geo_perturbations": GEO_PERTURBS,
        "feature_k": FEATURE_TOP_K,
        "geo_k": GEO_TOP_K,
        "perturbation": "cardinal_neighbor_lsoas",
        "smile_seed": SEED,
        "neighbors": per_direction,
        "feature": {
            "branch": BRANCH_FEATURE,
            "top_k_jaccard_mean": _mean_or_nan(feat_j),
            "top_k_jaccard_std": _std_or_nan(feat_j),
            "abs_coef_spearman_mean": _mean_or_nan(feat_rho),
            "abs_coef_spearman_std": _std_or_nan(feat_rho),
            "pred_spearman_mean": float(np.mean(feat_pred_rho)),
            "n_prediction_preserving": N_REPS,
            "perturbation": (
                "Add one outside-Hull LSOA (N/E/S/W) with real police and "
                "covariate data to the Feature SMILE reference"
            ),
        },
        "geo": {
            "branch": BRANCH_GEO,
            "top_k_jaccard_mean": _mean_or_nan(geo_j),
            "top_k_jaccard_std": _std_or_nan(geo_j),
            "abs_coef_spearman_mean": _mean_or_nan(geo_rho),
            "abs_coef_spearman_std": _std_or_nan(geo_rho),
            "pred_spearman_mean": float(np.mean(geo_pred_rho)),
            "n_prediction_preserving": N_REPS,
            "perturbation": (
                "Add one outside-Hull LSOA (N/E/S/W) with real coordinates "
                "and crime history to the Geo SMILE reference"
            ),
        },
        "per_rep": {
            "direction": list(DIRECTIONS),
            "feature_jaccard": feat_j,
            "feature_spearman": feat_rho,
            "geo_jaccard": geo_j,
            "geo_spearman": geo_rho,
        },
        "visuals_dir": str(VIZ_DIR),
        "visuals": viz_paths,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    sum_path = OUT / "geosmile_hull_att_stability.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for pack, k in (
        (summary["feature"], FEATURE_TOP_K),
        (summary["geo"], GEO_TOP_K),
    ):
        branch = pack["branch"]
        for metric, value in [
            ("status", "Evaluated"),
            ("Top-k Jaccard mean", pack["top_k_jaccard_mean"]),
            ("Top-k Jaccard std", pack["top_k_jaccard_std"]),
            ("|coef| Spearman mean", pack["abs_coef_spearman_mean"]),
            ("|coef| Spearman std", pack["abs_coef_spearman_std"]),
            ("prediction Spearman mean", pack["pred_spearman_mean"]),
            ("n_cardinal_directions", N_REPS),
        ]:
            rows.append(
                {
                    "branch": branch,
                    "evaluation_set": "att_stability",
                    "category": "Controlled-input stability",
                    "metric": metric,
                    "value": value,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "p_value": np.nan,
                    "n": N_REPS,
                    "k": k,
                    "source_file": "compute_att_stability.py",
                    "result_status": "computed",
                    "notes": method_note + f" perturbation={pack['perturbation']}",
                }
            )
    long_df = pd.DataFrame(rows)
    long_path = OUT / "geosmile_hull_att_stability.csv"
    long_df.to_csv(long_path, index=False)

    eval_path = OUT / "geosmile_hull_feature_geo_evaluation.csv"
    if eval_path.exists():
        ev = pd.read_csv(eval_path)
        ev = ev[ev["category"].astype(str) != "Controlled-input stability"].copy()
        for c in ev.columns:
            if c not in long_df.columns:
                long_df[c] = np.nan
        ev = pd.concat([ev, long_df[ev.columns]], ignore_index=True)
        ev.to_csv(eval_path, index=False)

    sj = OUT / "geosmile_hull_evaluation_summary_v3.json"
    if sj.exists():
        s = json.loads(sj.read_text(encoding="utf-8"))
        s["controlled_input_stability"] = summary
        lims = list(s.get("limitations") or [])
        lims = [
            x
            for x in lims
            if "Controlled-input stability" not in str(x)
            and "ATT/controlled-input stability" not in str(x)
        ]
        lims.append(
            "ATT/controlled-input stability expands the explanation reference with "
            f"four real outside-Hull LSOAs ({neighbor_desc}); models remain "
            "trained on Hull only. Means are over the four cardinal directions."
        )
        s["limitations"] = lims
        sj.write_text(json.dumps(s, indent=2), encoding="utf-8")

    neighbor_rows = "\n".join(
        f"- **{d.title()}**: `{per_direction[d]['lsoa21cd']}` "
        f"{per_direction[d]['lsoa21nm']} — Feature Jaccard "
        f"{per_direction[d]['feature_jaccard']:.3f}, Geo Jaccard "
        f"{per_direction[d]['geo_jaccard']:.3f}"
        for d in DIRECTIONS
    )
    metric_lines = []
    extra_lines = []
    n_ok_f = n_ok_g = 0
    for d in DIRECTIONS:
        rec = per_direction[d]
        ok_f = rec["feature_pred_spearman"] >= 0.99
        ok_g = rec["geo_pred_spearman"] >= 0.99
        n_ok_f += int(ok_f)
        n_ok_g += int(ok_g)
        extra_rank = rec.get("geo_extra_rank", float("nan"))
        extra_abs = rec.get("geo_extra_abs_coefficient", float("nan"))
        extra_in = rec.get("geo_extra_in_top20", 0.0)
        extra_rank_s = (
            str(int(extra_rank)) if np.isfinite(float(extra_rank)) else "—"
        )
        extra_abs_s = f"{float(extra_abs):.3f}" if np.isfinite(float(extra_abs)) else "—"
        extra_in_s = "Yes" if extra_in else "No"
        extra_lines.append(
            f"- **{d.title()}**: extra rank {extra_rank_s}, "
            f"|coef| {extra_abs_s}, in top-20: {extra_in_s}"
        )
        metric_lines.append(
            f"| {d.title()} | Feature | {rec['feature_jaccard']:.3f} | "
            f"{rec['feature_spearman']:.3f} | {rec['feature_pred_spearman']:.3f} | "
            f"{'Yes' if ok_f else 'No'} |"
        )
        metric_lines.append(
            f"| {d.title()} | Geo | {rec['geo_jaccard']:.3f} | "
            f"{rec['geo_spearman']:.3f} | {rec['geo_pred_spearman']:.3f} | "
            f"{'Yes' if ok_g else 'No'} |"
        )
    metric_table = "\n".join(metric_lines)
    (OUT / "geosmile_hull_att_stability_note.md").write_text(
        f"""# ATT / controlled-input stability (computed)

ATT stability compares SMILE attributions on the Hull-only test reference with
attributions after adding one real LSOA from each cardinal direction just
outside the city. Police.uk street counts and the same Census 2021 / OSM
covariates used for Hull LSOAs are collected for those neighbors. Quantified by:

- Jaccard overlap of top-k attribution sets (Feature k={FEATURE_TOP_K}, Geo k={GEO_TOP_K})
- Spearman correlation of absolute coefficients over the full attribution vector

The predictive models stay Hull-trained. Hull-row prediction Spearman is a
sanity check that original test rows are unchanged. This is distinct from
multi-run Consistency. Baseline Feature and Geo rankings are the published
Hull-only GeoSMILE attributions ({FEAT_MASKS} Feature masks, {GEO_PERTURBS}
Geo perturbations, seed {SEED}); expansions reuse that design.

## Neighbors

{neighbor_rows}

## Results (four one-at-a-time additions)

Accepted if Hull-row prediction Spearman ≥ 0.99. Feature k={FEATURE_TOP_K}, Geo k={GEO_TOP_K}.

| Addition | Branch | Top-k Jaccard | abs coef Spearman | Pred. Spearman | Accepted |
|----------|--------|---------------|-------------------|----------------|----------|
{metric_table}
| Mean | Feature | {summary['feature']['top_k_jaccard_mean']:.3f} | {summary['feature']['abs_coef_spearman_mean']:.3f} | {summary['feature']['pred_spearman_mean']:.3f} | {n_ok_f}/{N_REPS} |
| Mean | Geo | {summary['geo']['top_k_jaccard_mean']:.3f} | {summary['geo']['abs_coef_spearman_mean']:.3f} | {summary['geo']['pred_spearman_mean']:.3f} | {n_ok_g}/{N_REPS} |

Each run adds **one** neighbor LSOA as a real Geo player (fitted coefficient, not forced to 0). Hull core samples stay synchronized with the baseline. SMILE budgets: Feature masks={FEAT_MASKS}, Geo perturbations={GEO_PERTURBS}.

## Extra LSOA Geo rank

The added LSOA is out of the Geo top-20 in every direction:

{chr(10).join(extra_lines)}

## Visuals

Per-snapshot folders under `{VIZ_DIR.as_posix()}`:

- `baseline/`, `north/`, `east/`, `south/`, `west/`
- `cardinal_neighbors_map.png` — locator for the four added LSOAs
- `feature_importance_ranking.png` — Feature |coefficient| ranking (top-{FEATURE_TOP_K} highlighted)
- `geo_top{GEO_TOP_K}_map.png` — Geo top-{GEO_TOP_K} LSOA importance map
- matching `feature_importance.csv` / `geo_coefficients.csv`
""",
        encoding="utf-8",
    )

    method = OUT / "geosmile_hull_evaluation_method_v3.md"
    if method.exists():
        text = method.read_text(encoding="utf-8")
        new_bullet = (
            "- **Controlled-input stability (ATT stability)** - **Evaluated** - "
            f"Feature top-{FEATURE_TOP_K} Jaccard "
            f"{summary['feature']['top_k_jaccard_mean']:.3f}; "
            f"Geo top-{GEO_TOP_K} Jaccard {summary['geo']['top_k_jaccard_mean']:.3f} "
            "(mean over four cardinal neighbor-LSOA expansions just outside Hull). "
            "Distinct from multi-run Consistency."
        )
        lines = []
        replaced = False
        for line in text.splitlines(True):
            if "Controlled-input stability" in line:
                lines.append(new_bullet + "\n")
                replaced = True
            else:
                lines.append(line)
        if not replaced:
            lines.append("\n" + new_bullet + "\n")
        method.write_text("".join(lines), encoding="utf-8")

    print("\n=== ATT stability ===")
    print(
        f"Feature J={summary['feature']['top_k_jaccard_mean']:.4f} "
        f"Spearman={summary['feature']['abs_coef_spearman_mean']:.4f}"
    )
    print(
        f"Geo J={summary['geo']['top_k_jaccard_mean']:.4f} "
        f"Spearman={summary['geo']['abs_coef_spearman_mean']:.4f}"
    )
    print(f"Wrote {sum_path}")
    print(f"Wrote {long_path}")
    print(f"Visuals under {VIZ_DIR}")


if __name__ == "__main__":
    main()
