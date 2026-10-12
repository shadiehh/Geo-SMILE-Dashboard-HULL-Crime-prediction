"""Core vs. borderline consistency classification (item 8): for CatBoost and
XGBoost, rerun the R=10 SMILE-reseed consistency check (same seeds 42-51,
same N=1,024 Feature / N=4,000 Geo as everywhere else), this time saving
EACH reseed's own top-k membership -- the existing R=10 consistency numbers
in outputs/verify_metrics_all_models/<model>/results.json only ever kept
summary Jaccard/Spearman statistics, not which specific features/LSOAs were
in the top-k on each individual reseed.

Classification (membership-based, as specified): for each feature (Feature
branch, k=10) / LSOA (Geo branch, k=20):
  core       -- in the top-k on all 10/10 reseeds
  borderline -- in the top-k on 1-9 of the 10 reseeds
  (never)    -- in the top-k on 0/10 reseeds (omitted from the chart, not
                interesting on its own, but present in the full CSV)

Writes outputs/consistency_core_borderline/{feature,geo}_membership_<model>.csv
and .../core_borderline_<model>_<branch>.png (bar chart, top-k membership
frequency out of 10, core vs. borderline vs. never colour-coded).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_by_n_perturb import Y_COL
from geosmile_hull_twobranch import FORBIDDEN_FEATURE_TOKENS, SPATIAL_COLUMNS, choose_feature_players, geo_smile_coefficients, smile_global_feature, temporal_split
from hull_expert_topk import FEATURE_TOP_K, GEO_TOP_K
from model_factory import fit_model, make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "consistency_core_borderline"
PANEL_PATH = OUT / "hull_lsoa_month_panel_with_covariates.csv"

FEAT_N = 1024
GEO_N = 4000
RESEEDS = list(range(42, 52))  # R=10, same convention as everywhere else

INK, INK_MUTED = "#12171C", "#57646D"
CORE_COLOR = "#FF0051"
BORDERLINE_COLOR = "#FFB3CB"
NEVER_COLOR = "#D8DEE4"


def _log(msg: str) -> None:
    print(msg, flush=True)


def topk_membership_feature(model, X_test, players, baseline) -> pd.DataFrame:
    counts = {f: 0 for f in players}
    for seed in RESEEDS:
        imp, _, _, _, _ = smile_global_feature(model, X_test[players], players, baseline, FEAT_N, seed)
        top = imp.sort_values(["abs_coefficient", "feature"], ascending=[False, True], kind="mergesort").head(FEATURE_TOP_K)["feature"]
        for f in top:
            counts[f] += 1
    df = pd.DataFrame({"feature": list(counts.keys()), "topk_count": list(counts.values())})
    df["category"] = np.select([df["topk_count"] == len(RESEEDS), df["topk_count"] == 0], ["core", "never"], default="borderline")
    return df.sort_values("topk_count", ascending=False).reset_index(drop=True)


def topk_membership_geo(model, panel, test_mask, baseline) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for seed in RESEEDS:
        imp, _ = geo_smile_coefficients(model, panel, test_mask, baseline, GEO_N, seed)
        imp["lsoa21cd"] = imp["lsoa21cd"].astype(str)
        top = imp.sort_values(["geo_abs_coefficient", "lsoa21cd"], ascending=[False, True], kind="mergesort").head(GEO_TOP_K)["lsoa21cd"]
        for c in imp["lsoa21cd"]:
            counts.setdefault(c, 0)
        for c in top:
            counts[c] += 1
    df = pd.DataFrame({"lsoa21cd": list(counts.keys()), "topk_count": list(counts.values())})
    df["category"] = np.select([df["topk_count"] == len(RESEEDS), df["topk_count"] == 0], ["core", "never"], default="borderline")
    return df.sort_values("topk_count", ascending=False).reset_index(drop=True)


def plot_membership(df: pd.DataFrame, id_col: str, model_name: str, branch: str, path: Path, show_n: int) -> None:
    plot_df = df[df["category"] != "never"].head(show_n) if len(df[df["category"] != "never"]) > 0 else df.head(show_n)
    colors = plot_df["category"].map({"core": CORE_COLOR, "borderline": BORDERLINE_COLOR, "never": NEVER_COLOR})
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(plot_df) + 1.5)))
    y = np.arange(len(plot_df))[::-1]
    ax.barh(y, plot_df["topk_count"], color=colors, edgecolor="white", height=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df[id_col].astype(str), fontsize=8.5, color=INK)
    ax.set_xlim(0, len(RESEEDS) + 0.3)
    ax.set_xticks(range(0, len(RESEEDS) + 1))
    ax.set_xlabel(f"Reseeds in top-k (of {len(RESEEDS)})", fontsize=10, color=INK_MUTED)
    ax.axvline(len(RESEEDS), color=INK_MUTED, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    n_core = int((df["category"] == "core").sum())
    n_border = int((df["category"] == "borderline").sum())
    ax.set_title(f"{model_name}: {branch} top-k membership across R=10 reseeds\ncore={n_core}, borderline={n_border}",
                 fontsize=11.5, fontweight="bold", color=INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=CORE_COLOR, label="Core (10/10)"), Patch(color=BORDERLINE_COLOR, label="Borderline (1-9/10)")],
              loc="lower right", fontsize=8.5, frameon=False)
    fig.text(0.5, 0.01, f"GeoSMILE {branch} branch, N={FEAT_N if branch=='Feature' else GEO_N}, seeds 42-51. Model: {model_name}.",
              ha="center", fontsize=7.2, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    _log(f"Saved: {path}")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH)
    panel["month"] = pd.to_datetime(panel["month"])
    players = choose_feature_players(panel)
    bad = [p for p in players if any(tok in p.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise SystemExit(f"Spatial leak: {bad}")
    train, validation, test = temporal_split(panel)
    baseline = panel.loc[train, players + SPATIAL_COLUMNS].median()
    X_test = panel.loc[test].reset_index(drop=True)

    for model_name in ("catboost", "xgboost"):
        _log(f"\n=== [{model_name}] Feature branch, R=10 reseeds ===")
        model_feat = make_model(model_name, 42, n_jobs=2)
        fit_model(model_name, model_feat, panel.loc[train, players], panel.loc[train, Y_COL],
                  panel.loc[validation, players], panel.loc[validation, Y_COL])
        feat_df = topk_membership_feature(model_feat, X_test, players, baseline)
        feat_df.to_csv(DEST / f"feature_membership_{model_name}.csv", index=False)
        plot_membership(feat_df, "feature", model_name, "Feature", DEST / f"core_borderline_{model_name}_feature.png", show_n=len(players))

        _log(f"\n=== [{model_name}] Geo branch, R=10 reseeds ===")
        model_geo = make_model(model_name, 42, n_jobs=2)
        fit_model(model_name, model_geo, panel.loc[train, SPATIAL_COLUMNS], panel.loc[train, Y_COL],
                  panel.loc[validation, SPATIAL_COLUMNS], panel.loc[validation, Y_COL])
        geo_df = topk_membership_geo(model_geo, panel, test, baseline)
        geo_df.to_csv(DEST / f"geo_membership_{model_name}.csv", index=False)
        plot_membership(geo_df, "lsoa21cd", model_name, "Geo", DEST / f"core_borderline_{model_name}_geo.png", show_n=30)

    _log("\nDone.")


if __name__ == "__main__":
    main()
