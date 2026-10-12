"""ATT Faithfulness as the across-model Pearson correlation between predictive R2 and
attribution quality -- the KG-SMILE (2025, Sec. 5.3) definition adapted to Hull.

KG-SMILE defines ATT Faithfulness as Pearson r(x, y) over a set of models, with x an
external measure of model competence (there: benchmark accuracy) and y an attribution-
quality score (there: ATT-AUC). A strong positive r means the attribution metric tracks
model competence. Here, over every black box screened in all three families:

  x = predictive R2 on the test set
  y = ATT accuracy (overlap of GeoSMILE's top-k with the expert top-k)   primary
      ATT-AUC (AUROC of GeoSMILE scores against the expert labels)      closest to KG-SMILE
  faithfulness = Pearson r(x, y), per branch

Pearson r is the primary statistic, with its two-sided p; the one-sided p (H1: r > 0,
"attribution quality rises with R2") and Spearman rho are reported beside it. Replaces the
intervention definition (top-k minus random response, per model).

Families (as grouped in generate_feature_panel_figure.py / generate_geo_panel_figure.py):
  boosting  xgboost, lightgbm, catboost
  kernel    svr, gp
  neural    mlp, dl, ann, cnn, transformer

Inputs (read-only, no recomputation):
  - outputs/accuracy_faithfulness_nsweep/summary.csv: acc_TP / acc_k (overlap; Feature
    k=10, Geo k=20; XGBoost 6/10 and 15/20) and acc_ATT_AUROC_continuous (ATT-AUC).
    With k fixed, FP = FN = k - TP, so (TP+TN)/n is linear in the overlap and gives the
    same correlation.
  - outputs/eval_by_model/predictive_performance.csv: predictive R2.
No R2 gate: models with R2 <= 0.7 are scored too, since weak models are exactly where the
correlation should show. The sweep skipped them in the Geo branch, so their scores come
from the other runs that did explain them (GEO_FALLBACK); `score_n` records which N:
  mlp  verify_metrics_all_models/mlp/results.json   Geo N=4,000  11/20
  svr  eval_by_model/svr/n_2000/accuracy.json      Geo N=2,000  10/20  (only run for svr)
cnn was never explained in the Geo branch, so cnn/Geo has no score and is left out.

Primary result uses N=4,000 perturbations. The Geo sweep has N=4,000; the Feature sweep
ran N in {512, 1024, 2048, 4096}, so the Feature branch uses its nearest run, N=4,096
(only XGBoost was run at Feature N=4,000, and its overlap there is 6/10, as at 4,096).

Writes outputs/faithfulness_accuracy_r2/:
  model_table.csv           one row per (model, branch): family, R2, overlap, ATT-AUC, score_n
  faithfulness.csv          per branch x measure: Pearson r (two- and one-sided p), Spearman rho
  faithfulness_by_family.csv  the same within each family (families of 2 get no p-value)
  faithfulness_by_n.csv     the same at every N in the sweep
  accuracy_vs_r2.png/.pdf   scatter, rows = measure, columns = branch

Usage:  py -3.13 faithfulness_accuracy_r2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEST = OUT / "faithfulness_accuracy_r2"

FEAT_N_SELECTED = 4096  # nearest Feature sweep run to N=4,000; see docstring
GEO_N_SELECTED = 4000
# Geo scores for models the sweep skipped (R2 <= 0.7): (path, N it was computed at).
GEO_FALLBACK = {
    "mlp": ("verify_metrics_all_models/mlp/results.json", 4000),
    "svr": ("eval_by_model/svr/n_2000/accuracy.json", 2000),
}
MEASURES = {
    "att_accuracy": "ATT accuracy: overlap with expert top-k",
    "att_auc": "ATT-AUC vs expert labels",
}

FAMILY = {
    "xgboost": "boosting", "lightgbm": "boosting", "catboost": "boosting",
    "svr": "kernel", "gp": "kernel",
    "mlp": "neural", "dl": "neural", "ann": "neural", "cnn": "neural", "transformer": "neural",
}
# Categorical slots 1-3, fixed order; marker shape carries family too, so colour is never alone.
FAMILY_STYLE = {
    "boosting": ("#2a78d6", "o"),
    "kernel": ("#eb6834", "s"),
    "neural": ("#1baf7a", "^"),
}
INK, INK_MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _fallback_scores(path: Path) -> tuple[int, int, float]:
    """Geo top-k hits, k and ATT-AUC from a verify_metrics results.json or an eval_by_model accuracy.json."""
    d = json.loads(path.read_text(encoding="utf-8"))
    geo = d["metric1_att_accuracy"]["geo"] if "metric1_att_accuracy" in d else d["geo"]
    hits = geo.get("TP", geo.get("intersection"))
    auc = geo.get("ATT_AUROC_continuous", geo.get("auroc"))
    return int(hits), int(geo["k"]), float(auc)


def model_table(n_feature: int = FEAT_N_SELECTED, n_geo: int = GEO_N_SELECTED,
                any_fallback_n: bool = True) -> pd.DataFrame:
    """`any_fallback_n=False` (the N sweep) takes a fallback score only at its own N."""
    acc = pd.read_csv(OUT / "accuracy_faithfulness_nsweep" / "summary.csv")
    acc = acc[((acc["branch"] == "Feature") & (acc["n"] == n_feature)) |
              ((acc["branch"] == "Geo") & (acc["n"] == n_geo))]
    acc = acc[["model", "branch", "n", "acc_TP", "acc_k", "acc_ATT_AUROC_continuous"]].rename(
        columns={"n": "score_n", "acc_TP": "overlap_hits", "acc_k": "k", "acc_ATT_AUROC_continuous": "att_auc"})
    pred = pd.read_csv(OUT / "eval_by_model" / "predictive_performance.csv")
    t = pred[["model", "branch", "r2"]].rename(columns={"r2": "predictive_r2"}).merge(
        acc, on=["model", "branch"], how="left")
    t.insert(3, "n", np.where(t["branch"] == "Feature", n_feature, n_geo))
    for model, (rel, n_run) in GEO_FALLBACK.items():
        row = (t["model"] == model) & (t["branch"] == "Geo") & t["overlap_hits"].isna()
        if row.any() and (any_fallback_n or n_run == n_geo):
            hits, k, auc = _fallback_scores(OUT / rel)
            t.loc[row, ["overlap_hits", "k", "att_auc", "score_n"]] = [hits, k, auc, n_run]
    t["att_accuracy"] = t["overlap_hits"] / t["k"]
    t.insert(1, "family", t["model"].map(FAMILY))
    unknown = t.loc[t["family"].isna(), "model"].unique()
    if len(unknown):
        raise ValueError(f"no family for model(s): {', '.join(unknown)}")
    t["included"] = t["att_accuracy"].notna()
    return t.sort_values(["branch", "family", "model"]).reset_index(drop=True)


def _stats(g: pd.DataFrame, measure: str) -> dict:
    x, y = g["predictive_r2"], g[measure]
    if len(g) < 3:  # two points always correlate at +-1; no test is possible
        r = float(np.sign(np.corrcoef(x, y)[0, 1])) if len(g) == 2 else np.nan
        return {"pearson_r": r, "pearson_p": np.nan, "pearson_p_one_sided": np.nan,
                "spearman_rho": r, "spearman_p": np.nan}
    r, p = pearsonr(x, y)
    rho, sp = spearmanr(x, y)
    return {"pearson_r": r, "pearson_p": p, "pearson_p_one_sided": pearsonr(x, y, alternative="greater")[1],
            "spearman_rho": rho, "spearman_p": sp}


def correlate(t: pd.DataFrame, by_family: bool = False) -> pd.DataFrame:
    keys = ["branch", "family"] if by_family else ["branch"]
    rows = []
    for key, g in t[t["included"]].groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        for measure in MEASURES:
            rows.append({**dict(zip(keys, key)), "measure": measure, "n": int(g["n"].iloc[0]),
                         "n_models": len(g), "models": ",".join(g["model"]), **_stats(g, measure)})
    return pd.DataFrame(rows)


def by_n() -> pd.DataFrame:
    """The same correlation at every N in the sweep."""
    acc = pd.read_csv(OUT / "accuracy_faithfulness_nsweep" / "summary.csv")
    frames = []
    for branch in ("Feature", "Geo"):
        for n in sorted(acc.loc[acc["branch"] == branch, "n"].unique()):
            t = model_table(n_feature=n, n_geo=n, any_fallback_n=False)
            frames.append(correlate(t[t["branch"] == branch]))
    return pd.concat(frames, ignore_index=True)


def scatter(t: pd.DataFrame, faith: pd.DataFrame, note: str | None = None) -> None:
    branches = ["Feature", "Geo"]
    fig, axes = plt.subplots(len(MEASURES), 2, figsize=(11.0, 9.6))
    for (measure, ylabel), row_axes in zip(MEASURES.items(), axes):
        for ax, branch in zip(row_axes, branches):
            g = t[(t["branch"] == branch) & t["included"]]
            for family, (colour, marker) in FAMILY_STYLE.items():
                f = g[g["family"] == family]
                ax.scatter(f["predictive_r2"], f[measure], s=70, c=colour, marker=marker,
                           edgecolors="white", linewidths=2, zorder=3, label=family)
            # Near-coincident points would print their names between each other; a point
            # with a close neighbour to its right is labelled on its left instead, and a
            # label that would land on one already placed on the same side steps down a line.
            pts = g.sort_values("predictive_r2")
            x = (pts["predictive_r2"] / (np.ptp(pts["predictive_r2"]) or 1.0)).to_numpy()
            y = (pts[measure] / (np.ptp(pts[measure]) or 1.0)).to_numpy()
            placed: list[tuple[float, float, str]] = []
            for i, (_, row) in enumerate(pts.iterrows()):
                right_neighbour = any(0 <= x[j] - x[i] < 0.12 and abs(y[j] - y[i]) < 0.06
                                      for j in range(i + 1, len(pts)))
                ha = "right" if right_neighbour else "left"
                below = sum(abs(x[i] - px) < 0.12 and abs(y[i] - py) < 0.06 and side == ha
                            for px, py, side in placed)
                offset = ((-7 if ha == "right" else 7), 4 - 11 * below)
                placed.append((x[i], y[i], ha))
                name = row["model"] if row["score_n"] == row["n"] else f"{row['model']} (N={int(row['score_n']):,})"
                ax.annotate(name, (row["predictive_r2"], row[measure]), xytext=offset,
                            textcoords="offset points", fontsize=8, color=INK_MUTED, ha=ha)
            slope, icept = np.polyfit(g["predictive_r2"], g[measure], 1)
            xs = np.linspace(g["predictive_r2"].min(), g["predictive_r2"].max(), 2)
            ax.plot(xs, icept + slope * xs, color=INK_MUTED, lw=1.2, ls="--", zorder=2)
            f = faith[(faith["branch"] == branch) & (faith["measure"] == measure)].iloc[0]
            ax.set_title(f"{branch} branch (N={f['n']:,}, {f['n_models']} models)\n", loc="left",
                         fontsize=11, fontweight="bold", color=INK)
            ax.text(0.0, 1.02,
                    f"Pearson r = {f['pearson_r']:+.2f} (p = {f['pearson_p']:.2f}; one-sided "
                    f"{f['pearson_p_one_sided']:.2f})   Spearman ρ = {f['spearman_rho']:+.2f}",
                    transform=ax.transAxes, fontsize=9, color=INK_MUTED, va="bottom")
            ax.margins(x=0.12, y=0.1)
            ax.set_xlabel("Predictive R² (test)", color=INK)
            ax.set_ylabel(ylabel, color=INK)
            ax.grid(color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Model family", frameon=False, fontsize=9, title_fontsize=9,
               loc="center left", bbox_to_anchor=(1.0, 0.5))
    missing = t[~t["included"]]
    note = note or ("ATT Faithfulness (KG-SMILE Sec. 5.3) = Pearson r between predictive R² and attribution quality "
            "across models. Overlap: Feature k=10, Geo k=20. Feature uses the N=4,096 sweep run, nearest to 4,000.")
    if len(missing):
        note += " No score computed, left out: " + ", ".join(
            f"{m}/{b}" for m, b in zip(missing["model"], missing["branch"])) + "."
    fig.text(0.01, -0.01, note, fontsize=8, color=INK_MUTED, ha="left", va="top", wrap=True)
    fig.tight_layout()
    DEST.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEST / "accuracy_vs_r2.png", dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(DEST / "accuracy_vs_r2.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _log(f"Saved: {DEST / 'accuracy_vs_r2.png'}")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    t = model_table()
    faith = correlate(t)
    family = correlate(t, by_family=True)
    sweep = by_n()
    t.to_csv(DEST / "model_table.csv", index=False)
    faith.to_csv(DEST / "faithfulness.csv", index=False)
    family.to_csv(DEST / "faithfulness_by_family.csv", index=False)
    sweep.to_csv(DEST / "faithfulness_by_n.csv", index=False)
    scatter(t, faith)
    pd.set_option("display.width", 200)
    cols = ["measure", "n", "n_models", "pearson_r", "pearson_p", "pearson_p_one_sided", "spearman_rho", "spearman_p"]
    _log(faith[["branch"] + cols].round(3).to_string(index=False))
    _log("\nWithin each family:")
    _log(family[["branch", "family"] + cols].round(3).to_string(index=False))
    _log("\nRobustness across N:")
    _log(sweep[["branch"] + cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
