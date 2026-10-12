"""Geo accuracy of all ten models against the expert-highlighted regions.

Reference: the LSOAs inside the regions the expert highlighted as the areas whose
crime events inform their assessment of Hull's crime situation
(outputs/geo_area_data_models/top50_maps/highlighted_regions_lsoas.csv, built by
select_highlighted_region_lsoas.py). Model ranking: signed area-data Geo-SMILE
coefficient (outputs/geo_area_data_models/<model>.json, N=4,000, seed 42).

Accuracy@k = |model top-k  intersect  reference| / k -- the share of the model's k
most sensitive LSOAs that fall in the expert areas. Other metrics mirror
hull_geo_area_data_eval.accuracy (ATT-Acc, F1, Jaccard, recall) plus:
  chance      expected accuracy of a random top-k = |reference| / n
  lift        accuracy / chance
  p_hypergeom P(overlap >= observed) for a random top-k (one-sided)
  att_auroc   threshold-free: signed coefficient vs reference membership, all LSOAs

ATT Faithfulness (definition of faithfulness_accuracy_r2.py / Chapter 4): Pearson r,
across the models, between predictive R2 and attribution quality -- ATT-AUC primary,
top-k overlap secondary -- with two- and one-sided (H1: r > 0) p and Spearman rho.
Also reported without the two Keras models (dl, cnn), whose R2 is not reproducible
under a fixed seed and which alone span the low end of the R2 range.

Writes outputs/geo_area_data_models/expert_region_accuracy/. Not causal.

    python eval_expert_region_accuracy.py [--k 20 42 50 58]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import hypergeom
from sklearn.metrics import roc_auc_score

from faithfulness_accuracy_r2 import FAMILY, FAMILY_STYLE, INK, INK_MUTED, _stats
from map_geo_importance_ranks import rank_by

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "outputs" / "geo_area_data_models"
REF_PATH = SRC / "top50_maps" / "highlighted_regions_lsoas.csv"
DEST = SRC / "expert_region_accuracy"
MODELS = ("xgboost", "lightgbm", "catboost", "svr", "gp", "mlp", "dl", "ann", "cnn", "transformer")
KERAS_NONDETERMINISTIC = ("dl", "cnn")


def scores(ranked: pd.DataFrame, ref: set[str], k: int) -> dict:
    n = len(ranked)
    top = set(ranked.loc[ranked["rank"] <= k, "lsoa21cd"])
    tp = len(top & ref)
    chance = len(ref) / n
    return {
        "k": k,
        "hits": tp,
        "accuracy": tp / k,
        "chance": chance,
        "lift": (tp / k) / chance,
        "p_hypergeom": float(hypergeom.sf(tp - 1, n, len(ref), k)),
        "recall": tp / len(ref),
        "att_f1": 2 * tp / (len(ref) + k),
        "att_acc": (n - len(top ^ ref)) / n,
        "jaccard": tp / len(top | ref),
    }


def scatter(table: pd.DataFrame, corr: pd.DataFrame, measures: list[str], path: Path) -> None:
    labels = {"att_auroc": "ATT-AUC vs expert regions"}
    fig, axes = plt.subplots(1, len(measures), figsize=(4.6 * len(measures), 4.2), sharex=True)
    for ax, m in zip(axes, measures):
        for family, (colour, marker) in FAMILY_STYLE.items():
            f = table[table["model"].map(FAMILY) == family]
            ax.scatter(f["predictive_r2"], f[m], s=70, c=colour, marker=marker,
                       edgecolors="white", linewidths=2, zorder=3, label=family)
        for _, r in table.iterrows():
            ax.annotate(r["model"], (r["predictive_r2"], r[m]), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color=INK_MUTED)
        c = corr[corr["measure"] == m].iloc[0]
        title = labels[m] if m in labels else f"Overlap: model top {m.split('@')[1]} in expert regions"
        ax.set_title(
            f"{title}\nr = {c['pearson_r']:+.2f} (p = {c['pearson_p']:.2f}),  "
            f"Spearman ρ = {c['spearman_rho']:+.2f}",
            fontsize=9.5, color=INK,
        )
        ax.margins(x=0.12, y=0.15)
        ax.set_xlabel("Predictive R² (test)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, title="Model family", frameon=False, fontsize=8, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("ATT Faithfulness: predictive R² vs Geo accuracy against expert regions (M=10)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, nargs="+", default=None,
                   help="cut-offs; default 20, 42, 50 and the reference size (where accuracy = recall)")
    args = p.parse_args()

    ref_table = pd.read_csv(REF_PATH)
    ref = set(ref_table["lsoa21cd"].astype(str))
    args.k = args.k or [20, 42, 50, len(ref)]
    DEST.mkdir(parents=True, exist_ok=True)

    rows, summary = [], []
    for m in MODELS:
        res = json.loads((SRC / f"{m}.json").read_text(encoding="utf-8"))
        coef = pd.DataFrame(res["coefficients"]).assign(lsoa21cd=lambda d: d["lsoa21cd"].astype(str))
        ranked = rank_by(coef, "geo_coefficient", "rank")
        auroc = roc_auc_score(ranked["lsoa21cd"].isin(ref).astype(int), ranked["geo_coefficient"])
        for k in args.k:
            rows.append({"model": m, "predictive_r2": res["predictive_r2"], **scores(ranked, ref, k)})
        summary.append({"model": m, "predictive_r2": res["predictive_r2"], "att_auroc": auroc})

    long = pd.DataFrame(rows)
    long.to_csv(DEST / "accuracy_by_model_k.csv", index=False)
    wide = long.pivot(index="model", columns="k", values="accuracy").add_prefix("accuracy@").reindex(MODELS)
    hits = long.pivot(index="model", columns="k", values="hits").add_prefix("hits@").reindex(MODELS)
    table = pd.DataFrame(summary).set_index("model").join(hits).join(wide).reset_index()
    table.to_csv(DEST / "accuracy_summary.csv", index=False)

    measures = ["att_auroc"] + [c for c in table.columns if c.startswith("hits@")]
    corr = pd.DataFrame(
        [
            {"model_set": label, "measure": m, "M": len(t), **_stats(t, m)}
            for label, t in (
                ("all models", table),
                ("excl. dl, cnn", table[~table["model"].isin(KERAS_NONDETERMINISTIC)]),
            )
            for m in measures
        ]
    )
    corr.to_csv(DEST / "faithfulness_correlation.csv", index=False)
    scatter(table, corr[corr["model_set"] == "all models"], ["att_auroc", "hits@20", "hits@42"],
            DEST / "faithfulness_scatter.png")

    print(f"Reference: {len(ref)} LSOAs of {len(ranked)} (chance accuracy {len(ref) / len(ranked):.3f})")
    print(table.round(3).to_string(index=False))
    print(long[["model", "k", "hits", "accuracy", "lift", "p_hypergeom"]].round(4).to_string(index=False))
    cols = ["model_set", "measure", "M", "pearson_r", "pearson_p", "pearson_p_one_sided", "spearman_rho"]
    print(corr[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
