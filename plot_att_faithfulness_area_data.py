"""ATT Faithfulness figure for Chapter 4 with the area-data Geo branch.

ATT Faithfulness = across-model Pearson r between predictive R2 and ATT accuracy
(primary), with ATT-AUC secondary (faithfulness_accuracy_r2.py, KG-SMILE Sec. 5.3).

  Feature: unchanged, from outputs/faithfulness_accuracy_r2/model_table.csv (N=4,096).
  Geo:     area-data design, all ten models at N=4,000, accuracy = share of the
           top 20 inside the 58 expert-highlighted LSOAs; ATT-AUC against the same
           reference (outputs/geo_area_data_models/expert_region_accuracy/accuracy_summary.csv).

Reuses faithfulness_accuracy_r2.scatter for an identical layout. Writes
outputs/faithfulness_accuracy_r2/att_faithfulness_area_data.{png,pdf} and the
matching table att_faithfulness_area_data.csv; the old coordinates-design figure
(accuracy_vs_r2.png) is left untouched.

    python plot_att_faithfulness_area_data.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import faithfulness_accuracy_r2 as fa

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "faithfulness_accuracy_r2"
GEO_SUMMARY = ROOT / "outputs" / "geo_area_data_models" / "expert_region_accuracy" / "accuracy_summary.csv"
K_GEO = 20


def table() -> pd.DataFrame:
    feat = pd.read_csv(OUT / "model_table.csv")
    feat = feat[feat["branch"] == "Feature"].copy()
    g = pd.read_csv(GEO_SUMMARY)
    geo = pd.DataFrame({
        "model": g["model"],
        "family": g["model"].map(fa.FAMILY),
        "branch": "Geo",
        "predictive_r2": g["predictive_r2"],
        "n": 4000,
        "score_n": 4000.0,
        "overlap_hits": g[f"hits@{K_GEO}"].astype(float),
        "k": float(K_GEO),
        "att_auc": g["att_auroc"],
        "att_accuracy": g[f"hits@{K_GEO}"] / K_GEO,
        "included": True,
    })
    return pd.concat([feat, geo[feat.columns]], ignore_index=True)


def main() -> None:
    t = table()
    faith = fa.correlate(t)
    faith.to_csv(OUT / "att_faithfulness_area_data.csv", index=False)
    print(faith[["branch", "measure", "n_models", "pearson_r", "pearson_p", "spearman_rho"]].round(3).to_string())
    # scatter() always writes accuracy_vs_r2.*; render into a temporary folder and rename.
    tmp = OUT / "_area_data_tmp"
    fa.DEST = tmp
    fa.MEASURES.update({
        "att_accuracy": "ATT accuracy (primary)\nFeature: top-10 overlap | Geo: top-20 in expert regions",
        "att_auc": "ATT-AUC (secondary)",
    })
    fa.scatter(t, faith, note=(
        "ATT Faithfulness (KG-SMILE Sec. 5.3) = Pearson r, across the ten models, between predictive R² and "
        "ATT accuracy (primary; ATT-AUC secondary). Feature: top-10 overlap with the expert predictor reference, "
        "N=4,096 sweep run. Geo: area-data design, N=4,000, share of the top 20 inside the 58 expert-highlighted "
        "LSOAs; ATT-AUC against the same reference."))
    for ext in ("png", "pdf"):
        (tmp / f"accuracy_vs_r2.{ext}").replace(OUT / f"att_faithfulness_area_data.{ext}")
    tmp.rmdir()
    print(f"Saved {OUT / 'att_faithfulness_area_data.png'}")


if __name__ == "__main__":
    main()
