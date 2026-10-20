"""Dashboard files for the random-donor Hull Geo design, in the layout the dashboard reads.

The dashboard (dashboard/data/schema.py GEO_DESIGN_DIR) expects, per design folder:
  n_<N>/geo_coefficients.csv        lsoa21cd, geo_coefficient, geo_abs_coefficient, geo_score   (every N)
  n_<N>/metrics.json                smile_diagnostics, fidelity, accuracy, faithfulness, stability,
                                    consistency (+ at 4,000: fixed_kernel_west, concentration_*, exact_vs_smile_spearman)
  n_4000/consistency_per_lsoa.csv   per-LSOA seed runs (seeds 100-104)
  geo_coefficients_N4000.csv        ranked, with exact_w1 / exact_signed_shift, lsoa21nm, expert_reference
  spatial_patterns/                 LISA on the signed coefficient (compute_geo_spatial_patterns --signed)
  stability_four_directions_*.png, summary_by_n.csv
This script writes them to outputs/geo_random_donor_eval/ from the random-donor design
(geo_random_donor_eval.py; outputs/geo_random_donor_design/). Coefficient pools are replayed with the
same model, seeds and K = 50, which are deterministic; the N = 4,000 pool is checked against the
published xgboost/geo_coefficients_N4000.csv. The old area-data folder is not touched. Not causal.

    python build_random_donor_dashboard_files.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance

import hull_geo_area_data_eval as ev
from geo_random_donor_eval import CONS, Ctx, fit, setup
from hull_dehghani_eval import sparsity_entropy_metrics
from hull_expert_topk import geo_expert_binary_frame

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "outputs" / "geo_random_donor_design"
DEST = ROOT / "outputs" / "geo_random_donor_eval"
NS = (1000, 2000, 3000, 4000, 5000, 7500, 10000)
SEED = 42


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    k = int(json.loads((SRC / "k_selection.json").read_text())["chosen_K"])
    panel, cols, split, hull, names, refs, rules = setup()
    model = fit("xgboost", panel, cols, split)
    ctx = Ctx(model, panel, cols, hull)
    expert = geo_expert_binary_frame(hull)[["lsoa21cd", "expert_top10_reference"]]
    published = pd.read_csv(SRC / "xgboost" / "geo_coefficients_N4000.csv").set_index("lsoa21cd")["geo_coefficient"]
    rows = []
    for n in NS:
        coef, M, y, d, diag = ctx.pool(n, SEED, k)
        if n == 4000:
            gap = float((coef.set_index("lsoa21cd")["geo_coefficient"].loc[published.index] - published).abs().max())
            print(f"N=4000 replay vs published coefficients: max |difference| = {gap:.2e}")
            assert gap < 1e-9, "replayed pool does not reproduce the published random-donor coefficients"
        (DEST / f"n_{n}").mkdir(exist_ok=True)
        coef[["lsoa21cd", "geo_coefficient", "geo_abs_coefficient", "geo_score"]].to_csv(DEST / f"n_{n}" / "geo_coefficients.csv", index=False)
        src = json.loads((SRC / "xgboost" / f"n_{n}.json").read_text())
        metrics = {
            "n": n,
            "smile_diagnostics": {
                "method": "per_lsoa_random_donor_coordinate_masking", "n_spatial_units": len(hull), "n_perturbations": n,
                "n_sampled_lsoas": len(hull), "n_extra_lsoas": 0, "min_lsoas_per_perturbation": 5,
                "max_lsoas_per_perturbation": 10, "donor_draws_K": k, **diag,
                "response": "mean over K donor draws of wasserstein_1 on affected lsoa_month predictions",
                "perturbation": "move the coordinates of selected LSOAs' rows to the centroid of a random other Hull LSOA; all predictors kept",
                "model": "XGBoost on 30 predictors + BNG centroid easting/northing",
                "ranking": "signed coefficient (geo_score)"},
            "fidelity": src["fidelity"],
            "accuracy": ev.accuracy(coef, expert),
            "accuracy_expert_regions": src["accuracy"],
            "faithfulness": src["intervention"],
            "stability": src["stability"],
            "consistency": {"seeds": list(CONS), **src["consistency"]},
        }
        if n == 4000:
            metrics.update({k2: src[k2] for k2 in ("fixed_kernel_west", "concentration_exact_shift", "moran", "accuracy_k",
                                                   "accuracy_rule_robustness", "accuracy_other_maps", "rho_with_crime_level")})
            metrics["concentration_abs_coefficient"] = sparsity_entropy_metrics(coef["geo_abs_coefficient"].to_numpy())
            metrics["exact_vs_smile_spearman"] = src["exact_vs_signed_spearman"]
        (DEST / f"n_{n}" / "metrics.json").write_text(json.dumps(ev.to_json(metrics), indent=2, default=float), encoding="utf-8")
        rows.append({"N": n, "fidelity_r2": src["fidelity"]["R2"], "fidelity_weighted_r2": src["fidelity"]["Weighted R2"],
                     "accuracy_overlap": metrics["accuracy"]["intersection"], "att_auroc": metrics["accuracy"]["att_auroc"],
                     "regions_top20": src["accuracy"]["intersection"],
                     "faithfulness_top_minus_random": src["intervention"]["top_minus_random"],
                     "faithfulness_p": src["intervention"]["empirical_p_top_vs_random"],
                     "stability_jaccard": src["stability"]["jaccard_mean"], "stability_spearman": src["stability"]["spearman_mean"],
                     "consistency_mean_variance": src["consistency"]["mean_variance"], "consistency_mean_sd": src["consistency"]["mean_sd"],
                     "consistency_rank_spearman": src["consistency"]["rank_spearman_mean"]})
        print(f"N={n} written")
    pd.DataFrame(rows).to_csv(DEST / "summary_by_n.csv", index=False)

    # consistency per LSOA (seeds 100-104, N = 4,000)
    runs = {f"seed_{s}": ctx.pool(4000, s, k)[0].set_index("lsoa21cd")["geo_coefficient"] for s in CONS}
    mat = pd.DataFrame(runs)
    cons = pd.DataFrame({"mean_coefficient": mat.mean(axis=1), "variance": mat.var(axis=1, ddof=1), "sd": mat.std(axis=1, ddof=1)})
    cons["sd_pct_of_abs_mean"] = 100 * cons["sd"] / cons["mean_coefficient"].abs()
    cons["rank_by_mean"] = cons["mean_coefficient"].rank(ascending=False, method="first").astype(int)
    cons["expert_region"] = cons.index.isin(refs["map1"]).astype(int)
    cons = cons.join(mat).sort_values("rank_by_mean")
    cons.rename_axis("lsoa21cd").to_csv(DEST / "n_4000" / "consistency_per_lsoa.csv")
    src_cons = json.loads((SRC / "xgboost" / "n_4000.json").read_text())["consistency"]
    print(f"consistency replay: mean SD {cons['sd'].mean():.6f} (published {src_cons['mean_sd']:.6f})")

    # ranked table with the exact single-LSOA shift (mean over all other LSOAs as donors)
    ranked = pd.read_csv(SRC / "xgboost" / "geo_coefficients_N4000.csv")
    signed = {}
    for i, code in enumerate(ctx.lsoas[:ctx.n_core]):
        r = ctx.x[ctx.idx[i]]
        blocks = []
        for dnr in range(len(ctx.lsoas)):
            if dnr != i:
                p = r.copy()
                p[:, ctx.cix] = ctx.cent[dnr]
                blocks.append(p)
        pred = ctx.predict(np.vstack(blocks)).reshape(len(ctx.lsoas) - 1, len(r))
        signed[code] = float(np.mean(ctx.orig[ctx.idx[i]] - pred.mean(axis=0)))
    ranked["exact_signed_shift"] = ranked["lsoa21cd"].map(signed)
    ranked["expert_reference"] = ranked["lsoa21cd"].isin(set(expert.loc[expert["expert_top10_reference"] == 1, "lsoa21cd"]))
    ranked["expert_region_map1"] = ranked["lsoa21cd"].isin(refs["map1"])
    cols_out = ["lsoa21cd", "geo_coefficient", "geo_abs_coefficient", "geo_score", "geo_rank", "exact_w1", "exact_signed_shift",
                "lsoa21nm", "expert_reference", "expert_region_map1"]
    ranked[cols_out].to_csv(DEST / "geo_coefficients_N4000.csv", index=False)

    # figures and LISA
    shutil.copy2(SRC / "chapter4_figs" / "stability_four_directions_random_donor.png", DEST / "stability_four_directions_random_donor.png")
    shutil.copy2(SRC / "chapter4_figs" / "geo_map_random_donor.png", DEST / "geo_map_random_donor.png")
    subprocess.run([sys.executable, str(ROOT / "compute_geo_spatial_patterns.py"), "--signed",
                    "--coef", str(DEST / "n_4000" / "geo_coefficients.csv"), "--out-dir", str(DEST / "spatial_patterns")], check=True)
    print("done:", DEST)


if __name__ == "__main__":
    main()
