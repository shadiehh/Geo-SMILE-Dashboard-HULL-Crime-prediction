"""Re-render the 8 already-computed geo_top20_<model>.png maps on a SHARED
0-24 colour scale (no individual colorbar) for a small-multiple panel
figure, plus a standalone shared colorbar strip and the LaTeX (subcaption
package) source that assembles them as Figure~\\ref{fig:sel_maps}.

No SMILE recomputation: reads the coefficient tables generate_top_maps.py
already saved (outputs/top_maps/tables/geo_ranking_<model>.csv) and re-plots
them through the same save_geo_rank_map()/plot_topk_value_map() pipeline,
now with the shared vmin/vmax + show_colorbar=False options added to those
functions for this purpose. Rank labels (1-20) are unchanged -- only the
colour normalization and the per-map colorbar are affected.

Shared scale: vmin=0, vmax=24 (covers every model's max |coefficient|, the
highest being XGBoost/CatBoost at ~23.2-23.4; ANN/CNN top out around
~10.3-10.6 -- exactly the mismatch the shared scale is meant to fix).

Provenance tiers (marked with a symbol per model, mapped from this
project's own outputs/top_maps/qualifying_models.csv "source" column --
adjust SYMBOL_BY_MODEL below if Table~\\ref{tab:sel_provenance} defines
different tiers):
  dagger  (xgboost, lightgbm, catboost, gp, dl): saved from the original
    rigorous 7-model verification (verify_metrics_all_models.py) -- full
    5-metric suite (ATT accuracy, faithfulness, stability, consistency).
  ddagger (ann, transformer): fresh Ridge SMILE fit for this round of
    design experiments -- fidelity-focused only, no full 5-metric suite yet.
  section (cnn): same as ddagger, AND forced through despite the Geo
    branch failing its own R2>0.7 predictive guard (R2=0.661).

N mismatch flagged, not resolved here: these maps use Geo N=4,000 (this
project's current design, raised from the originally-published N=2,000
after a further XGBoost sweep found higher fidelity at N=4,000); if
Section~\\ref{sec:hull_quant_results} reports N=2,000 that section is
describing an earlier design point -- reconcile which N that section
should cite, or note the discrepancy in the caption, since this script has
no visibility into that section's content.

Writes outputs/top_maps/figures/panel/geo_top20_<model>_shared_scale.png,
outputs/top_maps/figures/panel/shared_colorbar.png, and
outputs/top_maps/figures/panel/sel_maps_figure.tex.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize

from compute_att_stability import save_geo_rank_map
from hull_boundaries import load_boundaries
from map_geo_importance_ranks import SEQ_GEO, INK_MUTED

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TABLE_DIR = OUT / "top_maps" / "tables"
PANEL_DIR = OUT / "top_maps" / "figures" / "panel"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

VMIN, VMAX = 0.0, 24.0
GEO_N = 4000  # see module docstring re: the N=2,000 mismatch this does not resolve

# Row 1: boosting + kernel ("saved" tier). Row 2: neural/attention family.
MODEL_ORDER = ["xgboost", "lightgbm", "catboost", "gp", "dl", "ann", "cnn", "transformer"]
DISPLAY_NAME = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost", "gp": "GP",
    "dl": "DL (Keras)", "ann": "ANN", "cnn": "CNN", "transformer": "Transformer",
}
SYMBOL_BY_MODEL = {
    "xgboost": r"\dagger", "lightgbm": r"\dagger", "catboost": r"\dagger",
    "gp": r"\dagger", "dl": r"\dagger",
    "ann": r"\ddagger", "transformer": r"\ddagger",
    "cnn": r"\S",
}
LABEL_LETTER = "abcdefgh"


def _log(msg: str) -> None:
    print(msg, flush=True)


def render_shared_colorbar(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 0.6))
    norm = Normalize(vmin=VMIN, vmax=VMAX)
    sm = plt.cm.ScalarMappable(cmap=SEQ_GEO, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_label("|Geo SMILE coefficient| (shared scale, 0-24)", fontsize=9, color=INK_MUTED)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)


def build_latex(dest: Path) -> None:
    rows = []
    for i, model in enumerate(MODEL_ORDER):
        letter = LABEL_LETTER[i]
        img = f"geo_top20_{model}_shared_scale.png"
        cap = f"{DISPLAY_NAME[model]}$^{{{SYMBOL_BY_MODEL[model]}}}$"
        rows.append(
            "  \\begin{subfigure}[t]{0.23\\textwidth}\n"
            f"    \\includegraphics[width=\\textwidth]{{{img}}}\n"
            f"    \\subcaption{{{cap}}}\n"
            f"    \\label{{fig:sel_maps_{model}}}\n"
            "  \\end{subfigure}"
        )
    grid = " \\hfill\n".join(rows[:4]) + "\n\n  \\vspace{0.6em}\n\n" + " \\hfill\n".join(rows[4:])

    tex = f"""% Requires \\usepackage{{subcaption}} in the preamble (and graphicx).
% Assumes the 8 geo_top20_<model>_shared_scale.png files and
% shared_colorbar.png sit alongside this .tex file (adjust the graphics
% path / \\graphicspath as needed for your document tree).
\\begin{{figure}}[htbp]
\\centering
{grid}

\\vspace{{0.8em}}
\\includegraphics[width=0.55\\textwidth]{{shared_colorbar.png}}

\\caption{{Geo SMILE top-20 LSOA attribution by candidate model, all eight
maps drawn on a shared |coefficient| colour scale (0--24) with one shared
colour bar so relative intensity is comparable across panels -- previously
each map was scaled to its own maximum (roughly 10 for ANN/CNN up to
roughly 23 for XGBoost/CatBoost), which would otherwise read as comparable
intensity when it was not. Rank labels 1--20 are unchanged. Geo $N{{=}}4{{,}}000$,
seed 42 for every panel; reconcile against Section~\\ref{{sec:hull_quant_results}}
if that section reports $N{{=}}2{{,}}000$, as that describes an earlier design
point in this project. Superscripts mark provenance per
Table~\\ref{{tab:sel_provenance}}: $\\dagger$ saved from the rigorous
7-model verification (full metric suite); $\\ddagger$ a fresh Ridge SMILE
fit for this round of design experiments (fidelity-focused only); $\\S$ as
$\\ddagger$, additionally forced through despite its own Geo-branch
predictive $R^2{{=}}0.661$ failing the $R^2>0.7$ guard.}}
\\label{{fig:sel_maps}}
\\end{{figure}}
"""
    dest.write_text(tex, encoding="utf-8")


def main() -> None:
    panel = pd.read_csv(OUT / "hull_lsoa_month_panel_with_covariates.csv")
    hull_lsoas = sorted(panel["lsoa21cd"].astype(str).unique())
    name_lookup = dict(zip(panel["lsoa21cd"].astype(str), panel["lsoa21nm"]))
    boundaries, _meta = load_boundaries(hull_lsoas)

    for model in MODEL_ORDER:
        src = TABLE_DIR / f"geo_ranking_{model}.csv"
        if not src.exists():
            raise SystemExit(f"Missing {src} -- run generate_top_maps.py first")
        coef = pd.read_csv(src)
        coef["lsoa21cd"] = coef["lsoa21cd"].astype(str)
        out_path = PANEL_DIR / f"geo_top20_{model}_shared_scale.png"
        save_geo_rank_map(
            coef, boundaries, out_path,
            title=DISPLAY_NAME[model],
            footer=f"GeoSMILE Geo branch, N={GEO_N}, seed=42. Shared 0-24 colour scale.",
            name_lookup=name_lookup,
            vmin=VMIN, vmax=VMAX, show_colorbar=False,
        )
        _log(f"Saved: {out_path}")

    cbar_path = PANEL_DIR / "shared_colorbar.png"
    render_shared_colorbar(cbar_path)
    _log(f"Saved: {cbar_path}")

    tex_path = PANEL_DIR / "sel_maps_figure.tex"
    build_latex(tex_path)
    _log(f"Saved: {tex_path}")


if __name__ == "__main__":
    main()
