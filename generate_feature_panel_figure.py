"""Emit the LaTeX (subcaption package) source for two SEPARATE small-multiple
panels of the already-generated feature_top10_<model>.png rankings -- no
image regeneration needed here, unlike the Geo panel: these are bar charts
(each auto-scaled to its own model's coefficient/importance magnitudes,
which is the correct way to read a ranking chart), not choropleth maps
sharing one absolute colour scale, so there is no shared-axis fix to make.

The two panels are kept separate because the underlying quantities differ,
not just the model:
  Panel 1 (8 models): the incumbent kernel-weighted Ridge SMILE coefficient,
    N=1,024 masks, seed=42. Title reports PREDICTIVE R^2.
  Panel 2 (2 models, mlp/dl): Random Forest surrogate feature_importances_
    (magnitude-only, unsigned -- not the officially adopted GeoSMILE
    attribution), used because the Ridge coefficient collapses for these
    two. Title reports FIDELITY R^2 instead of predictive R^2 -- a
    different quantity, which is exactly why these don't belong in the
    first panel. Budget and weighting also differ: MLP at N=8,192
    (kernel-weighted), DL at N=4,096 (unweighted) -- see
    outputs/design_experiments/smile_design_experiments_section.tex for why
    those specific (N, weighting) choices were selected as each model's
    best-found surrogate configuration.

Provenance symbols (mapped from this project's own outputs/top_maps/
qualifying_models.csv "source" column, same convention as
generate_geo_panel_figure.py -- adjust SYMBOL_BY_MODEL if
Table~\\ref{tab:sel_provenance} defines different tiers):
  dagger  (xgboost, lightgbm, catboost, svr, gp): saved from the rigorous
    7-model verification.
  ddagger (ann, cnn, transformer): fresh Ridge SMILE fit, fidelity-focused
    design experiments only.
  pilcrow (mlp, dl): Random Forest surrogate, not the adopted linear
    attribution -- a distinct tier from either Ridge group.

Writes outputs/top_maps/figures/panel/feature_top10_figure.tex.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
PANEL_DIR = ROOT / "outputs" / "top_maps" / "figures" / "panel"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

# Row 1: tree ensembles + SVR. Row 2: GP + attention/neural (Ridge-surrogate) family.
RIDGE_MODELS = ["xgboost", "lightgbm", "catboost", "svr", "gp", "ann", "cnn", "transformer"]
RF_MODELS = ["mlp", "dl"]

DISPLAY_NAME = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost", "svr": "SVR", "gp": "GP",
    "ann": "ANN", "cnn": "CNN", "transformer": "Transformer",
    "mlp": "MLP (shallow)", "dl": "DL (Keras)",
}
SYMBOL = {
    "xgboost": r"\dagger", "lightgbm": r"\dagger", "catboost": r"\dagger", "svr": r"\dagger", "gp": r"\dagger",
    "ann": r"\ddagger", "cnn": r"\ddagger", "transformer": r"\ddagger",
    "mlp": r"\P", "dl": r"\P",
}
RF_DETAIL = {
    "mlp": "Random Forest, $N{=}8{,}192$, kernel-weighted",
    "dl": "Random Forest, $N{=}4{,}096$, unweighted",
}
LABEL_LETTER = "abcdefghij"


def subfig(model: str, letter: str, width: str, extra_line: str = "") -> str:
    cap = f"{DISPLAY_NAME[model]}$^{{{SYMBOL[model]}}}$"
    if extra_line:
        cap += f" \\\\ {{\\scriptsize {extra_line}}}"
    return (
        f"  \\begin{{subfigure}}[t]{{{width}\\textwidth}}\n"
        f"    \\includegraphics[width=\\textwidth]{{feature_top10_{model}.png}}\n"
        f"    \\subcaption{{{cap}}}\n"
        f"    \\label{{fig:sel_feature_{model}}}\n"
        "  \\end{subfigure}"
    )


def build_ridge_panel() -> str:
    rows = [subfig(m, LABEL_LETTER[i], "0.23") for i, m in enumerate(RIDGE_MODELS)]
    grid = " \\hfill\n".join(rows[:4]) + "\n\n  \\vspace{0.6em}\n\n" + " \\hfill\n".join(rows[4:])
    return f"""\\begin{{figure}}[htbp]
\\centering
{grid}
\\caption{{Feature SMILE top-10 ranking by candidate model, kernel-weighted
Ridge surrogate ($N{{=}}1{{,}}024$ masks, seed 42). Title reports each
model's held-out \\emph{{predictive}} $R^2$ (not fidelity -- contrast
Figure~\\ref{{fig:sel_feature_rf}}, whose two panels report fidelity
$R^2$ instead, since that is the quantity their surrogate choice was
selected to fix). Superscripts mark provenance per
Table~\\ref{{tab:sel_provenance}}: $\\dagger$ saved from the rigorous
7-model verification (full metric suite); $\\ddagger$ a fresh Ridge SMILE
fit for this round of design experiments (fidelity-focused only).}}
\\label{{fig:sel_feature_ridge}}
\\end{{figure}}
"""


def build_rf_panel() -> str:
    rows = [subfig(m, LABEL_LETTER[i], "0.32", RF_DETAIL[m]) for i, m in enumerate(RF_MODELS)]
    grid = " \\hfill\n".join(rows)
    return f"""\\begin{{figure}}[htbp]
\\centering
{grid}
\\caption{{Feature ranking for the two models whose kernel-weighted Ridge
surrogate collapses (fidelity $R^2<0.13$, see
Table~\\ref{{tab:sel_provenance}}) -- Random Forest \\texttt{{feature\\_
importances\\_}} instead, magnitude-only (unsigned) and \\emph{{not}} the
officially adopted GeoSMILE attribution (tree ensembles have no
linear-coefficient equivalent). Title reports \\emph{{fidelity}} $R^2$, not
predictive $R^2$ -- the opposite of Figure~\\ref{{fig:sel_feature_ridge}} --
because predictive fit was never the problem for either model (both exceed
$R^2=0.85$ predictively); it is the surrogate's ability to reconstruct the
masking response that this figure reports instead. Budget and weighting
also differ between the two panels below, each being that model's
best-found configuration in the surrogate$\\times N$ sweep. Superscript
$\\P$ marks this distinct provenance tier.}}
\\label{{fig:sel_feature_rf}}
\\end{{figure}}
"""


def main() -> None:
    out_path = PANEL_DIR / "feature_top10_figure.tex"
    tex = (
        "% Requires \\usepackage{subcaption} in the preamble (and graphicx).\n"
        "% Assumes the feature_top10_<model>.png files sit alongside this .tex\n"
        "% file (adjust the graphics path / \\graphicspath as needed).\n\n"
        + build_ridge_panel()
        + "\n"
        + build_rf_panel()
    )
    out_path.write_text(tex, encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
