"""Model Diagnostics / Robustness tab: interactive views over the eval_by_n_perturb sweep.

Separate from the LLM stakeholder-explanation flow. Feature and Geo are always
plotted as two distinct series; never summed, averaged, or merged into one
line or figure. Expert-reference overlap is computed from the raw exported
rankings (see dashboard/data/sweep_loader.py), independent of the per-N
accuracy summary file.
"""

from __future__ import annotations

import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dashboard.data.sweep_loader import SweepBundle, load_sweep_bundle, topk_ranking
from dashboard.views.gating import persona_gated

BLUE, TEAL, PINK, INK = "#1E88E5", "#147D74", "#FF0051", "#12171C"
BRANCH_COLORS = {"Feature": BLUE, "Geo": TEAL}

METRIC_VIEWS: dict[str, dict] = {
    "surrogate_fidelity_r2": {
        "label": "Surrogate fidelity (R²)",
        "y_title": "Surrogate fidelity R² (held-out evaluation split)",
        "value_col": "surrogate_fidelity_r2",
        "error_col": None,
        "hover_extra": None,
        "fmt": ".4f",
    },
    "faithfulness_top_minus_random": {
        "label": "Faithfulness (top − random)",
        "y_title": "Top-minus-random response (Wasserstein distance)",
        "value_col": "faithfulness_top_minus_random",
        "error_col": None,
        "hover_extra": "faithfulness_empirical_p",
        "fmt": ".4f",
    },
    "att_stability_jaccard": {
        "label": "ATT stability (top-k Jaccard)",
        "y_title": "Mean top-k Jaccard (cardinal-neighbor expansion)",
        "value_col": "att_stability_jaccard",
        "error_col": None,
        "hover_extra": None,
        "fmt": ".3f",
    },
    "att_consistency_rank_spearman_mean": {
        "label": "ATT consistency (rank Spearman)",
        "y_title": "Mean rank Spearman across 5 seeds (± 1 SD)",
        "value_col": "att_consistency_rank_spearman_mean",
        "error_col": "att_consistency_rank_spearman_std",
        "hover_extra": None,
        "fmt": ".3f",
    },
    "expert_overlap_fraction": {
        "label": "Expert-reference overlap",
        "y_title": "Fraction of top-k matching the expert reference set",
        "value_col": "expert_overlap_fraction",
        "error_col": None,
        "hover_extra": "expert_overlap_count",
        "fmt": ".0%",
    },
}

FIDELITY_STAGES = [
    ("surrogate_fidelity_r2", "Surrogate fidelity (evaluation split)", BLUE),
    ("kernel_diagnostic_r2", "Kernel-selection diagnostic", PINK),
    ("consistency_fidelity_r2_mean", "Consistency-run mean", TEAL),
]

PUBLISHED_NOTE = (
    "This is the N used in the published GeoSMILE artefacts (the dashboard's map and "
    "evidence panel), not a sweep re-run at that N."
)


@st.cache_data(show_spinner="Loading robustness sweep…")
def _cached_sweep_bundle() -> SweepBundle:
    return load_sweep_bundle()


def _x_categories(n_grid: list[int], *, branch: str) -> list[str]:
    cats = []
    for n in n_grid:
        cats.append(str(n))
        if branch == "Feature" and n == 1000:
            cats.append("1024")
    return cats


def _hover_template(spec: dict, branch_label: str) -> str:
    base = f"{branch_label}<br>N=%{{x}}<br>" + spec["label"] + f": %{{y:{spec['fmt']}}}"
    if spec["hover_extra"] == "faithfulness_empirical_p":
        base += "<br>empirical p = %{customdata:.3f}"
    elif spec["hover_extra"] == "expert_overlap_count":
        base += "<br>%{customdata} of k matched"
    return base + "<extra></extra>"


def build_n_sensitivity_figure(bundle: SweepBundle, metric_key: str) -> go.Figure:
    spec = METRIC_VIEWS[metric_key]
    fig = go.Figure()
    for branch_label, color in BRANCH_COLORS.items():
        sub = bundle.metrics[
            (bundle.metrics["branch"] == branch_label) & (~bundle.metrics["is_published"])
        ].sort_values("n_perturb")
        x = [str(int(n)) for n in sub["n_perturb"]]
        y = sub[spec["value_col"]].tolist()
        err = sub[spec["error_col"]].tolist() if spec["error_col"] else None
        custom = sub[spec["hover_extra"]].tolist() if spec["hover_extra"] else None
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=branch_label,
                line=dict(color=color, width=2),
                marker=dict(size=8),
                error_y=dict(type="data", array=err, visible=True) if err else None,
                customdata=custom,
                hovertemplate=_hover_template(spec, branch_label),
            )
        )
        pub = bundle.metrics[
            (bundle.metrics["branch"] == branch_label) & (bundle.metrics["is_published"])
        ]
        if not pub.empty and pub.iloc[0][spec["value_col"]] is not None:
            pub_n = int(pub.iloc[0]["n_perturb"])
            pub_val = pub.iloc[0][spec["value_col"]]
            fig.add_trace(
                go.Scatter(
                    x=[str(pub_n)],
                    y=[pub_val],
                    mode="markers",
                    name=f"{branch_label} published (N={pub_n})",
                    marker=dict(symbol="star", size=16, color=color, line=dict(width=1, color=INK)),
                    hovertemplate=(
                        f"{branch_label} published run<br>N={pub_n}<br>"
                        + spec["label"]
                        + f": %{{y:{spec['fmt']}}}<br>"
                        + PUBLISHED_NOTE
                        + "<extra></extra>"
                    ),
                )
            )
    cat_array = _x_categories(bundle.n_grid, branch="Feature")
    fig.update_layout(
        xaxis=dict(
            title="Perturbation count N",
            type="category",
            categoryorder="array",
            categoryarray=cat_array,
        ),
        yaxis_title=spec["y_title"],
        height=440,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.22),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _render_n_sensitivity(bundle: SweepBundle) -> None:
    st.markdown("**N-sensitivity explorer**")
    st.caption(
        "How each evaluation metric moves with the perturbation/mask count N. "
        "Stars mark the published run. Feature and Geo are separate series, never combined."
    )
    metric_key = st.radio(
        "Metric",
        list(METRIC_VIEWS.keys()),
        format_func=lambda k: METRIC_VIEWS[k]["label"],
        horizontal=True,
        key="diag_metric_view",
    )
    fig = build_n_sensitivity_figure(bundle, metric_key)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_topk_bar(bundle: SweepBundle, *, branch: str, n_perturb: int | None) -> None:
    frame = topk_ranking(bundle, branch=branch, n_perturb=n_perturb)
    used_n = bundle.published_n[branch] if n_perturb is None else n_perturb
    label = "Feature top-10" if branch == "feature" else "Geo top-20"
    st.markdown(f"**{label} (N={used_n})**")
    if frame.empty:
        st.info("No ranking available for this N.")
        return
    frame = frame.sort_values("rank", ascending=False)
    colors = [PINK if hit else "#B7C0C7" for hit in frame["in_expert_reference"]]
    fig = go.Figure(
        go.Bar(
            x=frame["abs_coefficient"],
            y=frame["label"],
            orientation="h",
            marker_color=colors,
            customdata=list(
                zip(
                    frame["rank"],
                    frame["signed_coefficient"],
                    frame["in_expert_reference"].map({True: "yes", False: "no"}),
                )
            ),
            hovertemplate=(
                "%{y}<br>rank %{customdata[0]}<br>signed coefficient: %{customdata[1]:.4f}"
                "<br>|coefficient|: %{x:.4f}<br>in expert reference set: %{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=max(280, 26 * len(frame) + 80),
        margin=dict(l=10, r=10, t=8, b=24),
        xaxis_title="|coefficient|",
        yaxis=dict(automargin=True),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_topk_explorer(bundle: SweepBundle) -> None:
    st.markdown("**Top-k ranking explorer**")
    st.caption(
        "Ranked by |coefficient| from the raw exported rankings (feature_importance.csv / "
        "geo_coefficients.csv), computed independently of the per-N accuracy summary file. "
        "Pink bars are in the expert reference set."
    )
    options = ["published"] + [str(n) for n in bundle.n_grid]
    choice = st.selectbox(
        "Perturbation count N",
        options,
        format_func=lambda v: (
            f"Published (Feature N={bundle.published_n['feature']}, "
            f"Geo N={bundle.published_n['geo']})"
            if v == "published"
            else v
        ),
        key="diag_topk_n",
    )
    n_perturb = None if choice == "published" else int(choice)
    left, right = st.columns(2)
    with left:
        _render_topk_bar(bundle, branch="feature", n_perturb=n_perturb)
    with right:
        _render_topk_bar(bundle, branch="geo", n_perturb=n_perturb)


def build_fidelity_stage_figure(bundle: SweepBundle) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Feature", "Geo"), shared_yaxes=True)
    for col, branch_label in ((1, "Feature"), (2, "Geo")):
        sub = bundle.metrics[
            (bundle.metrics["branch"] == branch_label) & (~bundle.metrics["is_published"])
        ].sort_values("n_perturb")
        pub = bundle.metrics[
            (bundle.metrics["branch"] == branch_label) & (bundle.metrics["is_published"])
        ]
        x = [str(int(n)) for n in sub["n_perturb"]]
        for value_col, stage_label, color in FIDELITY_STAGES:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=sub[value_col],
                    mode="lines+markers",
                    name=stage_label,
                    legendgroup=stage_label,
                    showlegend=(col == 1),
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    hovertemplate=f"{stage_label}<br>N=%{{x}}<br>R²=%{{y:.4f}}<extra></extra>",
                ),
                row=1,
                col=col,
            )
            if not pub.empty and pub.iloc[0][value_col] is not None:
                pub_n = int(pub.iloc[0]["n_perturb"])
                fig.add_trace(
                    go.Scatter(
                        x=[str(pub_n)],
                        y=[pub.iloc[0][value_col]],
                        mode="markers",
                        marker=dict(symbol="star", size=14, color=color, line=dict(width=1, color=INK)),
                        showlegend=False,
                        hovertemplate=(
                            f"{stage_label} (published, N={pub_n})<br>R²=%{{y:.4f}}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=col,
                )
        cat_array = _x_categories(bundle.n_grid, branch=branch_label)
        fig.update_xaxes(
            title_text="Perturbation count N",
            type="category",
            categoryorder="array",
            categoryarray=cat_array,
            row=1,
            col=col,
        )
    fig.update_yaxes(title_text="R²", row=1, col=1)
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.28),
    )
    return fig


def _render_fidelity_stage(bundle: SweepBundle) -> None:
    st.markdown("**Fidelity by pipeline stage**")
    st.caption(
        "Three distinct fidelity R² figures, shown separately and never averaged: "
        "**Surrogate fidelity** (held-out evaluation split, fidelity.json), "
        "**kernel-selection diagnostic** (smile_diagnostics.json test_r2), and "
        "**consistency-run mean** (mean fidelity across 5 reseeded consistency reruns)."
    )
    fig = build_fidelity_stage_figure(bundle)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_diagnostics_section(stakeholder_id: str) -> None:
    """Model Diagnostics / Robustness tab. Visible by default for technical/governance."""
    show = persona_gated(
        stakeholder_id,
        default_personas={"technical", "governance"},
        toggle_label="Show diagnostics (Model Diagnostics / Robustness)",
        toggle_key="show_diagnostics_toggle",
        help_text=(
            "Robustness diagnostics — fidelity, faithfulness, ATT stability, ATT "
            "consistency, and expert-reference overlap — across the swept "
            "perturbation counts. Shown by default for Technical/Model Analyst and "
            "Governance/Assurance."
        ),
    )
    if not show:
        return

    st.divider()
    st.subheader("Model Diagnostics — robustness across perturbation count N")
    st.caption(
        "Diagnostic/robustness content. Separate from the stakeholder explanation above; "
        "it is not stakeholder-adapted text. Feature and Geo are always distinct series."
    )

    try:
        bundle = _cached_sweep_bundle()
    except FileNotFoundError as exc:
        st.warning(f"Robustness sweep data is not available: {exc}")
        return

    tab1, tab2, tab3 = st.tabs(
        ["N-sensitivity explorer", "Top-k ranking explorer", "Fidelity by pipeline stage"]
    )
    with tab1:
        _render_n_sensitivity(bundle)
    with tab2:
        _render_topk_explorer(bundle)
    with tab3:
        _render_fidelity_stage(bundle)
