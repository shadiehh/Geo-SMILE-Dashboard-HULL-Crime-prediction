"""Schematic rectangle panels for Feature attribution stability (4 directions).

Top-10 features: coloured squares sized by |SMILE coefficient|, with labels.
Remaining features: small neutral outline squares.
Layout: packed cluster (schematic, not a chart axis).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from geosmile_hull_twobranch import FEATURE_LABELS
from hull_boundaries import CARDINAL_DIRECTIONS
from hull_expert_topk import FEATURE_TOP_K

ROOT = Path(__file__).resolve().parent
VIZ_DIR = ROOT / "outputs" / "att_stability"
OUT_DIR = VIZ_DIR / "schematic"
DIRECTIONS = tuple(CARDINAL_DIRECTIONS.keys())
N_REPS = len(DIRECTIONS)
SEED = 42

BG = "#000000"
TOP_EDGE = "#FFFFFF"
NEUTRAL_EDGE = "#C8C8C8"
INK = "#FFFFFF"
INK_MUTED = "#B0B8C0"

# One hue per thesis category; shade varies by |coef| within the category.
CATEGORY_HUE = {
    "Crime / temporal": 355,  # crimson / red family
    "Economic / labour & tenure": 28,  # amber / orange
    "Social / demographic": 210,  # blue
    "Built environment / activity": 160,  # teal / green
    "Transport / accessibility": 275,  # violet
    "Environmental": 130,  # green
    "Structural/control": 0,  # grey via low saturation override
}
CATEGORY_MAP_PATH = (
    ROOT / "outputs" / "feature_category_alignment" / "feature_with_categories.csv"
)
EXPERT_RANK_PATH = ROOT / "outputs" / "feature_expert_ranking_v2.csv"


def load_feature_categories() -> dict[str, str]:
    if CATEGORY_MAP_PATH.exists():
        df = pd.read_csv(CATEGORY_MAP_PATH)
        col = "thesis_category" if "thesis_category" in df.columns else "expert_category"
        return dict(zip(df["feature"].astype(str), df[col].astype(str)))
    if EXPERT_RANK_PATH.exists():
        df = pd.read_csv(EXPERT_RANK_PATH)
        return dict(zip(df["feature"].astype(str), df["expert_category"].astype(str)))
    raise SystemExit("Missing feature category map CSV.")


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    """h in [0,360), s,l in [0,1] -> #RRGGBB."""
    h = h % 360.0
    s = float(np.clip(s, 0.0, 1.0))
    l = float(np.clip(l, 0.0, 1.0))
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = l - c / 2.0
    if h < 60:
        rp, gp, bp = c, x, 0.0
    elif h < 120:
        rp, gp, bp = x, c, 0.0
    elif h < 180:
        rp, gp, bp = 0.0, c, x
    elif h < 240:
        rp, gp, bp = 0.0, x, c
    elif h < 300:
        rp, gp, bp = x, 0.0, c
    else:
        rp, gp, bp = c, 0.0, x
    r = int(round((rp + m) * 255))
    g = int(round((gp + m) * 255))
    b = int(round((bp + m) * 255))
    return f"#{r:02X}{g:02X}{b:02X}"


def category_shade_scales(
    frames: list[pd.DataFrame], categories: dict[str, str]
) -> dict[str, tuple[float, float]]:
    """Per-category (min,max) |coef| across provided frames (shared across reps)."""
    bags: dict[str, list[float]] = {}
    for df in frames:
        for _, row in df.iterrows():
            feat = str(row["feature"])
            cat = categories.get(feat, "Structural/control")
            bags.setdefault(cat, []).append(float(row["abs_coefficient"]))
    scales = {}
    for cat, vals in bags.items():
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if hi <= lo:
            hi = lo + 1e-9
        scales[cat] = (lo, hi)
    return scales


def colour_for_feature(
    feature: str,
    abs_coef: float,
    categories: dict[str, str],
    scales: dict[str, tuple[float, float]],
) -> str:
    """Same category → same hue; stronger |coef| → darker / richer shade."""
    cat = categories.get(feature, "Structural/control")
    lo, hi = scales.get(cat, (0.0, max(abs_coef, 1e-9)))
    t = float(np.clip((abs_coef - lo) / (hi - lo), 0.0, 1.0))
    # Stronger attribution → darker, more saturated shade of the category hue.
    if cat == "Structural/control":
        # Grey family
        l = 0.72 - 0.38 * t
        return _hsl_to_hex(0.0, 0.05, l)
    hue = CATEGORY_HUE.get(cat, 200)
    sat = 0.45 + 0.45 * t
    light = 0.72 - 0.40 * t
    return _hsl_to_hex(hue, sat, light)


def _label(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " "))


def _short_label(name: str, max_chars: int = 14) -> str:
    lab = _label(name)
    return lab if len(lab) <= max_chars else lab[: max_chars - 1] + "…"


def _add_square(ax, x: float, y: float, half: float, *, face, edge, lw, z) -> None:
    ax.add_patch(
        Rectangle(
            (x - half, y - half),
            2 * half,
            2 * half,
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            zorder=z,
        )
    )


def pack_top(
    halves: np.ndarray,
    rng: np.random.Generator,
    *,
    n_iter: int = 500,
) -> np.ndarray:
    """Pack top-k squares: largest near centre, others as satellites."""
    n = len(halves)
    order = np.argsort(-halves)
    xy = np.zeros((n, 2), dtype=float)
    for k, idx in enumerate(order):
        if k == 0:
            xy[idx] = (0.0, 0.0)
            continue
        ang = 2 * np.pi * (k - 1) / max(n - 1, 1) + rng.uniform(-0.25, 0.25)
        ring = halves[order[0]] + halves[idx] * 0.45 + 0.14 * ((k - 1) % 3)
        xy[idx] = ring * np.array([np.cos(ang), np.sin(ang)])
        xy[idx] += rng.normal(0.0, 0.05, size=2)

    for _ in range(n_iter):
        xy[order[0]] *= 0.85
        for i in range(n):
            for j in range(i + 1, n):
                dx = float(xy[j, 0] - xy[i, 0])
                dy = float(xy[j, 1] - xy[i, 1])
                gap_x = halves[i] + halves[j] - abs(dx)
                gap_y = halves[i] + halves[j] - abs(dy)
                if gap_x > 0 and gap_y > 0:
                    # Separate along the smaller overlap axis.
                    if gap_x < gap_y:
                        push = 0.55 * gap_x * (1.0 if dx >= 0 else -1.0)
                        wi = halves[j] / (halves[i] + halves[j])
                        wj = halves[i] / (halves[i] + halves[j])
                        xy[i, 0] -= push * wi
                        xy[j, 0] += push * wj
                    else:
                        push = 0.55 * gap_y * (1.0 if dy >= 0 else -1.0)
                        wi = halves[j] / (halves[i] + halves[j])
                        wj = halves[i] / (halves[i] + halves[j])
                        xy[i, 1] -= push * wi
                        xy[j, 1] += push * wj
    return xy


def place_neutrals(
    top_xy: np.ndarray,
    top_h: np.ndarray,
    n_neutral: int,
    h_neu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Scatter small outline squares around / lightly over the cluster."""
    extent = float(np.max(np.linalg.norm(top_xy, axis=1) + top_h * np.sqrt(2))) + 0.15
    xy = np.zeros((n_neutral, 2), dtype=float)
    for i in range(n_neutral):
        for _ in range(40):
            if rng.random() < 0.35:
                ang = rng.uniform(0, 2 * np.pi)
                rad = rng.uniform(0.05, max(top_h.max() * 0.55, 0.1))
            else:
                ang = rng.uniform(0, 2 * np.pi)
                rad = rng.uniform(extent * 0.35, extent * 1.05)
            cand = rad * np.array([np.cos(ang), np.sin(ang)])
            if i == 0 or np.min(np.linalg.norm(xy[:i] - cand, axis=1)) > h_neu * 2.2:
                xy[i] = cand
                break
        else:
            ang = rng.uniform(0, 2 * np.pi)
            xy[i] = extent * rng.uniform(0.5, 1.1) * np.array([np.cos(ang), np.sin(ang)])
    return xy


def top_halves(abs_coef: np.ndarray, vmax: float) -> np.ndarray:
    s = np.sqrt(np.maximum(abs_coef, 0.0) / max(vmax, 1e-9))
    return 0.18 + 0.72 * s


def draw_panel(
    ax,
    df: pd.DataFrame,
    *,
    title: str,
    rng: np.random.Generator,
    shared_vmax: float,
    categories: dict[str, str],
    scales: dict[str, tuple[float, float]],
) -> None:
    ranked = df.sort_values(
        ["abs_coefficient", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    top = ranked.head(FEATURE_TOP_K).copy()
    rest = ranked.iloc[FEATURE_TOP_K:].copy()

    top_abs = top["abs_coefficient"].to_numpy(dtype=float)
    halves = top_halves(top_abs, shared_vmax)
    xy = pack_top(halves, rng)

    h_neu = 0.055
    neu_xy = place_neutrals(xy, halves, len(rest), h_neu, rng)

    ax.set_facecolor(BG)

    for (x, y) in neu_xy:
        _add_square(
            ax, x, y, h_neu, face="none", edge=NEUTRAL_EDGE, lw=1.0, z=1
        )

    for k, (_, row) in enumerate(top.iterrows()):
        x, y = xy[k]
        half = float(halves[k])
        feat = str(row["feature"])
        face = colour_for_feature(
            feat, float(row["abs_coefficient"]), categories, scales
        )
        _add_square(ax, x, y, half, face=face, edge=TOP_EDGE, lw=1.6, z=3)
        lab = _short_label(feat)
        val = float(row["abs_coefficient"])
        txt = f"{k + 1}. {lab}\n{val:.2f}"
        fs = 6.2 if half >= 0.45 else (5.2 if half >= 0.28 else 4.4)
        if half >= 0.26:
            ax.text(
                x,
                y,
                txt,
                ha="center",
                va="center",
                fontsize=fs,
                color=INK,
                zorder=4,
                linespacing=1.05,
                fontweight="bold" if k == 0 else "normal",
            )
        else:
            ang = np.arctan2(y, x + 1e-9)
            ox = x + (half + 0.10) * np.cos(ang)
            oy = y + (half + 0.10) * np.sin(ang)
            ax.text(
                ox,
                oy,
                txt,
                ha="center",
                va="center",
                fontsize=4.2,
                color=INK_MUTED,
                zorder=4,
                linespacing=1.0,
            )

    all_xy = np.vstack([xy, neu_xy]) if len(neu_xy) else xy
    all_h = (
        np.concatenate([halves, np.full(len(neu_xy), h_neu)]) if len(neu_xy) else halves
    )
    lim = float(np.max(np.linalg.norm(all_xy, axis=1) + all_h * np.sqrt(2))) * 1.18
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11, color=INK, pad=7, fontweight="bold")


def main() -> None:
    reps = []
    for r, direction in enumerate(DIRECTIONS, start=1):
        path = VIZ_DIR / direction / "feature_importance.csv"
        if not path.exists():
            raise SystemExit(f"Missing {path}; run compute_att_stability.py first.")
        reps.append((r, direction, pd.read_csv(path)))

    categories = load_feature_categories()
    scales = category_shade_scales([df for _, _, df in reps], categories)
    shared_vmax = max(float(df["abs_coefficient"].max()) for _, _, df in reps)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        N_REPS,
        figsize=(3.05 * N_REPS, 4.35),
        facecolor=BG,
    )
    if N_REPS == 1:
        axes = [axes]

    for ax, (r, direction, df) in zip(axes, reps):
        rng = np.random.default_rng(SEED + 17 * r)
        draw_panel(
            ax,
            df,
            title=direction.title(),
            rng=rng,
            shared_vmax=shared_vmax,
            categories=categories,
            scales=scales,
        )

    fig.suptitle(
        "Feature attribution stability — category shade clusters (top-10)",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "Rectangle size encodes |SMILE coefficient| (shared scale). "
        "Each category shares one hue; shade darkens with |coef|. "
        "Outline = features outside top-10. Cardinal neighbor LSOAs just outside Hull; not causal.",
        ha="center",
        fontsize=7.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0.01, 0.045, 0.99, 0.92))

    out_png = OUT_DIR / "feature_stability_schematic_4dirs.png"
    out_pdf = OUT_DIR / "feature_stability_schematic_4dirs.pdf"
    fig.savefig(out_png, dpi=280, facecolor=BG, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_pdf, facecolor=BG, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
