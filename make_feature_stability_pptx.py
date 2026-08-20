"""Editable PowerPoint: Geo top-20 ranking maps + Feature top-10 coefficient clusters.

One slide pair per cardinal-neighbor direction addition. Feature rectangles are
native PowerPoint shapes (rank, name, |coefficient|), matching the cluster layout.
Geo maps are the ranking choropleths with labels 1–20.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from compute_att_stability import (
    GEO_TOP_K,
    save_feature_importance_ranking,
    save_geo_rank_map,
)
from geosmile_hull_twobranch import FEATURE_LABELS
from hull_boundaries import load_boundaries
from hull_expert_topk import FEATURE_TOP_K
from make_feature_stability_schematic import (
    CATEGORY_HUE,
    DIRECTIONS,
    N_REPS,
    OUT_DIR,
    SEED,
    VIZ_DIR,
    category_shade_scales,
    colour_for_feature,
    load_feature_categories,
    pack_top,
    place_neutrals,
    top_halves,
)

ROOT = Path(__file__).resolve().parent
NEIGHBOR_GEOJSON = ROOT / "data" / "cardinal_neighbor_lsoas.geojson"

BG = RGBColor(0x00, 0x00, 0x00)
LIGHT_BG = RGBColor(0xF4, 0xF4, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x12, 0x17, 0x1C)
MUTED = RGBColor(0xB0, 0xB8, 0xC0)
MUTED_DARK = RGBColor(0x57, 0x64, 0x6D)
NEUTRAL_EDGE = RGBColor(0xC8, 0xC8, 0xC8)


def _rgb(hex_colour: str) -> RGBColor:
    h = hex_colour.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _label(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " "))


def _short_label(name: str, max_chars: int = 16) -> str:
    lab = _label(name)
    return lab if len(lab) <= max_chars else lab[: max_chars - 1] + "…"


def _contrast_ink(fill: RGBColor) -> RGBColor:
    y = 0.2126 * fill[0] + 0.7152 * fill[1] + 0.0722 * fill[2]
    return INK if y >= 145 else WHITE


def _set_slide_bg(slide, colour: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = colour


def _set_run(run, *, size_pt: float, bold: bool = False, colour: RGBColor = WHITE) -> None:
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.color.rgb = colour
    run.font.name = "Calibri"


def _add_textbox(
    slide,
    left,
    top,
    width,
    height,
    text: str,
    *,
    size_pt: float,
    bold: bool = False,
    colour: RGBColor = WHITE,
    align=PP_ALIGN.CENTER,
) -> None:
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    _set_run(run, size_pt=size_pt, bold=bold, colour=colour)
    return box


def _style_shape_fill_line(shape, *, fill: RGBColor | None, line: RGBColor, line_pt: float) -> None:
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(line_pt)


def _anchor_middle(shape) -> None:
    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    try:
        tf.anchor = MSO_ANCHOR.MIDDLE
    except Exception:
        pass
    body = tf._txBody
    bodyPr = body.find(qn("a:bodyPr"))
    if bodyPr is not None:
        bodyPr.set("anchor", "ctr")


def _set_rank_coef_text(
    shape,
    *,
    rank: int,
    label: str,
    coef: float,
    fill: RGBColor,
    side_in: float,
) -> None:
    """Rank + name on one line, |coefficient| large underneath (example cluster style)."""
    ink = _contrast_ink(fill)
    if side_in >= 1.35:
        name_pt, coef_pt, max_chars = 14, 28, 22
    elif side_in >= 0.95:
        name_pt, coef_pt, max_chars = 11, 18, 18
    elif side_in >= 0.65:
        name_pt, coef_pt, max_chars = 9, 14, 16
    else:
        name_pt, coef_pt, max_chars = 7, 10, 14
    lab = label if len(label) <= max_chars else label[: max_chars - 1] + "…"
    tf = shape.text_frame
    tf.clear()
    _anchor_middle(shape)
    p0 = tf.paragraphs[0]
    p0.alignment = PP_ALIGN.CENTER
    r0 = p0.add_run()
    r0.text = f"{rank}. {lab}"
    _set_run(r0, size_pt=name_pt, bold=False, colour=ink)
    p1 = tf.add_paragraph()
    p1.alignment = PP_ALIGN.CENTER
    r1 = p1.add_run()
    r1.text = f"{coef:.2f}"
    _set_run(r1, size_pt=coef_pt, bold=True, colour=ink)


def _panel_geometry(df: pd.DataFrame, *, rep: int, shared_vmax: float):
    ranked = df.sort_values(
        ["abs_coefficient", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    top = ranked.head(FEATURE_TOP_K).copy()
    rest = ranked.iloc[FEATURE_TOP_K:].copy()
    top_abs = top["abs_coefficient"].to_numpy(dtype=float)
    halves = top_halves(top_abs, shared_vmax)
    rng = np.random.default_rng(SEED + 17 * rep)
    xy = pack_top(halves, rng)
    h_neu = 0.055
    neu_xy = place_neutrals(xy, halves, len(rest), h_neu, rng)
    all_xy = np.vstack([xy, neu_xy]) if len(neu_xy) else xy
    all_h = (
        np.concatenate([halves, np.full(len(neu_xy), h_neu)]) if len(neu_xy) else halves
    )
    lim = float(np.max(np.linalg.norm(all_xy, axis=1) + all_h * np.sqrt(2))) * 1.18
    return top, rest, halves, xy, neu_xy, h_neu, lim


def _data_to_slide(
    x: float,
    y: float,
    half: float,
    *,
    lim: float,
    origin_left,
    origin_top,
    panel_w,
    panel_h,
):
    """Map data coords (origin centre, y up) -> slide EMUs (origin top-left, y down)."""
    u = (x + lim) / (2 * lim)
    v = (lim - y) / (2 * lim)
    s = panel_w / (2 * lim)
    cx = origin_left + u * panel_w
    cy = origin_top + v * panel_h
    side = 2 * half * s
    left = cx - side / 2
    top = cy - side / 2
    return left, top, side


def draw_cluster_on_slide(
    slide,
    df: pd.DataFrame,
    *,
    rep: int,
    shared_vmax: float,
    categories: dict[str, str],
    scales: dict[str, tuple[float, float]],
    origin_left,
    origin_top,
    panel_w,
    panel_h,
    title: str,
    label_prefix: str = "",
    show_neutrals: bool = True,
    title_colour: RGBColor = WHITE,
) -> None:
    top, rest, halves, xy, neu_xy, h_neu, lim = _panel_geometry(
        df, rep=rep, shared_vmax=shared_vmax
    )

    if title:
        _add_textbox(
            slide,
            origin_left,
            origin_top - Inches(0.32),
            panel_w,
            Inches(0.28),
            title,
            size_pt=14,
            bold=True,
            colour=title_colour,
        )

    if show_neutrals:
        for (x, y), feat in zip(neu_xy, rest["feature"].astype(str)):
            left, top_pos, side = _data_to_slide(
                float(x),
                float(y),
                h_neu,
                lim=lim,
                origin_left=origin_left,
                origin_top=origin_top,
                panel_w=panel_w,
                panel_h=panel_h,
            )
            shp = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE, left, top_pos, side, side
            )
            shp.name = f"{label_prefix}neutral_{feat}"
            _style_shape_fill_line(shp, fill=None, line=NEUTRAL_EDGE, line_pt=0.75)

    for k, (_, row) in enumerate(top.iterrows()):
        x, y = float(xy[k, 0]), float(xy[k, 1])
        half = float(halves[k])
        left, top_pos, side = _data_to_slide(
            x,
            y,
            half,
            lim=lim,
            origin_left=origin_left,
            origin_top=origin_top,
            panel_w=panel_w,
            panel_h=panel_h,
        )
        feat = str(row["feature"])
        face = _rgb(
            colour_for_feature(
                feat, float(row["abs_coefficient"]), categories, scales
            )
        )
        shp = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, left, top_pos, side, side
        )
        shp.name = f"{label_prefix}rank{k+1}_{feat}"
        _style_shape_fill_line(shp, fill=face, line=WHITE, line_pt=1.25)
        _set_rank_coef_text(
            shp,
            rank=k + 1,
            label=_label(feat),
            coef=float(row["abs_coefficient"]),
            fill=face,
            side_in=side / 914400,
        )


def _load_meta(direction: str) -> dict:
    path = VIZ_DIR / direction / "snapshot_meta.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _direction_caption(direction: str, meta: dict, idx: int) -> str:
    added = meta.get("added_lsoa", "")
    name = meta.get("added_name", "")
    bits = [f"Direction {idx} — {direction.title()} addition"]
    if added:
        bits.append(added)
    if name:
        bits.append(name)
    return "  ·  ".join(bits)


def ensure_geo_maps() -> dict[str, Path]:
    """Rebuild Geo top-20 ranking PNGs from saved coefficients (no SMILE recompute)."""
    base_geo = pd.read_csv(VIZ_DIR / "baseline" / "geo_coefficients.csv")
    hull_bounds, _ = load_boundaries(sorted(base_geo["lsoa21cd"].astype(str)))
    name_lookup = {
        str(c): str(n)
        for c, n in zip(
            base_geo["lsoa21cd"].astype(str),
            base_geo.get("lsoa21nm", base_geo["lsoa21cd"]).astype(str),
        )
    }
    neigh = None
    if NEIGHBOR_GEOJSON.exists():
        neigh = gpd.read_file(NEIGHBOR_GEOJSON)
        if neigh.crs is None:
            neigh = neigh.set_crs("EPSG:27700")
        else:
            neigh = neigh.to_crs(hull_bounds.crs)

    paths: dict[str, Path] = {}
    for idx, direction in enumerate(DIRECTIONS, start=1):
        folder = VIZ_DIR / direction
        geo = pd.read_csv(folder / "geo_coefficients.csv")
        meta = _load_meta(direction)
        hull_codes = set(hull_bounds["lsoa21cd"].astype(str))
        extra_codes = [
            str(c) for c in geo["lsoa21cd"].astype(str) if str(c) not in hull_codes
        ]
        map_bounds = hull_bounds[["lsoa21cd", "lsoa21nm", "geometry"]].copy()
        if extra_codes and neigh is not None:
            extra = neigh[neigh["lsoa21cd"].astype(str).isin(extra_codes)]
            keep = [c for c in ("lsoa21cd", "lsoa21nm", "geometry") if c in extra.columns]
            extra = extra[keep]
            if not extra.empty:
                map_bounds = gpd.GeoDataFrame(
                    pd.concat([map_bounds, extra], ignore_index=True),
                    crs=hull_bounds.crs,
                )
        if "lsoa21nm" in geo.columns:
            name_lookup.update(
                {
                    str(c): str(n)
                    for c, n in zip(
                        geo["lsoa21cd"].astype(str), geo["lsoa21nm"].astype(str)
                    )
                }
            )
        png = folder / f"geo_top{GEO_TOP_K}_map.png"
        added = meta.get("added_lsoa", "")
        added_name = meta.get("added_name", "")
        jacc = meta.get("geo_jaccard")
        footer_bits = [
            f"{direction.title()} addition",
            f"added {added}" if added else None,
            added_name or None,
            f"top-{GEO_TOP_K} Jaccard vs baseline={jacc:.3f}"
            if jacc is not None
            else None,
            "Not causal.",
        ]
        added_list = [str(added)] if added else extra_codes
        save_geo_rank_map(
            geo,
            map_bounds,
            png,
            title=(
                f"{direction.title()} addition — Geo SMILE top-{GEO_TOP_K} ranking"
            ),
            footer=". ".join(b for b in footer_bits if b),
            name_lookup=name_lookup,
            added_codes=added_list,
        )
        paths[direction] = png
        print(f"  Geo map {idx}/{len(DIRECTIONS)}: {png.name}")
    return paths


def ensure_feature_ranking_png(
    folder: Path,
    *,
    title: str,
    footer: str,
) -> Path:
    """Rebuild the Feature |coefficient| bar ranking from the saved CSV."""
    csv_path = folder / "feature_importance.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}")
    png = folder / "feature_importance_ranking.png"
    save_feature_importance_ranking(
        pd.read_csv(csv_path),
        png,
        title=title,
        footer=footer,
    )
    return png


def add_feature_ranking_slide(prs, blank, png: Path, *, title: str, footer: str) -> None:
    s = prs.slides.add_slide(blank)
    _set_slide_bg(s, LIGHT_BG)
    _add_textbox(
        s,
        Inches(0.4),
        Inches(0.08),
        Inches(12.5),
        Inches(0.38),
        title,
        size_pt=18,
        bold=True,
        colour=INK,
    )
    _add_picture_centered(
        s, png, top=Inches(0.48), height=Inches(6.55), slide_width=prs.slide_width
    )
    _add_textbox(
        s,
        Inches(0.4),
        Inches(7.12),
        Inches(12.5),
        Inches(0.28),
        footer,
        size_pt=11,
        colour=MUTED_DARK,
    )


def _fmt3(value: float) -> str:
    return f"{float(value):.3f}"


PRED_ACCEPT_MIN = 0.99


def add_metrics_table_slide(prs, blank, reps, metas) -> None:
    """Native PowerPoint table: Jaccard, |coef| Spearman, Pred. Spearman, Accepted."""
    s = prs.slides.add_slide(blank)
    _set_slide_bg(s, LIGHT_BG)
    _add_textbox(
        s,
        Inches(0.4),
        Inches(0.18),
        Inches(12.5),
        Inches(0.4),
        "ATT stability — four direction additions",
        size_pt=20,
        bold=True,
        colour=INK,
    )
    headers = [
        "Addition",
        "Added LSOA",
        "Branch",
        "Top-k Jaccard",
        "|coef| Spearman",
        "Pred. Spearman",
        "Accepted",
        "Extra rank",
        "In top-20",
    ]
    body: list[list[str]] = []
    feat_j, feat_s, feat_p = [], [], []
    geo_j, geo_s, geo_p = [], [], []
    n_ok_f = n_ok_g = n_extra_in = 0
    csv_rows = []
    for r, direction, _ in reps:
        m = metas[direction]
        code = str(m.get("added_lsoa", ""))
        name = str(m.get("added_name", "")).replace("East Riding of Yorkshire ", "")
        lsoa = f"{code}  {name}".strip()
        fj = float(m.get("feature_jaccard", float("nan")))
        fs = float(m.get("feature_spearman", float("nan")))
        fp = float(m.get("feature_pred_spearman", float("nan")))
        gj = float(m.get("geo_jaccard", float("nan")))
        gs = float(m.get("geo_spearman", float("nan")))
        gp = float(m.get("geo_pred_spearman", float("nan")))
        feat_j.append(fj)
        feat_s.append(fs)
        feat_p.append(fp)
        geo_j.append(gj)
        geo_s.append(gs)
        geo_p.append(gp)
        extra_rank = m.get("geo_extra_rank", float("nan"))
        extra_in = m.get("geo_extra_in_top20", 0.0)
        extra_rank_s = (
            str(int(extra_rank)) if np.isfinite(float(extra_rank)) else "—"
        )
        extra_in_s = "Yes" if extra_in else "No"
        ok_f = fp >= PRED_ACCEPT_MIN
        ok_g = gp >= PRED_ACCEPT_MIN
        n_ok_f += int(ok_f)
        n_ok_g += int(ok_g)
        n_extra_in += int(bool(extra_in))
        body.append(
            [
                direction.title(),
                lsoa,
                "Feature",
                _fmt3(fj),
                _fmt3(fs),
                _fmt3(fp),
                "Yes" if ok_f else "No",
                "—",
                "—",
            ]
        )
        body.append(
            [
                direction.title(),
                lsoa,
                "Geo",
                _fmt3(gj),
                _fmt3(gs),
                _fmt3(gp),
                "Yes" if ok_g else "No",
                extra_rank_s,
                extra_in_s,
            ]
        )
        csv_rows.append(
            {
                "addition": direction,
                "lsoa21cd": code,
                "lsoa21nm": m.get("added_name", ""),
                "branch": "Feature",
                "top_k_jaccard": fj,
                "abs_coef_spearman": fs,
                "pred_spearman": fp,
                "accepted": ok_f,
                "k": FEATURE_TOP_K,
            }
        )
        csv_rows.append(
            {
                "addition": direction,
                "lsoa21cd": code,
                "lsoa21nm": m.get("added_name", ""),
                "branch": "Geo",
                "top_k_jaccard": gj,
                "abs_coef_spearman": gs,
                "pred_spearman": gp,
                "accepted": ok_g,
                "k": GEO_TOP_K,
                "geo_extra_rank": extra_rank,
                "geo_extra_in_top20": bool(extra_in),
            }
        )
    n = len(reps)
    body.append(
        [
            "Mean",
            "Four additions",
            "Feature",
            _fmt3(float(np.mean(feat_j))),
            _fmt3(float(np.mean(feat_s))),
            _fmt3(float(np.mean(feat_p))),
            f"{n_ok_f}/{n}",
            "—",
            "—",
        ]
    )
    body.append(
        [
            "Mean",
            "Four additions",
            "Geo",
            _fmt3(float(np.mean(geo_j))),
            _fmt3(float(np.mean(geo_s))),
            _fmt3(float(np.mean(geo_p))),
            f"{n_ok_g}/{n}",
            "—",
            f"{n_extra_in}/{n}",
        ]
    )
    csv_path = VIZ_DIR / "direction_addition_metrics.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    nrows = 1 + len(body)
    table = s.shapes.add_table(
        nrows, len(headers), Inches(0.4), Inches(0.72), Inches(12.5), Inches(6.05)
    ).table
    widths = [1.15, 2.15, 1.05, 1.35, 1.5, 1.4, 1.05, 1.15, 1.1]
    for i, w in enumerate(widths):
        table.columns[i].width = Inches(w)

    def _paint(cell, text, *, header=False, mean=False, centre=False):
        cell.text = text
        fill = cell.fill
        fill.solid()
        if header:
            fill.fore_color.rgb = INK
            colour = WHITE
            bold = True
        elif mean:
            fill.fore_color.rgb = RGBColor(0xE8, 0xEC, 0xF0)
            colour = INK
            bold = True
        else:
            fill.fore_color.rgb = WHITE
            colour = INK
            bold = False
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER if centre or header else PP_ALIGN.LEFT
            for run in p.runs:
                run.font.size = Pt(12 if header else 13)
                run.font.bold = bold
                run.font.color.rgb = colour
                run.font.name = "Calibri"
        cell.text_frame.word_wrap = True

    for j, h in enumerate(headers):
        _paint(table.cell(0, j), h, header=True, centre=True)
    numeric = {3, 4, 5, 6, 7, 8}
    for i, row in enumerate(body, start=1):
        is_mean = row[0] == "Mean"
        for j, val in enumerate(row):
            _paint(
                table.cell(i, j),
                val,
                mean=is_mean,
                centre=j in numeric,
            )

    _add_textbox(
        s,
        Inches(0.4),
        Inches(6.88),
        Inches(12.5),
        Inches(0.45),
        "Accepted if Hull-row prediction Spearman ≥ 0.99. Feature k=10, Geo k=20. "
        "Same SMILE seed (42) as the Hull-only baseline. "
        "Added LSOA is a real Geo player (fitted coefficient). "
        "One addition at a time. Models remain Hull-trained. Not causal.",
        size_pt=11,
        colour=MUTED_DARK,
        align=PP_ALIGN.LEFT,
    )


def _add_picture_centered(slide, path: Path, *, top, height, slide_width) -> None:
    pic = slide.shapes.add_picture(str(path), Inches(0), top, height=height)
    pic.left = int((slide_width - pic.width) / 2)


def main() -> None:
    reps = []
    metas = {}
    for r, direction in enumerate(DIRECTIONS, start=1):
        path = VIZ_DIR / direction / "feature_importance.csv"
        if not path.exists():
            raise SystemExit(f"Missing {path}")
        reps.append((r, direction, pd.read_csv(path)))
        metas[direction] = _load_meta(direction)

    print("Building Geo top-20 ranking maps…")
    geo_pngs = ensure_geo_maps()

    baseline_feat_path = VIZ_DIR / "baseline" / "feature_importance.csv"
    if not baseline_feat_path.exists():
        raise SystemExit(f"Missing {baseline_feat_path}")
    baseline_df = pd.read_csv(baseline_feat_path)
    print("Building Feature importance ranking figures…")
    baseline_feat_png = ensure_feature_ranking_png(
        VIZ_DIR / "baseline",
        title="Feature SMILE importance ranking\nHull-only baseline",
        footer="Hull-only baseline. Top-10 highlighted. Not causal.",
    )
    print(f"  Baseline Feature ranking: {baseline_feat_png.name}")

    categories = load_feature_categories()
    feat_frames = [baseline_df] + [df for _, _, df in reps]
    scales = category_shade_scales(feat_frames, categories)
    shared_vmax = max(float(df["abs_coefficient"].max()) for df in feat_frames)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    sw = prs.slide_width

    # --- Title ---
    cover = prs.slides.add_slide(blank)
    _set_slide_bg(cover, BG)
    _add_textbox(
        cover,
        Inches(0.7),
        Inches(2.2),
        Inches(12),
        Inches(1.1),
        "Direction additions — Geo ranking maps and Feature coefficients",
        size_pt=28,
        bold=True,
        colour=WHITE,
    )
    _add_textbox(
        cover,
        Inches(0.7),
        Inches(3.4),
        Inches(12),
        Inches(1.4),
        "Four one-at-a-time cardinal-neighbor LSOA additions (N / E / S / W).\n"
        f"Geo: top-{GEO_TOP_K} LSOAs by |SMILE coefficient|, ranks labelled on the map.\n"
        f"Feature: top-{FEATURE_TOP_K} predictors as sized rectangles with rank, name, and |coefficient|.",
        size_pt=16,
        colour=MUTED,
        align=PP_ALIGN.LEFT,
    )

    add_metrics_table_slide(prs, blank, reps, metas)

    add_feature_ranking_slide(
        prs,
        blank,
        baseline_feat_png,
        title=f"Hull-only baseline — Feature SMILE importance ranking (top-{FEATURE_TOP_K} highlighted)",
        footer=(
            "Horizontal bars are |SMILE coefficient| for all Feature players. "
            "Pink bars are the top-10 set used in the Jaccard comparison. Not causal."
        ),
    )

    base_feat = prs.slides.add_slide(blank)
    _set_slide_bg(base_feat, BG)
    _add_textbox(
        base_feat,
        Inches(0.4),
        Inches(0.12),
        Inches(12.5),
        Inches(0.38),
        f"Hull-only baseline — Feature top-{FEATURE_TOP_K}",
        size_pt=20,
        bold=True,
        colour=WHITE,
    )
    draw_cluster_on_slide(
        base_feat,
        baseline_df,
        rep=0,
        shared_vmax=shared_vmax,
        categories=categories,
        scales=scales,
        origin_left=Inches(1.7),
        origin_top=Inches(0.72),
        panel_w=Inches(9.9),
        panel_h=Inches(6.05),
        title="",
        label_prefix="baseline_",
        show_neutrals=False,
    )
    _add_textbox(
        base_feat,
        Inches(0.4),
        Inches(7.08),
        Inches(12.5),
        Inches(0.32),
        "Hull-only baseline (no added LSOA). Each rectangle is an editable PowerPoint shape. "
        "Size = |SMILE coefficient|. Number on the box is the rank. Not causal.",
        size_pt=11,
        colour=MUTED,
    )

    # --- 2×2 Geo overview ---
    geo_ov = prs.slides.add_slide(blank)
    _set_slide_bg(geo_ov, LIGHT_BG)
    _add_textbox(
        geo_ov,
        Inches(0.35),
        Inches(0.12),
        Inches(12.6),
        Inches(0.4),
        f"Geo SMILE top-{GEO_TOP_K} ranking maps — all four direction additions",
        size_pt=18,
        bold=True,
        colour=INK,
    )
    positions = [
        (Inches(0.25), Inches(0.58)),
        (Inches(6.75), Inches(0.58)),
        (Inches(0.25), Inches(4.05)),
        (Inches(6.75), Inches(4.05)),
    ]
    for (r, direction, _), (left, top) in zip(reps, positions):
        pic = geo_ov.shapes.add_picture(
            str(geo_pngs[direction]), left, top, width=Inches(6.3)
        )
        # Keep maps from overflowing the lower half
        max_h = Inches(3.35)
        if pic.height > max_h:
            scale = max_h / pic.height
            pic.height = int(pic.height * scale)
            pic.width = int(pic.width * scale)

    # --- Full Geo map per direction ---
    for r, direction, _ in reps:
        meta = metas[direction]
        s = prs.slides.add_slide(blank)
        _set_slide_bg(s, LIGHT_BG)
        _add_textbox(
            s,
            Inches(0.4),
            Inches(0.08),
            Inches(12.5),
            Inches(0.38),
            f"{_direction_caption(direction, meta, r)}  —  Geo top-{GEO_TOP_K} ranking"
            + (
                f"  ·  extra rank {int(meta['geo_extra_rank'])}"
                f" ({'in' if meta.get('geo_extra_in_top20') else 'out of'} top-{GEO_TOP_K})"
                if np.isfinite(float(meta.get("geo_extra_rank", float("nan"))))
                else ""
            ),
            size_pt=18,
            bold=True,
            colour=INK,
        )
        _add_picture_centered(
            s, geo_pngs[direction], top=Inches(0.48), height=Inches(6.55), slide_width=sw
        )
        _add_textbox(
            s,
            Inches(0.4),
            Inches(7.12),
            Inches(12.5),
            Inches(0.28),
            "Coloured LSOAs are the 20 most important by |Geo SMILE coefficient|. "
            "Numbers on the map are ranks 1–20. Not causal.",
            size_pt=11,
            colour=MUTED_DARK,
        )

    # --- Feature 4-up overview ---
    slide = prs.slides.add_slide(blank)
    _set_slide_bg(slide, BG)
    _add_textbox(
        slide,
        Inches(0.3),
        Inches(0.12),
        Inches(12.7),
        Inches(0.38),
        f"Feature top-{FEATURE_TOP_K} — Hull-only baseline and four direction additions",
        size_pt=18,
        bold=True,
        colour=WHITE,
    )
    overview_panels = [(0, "baseline", baseline_df)] + list(reps)
    margin_l = Inches(0.16)
    margin_t = Inches(0.78)
    gap = Inches(0.10)
    n_ov = len(overview_panels)
    usable_w = prs.slide_width - 2 * margin_l - (n_ov - 1) * gap
    panel_w = usable_w / n_ov
    panel_h = Inches(5.75)
    for i, (r, direction, df) in enumerate(overview_panels):
        left = margin_l + i * (panel_w + gap)
        title = "Baseline" if direction == "baseline" else f"{r}. {direction.title()}"
        draw_cluster_on_slide(
            slide,
            df,
            rep=r,
            shared_vmax=shared_vmax,
            categories=categories,
            scales=scales,
            origin_left=left,
            origin_top=margin_t,
            panel_w=panel_w,
            panel_h=panel_h,
            title=title,
            label_prefix=f"overview_{direction}_",
            show_neutrals=False,
        )
    _add_textbox(
        slide,
        Inches(0.3),
        Inches(7.08),
        Inches(12.7),
        Inches(0.32),
        "Box size encodes |SMILE coefficient| (shared scale across directions). "
        "Each box: rank, feature name, coefficient. Editable shapes — not an image. Not causal.",
        size_pt=10,
        colour=MUTED,
    )

    # --- Full Feature cluster per direction (example layout) ---
    for r, direction, df in reps:
        meta = metas[direction]
        s = prs.slides.add_slide(blank)
        _set_slide_bg(s, BG)
        _add_textbox(
            s,
            Inches(0.4),
            Inches(0.12),
            Inches(12.5),
            Inches(0.38),
            f"{_direction_caption(direction, meta, r)}  —  Feature top-{FEATURE_TOP_K}",
            size_pt=20,
            bold=True,
            colour=WHITE,
        )
        draw_cluster_on_slide(
            s,
            df,
            rep=r,
            shared_vmax=shared_vmax,
            categories=categories,
            scales=scales,
            origin_left=Inches(1.7),
            origin_top=Inches(0.72),
            panel_w=Inches(9.9),
            panel_h=Inches(6.05),
            title="",
            label_prefix=f"{direction}_",
            show_neutrals=False,
        )
        _add_textbox(
            s,
            Inches(0.4),
            Inches(7.08),
            Inches(12.5),
            Inches(0.32),
            "Each rectangle is an editable PowerPoint shape. Size = |SMILE coefficient|. "
            "Number on the box is the rank; value underneath is the coefficient. Not causal.",
            size_pt=11,
            colour=MUTED,
        )

    # --- Legend ---
    leg = prs.slides.add_slide(blank)
    _set_slide_bg(leg, BG)
    _add_textbox(
        leg,
        Inches(0.5),
        Inches(0.25),
        Inches(12),
        Inches(0.4),
        "Feature category colour legend (one hue per category; darker = stronger |coef|)",
        size_pt=18,
        bold=True,
        colour=WHITE,
    )
    cat_order = list(CATEGORY_HUE.keys())
    for i, cat in enumerate(cat_order):
        y = Inches(0.85 + i * 0.75)
        _add_textbox(
            leg,
            Inches(0.5),
            y,
            Inches(4.2),
            Inches(0.35),
            cat,
            size_pt=13,
            bold=True,
            colour=WHITE,
            align=PP_ALIGN.LEFT,
        )
        lo, hi = scales.get(cat, (0.0, 1.0))
        for j, t in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
            abs_ex = lo + t * (hi - lo)
            feat = next((f for f, c in categories.items() if c == cat), "lsoa_area_km2")
            col = colour_for_feature(feat, abs_ex, {feat: cat}, {cat: (lo, hi)})
            x = Inches(5.0 + j * 0.7)
            shp = leg.shapes.add_shape(
                MSO_SHAPE.RECTANGLE, x, y, Inches(0.55), Inches(0.55)
            )
            shp.name = f"legend_{cat}_{j}"
            _style_shape_fill_line(shp, fill=_rgb(col), line=WHITE, line_pt=0.75)

    _add_textbox(
        leg,
        Inches(0.5),
        Inches(6.9),
        Inches(12),
        Inches(0.35),
        "Shade scale within each row: light (weaker |coef|) → dark (stronger |coef|).",
        size_pt=11,
        colour=MUTED,
        align=PP_ALIGN.LEFT,
    )

    out = OUT_DIR / "direction_addition_geo20_feature10.pptx"
    try:
        prs.save(out)
    except PermissionError:
        alt = OUT_DIR / "direction_addition_geo20_feature10_new.pptx"
        prs.save(alt)
        out = alt
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
