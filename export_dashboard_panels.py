"""Export the dashboard's stakeholder evidence panels as images for the Chapter 6 cards.

The cards show what the dashboard shows: each image is drawn by the same builder the dashboard
calls (dashboard.views.stakeholder_figures.STAKEHOLDER_PANELS) for the same stakeholder and LSOA,
and the caption is the dashboard's own caption text.

    python export_dashboard_panels.py --stakeholder police_short --lsoa E01033104

Panels are redrawn at PRINT_WIDTH inches (the dashboard draws them at 6.4), so their 8-10 pt
text stays readable when the card places them at half the page width.

Writes outputs/stakeholder_narratives/dashboard_<stakeholder>_<lsoa>_<k>.png (one per panel)
and dashboard_<stakeholder>_<lsoa>_captions.tex (\\renewcommand{\\dashCapA} ... per panel).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from dashboard.data.stakeholder_data import load_stakeholder_data
from dashboard.llm import card_mode as cm
from dashboard.views.stakeholder_figures import STAKEHOLDER_PANELS, is_path

DEST = Path(__file__).resolve().parent / "outputs" / "stakeholder_narratives"
PRINT_WIDTH = 4.0  # inches; placed at ~2.95 in (half an A4 text width), 8 pt text prints at ~6 pt


def export(stakeholder: str, lsoa: str, width: float = PRINT_WIDTH) -> list[Path]:
    d = load_stakeholder_data()
    contrast = d.default_contrast(lsoa)
    paths, captions = [], []
    for k, build in enumerate(STAKEHOLDER_PANELS[stakeholder](d, lsoa, contrast), start=1):
        obj, title, caption = build()
        path = DEST / f"dashboard_{stakeholder}_{lsoa}_{k}.png"
        if is_path(obj):
            path.write_bytes(Path(obj).read_bytes())
        else:
            w, h = obj.get_size_inches()
            obj.set_size_inches(width, h * width / w)
            obj.savefig(path, dpi=300, bbox_inches="tight", facecolor=obj.get_facecolor())
            plt.close(obj)
        paths.append(path)
        captions.append(rf"\textbf{{{cm.tex_escape(title)}}} {cm.tex_escape(caption)}")
        print(f"{path.name}: {title} {caption}")
    lines = [f"% Dashboard captions for {stakeholder} / {lsoa} (export_dashboard_panels.py). Do not edit by hand."]
    lines += [rf"\renewcommand{{\dashCap{chr(64 + k)}}}{{{c}}}" for k, c in enumerate(captions, start=1)]
    tex = DEST / f"dashboard_{stakeholder}_{lsoa}_captions.tex"
    tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("->", tex.name)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stakeholder", required=True, choices=sorted(STAKEHOLDER_PANELS))
    parser.add_argument("--lsoa", default=cm.CENTRE)
    parser.add_argument("--width", type=float, default=PRINT_WIDTH,
                        help="print width in inches; smaller gives larger text on the card")
    args = parser.parse_args()
    export(args.stakeholder, args.lsoa, args.width)
