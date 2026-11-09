"""Archive the detailed (comprehensive) Hull-wide narrative of a stakeholder role.

    python generate_detailed_narrative.py --stakeholder police --env-file path/to/.env

Uses the dashboard's pipeline for the comprehensive mode: the structured evidence, the master prompt
plus the role's comprehensive addendum, GPT-4.1 at temperature 0, and the same figure-by-figure
checks as the cards. Writes outputs/dashboard_final/cards/<stakeholder>_detailed_llm.json, which the
dashboard shows without calling an API.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from dashboard.data.loader import load_hull_bundle
from dashboard.data.schema import CARDS_DIR, STAKEHOLDERS
from dashboard.data.stakeholder_data import load_stakeholder_data
from dashboard.llm import card_mode as cm
from dashboard.llm.attribution_evidence import build_local_attribution_evidence
from dashboard.llm.evidence_builder import build_evidence
from dashboard.llm.openai_provider import OpenAIProvider
from dashboard.llm.prompt import GEOSMILE_SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION
from dashboard.views.stakeholder_figures import STAKEHOLDER_PANELS

ROOT = Path(__file__).resolve().parent
MODEL = "gpt-4.1"
MAX_WORDS = 200
ATTEMPTS = 4
LABEL_AND_SHARE_RULE = """
Crime history is [[n:63%]] of local importance (it is 63.47%, so never write 64%). Call the leading feature
"Crime, 6-month average", never "Crime roll mean 6".
"""
ROLE_RULES = {
    "police": """
The largest rise is 90%, computed from 10.5 to 20.0 crimes a month: never write 91%.
""",
    "manager": LABEL_AND_SHARE_RULE + """
The top area is about [[n:2.6]] times the next: write "about", never "over" or "more than".
The [[n:8%]] combined share belongs to all twelve low-weight features together. Say that twelve features
together carry it, and do not attach it to a shorter list of named features.
""",
    "researcher": LABEL_AND_SHARE_RULE + """
The [[n:128]] of [[n:168]] count is of negative coefficients only: write "negative", never "near-zero".
This role overrides the paragraph rule above: write four short paragraphs of about 45 words each. The fourth
covers explanation quality: Geo surrogate R2 [[n:0.93]], Geo rank Spearman across seeds [[n:0.904]],
neighbour-addition Jaccard [[n:1.0]] and [[n:16/20]] ATT Accuracy, then ends with one limitation.
""",
}
EXTENDED_RULE = """

FINAL LENGTH RULE, overriding every word limit stated above: at most 180 words in total, as three
paragraphs of about 60 words each. Keep only the strongest finding of each numbered point and drop
the rest. Use 4-8 [[t:...]] spans in the whole text. Wrap every figure in [[n:...]]; write small
counts outside a citation as words ("one of the ten"), and do not cite years or dates.
"""
CUT_REQUEST = """

Below is a draft that failed these checks: {problems}.

Return a shortened version of the draft and nothing else. The total must be at most 170 words: delete
whole sentences and lists until it fits, keeping three short paragraphs and the closing limitation.
Add no new claim or figure. Every remaining sentence must mean what it meant in the draft: never merge
two groups of features or places, or move one into a list it was not in. Keep at most 8 [[t:...]] spans and remove the markup from any others.
Every digit must sit inside [[n:...]], apart from LSOA codes such as 029E; delete or spell out any
other number, including ranks, counts of runs and p-value permutations.

DRAFT:
{draft}
"""


def mark_bare_figures(text: str) -> str:
    """Wrap stray numbers as [[n:...]], so that the figure check has to trace them to the evidence too."""
    names = r"Level 4\+?|\b\d{3}[A-Z]\b|\b[Pp]anel \d+|\b(?:W1|R2|W_1)\b|[Tt]op[- ]?\d+|\b\d+(?:st|nd|rd|th)\b"
    bare = re.compile(rf"(?:{names})|(?<![\w.])(\d+(?:\.\d+)?%?)")
    parts = re.split(r"(\[\[[nt]:.+?\]\])", text)
    parts[::2] = [bare.sub(lambda m: f"[[n:{m[1]}]]" if m[1] else m[0], part) for part in parts[::2]]
    return "".join(parts)


def evidence_for(stakeholder: str) -> dict:
    bundle = load_hull_bundle()
    data = load_stakeholder_data()
    spec = cm.SPECS[stakeholder]
    evidence = build_evidence(bundle, stakeholder_id=stakeholder, current_map_view="geo_sensitivity", scope="overall",
                              selected_lsoa=None, user_question=spec["question"])
    cm.add_evidence(evidence, data, stakeholder, "comprehensive")
    evidence["local_attribution_patterns"] = build_local_attribution_evidence(
        bundle.local_features, bundle.global_features, bundle.geometry, tuple(bundle.feature_names), None)
    focus, contrast = cm.detailed_focus(data, stakeholder)
    figures = []
    for build in STAKEHOLDER_PANELS[stakeholder](data, focus, contrast):
        obj, title, caption = build()
        if hasattr(obj, "savefig"):
            plt.close(obj)
        figures.append({"title": title.rstrip("."), "caption": caption})
    evidence["dashboard_figures"] = {"lsoa": focus, "contrast_lsoa": contrast, "figures": figures}
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stakeholder", required=True, choices=sorted(cm.SPECS))
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    args = parser.parse_args()
    load_dotenv(args.env_file, override=True)

    stakeholder = args.stakeholder
    spec = cm.SPECS[stakeholder]
    evidence = evidence_for(stakeholder)
    addendum = spec["addenda"]["comprehensive"] + EXTENDED_RULE + ROLE_RULES.get(stakeholder, "")
    suffix = None
    for attempt in range(1, ATTEMPTS + 1):
        response = OpenAIProvider(model=MODEL).generate(
            GEOSMILE_SYSTEM_PROMPT, evidence, {"id": stakeholder, "label": STAKEHOLDERS[stakeholder]}, spec["question"],
            addendum=addendum, user_suffix=suffix)
        if not response.ok:
            raise SystemExit(f"{stakeholder}: {response.error}")
        response.text = mark_bare_figures(response.text)
        checks = cm.check(response.text, evidence, "comprehensive", {"text": MAX_WORDS})
        print(f"{stakeholder} attempt {attempt}: {checks}", flush=True)
        if checks["ok"]:
            break
        suffix = CUT_REQUEST.format(draft=response.text, problems="; ".join(checks["problems"]))
    else:
        raise SystemExit(f"{stakeholder}: failed the checks after {ATTEMPTS} attempts; nothing archived")
    record = {"stakeholder": stakeholder, "evidence": evidence, "addendum": addendum,
              "run": {"ok": True, "model": response.model, "timestamp": response.timestamp,
                      "prompt_version": f"{SYSTEM_PROMPT_VERSION}+{spec['version']['comprehensive']}+extended-{MAX_WORDS}{'-corrected' if stakeholder in ROLE_RULES else ''}",
                      "text": response.text, "checks": checks}}
    path = ROOT / CARDS_DIR / f"{stakeholder}_detailed_llm.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"checks: {checks}")


if __name__ == "__main__":
    main()
