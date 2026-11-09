"""Live, evidence-grounded explanations with automatic validation and repair.

    selected area + role -> Geo-SMILE evidence -> AI draft (JSON) -> deterministic repair -> validation
        valid   -> shown, cached for identical requests
        invalid -> regenerated with the problems listed, at most ATTEMPTS times
        still invalid -> an evidence-only summary is shown instead of AI text

Deterministic repair: the interface, not the model, sets the highlights; bare figures are wrapped so the figure
check must trace them; detailed text is cut to its word budget by whole sentences.
Validation: every figure traced to the evidence, every rank present in the evidence, no evaluative word without a
supplied reference, the role's wording rules, length and repetition limits. Validation catches known errors; it
cannot prove that an interpretation is right.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from dashboard.data.schema import STAKEHOLDERS, human_label
from dashboard.llm import card_mode as cm
from dashboard.llm.attribution_evidence import build_local_attribution_evidence
from dashboard.llm.base import LLMResponse
from dashboard.llm.evidence_builder import build_evidence
from dashboard.views.explanation_panel import AREA_RETRY_NOTE, DETAILED_AREA_RULE, _generate, _providers_for
from dashboard.views.stakeholder_evidence import figure_evidence, stakeholder_data, stakeholder_panels

ATTEMPTS = 3
DETAILED_WORDS = 200
DETAILED_PARTS = ("part_1", "part_2", "part_3")
HEADINGS = {
    "public": ("Across Hull", "In this area", "Where location matters"),
    "planner": ("Where location matters", "What the model relies on", "This area"),
    "police": ("Signals across Hull", "Recorded change", "This area"),
}
ROLE_TERMS = {"public": ("crime history", "location"), "planner": ("built environment", "GeoSensitivity"),
              "police": ("recent months", "recorded crime")}
EVALUATIVE = re.compile(r"\b(modest(?:ly)?|highly|strongly|substantial(?:ly)?|considerabl[ey]|notabl[ey]|sharp(?:ly)?|"
                        r"dramatic(?:ally)?|markedly|remarkabl[ey]|very)\b", re.I)
ORDINALS = {w: i for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth "
    "fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split(), start=1)}
RANKS = (re.compile(r"\brank(?:s|ed|ing)?(?:\s+it)?(?:\s+at)?\s+(\d+|[a-z]+)\b", re.I),
         re.compile(r"\b(\d+)(?:st|nd|rd|th)\b"))
GROUNDING = """

GROUNDING RULES: every comparison (higher, lower, above, below, more, less) names its reference value from the
evidence, such as the Hull median or a rank. Do not use modest, strong, substantial, notable, sharp or very. Cite a
rank only if that exact rank appears in the evidence. Do not use [[t:...]] markup: the interface adds highlights.
"""
DETAILED_JSON = GROUNDING + """
OUTPUT FORMAT, overriding the paragraph instructions above: return ONLY one JSON object
{"part_1": "...", "part_2": "...", "part_3": "...", "limitation": "...", "evidence_fields": ["...", "..."]}.
part_1, part_2 and part_3 are the three numbered points above, about 50 words each; limitation is one sentence.
evidence_fields lists the EVIDENCE_JSON field paths you used, for example "police_signals.focus_lsoa.recent_share".
Wrap every figure in [[n:...]].
"""


@dataclass
class LiveResult:
    status: str                      # "ai", "evidence_only", "limit" or "error"
    content: dict | None = None      # validated, repaired content (short card fields, or detailed parts)
    response: LLMResponse | None = None
    problems: list[str] = field(default_factory=list)
    attempts: int = 0


def _evidence(bundle, role: str, lsoa: str, mode: str) -> tuple[dict, str]:
    spec, data = cm.SPECS[role], stakeholder_data()
    if mode == "short":
        evidence = build_evidence(bundle, stakeholder_id=role, current_map_view="geo_sensitivity", scope="lsoa",
                                  selected_lsoa=lsoa, user_question=spec["question"], output_mode="dashboard_mode")
        cm.add_evidence(evidence, data, role, "short", focus=lsoa)
        if role == "police":
            evidence["police_signals"]["crime_signals"].pop("hull_median_crime_history_share", None)
        _exact_shares(evidence, role)
        return evidence, cm.AREA_RULES[role] + GROUNDING
    from generate_detailed_narrative import EXTENDED_RULE, ROLE_RULES

    evidence = build_evidence(bundle, stakeholder_id=role, current_map_view="geo_sensitivity", scope="lsoa",
                              selected_lsoa=lsoa, user_question=spec["question"])
    cm.add_evidence(evidence, data, role, "comprehensive", focus=lsoa)
    _exact_shares(evidence, role)
    evidence["local_attribution_patterns"] = build_local_attribution_evidence(
        bundle.local_features, bundle.global_features, bundle.geometry, tuple(bundle.feature_names), lsoa)
    contrast = data.default_contrast(lsoa)
    evidence["dashboard_figures"] = {"lsoa": lsoa, "contrast_lsoa": contrast if role == "planner" else None,
                                     "figures": figure_evidence(stakeholder_panels(role, lsoa, contrast))}
    rule = (EXTENDED_RULE + ROLE_RULES.get(role, "") + DETAILED_AREA_RULE.format(name=data.short(lsoa), place=data.place(lsoa) or "Hull")
            + DETAILED_JSON)
    return evidence, rule


def _exact_shares(evidence: dict, role: str) -> None:
    """Hull category shares to four decimals: at three, crime history is 0.635 and rounds to 64%, not its 63.47%.
    Live evidence only, so the archived cards' evidence stays reproducible."""
    shares = {cm.CATEGORIES[k]: round(float(v), 4)
              for k, v in cm.category_shares(stakeholder_data()).sort_values(ascending=False).items()}
    if role == "planner":
        evidence["built_environment_share"]["hull_feature_category_shares"] = shares
    elif role == "public":
        evidence["public_overview"]["hull"]["feature_category_shares"] = shares


# ---------------------------------------------------------------- deterministic repair
NAMES = r"Level 4\+?|\b\d{3}[A-Z]\b|\b[Pp]anel \d+|\b(?:W1|R2|W_1)\b|[Tt]op[- ]?\d+|\b\d+(?:st|nd|rd|th)\b|\b\d+-[a-z]+"
BARE = re.compile(rf"(?:{NAMES})|(?<![\w.])(\d+(?:\.\d+)?%?)(?![\w-]|\.\d)")


def _plain(text: str) -> str:
    """Drop highlight markup, and figure markup around words that are not figures."""
    text = re.sub(r"\[\[t:(.+?)\]\]", r"\1", str(text))
    # A "figure" holding a word (a place name, or an LSOA code with its town) is a name, not a figure.
    return re.sub(r"\[\[n:([^\]]+?)\]\]", lambda m: m[1] if re.search(r"[A-Za-z]{2,}", m[1]) else m[0], text)


def _mark_bare(text: str) -> str:
    """Wrap stray figures as [[n:...]] so the figure check traces them; numbers inside labels such as
    "6-month" and names such as 029E stay as they are."""
    parts = re.split(r"(\[\[[nt]:.+?\]\])", text)
    parts[::2] = [BARE.sub(lambda m: f"[[n:{m[1]}]]" if m[1] else m[0], part) for part in parts[::2]]
    return "".join(parts)


def _words(text: str) -> int:
    return len(cm._bare(text).split())


def _trim(parts: dict, keys, budget: int) -> None:
    """Cut whole sentences from the longest part until the total fits."""
    while sum(_words(parts[k]) for k in keys) > budget:
        key = max(keys, key=lambda k: _words(parts[k]))
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", parts[key].strip())
        if len(sentences) <= 1:
            return
        parts[key] = " ".join(sentences[:-1])


def _highlight(text: str, terms: list[str], limit: int) -> str:
    """Mark the first plain-text occurrence of up to `limit` terms, in the order given."""
    marked = 0
    for term in terms:
        if marked >= limit or not term:
            continue
        pieces = re.split(r"(\[\[[nt]:.+?\]\])", text)
        for i in range(0, len(pieces), 2):
            match = re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", pieces[i], re.I)
            if match:
                pieces[i] = f"{pieces[i][:match.start()]}[[t:{match.group(0)}]]{pieces[i][match.end():]}"
                marked += 1
                break
        text = "".join(pieces)
    return text


def _terms(role: str, lsoa: str) -> list[str]:
    d = stakeholder_data()
    features = list(d.abs_local(lsoa).index[:3])
    names = [d.place(lsoa), d.short(lsoa)] + [cm.plain_label(f) for f in features] + [human_label(f) for f in features]
    return list(dict.fromkeys([*names, *ROLE_TERMS[role], "Old Town"]))


def _repair(raw: str, role: str, lsoa: str, mode: str) -> dict:
    mark_bare_figures = _mark_bare
    content = cm.parse_card(raw)
    terms = _terms(role, lsoa)
    if mode == "short":
        for key in ("headline", "section_01", "section_02", "footer"):
            content[key] = mark_bare_figures(_plain(content.get(key, ""))) if key != "footer" else _plain(content.get(key, ""))
        content["headline"] = _highlight(content["headline"], terms, 1)
        for key in ("section_01", "section_02"):
            content[key] = _highlight(content[key], terms, 3)
        content["tiles"] = [{"value": cm._bare(str(t.get("value", ""))), "label": cm._bare(str(t.get("label", "")))}
                            for t in content.get("tiles", [])]
        return content
    for key in (*DETAILED_PARTS, "limitation"):
        content[key] = mark_bare_figures(_plain(content.get(key, "")))
    _trim(content, DETAILED_PARTS, DETAILED_WORDS - _words(content["limitation"]))
    for key in DETAILED_PARTS:
        content[key] = _highlight(content[key], terms, 3)
    return content


# ---------------------------------------------------------------- validation
def _untraced_ranks(text: str, pool: list[float]) -> list[str]:
    bare, bad = cm._bare(text), []
    for pattern in RANKS:
        for match in pattern.finditer(bare):
            token = match.group(1).lower()
            value = int(token) if token.isdigit() else ORDINALS.get(token)
            if value is not None and not any(abs(p - value) < 1e-9 for p in pool):
                bad.append(f"rank not in the evidence: {match.group(0)!r}")
    return bad


def _validate(content: dict, role: str, mode: str, evidence: dict) -> list[str]:
    spec = cm.SPECS[role]
    if mode == "short":
        checked = cm.check(content, evidence, "short", spec.get("limits"))
        text = " ".join(content.get(k, "") for k in ("headline", "section_01", "section_02", "footer"))
    else:
        text = "\n\n".join(content[k] for k in (*DETAILED_PARTS, "limitation"))
        checked = cm.check(text, evidence, "comprehensive", {"text": DETAILED_WORDS})
    problems = [p for p in checked["problems"] if "key terms" not in p]   # the interface sets the highlights
    problems += cm.area_card_problems(role, text, evidence)
    supported = {(evidence.get("selected_lsoa") or {}).get("geo_magnitude_class")}
    problems += [f"unsupported evaluative word: {w!r}; compare with a supplied reference value instead"
                 for w in sorted({m.group(0).lower() for m in EVALUATIVE.finditer(cm._bare(text))})
                 if w.rstrip("ly") not in supported and w not in supported]
    problems += _untraced_ranks(text, cm._evidence_numbers(evidence))
    return list(dict.fromkeys(problems))


def explain(bundle, role: str, lsoa: str, mode: str, regenerate: bool = False) -> LiveResult:
    """mode is "short" (card) or "comprehensive" (detailed narrative)."""
    evidence, rule = _evidence(bundle, role, lsoa, mode)
    provider = _providers_for("openai", mode)[0]
    stakeholder = {"id": role, "label": STAKEHOLDERS[role]}
    note, problems, response = "", [], None
    for attempt in range(1, ATTEMPTS + 1):
        response = _generate(provider, evidence, stakeholder, cm.SPECS[role]["question"], force=regenerate or attempt > 1,
                             mode=mode, area_rule=rule, retry_note=note)
        if not response.ok:
            status = "limit" if "limit" in (response.error or "") else "error"
            return LiveResult(status, response=response, problems=[response.error or ""], attempts=attempt)
        try:
            content = _repair(response.text, role, lsoa, mode)
            problems = _validate(content, role, mode, evidence)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            content, problems = None, [f"the answer was not the requested JSON ({type(exc).__name__})"]
        if not problems:
            return LiveResult("ai", content=content, response=response, attempts=attempt)
        note = AREA_RETRY_NOTE.format(problems="; ".join(problems), previous=response.text)
    return LiveResult("evidence_only", response=response, problems=problems, attempts=ATTEMPTS)
