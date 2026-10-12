"""Export six-stakeholder GeoSMILE narrative comparison tables to LaTeX.

Standalone script (no Streamlit) for thesis use. Reuses the same evidence
package and OpenAI call path as dashboard/views/explanation_panel.py's
"Stakeholder narrative comparison" table, for a fixed set of representative
scenarios, and writes ready-to-\\input LaTeX tables matching the tabularx /
booktabs convention already used in outputs/geosmile_hull_chapter4_updated.tex.

Run from the hull folder:

    python generate_stakeholder_narrative_table.py

Requires OPENAI_API_KEY in .env or the environment. Responses are cached
under outputs/dashboard_llm_cache/ (same cache the dashboard uses), so a
repeat run only pays for scenarios/stakeholders not already generated.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    pass

from dashboard.data.loader import load_hull_bundle
from dashboard.data.schema import LSOA_ID, LSOA_NAME, STAKEHOLDERS
from dashboard.llm.attribution_evidence import build_local_attribution_evidence
from dashboard.llm.base import LLMResponse
from dashboard.llm.cache import cache_get, cache_put, evidence_hash
from dashboard.llm.evidence_builder import build_evidence
from dashboard.llm.openai_provider import OpenAIProvider
from dashboard.llm.prompt import GEOSMILE_SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION

CURRENT_MAP_VIEW = "geo_sensitivity"
OUT_DIR = ROOT / "outputs" / "stakeholder_narratives"


@dataclass
class Scenario:
    id: str
    lsoa: str | None
    label: str
    rationale: str


SCENARIOS: list[Scenario] = [
    Scenario(
        id="overall",
        lsoa=None,
        label="Overall Hull",
        rationale=(
            "City-wide scope, no LSOA selected: the evidence package summarises "
            "the GeoSensitivity distribution and spatial pattern across all Hull LSOAs."
        ),
    ),
    Scenario(
        id="top_magnitude",
        lsoa="E01033104",
        label="Kingston upon Hull 029E (E01033104)",
        rationale=(
            "Selected as the Hull LSOA with the single largest absolute GeoSensitivity "
            "value, to illustrate the largest-magnitude case."
        ),
    ),
    Scenario(
        id="rank_vs_magnitude",
        lsoa="E01012761",
        label="Kingston upon Hull 015A (E01012761)",
        rationale=(
            "Selected because it ranks in the top 20 Hull LSOAs by GeoSensitivity "
            "while its absolute magnitude is only moderate, illustrating that rank "
            "is not the same as magnitude."
        ),
    ),
    Scenario(
        id="hh_cluster",
        lsoa="E01012891",
        label="Kingston upon Hull 030C (E01012891)",
        rationale=(
            "Selected as a member of a statistically significant High-High "
            "GeoSensitivity cluster (LISA)."
        ),
    ),
    Scenario(
        id="typical",
        lsoa="E01035471",
        label="Kingston upon Hull 035B (E01035471)",
        rationale=(
            "Selected as a typical, non-clustered LSOA with GeoSensitivity close "
            "to the Hull median, for contrast with the extreme examples above."
        ),
    ),
]


def _generate(
    provider: OpenAIProvider,
    evidence: dict,
    stakeholder: dict,
    *,
    force: bool,
) -> LLMResponse:
    key = evidence_hash(
        provider=provider.name,
        model=provider.model,
        stakeholder=stakeholder["id"],
        scope=evidence.get("scope", ""),
        selected_lsoa=(evidence.get("selected_lsoa") or {}).get("id"),
        current_map=evidence.get("current_map_view", ""),
        evidence=evidence,
        user_question=None,
        prompt_version=SYSTEM_PROMPT_VERSION,
    )
    if not force:
        cached = cache_get(key)
        if cached is not None:
            return cached
    response = provider.generate(GEOSMILE_SYSTEM_PROMPT, evidence, stakeholder, None)
    if response.ok:
        cache_put(key, response)
    return response


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(text: str) -> str:
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in text)


def _narrative_to_latex(text: str) -> str:
    parts = _BOLD_RE.split(text)
    # re.split with a capturing group interleaves [plain, bold, plain, bold, ...]
    out = []
    for i, part in enumerate(parts):
        escaped = _latex_escape(part)
        out.append(r"\textbf{" + escaped + "}" if i % 2 == 1 else escaped)
    return "".join(out)


def _table_latex(scenario: Scenario, results: dict[str, LLMResponse]) -> str:
    caption = f"Stakeholder narrative comparison for {scenario.label}. {scenario.rationale}"
    label = f"tab:stakeholder_narratives_{scenario.id}"
    rows = []
    for stakeholder_id, persona_label in STAKEHOLDERS.items():
        response = results[stakeholder_id]
        if response.ok:
            body = _narrative_to_latex(response.text)
        else:
            body = r"\textit{Not generated: " + _latex_escape(response.error or "unknown error") + "}"
        rows.append(f"{_latex_escape(persona_label)} &\n{body} \\\\")
    body_rows = "\n".join(rows)
    return f"""\\begin{{table}}[H]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\small
\\begin{{tabularx}}{{\\textwidth}}{{p{{0.20\\textwidth}} X}}
\\toprule
Stakeholder & Narrative (OpenAI, {{\\ttfamily {results[next(iter(STAKEHOLDERS))].model}}}) \\\\
\\midrule
{body_rows}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}"""


def main(force: bool = False) -> None:
    bundle = load_hull_bundle(ROOT)
    provider = OpenAIProvider()

    all_results: dict[str, dict[str, LLMResponse]] = {}
    failures: list[str] = []

    for scenario in SCENARIOS:
        scope = "overall" if scenario.lsoa is None else "lsoa"
        local_patterns = build_local_attribution_evidence(
            bundle.local_features,
            bundle.global_features,
            bundle.geometry,
            tuple(bundle.feature_names),
            scenario.lsoa,
        )
        scenario_results: dict[str, LLMResponse] = {}
        for stakeholder_id, persona_label in STAKEHOLDERS.items():
            evidence = build_evidence(
                bundle,
                stakeholder_id=stakeholder_id,
                current_map_view=CURRENT_MAP_VIEW,
                scope=scope,
                selected_lsoa=scenario.lsoa,
                output_mode="comparison_table_mode",
            )
            evidence["local_attribution_patterns"] = local_patterns
            stakeholder = {"id": stakeholder_id, "label": persona_label}
            print(f"[{scenario.id}] {persona_label} ...", end=" ", flush=True)
            response = _generate(provider, evidence, stakeholder, force=force)
            scenario_results[stakeholder_id] = response
            if response.ok:
                print(f"ok ({len(response.text.split())} words)")
            else:
                print(f"FAILED: {response.error}")
                failures.append(f"{scenario.id}/{stakeholder_id}: {response.error}")
        all_results[scenario.id] = scenario_results

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUT_DIR / "stakeholder_narratives.json"
    json_path.write_text(
        json.dumps(
            {
                sid: {pid: r.to_dict() for pid, r in res.items()}
                for sid, res in all_results.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    tex_tables = "\n\n".join(
        _table_latex(scenario, all_results[scenario.id]) for scenario in SCENARIOS
    )
    tex_path = OUT_DIR / "stakeholder_narrative_tables.tex"
    header = (
        "% Generated by generate_stakeholder_narrative_table.py\n"
        f"% Model: {provider.model} | System prompt version: {SYSTEM_PROMPT_VERSION}\n"
        "% Requires: booktabs, tabularx, float (\\usepackage{booktabs,tabularx,float}).\n"
        "% Each \\begin{table} block is self-contained; \\input{} them individually\n"
        "% or use this whole file directly.\n\n"
    )
    tex_path.write_text(header + tex_tables + "\n", encoding="utf-8")

    print(f"\nWrote {json_path}")
    print(f"Wrote {tex_path}")
    if failures:
        print(f"\n{len(failures)} generation(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    force = "--force" in sys.argv
    main(force=force)
