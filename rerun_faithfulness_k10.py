"""Re-run model-based faithfulness at Feature k=10 / Geo k=20 and sync outputs.

Replaces archived Feature k=6 faithfulness values, removes archive_note mismatch rows,
and updates method/summary text so Feature attribution accuracy and intervention share k=10.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hull_eval_v3 import model_based_faithfulness, write_method_md
from hull_eval_v3 import dehghani_expert_faithfulness_status, controlled_input_stability_status
from hull_expert_topk import FEATURE_TOP_K, GEO_TOP_K, BRANCH_FEATURE, BRANCH_GEO

OUT = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    assert FEATURE_TOP_K == 10, FEATURE_TOP_K
    assert GEO_TOP_K == 20, GEO_TOP_K

    rows, summary = model_based_faithfulness(n_random=1000, seed=42)
    mb = pd.DataFrame(rows)
    print("\n=== Recomputed faithfulness ===")
    print(mb[["branch", "metric", "value", "k"]].to_string(index=False))

    # --- update full_v3 ---
    v3_path = OUT / "geosmile_hull_evaluation_full_v3.csv"
    v3 = pd.read_csv(v3_path)
    keep = v3[v3["category"].astype(str) != "Model-based attribution faithfulness"].copy()
    v3_new = pd.concat([keep, mb], ignore_index=True)
    v3_new.to_csv(v3_path, index=False)
    print(f"updated {v3_path}")

    # --- update consolidated evaluation ---
    eval_path = OUT / "geosmile_hull_feature_geo_evaluation.csv"
    ev = pd.read_csv(eval_path)
    drop = (
        (ev["category"].astype(str) == "Model-based attribution faithfulness")
        | (ev["evaluation_set"].astype(str) == "archive_note")
    )
    ev = ev.loc[~drop].copy()
    mb2 = mb.copy()
    mb2["evaluation_set"] = "full_v3"
    mb2["result_status"] = "recomputed"
    # ensure schema columns
    for c in ev.columns:
        if c not in mb2.columns:
            mb2[c] = pd.NA
    ev = pd.concat([ev, mb2[ev.columns]], ignore_index=True)
    ev.to_csv(eval_path, index=False)
    print(f"updated {eval_path} rows={len(ev)}")

    # --- update summary JSON ---
    sum_path = OUT / "geosmile_hull_evaluation_summary_v3.json"
    s = json.loads(sum_path.read_text(encoding="utf-8"))
    s["model_based_attribution_faithfulness"] = {
        "feature": {**summary["feature"], "design_k": 10, "archive_status": "recomputed_at_k10"},
        "geo": {**summary["geo"], "design_k": 20, "archive_status": "recomputed_at_k20"},
    }
    # drop known inconsistency about k=6
    notes = [
        n
        for n in s.get("known_inconsistencies", [])
        if "k=6" not in str(n) and "k = 6" not in str(n)
    ]
    s["known_inconsistencies"] = notes
    if "expert_topk_definition" in s:
        s["expert_topk_definition"]["feature_k"] = FEATURE_TOP_K
        s["expert_topk_definition"]["geo_k"] = GEO_TOP_K
        s["expert_topk_definition"]["note"] = (
            f"Expert top sets: Feature top-{FEATURE_TOP_K} of 30, Geo top-{GEO_TOP_K} of 168. "
            f"Model-based faithfulness interventions use the same cardinalities."
        )
    sum_path.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(f"updated {sum_path}")

    # --- rewrite method md without mismatch paragraph ---
    dehghani = dehghani_expert_faithfulness_status()
    stability = controlled_input_stability_status()
    write_method_md(OUT / "geosmile_hull_evaluation_method_v3.md", dehghani, stability)
    # append aligned intervention note
    method = OUT / "geosmile_hull_evaluation_method_v3.md"
    text = method.read_text(encoding="utf-8")
    # replace any leftover archived-k6 wording if template still has it
    text = text.replace(
        f"- **Model-based attribution faithfulness** — top vs random vs bottom interventions at fixed cardinality (WD response).\n"
        f"  - Design/method target: Feature \\(k={FEATURE_TOP_K}\\), Geo \\(k={GEO_TOP_K}\\).\n"
        f"  - Archived Feature results were produced at \\(k=6\\) and are reported as archived; they are not at a common cardinality with Feature attribution accuracy (\\(k={FEATURE_TOP_K}\\)).\n"
        f"  - Archived Geo results use \\(k={GEO_TOP_K}\\) and are aligned with Geo attribution accuracy.\n"
        f"  - Re-running the Feature intervention at \\(k={FEATURE_TOP_K}\\) would align the Feature columns.\n",
        f"- **Model-based attribution faithfulness** — top vs random vs bottom interventions at fixed cardinality "
        f"(Feature={FEATURE_TOP_K}, Geo={GEO_TOP_K}; WD response), matching the expert reference sets.\n",
    )
    if "Archived Feature results were produced at" in text:
        # fallback cleanup
        lines = []
        skip_prefixes = (
            "  - Design/method target:",
            "  - Archived Feature results",
            "  - Archived Geo results",
            "  - Re-running the Feature intervention",
        )
        for line in text.splitlines(True):
            if line.startswith(skip_prefixes):
                continue
            if "Archived Feature intervention cardinality" in line or line.startswith("## Archived Feature"):
                continue
            lines.append(line)
        text = "".join(lines)
    # remove dedicated archived section if present
    if "## Archived Feature intervention cardinality" in text:
        pre, _, rest = text.partition("## Archived Feature intervention cardinality")
        # drop until next ## or end
        idx = rest.find("\n## ")
        text = pre + (rest[idx + 1 :] if idx >= 0 else "")
    if "matching the expert reference sets" not in text:
        text = text.replace(
            f"(Feature={{{FEATURE_TOP_K}}}, Geo={{{GEO_TOP_K}}}; WD response).",
            f"(Feature={FEATURE_TOP_K}, Geo={GEO_TOP_K}; WD response), matching the expert reference sets.",
        )
    method.write_text(text, encoding="utf-8")

    # replace old mismatch note file with alignment confirmation
    note_path = OUT / "geosmile_hull_feature_intervention_cardinality_note.md"
    note_path.write_text(
        f"""# Feature intervention cardinality

Feature model-based attribution faithfulness has been recomputed at **k = {FEATURE_TOP_K}**, matching the Feature expert-reference top-{FEATURE_TOP_K} set.

- Feature attribution accuracy: top {FEATURE_TOP_K} of 30
- Feature faithfulness intervention: masks the {FEATURE_TOP_K} highest-ranked predictors
- Geo attribution accuracy and faithfulness: both **k = {GEO_TOP_K}**

The Feature columns are therefore evaluated at a common cardinality.
""",
        encoding="utf-8",
    )

    # compact dump of Feature values for thesis check
    feat = summary["feature"]
    print("\nFeature faithfulness @ k=10:")
    for k in (
        "k",
        "top_response",
        "random_mean",
        "bottom_response",
        "top_minus_random",
        "top_minus_bottom",
        "empirical_p_top_vs_random",
        "random_ci95_low",
        "random_ci95_high",
    ):
        print(f"  {k}: {feat[k]}")
    print("Geo faithfulness @ k=20:")
    geo = summary["geo"]
    for k in ("k", "top_response", "random_mean", "bottom_response", "empirical_p_top_vs_random"):
        print(f"  {k}: {geo[k]}")


if __name__ == "__main__":
    main()
