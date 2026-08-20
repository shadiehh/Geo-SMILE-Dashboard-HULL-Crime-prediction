"""Record Feature faithfulness archive cardinality inconsistency (k=6 vs design k=10)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def main() -> None:
    sum_path = OUT / "geosmile_hull_evaluation_summary_v3.json"
    s = json.loads(sum_path.read_text(encoding="utf-8"))
    fb = s.setdefault("model_based_attribution_faithfulness", {}).setdefault("feature", {})
    fb["k"] = 6
    fb["design_k"] = 10
    fb["archive_status"] = "archived_at_k6"
    fb["notes"] = (
        "Archived fixed-cardinality feature ablation at k=6 (WD on test predictions). "
        "Method/design target is Feature k=10 to match the expert top-10 reference; "
        "attribution accuracy uses k=10, so Feature faithfulness and attribution accuracy "
        "are not at a common cardinality. Geo intervention remains k=20 and is aligned. "
        "Re-run Feature intervention at k=10 to align Feature columns."
    )
    gb = s.setdefault("model_based_attribution_faithfulness", {}).setdefault("geo", {})
    gb["k"] = 20
    gb["design_k"] = 20
    gb["archive_status"] = "archived_at_k20_aligned"
    notes = s.setdefault("known_inconsistencies", [])
    msg = (
        "Feature model-based faithfulness archived at k=6 while expert-reference attribution "
        "accuracy uses Feature top-10; Geo both use k=20. Re-run Feature intervention at k=10 to align."
    )
    if msg not in notes:
        notes.append(msg)
    sum_path.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(f"updated {sum_path}")

    eval_path = OUT / "geosmile_hull_feature_geo_evaluation.csv"
    df = pd.read_csv(eval_path)
    mask = (df["branch"].astype(str).str.contains("Feature")) & (
        df["category"].astype(str) == "Model-based attribution faithfulness"
    )
    df.loc[mask, "k"] = 6
    df.loc[mask, "notes"] = (
        "Archived Feature intervention at k=6; method target k=10 (matches expert top-10). "
        "Not comparable at common cardinality with Feature attribution accuracy."
    )
    doc_metric = "Feature intervention cardinality note"
    if not (df["metric"].astype(str) == doc_metric).any():
        rows = [
            {
                "branch": "Feature (location excluded)",
                "evaluation_set": "archive_note",
                "category": "Model-based attribution faithfulness",
                "metric": doc_metric,
                "value": 6,
                "ci95_low": pd.NA,
                "ci95_high": pd.NA,
                "p_value": pd.NA,
                "n": 30,
                "k": 6,
                "source_file": "geosmile_hull_evaluation_full_v3.csv / geosmile_hull_evaluation_summary_v3.json",
                "result_status": "archived",
                "notes": (
                    "Archived faithfulness intervention k=6; attribution accuracy uses k=10. "
                    "Design/method specifies Feature k=10. Geo branch aligned at k=20. "
                    "Re-run Feature intervention at k=10 to align Feature columns."
                ),
            },
            {
                "branch": "Feature (location excluded)",
                "evaluation_set": "archive_note",
                "category": "Model-based attribution faithfulness",
                "metric": "Feature intervention design cardinality",
                "value": 10,
                "ci95_low": pd.NA,
                "ci95_high": pd.NA,
                "p_value": pd.NA,
                "n": 30,
                "k": 10,
                "source_file": "geosmile_hull_evaluation_method_v3.md",
                "result_status": "design_target",
                "notes": "Method target matching expert-reference Feature top-10.",
            },
            {
                "branch": "Geo",
                "evaluation_set": "archive_note",
                "category": "Model-based attribution faithfulness",
                "metric": "Geo intervention cardinality",
                "value": 20,
                "ci95_low": pd.NA,
                "ci95_high": pd.NA,
                "p_value": pd.NA,
                "n": 168,
                "k": 20,
                "source_file": "geosmile_hull_evaluation_summary_v3.json",
                "result_status": "archived",
                "notes": "Archived Geo intervention k=20; aligned with Geo expert top-20.",
            },
        ]
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(eval_path, index=False)
    print(f"updated {eval_path} rows={len(df)}")

    note_path = OUT / "geosmile_hull_feature_intervention_cardinality_note.md"
    note_path.write_text(
        """# Feature intervention cardinality note

The v3 method description specifies a Feature intervention cardinality of **ten**, matching the ten-item expert reference set. The **archived** model-based attribution-faithfulness results were produced at **k = 6** and are reported as archived.

- Feature attribution accuracy: assessed over the expert/model **top 10** of 30.
- Feature archived faithfulness intervention: masks the **six** highest-ranked predictors.
- Therefore the two Feature evaluations are **not at a common cardinality**.
- This affects only comparability of the Feature columns.
- Geo attribution accuracy and Geo archived faithfulness both use **k = 20** and are aligned.
- Re-running the Feature intervention at **k = 10** would align the Feature columns.
""",
        encoding="utf-8",
    )
    print(f"wrote {note_path}")


if __name__ == "__main__":
    main()
