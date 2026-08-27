# Report: Hull dashboard system prompt

**Document:** GeoSMILE Explain — system policy  
**Version:** `geosmile-hull-v1`  
**Files:** `llm/prompt.py` and `dashboard/llm/prompt.py` (identical `GEOSMILE_SYSTEM_PROMPT` strings)  
**Size:** 22,045 characters; 3,084 words; 16 numbered sections  
**Providers:** OpenAI and Anthropic use this **one** prompt. Both overwrite any passed string with `GEOSMILE_SYSTEM_PROMPT` at call time.  
**Date of this report:** 21 August 2026

---

## 1. Purpose

The dashboard system prompt is the **policy layer** between published GeoSMILE artefacts and stakeholder-facing text. It does not compute GeoSMILE, does not read map pixels, and does not retrieve external Hull context. It instructs a large language model to **explain a compact JSON evidence package** already built from published tables.

The assistant persona is **GeoSMILE Explain**. Its job is faithful explanation, not persuasion, causation, or policy.

Design rule: **computation discovers the evidence; the model explains the evidence.** That order is not reversible.

---

## 2. Where it lives and how it is used

| Item | Detail |
|---|---|
| Canonical text | `GEOSMILE_SYSTEM_PROMPT` in `llm/prompt.py` and `dashboard/llm/prompt.py` |
| Version stamp | `SYSTEM_PROMPT_VERSION = "geosmile-hull-v1"` |
| Runtime import | `from dashboard.llm.prompt import GEOSMILE_SYSTEM_PROMPT` |
| OpenAI | Responses API (Chat Completions fallback); `instructions` / system role = master prompt |
| Anthropic | Messages API; `system=` = master prompt |
| User message | Compact JSON evidence + optional follow-up; not the system policy |
| Generate | Only on explicit dashboard button; results cached by evidence hash + prompt version |
| Integrity check | `assert_prompt_semantics()` requires the core “is not” rules to remain in the text |

Both copies must stay identical. Tests confirm `dashboard/llm/prompt.py` contains the full policy (not a loader) and that the repo-root copy matches.

---

## 3. Pipeline the prompt encodes

```
Crime prediction model
        ↓
     GeoSMILE
        ↓
Structured GeoSensitivity / feature evidence
        ↓
Spatial-pattern analysis
        ↓
Structured evidence package (JSON)
        ↓
Stakeholder profile
        ↓
ONE system prompt
        ↓
OpenAI and/or Anthropic explanation
```

There is no RAG layer and no web search. The model may not invent deprivation, lighting, policing, land use, ethnicity, income, or urban-form stories.

---

## 4. Four spatial objects (must never be merged)

These are the prompt’s conceptual core. Each has a distinct provenance.

| Object | Source | What it is | What it is not |
|---|---|---|---|
| **GeoSensitivity** | Geo-branch location-masking SMILE (`geo_coefficient`) | Per-area attribution of the **shift in target predictions** when geographic units are masked | Latitude/longitude importance; crime cause; hotspot |
| **Latitude feature effect** | Feature SMILE with `centroid_latitude` as a player | How much the **latitude feature** is attributed in that unit’s prediction explanation | GeoSensitivity |
| **Longitude feature effect** | Feature SMILE with `centroid_longitude` as a player | How much the **longitude feature** is attributed in that unit’s prediction explanation | GeoSensitivity |
| **Spatial clustering of \|GeoSensitivity\|** | Global Moran and LISA on published \|GeoSensitivity\| | Statistical clustering / outlier class of GeoSensitivity magnitudes | Clustering of crime; clustering of lat/lon effects; map appearance |

Map view (`geo_sensitivity`, `latitude_effect`, `longitude_effect`, `spatial_pattern`) changes **emphasis**, not the underlying objects.

---

## 5. Operational definition of GeoSensitivity (in the prompt)

The prompt now states the measurement, not a metaphor about “moving one area’s location.”

1. **Intervention:** perturb geographic units (replace selected units’ coordinates with the training median).  
2. **Response:** shift in the Geo model’s **target predictions** (Wasserstein-1 between original and masked predicted-value distributions on affected units).  
3. **Score:** per-unit SMILE coefficient attributing that prediction shift to each geographic unit.

A large \|GeoSensitivity\| means that unit **contributes more to the measured shift in predicted target** when locations are masked. It does not mean that unit’s coordinates were moved in isolation, and it is not a lat/lon feature effect.

---

## 6. Section-by-section digest

| § | Title | Function |
|---|---|---|
| 1 | ROLE | Explain existing evidence; not a model, not a causal engine, not a policy authority |
| 2 | CONTEXT | Hull next-month crime at LSOA 2021; four spatial objects; no RAG |
| 3 | OBJECTIVE | Faithful, stakeholder-appropriate, magnitude-aware, non-causal |
| 4 | INPUT / EVIDENCE CONTRACT | Use only the JSON + this policy; missing fields stay missing |
| 5 | EVIDENCE HIERARCHY | Numbers → spatial stats → evaluation → metadata → cautious inference → limitation |
| 6 | PROVENANCE RULES | Name the branch/source; map view sets lead layer only |
| 7 | GEOSMILE SEMANTICS | Attribution ≠ causation; rank ≠ magnitude; GeoSensitivity operational definition; evaluation is not truth |
| 8 | SPATIAL-PATTERN SEMANTICS | Moran/LISA only; no visual clustering; `pattern_scale` only if established |
| 9 | FIVE STAKEHOLDER CONFIGURATIONS | Same evidence; different structure and language |
| 10 | LOW-MAGNITUDE / RANK RULES | High rank + small \|value\| is comparative, not “strong” |
| 11 | UNCERTAINTY | Report supplied metrics only; Consistency ≠ ATT stability |
| 12 | RECOMMENDATION POLICY | Evidence next steps only; no policing or intervention targeting |
| 13 | LANGUAGE CALIBRATION | Preferred terms; forbidden collapses (“hotspot”, “location importance”) |
| 14 | OUTPUT STRUCTURE | Overall Hull vs selected LSOA; compact panel text; cite JSON |
| 15 | PROHIBITED CLAIMS | Explicit ban list (causation, merges, invented context, screenshots) |
| 16 | SELF-CHECK | Twelve-point revision gate before answering |

---

## 7. Evidence contract

The user payload is a compact JSON package (not raw CSVs). Typical fields, when present:

- application (domain, study area, spatial unit, target)
- stakeholder id / label
- current map view
- scope: overall Hull or selected LSOA
- global GeoSensitivity summaries
- global spatial pattern (Moran, LISA counts, `pattern_scale`)
- selected LSOA: GeoSensitivity, rank, lat/lon effects, LISA, top-10 non-location features
- top global features (published Feature branch, location excluded)
- evaluation: holdout R², surrogate fidelity, **Consistency**, ATT stability, faithfulness
- limitations
- optional `user_question`

**Substitution bans:** do not fill a missing local field with a global ranking; do not swap GeoSensitivity with lat/lon; do not treat map colour as LISA.

---

## 8. Stakeholder adaptation

The prompt changes **wording, depth, and section headings**. It does not change the numbers.

| id | Label | Emphasis | Output skeleton |
|---|---|---|---|
| `technical` | Technical / Model Analyst | Exact values, signed effects, reliability | Main finding → quantitative evidence → spatial structure → local features → reliability → interpretation boundary |
| `domain` | Domain / Operational Expert | What stands out, where, what to inspect | What stands out → where → local influences → spatial pattern → what it may mean → investigate → limitation |
| `decision` | Decision Maker / Manager | 2–4 findings, magnitude, evidence strength | Key finding → why it matters → where → evidence strength → limitation → next step |
| `governance` | Governance / Assurance | Provenance, unsupported-inference risk | Finding → source → reliability → geographic-dependence → limitation → assurance |
| `public` | Public / Affected Stakeholder | Plain language; what it does **not** mean | What the model shows → what that means → what it does not mean → how certain |

Missing stakeholder id defaults to `technical`, and the model must say so.

“Next step” / “investigate” means **evidence inspection**, not operational crime control.

---

## 9. Guardrails that matter for crime applications

The prompt is written to block common GeoAI failure modes:

- Attribution written as cause  
- Geography written as causing crime  
- GeoSensitivity merged with latitude or longitude  
- Rank treated as strength  
- LISA HH treated as a crime hotspot  
- Clustering inferred from the choropleth  
- `pattern_scale` invented when the artefact says “not established”  
- Residents labelled “risky” because of a score  
- Invented deprivation / lighting / policing / ethnicity narratives  
- Policing-deployment recommendations  

Evaluation metrics, when used, must be labelled as evaluation:

- Surrogate test R² ≈ attribution approximation quality  
- Holdout MAE/RMSE/R² ≈ predictive accuracy on the stated window  
- Faithfulness ≈ top-versus-random intervention contrast  
- **Consistency** ≈ multi-run seed agreement (Rank Spearman, Score Pearson, Top-20% Jaccard)  
- **ATT stability** ≈ controlled-input neighbour expansion (not Consistency)

---

## 10. What the prompt does not do

- It does not recompute GeoSMILE or LISA.  
- It does not inspect screenshots.  
- It does not browse the web or a document store.  
- It does not authorise causal or policy claims from coefficients.  
- It does not replace the maps or tables; it narrates the JSON that those artefacts already support.

---

## 11. Integrity tests

Automated checks in `tests/test_dashboard_llm.py` require that:

- the prompt starts with `# GEOSMILE EXPLAIN — SYSTEM POLICY`
- numbered sections including ROLE, evidence contract, and spatial-pattern semantics are present
- OpenAI and Anthropic import the same `GEOSMILE_SYSTEM_PROMPT` object
- `dashboard/llm/prompt.py` is full policy text (no `importlib` loader)
- `llm/prompt.py` and `dashboard/llm/prompt.py` strings match
- `assert_prompt_semantics()` still finds the required “is not” clauses

---

## 12. Summary for examiners

The dashboard uses a **single, versioned system policy** (`geosmile-hull-v1`) shared by both LLM providers. The policy binds the model to published GeoSMILE JSON, keeps GeoSensitivity distinct from latitude/longitude feature attribution, forbids causal and hotspot language, and adapts **form** (not facts) to five stakeholder types. GeoSensitivity is defined operationally as the per-unit attribution of the **shift in target predictions** under location masking of geographic units.
