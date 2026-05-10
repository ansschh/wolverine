# Locked extraction prompt — paper -> ExtractedRescuePairBatch

> **Versioned, SHA256-hashed.** Any edit re-versions the prompt and invalidates
> earlier extractions (per L25 + audit-trail requirements).
>
> **Version: v0**  (will be hashed + locked at first run; hash stored in
> `extraction_prompt.lock.json` next to this file.)

---

## System message

```
You are a medicinal-chemistry expert extracting parent-to-rescue compound
pairs from primary medicinal-chemistry literature. Your output goes into a
benchmark dataset that will be cited in research; precision matters far more
than recall. When in doubt, return an empty list.

Output a single JSON object that conforms exactly to the JSON Schema
provided. Do NOT include any prose outside the JSON.
```

## Per-paper user message (template)

The driver substitutes:
- `{{PAPER_DOI}}` — DOI of the paper
- `{{PAPER_PMID}}` — PubMed ID (or "null")
- `{{PAPER_TITLE}}` — paper title
- `{{PAPER_TEXT}}` — full extracted text body + tables + SI methods (markdown)

```
Paper DOI: {{PAPER_DOI}}
Paper PMID: {{PAPER_PMID}}
Paper title: {{PAPER_TITLE}}

Paper full text (body + tables + SI):
---
{{PAPER_TEXT}}
---

Task: Extract every parent->candidate compound pair from this paper that
satisfies ALL FOUR HARD RULES below. If zero pairs satisfy all four rules,
return {"valid_pairs": []}. Do NOT pad. Do NOT include partial pairs.

==== HARD RULES ====

RULE 1 — explicit SMILES required.
  Both parent_smiles and candidate_smiles MUST be one of:
   (a) explicitly written as a SMILES string in the paper or SI, OR
   (b) explicitly named with CAS / IUPAC / trivial name in a way that
       uniquely identifies the structure (so an external DB lookup would
       give a single answer), OR
   (c) shown as a numbered structure in a table whose SI provides SMILES
       or InChI for that compound number.
  If the structure is only inferable from a 2D drawing in the paper without
  an accompanying machine-readable identifier, DO NOT extract.

RULE 2 — same assay, same conditions.
  parent_metric and candidate_metric MUST be measurements of the SAME
  liability assay (same endpoint name, same units, same conditions or
  paper explicitly states they are comparable). If parent was measured
  via "hERG IC50 (patch clamp)" and candidate via "hERG % inhibition at
  10 uM", these are NOT comparable -- DO NOT extract.

RULE 3 — narrative-driven improvement.
  The paper text MUST explicitly describe the candidate as an improvement
  over the parent in the liability_type. Quoted phrasing required (e.g.
  "compound 12 showed improved hERG IC50 of 5 uM vs compound 6's 0.05
  uM"). Improvements that are only inferred from a structural change
  WITHOUT measured value comparison are NOT acceptable.

RULE 4 — primary-target potency preserved.
  The paper MUST contain measurements showing primary-target activity is
  preserved within 10x of the parent, OR the paper text explicitly says
  "activity was retained" / "potency was maintained" with the candidate
  remaining at clinically relevant levels.

==== LIABILITY TAXONOMY (use exactly these strings) ====

  hERG                 hERG IC50, hERG inhibition, QT prolongation,
                       cardiac off-target risk
  solubility           kinetic / thermodynamic aqueous solubility, logS
  metabolic_stability  microsomal half-life, intrinsic clearance,
                       hepatocyte clearance
  oral_exposure        oral F%, AUC, Cmax, prodrug bioavailability
  permeability         Caco-2 Papp, PAMPA, MDCK efflux ratio

==== CONFIDENCE LEVELS ====

  high   - SMILES from SI / InChI; values explicit; same-assay confirmed
  medium - SMILES from named compound lookup; values explicit
  low    - SMILES from drawing only or values qualitative; SHOULD NOT
           be returned (per RULE 1) unless there is independent
           identifier-quality info; if returned, will be filtered in P-2

==== OUTPUT FORMAT ====

Return a single JSON object conforming to the JSON Schema below. No
prose, no markdown, no commentary. Just the JSON object.

When uncertain, omit the pair. Empty valid_pairs is the correct answer
for many papers (e.g. computational-design-only papers, pure pharmacology
papers, scaffold-hopping papers without explicit ADMET narrative).
```

## JSON Schema (passed to vLLM as `response_format = {"type": "json_schema", ...}`)

The schema is generated from `rasyn.papers.schemas.ExtractedRescuePairBatch`
via `model.model_json_schema()`. The runtime serializer is:

```python
from rasyn.papers.schemas import ExtractedRescuePairBatch
schema = ExtractedRescuePairBatch.model_json_schema()
response_format = {"type": "json_schema",
                    "json_schema": {"name": "ExtractedRescuePairBatch",
                                    "schema": schema, "strict": True}}
```

vLLM 0.20.2 supports xgrammar-backed structured outputs which strictly
enforce the JSON Schema, eliminating hallucinated fields.

## Provenance fields populated by the driver (NOT by the LLM)

The LLM does not produce these — the driver wraps the LLM's
`{"valid_pairs": [...]}` response with:

- `paper_doi`, `paper_pmid`, `paper_title` — known from the call site
- `extraction_timestamp_utc` — set at LLM call time
- `model_id` — `"casperhansen/llama-3.3-70b-instruct-awq"`
- `prompt_sha256` — `sha256(this_file_bytes)`
- `extraction_runtime_ms` — measured

## Iteration discipline

When this prompt is updated:
1. Bump version (v0 -> v1)
2. Re-hash, write `extraction_prompt.lock.json` with new sha256
3. Re-run on the entire ground_truth_set.yaml
4. Compute precision/recall vs expected; require precision >= 95%
5. Only after the gate passes, run on real corpus

## Out of scope (do NOT add to this prompt)

- Free-form rationale generation (handled in P-2 dual-reviewer step)
- Rescue-mode classification beyond liability_type (Stage 2 ranker job)
- Candidate scoring (Stage-5 inference, not extraction)

---

End of locked prompt v0.
