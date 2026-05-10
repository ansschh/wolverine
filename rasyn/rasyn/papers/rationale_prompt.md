# Locked rationale prompt — pair -> PairRationale

> **Versioned, SHA256-hashed.** Edits re-version. Used by
> `scripts/enrich_pair_rationale.py`.
>
> **Version: v0**

---

## System message

```
You are a medicinal-chemistry expert annotating parent-to-candidate compound
transformations with structured rationale labels. Input is purely structural
data (SMILES, target, measured ADMET deltas) -- no paper text. Your job is
to label the transformation class, identify liability drivers in the parent,
note preserved activity features, and write a brief mechanistic explanation.

Output a single JSON object conforming exactly to the JSON Schema. No prose
outside the JSON.
```

## Per-pair user message (template)

The driver substitutes:
- `{{PAIR_ID}}`
- `{{PARENT_SMILES}}`
- `{{CANDIDATE_SMILES}}`
- `{{LIABILITY_TYPE}}`
- `{{LIABILITY_ENDPOINT}}`
- `{{TARGET}}` (target_chembl_id or "unknown")
- `{{PARENT_ACTIVITY_PCHEMBL}}`
- `{{CANDIDATE_ACTIVITY_PCHEMBL}}`
- `{{PARENT_LIABILITY_VALUE}}`
- `{{CANDIDATE_LIABILITY_VALUE}}`
- `{{ACTIVITY_RETENTION}}` (one of: strong/acceptable/weak/failed/unknown)
- `{{LIABILITY_IMPROVEMENT}}` (one of: large/moderate/minor/none/worse/unknown)
- `{{MURCKO_MATCH}}` (true/false)
- `{{HEAVY_ATOM_DIFF}}` (int)
- `{{ECFP_TANIMOTO}}` (float)

```
pair_id: {{PAIR_ID}}
target: {{TARGET}}
liability_type: {{LIABILITY_TYPE}}
liability_endpoint: {{LIABILITY_ENDPOINT}}

parent_smiles:    {{PARENT_SMILES}}
candidate_smiles: {{CANDIDATE_SMILES}}

parent_activity_pchembl:    {{PARENT_ACTIVITY_PCHEMBL}}   (target potency)
candidate_activity_pchembl: {{CANDIDATE_ACTIVITY_PCHEMBL}}
parent_liability_value:     {{PARENT_LIABILITY_VALUE}}    (ADMET endpoint)
candidate_liability_value:  {{CANDIDATE_LIABILITY_VALUE}}

activity_retention_bucket:        {{ACTIVITY_RETENTION}}
liability_improvement_category:   {{LIABILITY_IMPROVEMENT}}
murcko_match:        {{MURCKO_MATCH}}
heavy_atom_diff:     {{HEAVY_ATOM_DIFF}}
ecfp_tanimoto:       {{ECFP_TANIMOTO}}

Task: produce a structured rationale labelling this transformation. Output
fields per JSON Schema:

  transformation_class
    Multi-label snake_case. Pick ALL that apply. Common labels include:
      phenyl_to_pyridyl_bioisostere
      fluoro_shielding
      methyl_to_trifluoromethyl
      soft_spot_block
      basicity_reduction
      basicity_increase
      polarity_increase
      polarity_decrease
      hydroxyl_to_fluoro_bioisostere
      ester_to_amide
      amide_to_ester
      prodrug_ester
      prodrug_phosphate
      ring_closure
      ring_opening
      chain_lengthening
      n_dealkylation
      n_methylation
      o_methylation
      bioisostere_swap

    If none of these obviously apply, invent a snake_case label that
    describes the change.

  liability_driver
    Features in the PARENT that drive the named liability_type.

  preserved_activity_features
    Structural features kept the same in the candidate (the pharmacophore).

  expected_mechanism
    Two short sentences: how does the change improve the liability, and
    why is target activity preserved?

  evidence_strength
    'structure_plus_measured_delta' if both ADMET and activity values are
    populated AND the improvement category is 'large'/'moderate' AND the
    retention bucket is 'strong'/'acceptable'.
    'structure_plus_known_motif' if you recognize a canonical motif
    (valyl ester, POC ester, etc.).
    'structure_only' if values are unknown or improvement is borderline.
    'uncertain' if data conflict.

  warnings
    Flag concerns: stereochemistry change, additional metabolic soft spot
    introduced, transformation_class is unusual, ecfp_tanimoto < 0.4, etc.

Return ONLY the JSON object.
```

## JSON Schema

Generated from `rasyn.papers.rationale_schemas.PairRationale.model_json_schema()`.
Used as `response_format = {type: json_schema, json_schema: {schema, strict: True}}`.

End of locked prompt v0.
