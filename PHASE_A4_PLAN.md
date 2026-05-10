# Phase A-4 detailed sub-plan — Build the rescue-pair dataset (the missing critical path)

> **Status:** Draft 2026-05-10. Plan-aligned recovery after deviation (L25/L26).
> **Read order on resume:** PLAN.md (§5 Track A, §3 architecture) → spec `rasyn_curating_the_dataset.md` (the source of truth for this phase) → this file → MEMORY.md.

## Why this exists

Phase A-4 was treated as a skeleton in `data/curate/passes.py`. Only Pass 0 (decontam) and an early canonicalization step ran. **The rescue-pair dataset that Stage 2 trains on does not exist** — that's why I had to fall back to a HeuristicRanker (deviation). The whole chain needs Phase A-4 outputs to function as the plan describes.

## What we currently have on disk + on git

**Local artifacts (`A:/rasyn-case-studies/artifacts/`, ~6.5 GB):**
- `datasets/molecules_canonical.parquet` — 2.47M canonicalized + decontaminated SMILES (Pass 1-2 partial). Columns: `chembl_id, canonical_smiles, inchi_key, max_phase`. **Missing: assay/activity/target context columns.**
- `registry/sealed_case_registry.yaml` — populated for ADMET-001/002 (parent + answer); ADMET-003 partial (parent OXS007570 from CHEMBL4853189; answer OXS008474 not in any public DB).
- Aux ADMET predictor checkpoints (8 from-scratch + 4 finetuned) — useful as evidence-builder ML inputs for Pass 9 (liability improvement labeling) and at inference time.
- Stage-1 SMILES LM backbone (`smiles_lm_pretrain_chembl.pt`) — needed for Stage-2 ranker init.

**What we DON'T have (the gaps):**
- ChEMBL bulk SQLite is gone (H200 was terminated). Need to re-download to get activity/assay/target/document tables.
- PubChem BioAssay subsets — never extracted.
- Analog graph (Pass 6).
- `rescue_pairs.parquet` (Pass 7) — the Stage-2 training set.
- `assay_facts.parquet` (Pass 1, 2) — joins for activity context.
- `candidate_sets.parquet` (Pass 11) — local ranking tasks.
- Hard negatives (Pass 10).
- Quality-tier assignment (Pass 12).
- Structured rationales (Pass 13).

## Spec reference (source of truth)

Per `rasyn_curating_the_dataset.md`:
- Four-table schema: **molecule, assay_fact, rescue_pair, candidate_set** (we have molecule only).
- 13-pass mining workflow.
- Hard negatives (5 types) are MANDATORY (lines 775–834 of spec).
- Quality tiers gold/silver/bronze (lines 837–866).
- Structured rationales auto-generated (lines 1067–1118).

Pass 4 (papers) is deferred per L16 — confirmed by user. Pass 5 (internal data) skipped. The remaining 11 passes ALL must run.

## The 11-pass execution plan

For each pass: **purpose, input, output, principal risk, work estimate**.

### Pass 1 — ChEMBL activity contexts

- **Purpose:** Build per-(molecule, target, assay, document) activity records. Needed for Pass 7 to know which target a molecule was tested against.
- **Input:** ChEMBL 35 bulk SQLite (re-download; ~5 GB tar.gz → ~30 GB SQLite).
- **Output:** `assay_facts.parquet` columns: `molecule_chembl_id, target_chembl_id, target_pref_name, assay_chembl_id, assay_type, document_chembl_id, doi, standard_type, standard_relation, standard_value, standard_units, pchembl_value, valid_units_flag`.
- **Filter:** `standard_relation = '='` preferred; convertible units only; `data_validity_comment IS NULL`.
- **Risk:** ChEMBL 35 has ~21M activity rows. Streaming + filtering needed; can't fit in RAM.
- **Work:** Re-download ChEMBL on a pod (~5 min); stream activities table → parquet (~10–20 min CPU).

### Pass 2 — PubChem ADMET/toxicity facts

- **Purpose:** Auxiliary toxicity / hERG / cytotoxicity labels not in ChEMBL.
- **Input:** PubChem BioAssay AIDs from `LIABILITY_AID_HINTS` (in `rasyn/data/sources/pubchem.py`) — hERG (1903, 588834), solubility (1996, 603846), metabolic stability (1645841, 1645842).
- **Output:** rows in `assay_facts.parquet` with `source='pubchem'`.
- **Filter:** active/inactive labels only; merge via canonical-SMILES → InChIKey → `molecule_chembl_id` lookup against Pass 1.
- **Risk:** PubChem CIDs don't always map cleanly to ChEMBL IDs — orphan records.
- **Work:** ~30 min on a pod with PubChem PUG REST + canonicalization.

### Pass 3 — TDC + MoleculeNet auxiliary endpoints

- **Purpose:** ADMET predictions and per-molecule property labels for evidence builder + later weak-label generation.
- **Input:** TDC's 22 ADMET datasets + relevant MoleculeNet (BBBP, Tox21, FreeSolv).
- **Output:** rows in `assay_facts.parquet` with `source='tdc'` or `source='molnet'`. Decontaminated against sealed cases.
- **Risk:** TDC datasets may include sealed-case answer molecules (fexofenadine, valacyclovir likely in solubility/permeability sets).
- **Work:** ~10 min Python — already partially done (we used TDC for aux training); now cement to parquet.

### Pass 4 — Paper rows (DEFERRED per L16)

- **Status:** Skipped at v1 per user scope lock.
- **Note:** Methodology workstream (P-0 through P-5) deferred to post-Layer-3.

### Pass 5 — Internal data (SKIPPED per L2)

- **Status:** No internal data available per scope lock.

### Pass 6 — Analog graph

- **Purpose:** Build the parent ↔ candidate edge set on which Pass 7 generates rescue pairs.
- **Inputs:** `molecules_canonical.parquet` + `assay_facts.parquet`.
- **Output:** `analog_edges.parquet` columns: `parent_id, candidate_id, ecfp_tanimoto, murcko_match, mcs_atom_count, mmp_transformation_class, heavy_atom_diff, same_target_set, same_document_set`.
- **Edge criterion:** ECFP4 Tanimoto ≥ 0.5 OR (Murcko match AND ≤6 heavy atom diff) OR MMP-detectable (matched-pair single-fragment substitution).
- **Risk:** Combinatorial blow-up. 2.47M molecules × 2.47M = infeasible. Must restrict to within-target groups (same `target_chembl_id` set from Pass 1).
- **Work:** ~30–60 min on multi-core CPU (or A100 with FAISS-GPU for the Tanimoto step). Estimated ~5–50M analog edges depending on threshold.

### Pass 7 — Pair generation

- **Purpose:** Convert analog edges into oriented rescue-pair candidates. Each edge becomes (P, C) with rescue context (target activity + liability deltas).
- **Input:** `analog_edges.parquet` + `assay_facts.parquet`.
- **Output:** `rescue_pair_candidates.parquet` columns: `pair_id, parent_chembl_id, candidate_chembl_id, target_chembl_id, liability_type, parent_activity, candidate_activity, parent_liability, candidate_liability, mmp_transformation_class, scaffold_tanimoto, ...`.
- **Orientation rule:** P → C if both have measured activity at same target (within 1 unit log scale) AND C has measured liability improvement (or vice versa).
- **Risk:** Spurious orientation when assay types mismatch.
- **Work:** ~20–40 min CPU.

### Pass 8 — Activity-retention bucketing

- **Purpose:** Label each pair with retention bucket (strong/acceptable/weak/failed/unknown) per spec §11 / `rasyn_admet_conditioning_architecture_benchmark_spec.md` thresholds (3×/10×/100× potency).
- **Input:** `rescue_pair_candidates.parquet`.
- **Output:** adds column `activity_retention_bucket`.
- **Risk:** Censored values (`>10 µM`) need bucket-based handling per spec.
- **Work:** ~5 min Python.

### Pass 9 — Liability-improvement category labeling

- **Purpose:** Label each pair with `liability_improvement_category` (large/moderate/minor/none/worse) per spec §3.4 (PLAN.md §3 "Liability-improvement targets").
- **Input:** `rescue_pair_candidates.parquet`.
- **Output:** adds column `liability_improvement_category`.
- **Risk:** Endpoint-specific thresholds need per-liability dispatch (solubility ≥5–10×, ER<0.3, hERG risk-category change).
- **Work:** ~5 min Python.

### Pass 10 — Hard-negative construction (5 types)

Per spec §15 / `rasyn_curating_the_dataset.md` lines 775–834. **MANDATORY.**

- **Type 1 — ADMET-improved-but-activity-lost:** liability fixed, but activity retention is `failed` (>100× drop).
- **Type 2 — Activity-retained-but-liability-unfixed:** activity strong/acceptable, but liability `none` or `worse`.
- **Type 3 — Wrong-liability-improved:** improved a different liability than the one specified by `rescue_mode`.
- **Type 4 — New-liability-introduced:** parent had no `hERG` issue but candidate does.
- **Type 5 — Heuristic trap:** candidates a naive ranker (similarity-only, polarity-only) would pick but that fail real rescue criteria.
- **Output:** adds column `hard_negative_type` (or `null` for true positives).
- **Risk:** Class imbalance — too many easy negatives swamp signal.
- **Work:** ~30 min CPU (per-type extraction + label).

### Pass 11 — Local ranking-task assembly

- **Purpose:** Group candidates per (parent, liability) into ranking tasks. Stage 2 trains a local ranking objective.
- **Input:** `rescue_pair_candidates.parquet` + `analog_edges.parquet`.
- **Output:** `candidate_sets.parquet` columns: `ranking_task_id, parent_id, liability_type, candidate_ids[list], rescue_labels[list], hard_negative_types[list]`.
- **Per-task size:** ~10–50 candidates per parent (mix of positives + hard negatives).
- **Work:** ~10 min Python.

### Pass 12 — Quality-tier assignment

- **Purpose:** Mark each pair gold/silver/bronze per spec quality tiers.
- **Tier rules (from spec):**
  - **Gold:** measured both sides, same/compatible assay, curator-reviewed → 0 (no curator at v1; gold tier remains empty until paper-extraction methodology lands).
  - **Silver:** measured both sides, heterogeneous assays, plausible analog (most ChEMBL pairs).
  - **Bronze:** predicted liabilities, weak supervision (TDC/MolNet sourced).
- **Input:** all prior parquets.
- **Output:** adds `quality_tier` column.
- **Work:** ~5 min Python.

### Pass 13 — Structured-rationale auto-generation

- **Purpose:** Produce structured rationale fields per spec §4.7 (NOT free-form text). Fields: `liability_driver_features, modified_features, preserved_activity_features, transformation_class, expected_delta_direction, failure_mode_risks`.
- **Input:** `rescue_pair_candidates.parquet` + RDKit-derived liability driver detection (already exists in `rasyn/evidence/liability_drivers.py`).
- **Output:** rationale columns added to `rescue_pair_candidates.parquet`.
- **Work:** ~15 min Python.

## Acceptance gates for Phase A-4 completion

Per PLAN.md §5 / spec quality targets:

1. **Silver pairs: 10K–100K** (target). With ChEMBL alone we should hit the lower end.
2. **Hard negatives: ≥30%** of total pair set across all 5 types.
3. **Decontamination integrity:** all sealed-case parents/answers + Tanimoto-≥0.85 neighbors absent from `rescue_pair_candidates.parquet`. Re-run canary audit.
4. **Per-target balance:** no single target with >20% of pairs (avoid hERG-data-only over-fit).
5. **Schema validates:** all rows pass Pydantic models.
6. **Manifest hashed:** `dataset_manifest.json` produced with SHA256s of every parquet.

## Compute / wall-clock estimate

| Step | Pod | Compute | Wall-clock |
|---|---|---|---|
| Re-download ChEMBL bulk | 1 A100 pod (or any with disk) | network + I/O | ~10 min |
| Pass 1 (ChEMBL activity stream) | same pod, CPU | streaming SQL | ~20 min |
| Pass 2 (PubChem subsets) | same pod, CPU | network | ~30 min |
| Pass 3 (TDC + MolNet stamp) | same pod, CPU | already cached | ~10 min |
| Pass 6 (analog graph) | A100 with FAISS-GPU OR 16+ CPU cores | Tanimoto + Murcko | ~30–60 min |
| Pass 7 (pair generation) | CPU | join + filter | ~20 min |
| Pass 8–13 (labeling + tiering + rationales) | CPU | per-row ops | ~60 min total |
| **Total** | one A100 pod | ~3–4 hours wall-clock |

## Outputs that drop into Stage 2 directly

- `rasyn/data/clean/molecules_canonical.parquet` (already exists ✅)
- `rasyn/data/clean/assay_facts.parquet` (NEW — Passes 1+2+3)
- `rasyn/data/clean/analog_edges.parquet` (NEW — Pass 6)
- `rasyn/data/clean/rescue_pair_candidates.parquet` (NEW — Passes 7–10, 12, 13)
- `rasyn/data/clean/candidate_sets.parquet` (NEW — Pass 11)
- `rasyn/data/clean/dataset_manifest.json` (NEW — frozen + hashed)

## What this UNBLOCKS

- **Stage 2 (B-7 trained ranker):** train pairwise rescue ranker on `rescue_pair_candidates.parquet` with hard-negative mining from Pass 10.
- **B-6 channel 4 (learned inverse-delta proposer):** train on (parent, candidate, delta, ctx) tuples with rescue labels.
- **Layer-2 verification:** "tiny ranker beats random + similarity baselines on held-out hERG pairs" — only possible with a real held-out set, which is in `candidate_sets.parquet`.
- **Layer-3 preflight:** scaling check.
- **Stage 5 sealed-case inference (REAL VERSION):** trained ranker + trained generator, replaces the heuristic-ranker deviation.

## Risk register

| Risk | Mitigation |
|---|---|
| ChEMBL re-download is slow on a fresh pod | Run on a single pod with persistent disk; SCP final parquets back to local |
| Analog graph blows up combinatorially | Restrict to within-target groups; use FAISS for Tanimoto NN search |
| Hard negatives too imbalanced | Sample 1:1 with positives in Stage-2 batch construction (not at dataset level) |
| Decontamination misses something | Re-run canary audit on `rescue_pair_candidates.parquet` (mandatory acceptance gate) |
| OXS-related contamination through ChEMBL doc joins | Already covered by sealed-case-registry document-id quarantine |

## Order of operations

I will:

1. **First:** spin up a single A100 pod (one of the 4 we have keys for); confirm with you which IP/credentials.
2. **Second:** write `rasyn/scripts/build_rescue_pair_dataset.py` (the orchestrator that runs passes 1–13, skipping 4 and 5).
3. **Third:** run Pass 1 (ChEMBL re-download + activity extraction) and SCP `assay_facts.parquet` back to local before running Pass 6+.
4. **Fourth:** run Passes 6–13 in sequence; SCP each artifact back to local as it lands.
5. **Fifth:** verify acceptance gates (especially canary audit on `rescue_pair_candidates.parquet`).
6. **Stop.** Report results + manifest hashes. Do not proceed to Stage 2 until you confirm Phase A-4 is good.

## Hard rule reminder (from L25)

I will NOT:
- Skip any pass and use a placeholder.
- Use existing aux ADMET predictor outputs as substitutes for Pass 9 measured-data labels (they're separate things — predictions are evidence-time; Pass 9 labels are training-time ground truth).
- Lower the hard-negative count to "ship faster."
- Loosen quality tier rules.
- Reuse the molecules_canonical.parquet decontamination as a substitute for re-running canary audit on the rescue_pair_candidates.parquet.

## Question for you before I start

Two confirmations needed (since this is a substantial wall-clock + compute commitment):

1. **Which pod do I run Phase A-4 on?** Pod C (`161.33.194.122`, asia-northeast-2) already has `molecules_canonical.parquet` on disk and the venv set up — least friction. OK to proceed there?
2. **Do you confirm Pass 4 (paper extraction) stays deferred per L16?** Or do you want me to also build the paper extraction methodology (P-0 through P-5 from PLAN.md §16) as part of this same work session? It's a separate large workstream (~3–6 sessions / 600–1500 TC of additional implementation, plus 10K–100K paper review effort).
