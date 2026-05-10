# Rasyn ADMET Rescue — Master Plan

> **Status:** Draft v0.4 (2026-05-10). Owner: Ansh. Implementer: Claude (Opus 4.7, 1M ctx).
> **Remote:** https://github.com/ansschh/wolverine — branch `main`. The full repo (specs + plan + memory + rasyn/ code) is mirrored there.
> **v0.4 changes:** Aggressive parallel-track shipping in one session. Now complete:
>   - Phase B-0 schemas (11 modules, ~30 tests)
>   - Phase A-0 sealed-case registry stub + populator + canary generator
>   - Utils (canonicalize/similarity/descriptors/identifiers)
>   - Data sources (ChEMBL/PubChem/TDC/MoleculeNet adapters)
>   - Decontamination (Pass-0 quarantine + canary audit)
>   - Evidence builder (rule-based v1 with liability driver detection)
>   - 6 proposer channels (3 deterministic complete, 3 ML-bearing scaffolded)
>   - 8 baselines
>   - Eval harness (Mode A + Mode B + functional-recovery scorer + metrics)
>   - Heuristic ranker (full Ranker contract; usable BEFORE training)
>   - Torch ranker scaffold (ConcatMLP architecture + featurizer; ready for training)
>   - Auxiliary ADMET predictor scaffold
>   - Training entry points (pretrain/train_ranker/finetune/calibrate)
>   - RescuePairDataset + parquet loader scaffold
>   - Audit pack (locked-prediction I/O with hash verify; full pack assembler)
>   - Reports (per-case card + investor 1-slide + 13-section technical appendix)
>   - 12-pass curation orchestrator skeleton
>   - Layer-1 smoke script + populate/canary scripts
>   - GitHub Actions CI (chemistry-free test suite)
>   - ~80 tests passing (chemistry-touching tests skip if RDKit absent)
> **v0.3 changes:** (a) Phase B-0 (scaffold + frozen schemas + tests) shipped; (b) Phase A-0 (sealed-case registry stub) shipped; (c) GitHub remote configured.
> **v0.2 changes:** (a) papers deferred to a methodology-first workstream (§16); (b) compute estimate added (§17).
> **Read order on resume:** this file → `MEMORY.md` → the 5 spec `.md`s at repo root.
> **Code lives in** `A:/rasyn-case-studies/rasyn/`. Specs stay at repo root, untouched.

---

## 0. North Star

Build a chemistry AI ("Rasyn") that, in a contamination-controlled blind benchmark, **discovers the held-out solution to three sealed ADMET-rescue cases**:

| Case | Parent → Answer | Liability | Rescue mode |
|---|---|---|---|
| ADMET-001 | Terfenadine → Fexofenadine | hERG / cardiac | active-metabolite / polarity ↑ |
| ADMET-002 | Acyclovir → Valacyclovir | Oral exposure | L-valyl ester prodrug |
| ADMET-003 | OXS007570 → OXS008474 | Solubility + metabolic stability | Direct analog (N-insertion, ↓clogP) |

Headline claim Rasyn must support after evaluation:
> *"Rasyn discovered all three held-out ADMET rescue solutions in a contamination-controlled blind benchmark — answers hidden before training, predictions locked before reveal."*

The 8-condition definition of "discovered" in `rasyn_heldout_discovery_demo_context.md` is binding.

---

## 1. Scope

### In scope (this project, this timeline)
- 3 sealed ADMET cases above
- Dataset curation from public sources: **ChEMBL, PubChem BioAssay, TDC, MoleculeNet** (papers deferred — see Out of scope and §16)
- 6-channel proposer ensemble (analog retrieval, MMP, liability rules, learned inverse-delta, forward-reward, learned novelty)
- Pairwise rescue ranker (parent–candidate)
- Evidence builder + auxiliary "clean" predictors
- 8 baselines
- 2-mode eval harness (open proposer, closed hard-ranking)
- Decontamination + canary + nearest-neighbor audit
- Locked-prediction ledger + 13-section technical appendix + 1-slide investor summary

### Explicitly OUT of scope (deferred or skipped)
- ❌ **Antibiotic discovery** (3 cases) — deferred. Plumb proposer/ranker generically so this can be added later without rewrites.
- ❌ **NMR / spectra suite** — deferred.
- ❌ **Internal / proprietary chemistry data** — none available. All sources public.
- ❌ **Wet-lab validation** — not running any experiments.
- ❌ **Custom decontamination thresholds** — using spec defaults (Tanimoto ≥0.85 to answer; ≥0.65 with same Murcko + same target/liability). Revisit only if audit flags problems.
- ⏸ **Paper SAR extraction (gold-tier from PDFs)** — *deferred*, not cancelled. Treated as a **methodology-first workstream** (§16): we must design + validate a *safe* extraction pipeline (likely structured-LLM extract → dual-reviewer → provenance hashing → decontamination cross-check) before any paper-derived row is allowed into training. Track A v1 ships **silver-tier-only** without papers; gold tier promotes paper rows in once the methodology lands.

---

## 2. Hard constraints

1. **Extreme time crunch.** Optimize for parallel work. Block as little as possible on synchronous decisions.
2. **Plan-first.** This document and `MEMORY.md` get updated **before** code lands.
3. **Continuity.** Every meaningful decision/observation goes in `MEMORY.md`. Assume the next agent starts cold.
4. **No internal data, no wet-lab, defaults are fine.** Don't burn time re-litigating these.
5. **No invented credentials, no fake URLs, no guessed dataset versions.** Anything that needs a human action gets flagged in §10.
6. **Decontamination is sacred.** Sealed-case Pass-0 quarantine runs *before* any pair mining. Canaries inserted before cleaning, verified removed after.
7. **Architecture-agnostic until Layer 2.** The spec deliberately defers neural family choice — we keep that option open until empirical signal forces a pick.

---

## 3. System architecture (synthesized from the 5 spec files)

```
                ┌─────────────────────────────────────────────┐
                │ Sealed-case registry  (forbidden_entities)  │
                └─────────────────────┬───────────────────────┘
                                      │ Pass-0 quarantine
                                      ▼
   ChEMBL ─┐                    ┌─ canonicalize ─┐
   PubChem ┼──── raw_corpus ───►│   desalt       ├─► clean_corpus ─► decontam_audit
   TDC     │                    │   tautomer     │            │
   Mol-Net │                    │   InChIKey     │            │ canary check
   Papers ─┘                    └────────────────┘            ▼
                                                       4-table dataset
                                                       ┌──────────────┐
                                                       │ molecule     │
                                                       │ assay_fact   │
                                                       │ rescue_pair  │
                                                       │ candidate_set│
                                                       └──────┬───────┘
                                                              │
                                                              ▼
   ADMETChallengePacket ──► Proposer ensemble (6 ch) ──► raw 5–20k cands
                                                              │
                                              filters (validity, parent-relative,
                                              rescue-mode, decontam) → 500–2000
                                                              │
                                                              ▼
                                       Evidence builder + auxiliary "clean" predictors
                                                              │
                                                              ▼
                                          Pairwise Rescue Ranker (PyTorch)
                                                              │
                                            ┌─────────────────┴──────────────────┐
                                            ▼                                    ▼
                                     Mode A: open                      Mode B: closed hard-ranking
                                     (rank top-5/10/20)                (rank sealed pool incl. answer)
                                                              │
                                                              ▼
                                          Locked predictions  +  audit pack
```

### Frozen schemas (Phase 0 deliverable)
`ADMETChallengePacket`, `ProposerRequest`, `ProposerOutput`, `CandidateEvidencePacket`, `RankerInput`, `RankerOutput`, `LockedPrediction`, `SealedCaseRegistry`, `ForbiddenEntities`, `DecontaminationConfig`, `BaselineConfig`, `DatasetManifest` — all Pydantic v2 + JSON Schema export, hash-stable serialization.

### Activity-retention buckets (default, overrideable per case)
- `strong_retention`: ≤3× parent potency
- `acceptable_retention`: ≤10×
- `weak_retention`: 10–100×
- `failed_retention`: >100×

### Liability-improvement targets (default v1)
- Solubility: ≥5–10× improvement
- Metabolic stability (ER): move toward <0.3
- hERG: risk-category reduction (no worsening)

### Quality tiers
- **Gold** (target 500–2,000 pairs): measured both sides, same/compatible assay, curator-reviewed. Source: curated paper rows + same-document ChEMBL.
- **Silver** (10k–100k): measured both sides, heterogeneous assays. Source: ChEMBL + PubChem joins, ChEMBL + TDC overlap.
- **Bronze** (millions): predicted liabilities, weak supervision only.

---

## 4. Parallel-track strategy

We have two tracks that can run almost fully in parallel and converge on Layer-1 verification.

```
       T0                                                    T_converge
        │                                                         │
Track A │ ingest ─► canonicalize ─► quarantine ─► tables ─► v0.1  │
─data─  │                                                         │ ─► Layer-1 smoke test
Track B │ schemas ─► baselines + proposer 1–3 ─► eval harness     │
─arch── │                  └─ synthetic-data fixture ─────────────┘
```

**Track A — Data curation.** Network/IO-bound. Long wall-clock for downloads.
**Track B — Architecture.** Compute-light. Runs against synthetic fixtures until A produces v0.1.

The "easy-to-get data converted into our model-compatible format" the user described is the **synthetic-data fixture** in Track B (Phase B-2) — small, hand-constructed parent-candidate pairs that match the real schema, used to smoke-test every component before A is done.

---

## 5. Track A — Data curation pipeline

### A-0. Sealed-case + forbidden-entities lock (BLOCKS EVERYTHING)
Generate `sealed_case_registry.yaml` with:
- For each of the 3 cases: parent + answer SMILES, InChIKey, salts, tautomers, stereo variants, synonyms (incl. CAS, ChEMBL IDs, PubChem CIDs), associated DOIs/PMIDs (incl. **`10.1039/d4md00275j`** for OXS), ChEMBL doc IDs, assay IDs, patents.
- Quarantine radii: spec defaults.
- Insert ≥30 canaries per case (synthetic SMILES + synthetic doc IDs + synthetic synonyms) into raw data.

**Output:** `rasyn/data/registry/sealed_case_registry.yaml`, `forbidden_entities.json`.

### A-1. Source ingestion adapters
- **ChEMBL** (≈ChEMBL 35, free, no auth): bulk SQLite dump preferred over API for full-corpus mining.
- **PubChem BioAssay**: PUG REST for targeted pulls + FTP bulk for broad screens.
- **TDC**: `tdc.single_pred.ADME`, `Tox` etc.
- **MoleculeNet**: via DeepChem loaders.
- **Papers**: ⏸ **deferred**. No paper-derived rows in v1. See §16 for the methodology-first workstream that must complete before paper data is allowed in.

**Output:** `rasyn/data/raw/{chembl,pubchem,tdc,molnet,papers}/` + per-source manifest with hashes.

### A-2. Canonicalization
RDKit + `chembl_structure_pipeline`: standardize, desalt, neutralize, tautomer normalize, InChIKey, Murcko scaffold, ECFP4/Morgan-1024, descriptor block (MW, logP, TPSA, HBD/HBA, RotB, fsp3, aromatic rings, formal charge).

**Output:** `rasyn/data/clean/molecules.parquet`.

### A-3. Pass-0 decontamination
Apply quarantine BEFORE any pair mining:
- Exact match (canonical SMILES, InChIKey, salt/tautomer/stereo variants of forbidden list).
- Synonym/identifier scrub across documents and assay metadata.
- Document/assay quarantine (DOI/PMID/ChEMBL doc/assay IDs).
- Neighborhood removal: Tanimoto ≥0.85 to any forbidden molecule, OR (≥0.65 AND same Murcko AND same target/liability).
- Canary verification: 100% of seeded canaries removed → halt if any survive.

**Output:** `rasyn/data/clean/decontam_audit_pre.json` + nearest-neighbor table.

### A-4. 4-table build (13-pass workflow from `rasyn_curating_the_dataset.md`)
Passes:
1. ChEMBL activity contexts (target × standard_type × document)
2. PubChem ADMET/toxicity facts
3. TDC + MoleculeNet auxiliary endpoints
4. ⏸ Paper rows — **skipped at v1** (deferred per §16; silver only without papers)
5. (skipped — internal data)
6. Analog graph (ECFP-Tanimoto + Murcko + MMP + MCS)
7. Pair generation (parent-candidate edges with activity context + liability)
8. Activity-retention bucketing
9. Liability-improvement category labeling
10. Hard-negative construction (5 types: ADMET-improved-but-activity-lost, activity-retained-but-liability-unfixed, wrong-liability-improved, new-liability-introduced, heuristic trap)
11. Local ranking-task assembly
12. Quality-tier assignment
13. Structured-rationale auto-generation (rule-based → reviewer-promoted)

**Output:** `rasyn/data/clean/{molecules,assay_facts,rescue_pairs,candidate_sets}.parquet` + `dataset_manifest.json` (with hashes + leakage audit).

### A-5. Final decontamination audit + freeze
Re-run quarantine on derived tables, regenerate canaries report, freeze hashes.

**Output:** `rasyn/data/clean/decontam_audit_post.json`, `dataset_manifest.json` (frozen).

---

## 6. Track B — Architecture implementation

### B-0. Repo scaffold + frozen schemas
Pydantic v2 models for every schema in §3, JSON Schema export, round-trip tests, deterministic hashing utility.

**Output:** `rasyn/schemas/*.py`, `rasyn/tests/test_schemas.py`.

### B-1. Synthetic data fixture
Hand-construct ~100 parent-candidate pairs covering all rescue modes + all hard-negative classes. Match real schema. Inject canaries.

**Output:** `rasyn/tests/fixtures/synthetic_v0.parquet`.

### B-2. Proposer channels 1–3 (deterministic, no ML)
- Analog retrieval (FAISS over Morgan fingerprints)
- MMP transformer (mining + applying)
- Liability-specific rule packs (hERG, solubility, metabolic stability, prodrug)

These run on synthetic fixture and silver-tier data alike.

**Output:** `rasyn/proposer/{analog,mmp,rules}.py`.

### B-3. Evidence builder + auxiliary clean predictors
Descriptor deltas, similarity stack, pharmacophore proxy (RDKit feature maps), shape similarity (RDKit ROC-mode or skipped at v1), liability driver detection (rule-first), synthesizability (SAScore).

Auxiliary predictors trained on TDC/MoleculeNet endpoints — these are the "clean" predictors used as evidence at inference. Decontaminated training set.

**Output:** `rasyn/evidence/*.py`, `rasyn/aux_models/*.py`.

### B-4. 8 baselines
random, similarity-only, most-polar, liability-only-property, activity-only, weighted-property, MMP-frequency, medchem-heuristic. None require training.

**Output:** `rasyn/baselines/*.py`.

### B-5. Eval harness
Mode A (open proposer): generate → filter → rank → top-k.
Mode B (closed hard-ranking): rank a sealed pool with answer + decoys.
Metrics: exact-recall@{5,10,20}, functional-recall@{5,10,20}, MRR, invalid rate, novelty rate, diversity, failure-mode calibration.
Per-proposer-channel attribution.

**Output:** `rasyn/eval/{harness,metrics,functional_recovery}.py`.

### B-6. Proposer channels 4–6 (ML-bearing, scaffolded only until Layer 2)
- Learned inverse-delta proposer (P + Δ + ctx → C)
- Forward-reward optimizer
- Pure learned novelty proposer

Code complete with training entry-points, but **no actual training** until Layer-2 verification gate.

**Output:** `rasyn/proposer/{inverse,forward_opt,novelty}.py`, `rasyn/training/*.py`.

### B-7. Pairwise rescue ranker (scaffolded)
Hybrid input: parent-candidate pair + structured evidence + structured rationale fields. Pluggable backbone (start with concat-MLP; design admits GNN / transformer / fingerprint-MLP ensemble swap).
Outputs: `rescue_score`, `rescue_label_probs`, `activity_retention_pred`, `liability_improvement_pred`, `failure_mode_probs`, `confidence`.

**Output:** `rasyn/ranker/*.py`.

### B-8. Locked-prediction ledger + audit pack
Hash-stable JSON, immutable timestamps, reveal-protocol enforcement, 13-section appendix template.

**Output:** `rasyn/audit/*.py`, `rasyn/reports/template_appendix.md`.

---

## 7. Convergence: three-layer verification

The user-requested "test the waters before burning compute" gate. Each layer must pass before the next. **No full training run starts until Layer 3 is green.**

### Layer 1 — Smoke / E2E plumbing (model-execution: ~10–30 min per loop, CPU)
Goal: prove every box in §3 is wired correctly.
Inputs: synthetic fixture (B-1) only.
Pass criteria:
- All 6 proposer channels produce schema-valid output for all 3 sealed cases (using *fake* sealed cases for synthetic).
- Evidence builder produces well-formed `CandidateEvidencePacket` for every candidate.
- Ranker (random init) returns valid scores; eval harness computes recall@K without crashing.
- All 8 baselines run.
- Decontamination removes 100% of seeded canaries.
- Locked-prediction format round-trips with stable hash.
- ≥95% of unit tests green.

### Layer 2 — Single-target real-data slice (model-execution: ~1–4 hr; small GPU helpful, CPU acceptable)
Goal: prove the SIGNAL exists and is learnable on real data.
Inputs: ONE liability family end-to-end (likely **hERG**, since terfenadine/fexofenadine maps cleanly and ChEMBL hERG data is rich). Silver-tier only.
Pass criteria:
- Tiny ranker (small embedding + MLP, <5M params) trains for ~minutes and **beats `random` and `similarity-only` baselines** on held-out hERG pairs.
- Decontamination + Pass-0 successfully blocks terfenadine→fexofenadine pair from training set (verified by canary).
- Functional-recovery scoring distinguishes plausible rescues from hard negatives on a synthetic mini-bench.
- Activity-retention buckets + liability buckets show non-trivial calibration (not all "uncertain").

### Layer 3 — Pre-flight at scale (model-execution: ~6–24 hr; GPU required)
Goal: catch dataset/pipeline bugs before burning the full compute budget.
Inputs: ~10% of full clean dataset, all 3 liability families, all rescue modes.
Pass criteria:
- "Production-architecture-but-smaller" model trains end-to-end without OOM, NaN, divergence.
- Outperforms ALL 8 baselines on held-out (non-sealed) pairs.
- Decontamination audit clean (zero canary survivors, NN audit all justified).
- Locked-prediction dry-run for ADMET-001 produces well-formed output.
- Per-channel proposer attribution shows non-degenerate distribution (no single channel dominating to the exclusion of others).

**Only after Layer 3 passes:** Phase 8.

---

## 8. Full training stack (post-Layer-3)

This is the part with the real human gate (compute provisioning).

### Stage 1 — Pretraining (auxiliary heads on clean public data)
- Mask-style + property-prediction multi-task on decontaminated ChEMBL + TDC + MoleculeNet
- Builds the molecule encoder backbone
- Outputs feed evidence builder + ranker + proposer 4–6

### Stage 2 — Main training (rescue ranker + proposers 4–6)
- Local ranking loss + auxiliary heads (retention bucket, improvement bucket, failure-mode, rationale fields)
- Hard-negative mining built into batch construction
- Training-time decontamination check on every batch (sanity)

### Stage 3 — Finetuning (per-rescue-mode specialization)
- Direct-analog mode head
- Prodrug mode head (separate scoring formula — delivery probability, not direct potency)
- Active-metabolite mode head

### Stage 4 — Calibration
- Failure-mode probability calibration (isotonic / Platt on held-out pairs)
- Confidence calibration

### Stage 5 — Sealed-case inference + lock
1. Generate `ADMETChallengePacket` for each of the 3 cases (no answer leakage).
2. Run Mode A (open proposer) — 5–20k raw → 500–2000 filtered → top 5/10/20.
3. Run Mode B (closed hard-ranking) — sealed pool with answer + decoys.
4. Lock predictions (timestamp, model hash, input hash, output hash) — IMMUTABLE.
5. Reveal answers, score, build audit pack.

---

## 9. Evaluation & audit deliverables

- `eval/results/admet_001.json` × {Mode A, Mode B}
- `eval/results/admet_002.json` × {Mode A, Mode B}
- `eval/results/admet_003.json` × {Mode A, Mode B}
- `audit/sealed_case_registry.yaml` (frozen)
- `audit/decontam_audit_pre.json`, `audit/decontam_audit_post.json`
- `audit/canary_report.json`
- `audit/nearest_neighbor_table.csv`
- `audit/dataset_manifest.json` (with hashes)
- `audit/training_manifest.json` (with checkpoints + hashes)
- `audit/locked_predictions/{admet_001,002,003}.json` (immutable)
- `reports/technical_appendix.md` (13 sections)
- `reports/investor_one_slide.md`
- `reports/per_case_card_{001,002,003}.md`

---

## 10. Phase-by-phase plan with model-execution estimates

Estimates are in tool-calls (TC) and conversation-sessions (S ≈ 2 hr). Wall-clock for downloads/training is flagged separately as **WC**. Human gates flagged 🚧.

| Phase | Track | Description | Model TC | Sessions | Wall-clock | Human gate |
|---|---|---|---|---|---|---|
| B-0 | B | Scaffold + schemas + tests | 80–200 | 1 | — | none |
| B-1 | B | Synthetic fixture | 60–120 | 0.5 | — | none |
| B-2 | B | Proposers 1–3 | 200–400 | 1–2 | — | none |
| B-3 | B | Evidence builder + aux predictors (code only) | 250–500 | 1.5–2 | — | none |
| B-4 | B | 8 baselines | 150–300 | 1 | — | none |
| B-5 | B | Eval harness + metrics | 200–400 | 1–2 | — | none |
| **L1** | both | **Layer-1 smoke verification** | 80–200 | 1 | minutes | none |
| A-0 | A | Sealed-case registry + canaries | 100–250 | 1 | — | none (defaults) |
| A-1 | A | Source adapters | 200–400 | 1–2 | **WC: hours of download** | 🚧 disk space (~100 GB) |
| A-2 | A | Canonicalization | 150–300 | 1 | WC: 1–4 hr CPU | none |
| A-3 | A | Pass-0 decontam | 100–200 | 0.5 | WC: minutes | none |
| A-4 | A | 4-table build (12 passes; pass-4 papers skipped) | 500–1000 | 2.5–4 | WC: hours CPU | none |
| A-5 | A | Final decontam + freeze | 100–200 | 0.5 | — | none |
| B-6 | B | Proposers 4–6 (code) | 400–800 | 2–3 | — | none |
| B-7 | B | Ranker (code) | 300–600 | 2 | — | none |
| **L2** | both | **Layer-2 single-target real-data** | 200–500 | 1–2 | **WC: 1–4 hr GPU** | 🚧 GPU access |
| **L3** | both | **Layer-3 pre-flight at scale** | 200–500 | 1–2 | **WC: 6–24 hr GPU** | 🚧 GPU budget |
| Stage 1–4 | train | Full pretrain → train → finetune → calibrate | 200–500 | 1–2 | **WC: days GPU** | 🚧 full GPU budget; 🚧 architecture-family pick |
| Stage 5 | train | Sealed-case inference + lock | 100–200 | 1 | WC: minutes | none |
| B-8 | B | Audit pack + reports | 250–500 | 1–2 | — | none |
| Final | both | Reveal + technical appendix + investor slide | 150–300 | 1 | — | none |

**Total model work, all phases except training itself: ~4,000–8,000 TC ≈ 20–40 sessions.**
**Wall-clock dominated by:** ChEMBL download (hours), 4-table build (hours), training stages (days).

---

## 11. Human gates remaining

These block specific phases — none block planning or B-0–B-5.

| # | Gate | Blocks | Default if not answered | When needed by |
|---|---|---|---|---|
| 1 | Disk + network access for ChEMBL/PubChem bulk | A-1 | n/a | start of Track A |
| 2 | ~~GPU access~~ | ~~L2, L3, training~~ | **✅ RESOLVED 2026-05-10: 16× H100 + 16× A100 = 32 GPUs total** (see §17.A, §17.B) | n/a |
| 3 | Architecture family pre-commit | training Stage 1 | **with 32-GPU access: pre-commit transformer family (~200–500M, escalate to ~1B if signal supports) at Layer-3.** Concat-MLP only at Layer-2 for fast signal check. | Stage 1 |
| 4 | ~~Compute budget cap~~ | ~~training~~ | **✅ "No cap, gate per major run"** — Claude must check in with user before launching: Stage-1 pretrain, each Stage-2 sweep batch, Stage-3 finetune, Stage-5 sealed-case inference + lock. Report estimated cluster-hours + intent before each launch. | per-launch |
| 5 | Paper extraction methodology lock (§16) | gold-tier promotion only | silver-only v1; gold deferred to post-§16 workstream | post-Layer-3 |
| 6 | Functional-recovery judge (auto vs. human) | eval | auto-only, pre-registered criteria | post-Layer-3 |
| 7 | Decontamination audit acceptance | freeze | self-audit + canary report; flag for review | A-5, freeze |
| 8 | Sign-off on locked predictions before reveal | Stage 5 | block reveal until user OKs | Stage 5 |

---

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Decontamination leakage via paper text in pretraining corpus | Document/DOI/PMID + assay-ID quarantine + neighborhood removal + canaries; regen if any canary survives. |
| Activity-cliff bias in retention buckets | Buckets are coarse (3×/10×/100×); per-target threshold override allowed; flagged in audit. |
| Prodrug case (ADMET-002) requires "delivery probability" — no measured PK | Rule-based prodrug-motif detector + structural plausibility; mark predictions with prodrug-mode flag; do NOT score on direct potency. |
| Forward-reward proposer hacking the reward | Validity + parent-relative + uncertainty penalties; flag unusually high-reward candidates for review; track invalid-rate per channel. |
| Canary survives decontamination | Pipeline halts; debug; do not proceed. |
| Layer-2 fails to beat baselines | Means signal isn't there at this slice — investigate (data quality, label noise, threshold choice) before scaling. Cheap to redo. |
| Layer-3 OOM / NaN at small scale | Catch BEFORE full training. Re-architect or shrink. |
| OXS paper (DOI `10.1039/d4md00275j`) appears in pretraining corpus | Explicit DOI quarantine in A-0; canary check on author/title fragments. |
| Single-channel proposer dominates | Per-channel attribution metric; rebalance during Stage 2. |
| Confidence miscalibration | Stage 4 calibration with held-out pairs; isotonic regression. |
| Spec drift between PLAN.md and 5 source docs | Keep PLAN.md as the synthesis, link back to source line-numbers in MEMORY.md when decisions are made. |

---

## 13. Directory layout

```
A:/rasyn-case-studies/
  PLAN.md                                  # this file
  MEMORY.md                                # running log
  proposer_system_test_cases.md            # spec (locked)
  rasyn_admet_conditioning_architecture_benchmark_spec.md
  rasyn_admet_rescue_architecture_context.md
  rasyn_curating_the_dataset.md
  rasyn_heldout_discovery_demo_context.md
  rasyn/
    pyproject.toml
    schemas/             # Pydantic v2 models
    data/
      registry/          # sealed_case_registry, forbidden_entities
      raw/{chembl,pubchem,tdc,molnet,papers}/
      clean/             # 4 parquet tables + manifests + audits
    proposer/            # 6 channels
    evidence/            # builder + similarity + descriptors
    aux_models/          # clean ADMET / activity-retention predictors
    ranker/
    baselines/           # 8 baselines
    eval/                # harness, metrics, functional-recovery scorer
    training/            # pretrain / train / finetune / calibrate
    audit/               # canary, NN audit, manifest, locked-pred ledger
    reports/             # technical appendix, investor slide, per-case cards
    tests/
      fixtures/          # synthetic_v0.parquet, etc
```

---

## 14. How to resume this project (cold-start protocol for a future agent)

1. Read `PLAN.md` (this file) end-to-end.
2. Read `MEMORY.md` bottom-up by date (most recent first) until you have ~10 sessions of context.
3. Skim the 5 spec markdowns at repo root — they are the source of truth; if PLAN.md disagrees with a spec, the spec wins (and update PLAN.md + log it in MEMORY.md).
4. Run `pytest rasyn/tests` — tests document the contract.
5. Check current state via `git log` (once initialized) and the latest `dataset_manifest.json` + `training_manifest.json` if present.
6. Find the most recent "next step" entry in MEMORY.md and continue.
7. Update MEMORY.md with every meaningful action, observation, decision, or failure as you go. Brevity is fine; completeness over polish.

---

## 16. Deferred — Paper SAR extraction methodology workstream

**Why deferred (per user 2026-05-10):** *"For papers we will have to decide how to extract safely so that the model gets trustable structured data."* Cheap PDF-to-table extraction is unsafe for two reasons:

1. **Hallucinated structure.** Free-form LLM extraction of (parent SMILES, candidate SMILES, activity value, units, liability value) from PDFs is error-prone. A wrong SMILES or off-by-an-order-of-magnitude potency goes straight into gold-tier training and corrupts the rescue label.
2. **Decontamination escape hatch.** Papers carry author names, synonyms, structural drawings, and partial SMILES that may slip past sealed-case quarantine — esp. for OXS where related medchem papers from the same group exist.

Until the methodology below is locked + dry-run-validated, **no paper-derived row enters any training set**.

**Methodology to design (separate workstream, not blocking v1):**

Phase P-0. **Source whitelist + sealed-case cross-check.**
- Define explicit DOI/PMID whitelist of papers eligible for extraction (NOT the sealed-case papers; NOT obvious neighbors).
- Run sealed-case quarantine *first* against the whitelist — drop any paper whose abstract/title/authors trip the forbidden-entities scrub.

Phase P-1. **Structured extraction pipeline.**
- Tier-1 (mechanical): RDKit-friendly supplementary tables (CSV/XLSX) → direct parse with schema validation.
- Tier-2 (LLM-structured): for body-text SAR tables, structured-output LLM call (JSON schema fixed: parent_smiles, candidate_smiles, activity_endpoint, activity_value, activity_unit, liability_endpoint, liability_value, liability_unit, transformation_class, page_ref). Reject any row where SMILES doesn't round-trip through RDKit canonicalization.
- Tier-3 (figure OCR): explicitly skip at v2; revisit later.

Phase P-2. **Dual-reviewer with disagreement adjudication.**
- Two independent extractions per row (could be 2 LLM calls with different prompts, or LLM + rule-based parser).
- Auto-promote to silver if both agree byte-for-byte after canonicalization.
- Auto-reject if they disagree on SMILES or differ >2× on numeric value.
- Flag for human review only the residual.

Phase P-3. **Provenance hashing + traceability.**
- Every extracted row stores: source DOI, page, paragraph hash, extraction model + version, extraction timestamp, dual-reviewer agreement bits.
- Allows full retraction of a paper's rows if a leakage incident is found later.

Phase P-4. **Validation dry-run.**
- Run pipeline against ~10 known papers where ground-truth structured tables already exist (e.g., from ChEMBL document mappings).
- Acceptance: ≥98% SMILES match, ≥95% activity-value match within unit-conversion tolerance, zero leakage of sealed-case structures.

Phase P-5. **Promotion to gold tier.**
- Only papers that pass P-0–P-4 get rows promoted from silver to gold.
- Paper rows always carry a `source=paper_extracted` flag in `rescue_pair_table` so they can be ablated out.

**Estimated effort for the methodology workstream itself:** ~600–1500 TC (~3–6 sessions) for design + Phase P-0 to P-4. Wall-clock dominated by LLM extraction calls (free if using Claude in-conversation; rate-limited if API-based).

**When to start:** post-Layer-3, once the silver-only baseline is established. That way we can A/B the gold-paper-tier addition cleanly and prove papers actually move metrics.

---

## 17. Compute & infrastructure estimate

Real numbers so cloud / hardware decisions can be made now. Estimates are honest ranges; tighter numbers possible after Layer-2 fixes the architecture choice.

### Storage
| Item | Size |
|---|---|
| Raw downloads (ChEMBL bulk SQLite ~20 GB; PubChem BioAssay subset ~50–100 GB; TDC + MoleculeNet ~few GB) | **~100–200 GB** |
| Clean tables (4 parquet) | ~20–50 GB |
| Model checkpoints + logs | ~5–20 GB |
| Audit pack + locked predictions | <1 GB |
| **Total disk required** | **~150–300 GB** (SSD strongly preferred) |

### RAM (preprocessing)
- Canonicalization + fingerprinting + MMP mining: **64–128 GB ideal**; 32 GB workable with chunking.
- Analog graph (FAISS over Morgan fingerprints): 16–32 GB.

### CPU (preprocessing)
- 16–32 cores recommended for ChEMBL canonicalization + analog graph build.
- Wall-clock with 32 cores: **~24–72 hours** for full pipeline.
- Wall-clock with 8 cores: ~5–10 days.

### GPU (training stages) — three options to choose from

| Option | GPU | Pretrain (Stage 1) | Main train + 5–10 sweep runs (Stage 2) | Finetune (Stage 3) | Total wall-clock | Total cost |
|---|---|---|---|---|---|---|
| **Recommended** | 1× A100 80 GB (cloud) | ~50–100 hr | ~50–100 hr | ~10–20 hr | **~5–10 days** | **~$240–700** @ ~$2/hr |
| **Faster** | 2× A100 80 GB or 1× H100 (cloud) | ~25–50 hr | ~25–50 hr | ~5–10 hr | **~3–5 days** | **~$300–1500** |
| **Owned** | 1× RTX 4090 24 GB | ~150–300 hr | ~150–300 hr | ~30–60 hr | **~3–6 weeks** continuous | $0 marginal |

### Sanity totals
- **Model GPU compute: ~120–350 single-A100-equivalent hours** for full pretrain → train → finetune → calibrate stack with sweep.
- **Cloud cost (A100 @ ~$2/hr): ~$240–700** mid-estimate.
- **Cloud cost (H100 @ ~$4/hr, less wall-clock): ~$200–800** mid-estimate.
- **Wall-clock from compute go-ahead to locked predictions: ~1–2 weeks** with cloud, ~4–6 weeks on owned 4090.

### Caveats / sensitivity
- Numbers assume **~50–150M-param ranker** (concat-MLP backbone + small molecular encoder). If Layer-3 forces escalation to a 500M+ transformer, **multiply by ~3–5×**.
- Hyperparameter sweep budget assumes ~5–10 runs across LR, batch size, hard-negative ratio, evidence weighting.
- Auxiliary "clean" predictors (Phase B-3) add ~10–20 GPU-hr — folded into pretrain budget above.
- Layer-2 verification: **~1–4 GPU-hr** (fits on free Colab T4 if needed).
- Layer-3 verification: **~6–24 GPU-hr** (needs proper GPU).

### What can compress this
- **Use a public pretrained molecular encoder** (Uni-Mol, MolFormer, ChemBERTa) and skip Stage 1 → ~50% time savings. ⚠️ But every public encoder must be re-checked for sealed-case leakage; many saw fexofenadine + valacyclovir during their pretraining. Net savings only realized if a leakage-free public checkpoint is found.
- **Skip novelty proposer (channel 6)** at v1 → ~10–15% training time saved; lose moonshot channel.
- **Single-mode instead of per-mode finetuning** → saves Stage 3; lose specialization edge for prodrug case.
- **Skip 3D / shape similarity** → already the default at v1; explicitly noted to keep flexibility.

### What can blow this up
- 500M+ params: ~3–5× cost.
- Adding 3D / shape similarity at full quality: ~2× preprocessing + ~50% inference.
- Decontamination miss → forced retrain from scratch: 2× total compute.
- Iteration loops between Layer-3 and full training (e.g., ranker underperforms, swap architecture): each redo is ~50–100% of Stage-2 budget.

### Hard ask for human (gates G2 + G4)

1. **Confirm a compute provider** before Layer-2 starts. Recommended: **Lambda Labs** or **RunPod** (simplest A100 access, ~$1.50–2.50/hr). AWS/GCP are fine but heavier setup. Owned 4090 viable but multiplies wall-clock 3–6×.
2. **Approve a compute budget cap.** Suggested: **$1000 ceiling** on cloud (~30% above mid-estimate, hard halt at cap with mandatory user re-auth to overrun).
3. **Decide preprocessing host.** Either same cloud GPU instance (CPU-bound preprocessing on a non-GPU instance is cheaper, ~$0.10–0.50/hr), or a local 32-core machine with 128 GB RAM if available.

### Bottom line
**$240–700 cloud cost + ~1–2 weeks wall-clock** is the realistic mid-estimate to take Rasyn from "Layer-3 green" to "locked predictions on all 3 sealed cases."

---

### 17.A — ACTUAL compute available (resolved 2026-05-10): **16× H100**

User has access to **16× H100 80GB**. Strategic shifts:

**Throughput.** 16-H100 cluster ≈ ~12–25× single A100 with reasonable DDP/FSDP scaling. Total training wall-clock collapses:

| Stage | Was (1× A100) | Now (16× H100) |
|---|---|---|
| Pretrain (Stage 1) | 50–100 hr | **~3–8 hr** |
| Main train + 5–10 sweep runs (Stage 2) | 50–100 hr (sequential) | **~3–8 hr** sequential, or **~1–2 hr** if 4×4-GPU jobs in parallel |
| Finetune (Stage 3) | 10–20 hr | **~1–2 hr** |
| Layer-2 verification | 1–4 hr | **~10–30 min** (1 GPU) |
| Layer-3 verification | 6–24 hr | **~30 min – 2 hr** |
| **Total training wall-clock** | **5–10 days** | **~6–20 hours** |

**Cost.** If the 16 H100s are owned / pre-paid / sponsored: ~$0 marginal. If renting equivalent cluster (~$25–40/hr on H100): ~$150–800 total — still well under any cap.

**The bottleneck shifts.** With training collapsing to ~hours, the long pole becomes:
1. **Preprocessing wall-clock** (~24–72 hr CPU). Now > training wall-clock. Front-load aggressively. Run on a separate non-GPU instance to avoid burning H100-hours on CPU work.
2. **Pipeline correctness.** Bugs caught at full scale waste >12× more aggregate compute than at single-A100 scale. Verification discipline (Layers 1–3) matters MORE, not less.
3. **Decontamination integrity.** A retrain due to leakage discovery now still costs ~1 day instead of ~2 weeks, but it's avoidable — get it right the first time.

**Strategic plan revisions enabled by 16× H100:**

1. **Architecture escalation is cheap.** Default starting architecture for **Layer-3 onward escalates from concat-MLP → real transformer (~200–500M params)** without budget anxiety. Layer-2 still uses small model for fast signal-check.
2. **Parallel ablations replace sequential sweep.** Stage-2 sweep runs as **4 parallel 4-GPU jobs** instead of sequential, cutting sweep wall-clock 4×.
3. **Pretrain can be richer.** Full ChEMBL + augmented data + longer schedule + larger model all fit comfortably. Explicitly: do multi-task pretrain (mask + property heads on TDC's 22 endpoints + activity-context conditioning) instead of cheaper single-task.
4. **Per-rescue-mode finetuning is essentially free.** Run all 3 mode-specific finetunes in parallel (3 GPUs each, 13 GPUs total).
5. **Multi-seed evaluation.** Train 3–5 seeds of the final model in parallel for variance estimates. Investor-credibility bonus, near-zero marginal cost.

**Plan adjustments to apply downstream:**
- §7 Layer-3 gate: now also checks DDP/FSDP scaling efficiency before committing full-cluster runs.
- §8 Stage 1: target ≥200M-param backbone (vs. earlier 50–150M).
- §8 Stage 2: parallel sweep, not sequential.
- §11 G2 (GPU access): **resolved.** G3 (architecture pre-commit): can pre-commit transformer family at L3 with high confidence.
- Wall-clock from compute go-ahead → locked predictions: **~3–5 days total** (preprocessing-bound), not 1–2 weeks.

**What's still slow.** Preprocessing (~24–72 hr) and human-loop iterations (decontamination audit review, sealed-prediction sign-off). Compute is no longer the bottleneck. **Front-load preprocessing in Track A as priority #1 the moment scaffold is ready.**

**Budget-gating discipline (per user choice "no cap, ask before each major run"):**
Claude must check in with user before launching:
- Stage 1 pretrain
- Each Stage-2 sweep batch
- Stage 3 finetune
- Stage 5 sealed-case inference + lock
Report estimated cluster-hours + intent before each launch; await OK.

---

### 17.B — UPDATED total compute (resolved 2026-05-10): **16× H100 + 16× A100 = 32 GPUs**

User clarified: in addition to the 16× H100, also has **16× A100**. **Total cluster: 32 GPUs.** Aggregate ≈ 16×H100 + 16×A100 ≈ ~56 single-A100-equivalent throughput.

**This makes compute essentially free for this project.** Strategy shifts further:

**Cluster partition (proposed default):**

| Workload | Allocation | Why |
|---|---|---|
| **Pretrain (Stage 1)** | 16× H100 (full H100 cluster) | Throughput-bound; H100 FlashAttention/transformer perf wins. |
| **Aux predictor training (Phase B-3)** | 4× A100 in parallel with pretrain | TDC's 22 endpoints → 22 small heads; trivially parallel; doesn't need H100s. |
| **Preprocessing acceleration (descriptors, fingerprints, MMP mining)** | 4× A100 (chemistry libs that have GPU paths) + CPU host | Cuts the ~24–72 hr preprocessing wall-clock by ~30–50%. RDKit-CUDA / cuML / FAISS-GPU. |
| **Stage-2 main training + sweep** | Split: 8 parallel jobs of 2× H100 each (or 4× 4-GPU jobs) | Run all hyperparameter variants concurrently; sweep finishes in ~one job's wall-clock. |
| **Stage-3 per-mode finetune (3 modes)** | 3 parallel jobs of 4–8 GPUs each | All 3 modes trained simultaneously. |
| **Multi-seed final training (3–5 seeds)** | 3–5 parallel jobs of 4–8 GPUs | Variance estimates for free. |
| **Layer-1, Layer-2, ablations, debugging** | 1–4 A100s as needed | Spare capacity for ad-hoc work without touching the H100 cluster. |

**Updated total wall-clock:**

| Phase | Earlier (1× A100) | 16× H100 only | **32 GPUs (current)** |
|---|---|---|---|
| Preprocessing | 24–72 hr CPU | 24–72 hr CPU | **~12–36 hr** with GPU-accelerated descriptors/FAISS |
| Pretrain Stage 1 | 50–100 hr | 3–8 hr | **~3–8 hr** (already H100-bound) |
| Stage 2 main + sweep (10 runs) | 50–100 hr sequential | 3–8 hr sequential, 1–2 hr parallel | **~1–2 hr** parallel + aux model training overlapped |
| Stage 3 finetune (3 modes) | 10–20 hr sequential | 1–2 hr sequential | **~30 min** parallel |
| Multi-seed retraining (5 seeds) | not budgeted | optional | **~3–8 hr** parallel — now feasible |
| Layer-1 / Layer-2 / Layer-3 | hours–day | minutes–hour | minutes–hour (spare A100s) |
| **Total wall-clock from "go" → locked predictions** | **~5–10 days** | **~3–5 days** | **~1.5–2.5 days** (preprocessing-bound) |

**The bottleneck is now unambiguously preprocessing + human review loops.** Compute is solved. To compress further:
1. GPU-accelerate preprocessing (cuML for descriptors, FAISS-GPU for fingerprint similarity, parallel chunked canonicalization across 4 A100 hosts).
2. Front-load preprocessing the moment scaffold is ready — Track A is now **the critical path**, Track B is overlapped.
3. Parallelize human review (decontamination audit) by giving user pre-summarized findings, not raw logs.

**New strategic options unlocked by 32-GPU access:**

1. **Architecture ambition.** Pre-commit transformer at ~500M–1B params for the ranker backbone. Ablations against smaller variants run in parallel for free.
2. **Encoder choice.** Train our own molecular encoder from scratch on decontaminated ChEMBL (no public-encoder leakage risk) — was previously a tradeoff, now obviously the right call.
3. **Multi-seed everything.** All Layer-3 + final results reported with seed variance; investor-credibility multiplier.
4. **Wide hyperparameter sweep.** ~16+ configurations tested simultaneously instead of 5–10 sequentially.
5. **Per-rescue-mode dedicated heads from Stage 1.** Joint-training the mode-specific heads alongside the backbone (instead of finetuning later) becomes feasible with this much compute.
6. **Multiple full pretrain runs.** Can run 2–3 pretrain variants (different masking strategies, different aux objectives) and pick the best — was previously infeasible.
7. **Live retraining if decontamination must be redone.** A leakage discovery → retrain costs ~1 day instead of ~2 weeks. De-risks the "must we redo?" decision.

**What does NOT change with more GPUs:**
- Verification discipline (Layers 1–3) is **still mandatory.** Spending 32-GPU-hours on a buggy pipeline still wastes a day. Cheap-fail-fast at small scale is still the right pattern.
- Decontamination still must be perfect before launching any pretrain.
- Sealed-case lock still must be human-signed before reveal.

**Plan adjustments to apply (supersedes §17.A in part):**
- §8 Stage 1: target **~500M–1B-param transformer backbone**; train our own (no public encoder).
- §8 Stage 2 + Stage 3: parallelize aggressively. All sweep + all finetuning in one wall-clock window.
- §8 add Stage 1.5: **multi-pretrain-variant selection** (run 2–3 variants in parallel, keep best at Layer-3).
- §10 phase table: training-stage estimates collapse to hours, not days.
- Track A becomes the critical path. Front-load A-1 (ingestion) and A-2 (canonicalization) the moment B-0 scaffold lands.

---

## 18. What's next (immediate)

After user OK on this plan:
- **Phase B-0** (scaffold + schemas) — start in next conversation, ~1 session, no human gates.
- In parallel kickoff, define `sealed_case_registry.yaml` (Phase A-0) — also no gates.
- Then unblock Track A by starting source adapters (Phase A-1) — flag disk-space / network requirement at that point.

— end PLAN.md v0.1 —
