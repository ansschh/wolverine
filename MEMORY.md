# Rasyn Project — Running Memory Log

> **Purpose:** This is the project's continuous memory. Every meaningful decision, observation, attempt, failure, breakthrough, and open question goes here so that **if this conversation runs out of context, the next agent has the complete history**.
>
> **This is project-level memory.** Distinct from any model-personal memory system. Treat this file as the single source of project history.

---

## How to read this file

- **Newest first.** New entries go at the top.
- **Resume protocol:** read `PLAN.md` first, then this file top-down until you have ~5–10 entries of context.
- **If a spec disagrees with PLAN.md or with this log, the source spec wins.** Update PLAN.md + log the discrepancy here.

## How to write entries (template)

```
### YYYY-MM-DD — short title
**Type:** decision | observation | attempt | result | question | hypothesis | gate-status
**Phase:** A-x / B-x / Lx / Stage-x / planning / meta
**Context:** one sentence
**Detail:**
- ...
**Outcome / next:** ...
**Refs:** PLAN.md §x, file:line, spec-name §x
```

Use headings, bullet lists, and tables freely. Do NOT prune old entries — append-only. If something supersedes a prior entry, link to it: "supersedes 2026-05-10 entry on X".

---

## Locked decisions snapshot (kept current)

> Update this section every time a hard decision is made. Source of truth for "what's settled".

| # | Decision | Date | Locked by |
|---|---|---|---|
| L1 | ADMET-only scope. Antibiotic + NMR/spectra deferred. | 2026-05-10 | user |
| L2 | No internal/proprietary data. Public sources only (ChEMBL, PubChem, TDC, MoleculeNet, papers). | 2026-05-10 | user |
| L3 | No wet-lab validation, ever, in this project. | 2026-05-10 | user |
| L4 | Default decontamination thresholds (Tanimoto ≥0.85 to answer; ≥0.65 + same Murcko + same target/liability). Revisit only if audit flags problems. | 2026-05-10 | user |
| L5 | Plan-first, then code. PLAN.md and MEMORY.md are kept current. | 2026-05-10 | user |
| L6 | Parallel tracks: Track A = data, Track B = architecture against synthetic fixture. Converge at Layer-1 smoke test. | 2026-05-10 | user |
| L7 | Three-layer verification before any full training run. Layers in PLAN.md §7. | 2026-05-10 | user |
| L8 | Code lives in `A:/rasyn-case-studies/rasyn/`. Specs at repo root untouched. | 2026-05-10 | user |
| L9 | 3 sealed ADMET cases per `proposer_system_test_cases.md` §1007–1049: terfenadine→fexofenadine, acyclovir→valacyclovir, OXS007570→OXS008474. | 2026-05-09 | spec |
| L10 | OXS case requires DOI `10.1039/d4md00275j` quarantine. | 2026-05-09 | spec |
| L11 | Activity-retention buckets: ≤3× strong, ≤10× acceptable, 10–100× weak, >100× failed. | 2026-05-09 | spec |
| L12 | Liability-improvement targets v1: solubility ≥5–10×, ER<0.3, hERG no-worsening. | 2026-05-09 | spec |
| L13 | Quality tiers: gold 500–2k, silver 10k–100k, bronze millions. | 2026-05-09 | spec |
| L14 | Architecture family deferred until Layer-3. Default starting point: concat-MLP ranker. | 2026-05-10 | implementer (revisit at L3 gate) |
| L15 | "Discovered" claim must satisfy all 8 conditions in `rasyn_heldout_discovery_demo_context.md` §65–76. Non-negotiable. | 2026-05-09 | spec |
| L16 | **Paper SAR extraction deferred** to a separate methodology-first workstream (PLAN.md §16). v1 ships silver-only without papers. Promote to gold tier post-Layer-3 once safe extraction pipeline (structured-LLM extract → dual-reviewer → provenance hashing → canonicalization round-trip → leakage cross-check) is locked + dry-run-validated. | 2026-05-10 | user |
| L17 | ~~Compute estimate~~ — superseded by L18. | 2026-05-10 | superseded |
| L18 | **Compute resolved: 32 GPUs total = 16× H100 + 16× A100.** No cap; gate per major run (Stage-1 pretrain, each Stage-2 sweep batch, Stage-3 finetune, Stage-5 sealed-case lock). Wall-clock from "go" → locked predictions: **~1.5–2.5 days, preprocessing-bound** (training compresses to hours). Strategy shifts: pre-commit transformer (~500M–1B params), train our own encoder (no public-encoder leakage risk), parallel sweep + parallel per-mode finetune + multi-seed evaluation all feasible. Track A (data) becomes the critical path. Detail: PLAN.md §17.A + §17.B. | 2026-05-10 | user |
| L19 | **GitHub remote: https://github.com/ansschh/wolverine** (branch `main`). Full project mirrored (specs + plan + memory + rasyn/ code). `.claude/` excluded via .gitignore. | 2026-05-10 | user |
| L20 | **Phase B-0 + A-0 shipped** (commit 1, 2026-05-10). 11 frozen Pydantic schemas + ~30 round-trip/invariant tests + sealed-case registry stub with 3 ADMET cases + OXS DOI quarantine. Identifiers (canonical_smiles, inchi_key, ChEMBL/PubChem IDs) intentionally null pending populator script — avoids hand-typed SMILES corruption. | 2026-05-10 | implementer |

---

## Open questions / pending gates (kept current)

> Mirror of PLAN.md §11 with current status.

| # | Gate | Status | Blocker for | Notes |
|---|---|---|---|---|
| G1 | Disk + network for bulk downloads (~100 GB) | open | Phase A-1 | Will surface when A-1 starts. |
| G2 | ~~GPU access~~ | ✅ RESOLVED 2026-05-10 | n/a | 32 GPUs total: 16× H100 + 16× A100. See L18, PLAN.md §17.A + §17.B. |
| G3 | Architecture family pre-commit | deferred to L3 | training Stage 1 | Starting concat-MLP. |
| G4 | ~~Compute budget cap~~ | ✅ RESOLVED 2026-05-10 — "no cap, gate per major run" | per-launch | Claude reports cluster-hours + intent before each major training run; awaits user OK. Applies to Stage-1, each Stage-2 sweep batch, Stage-3, Stage-5. |
| G5 | Paper extraction methodology lock (PLAN.md §16) | deferred — silver-only v1 | gold-tier paper rows | Post-Layer-3 workstream P-0 to P-5; not blocking. |
| G6 | Functional-recovery judge auto-vs-human | auto-only default | post-L3 eval | Pre-register criteria. |
| G7 | Decontam audit acceptance | self-audit + canary | A-5 freeze | Flag for review. |
| G8 | Sign-off on locked predictions before reveal | required | Stage 5 reveal | Hard gate. |

---

## Conventions

- **Hashing.** All frozen artifacts (configs, datasets, model checkpoints, locked predictions) get SHA256 hashes. Hash function: `sha256` over canonical JSON (sorted keys, no whitespace) for configs; over file bytes for datasets/models.
- **SMILES.** Canonical form is RDKit canonical SMILES with `chembl_structure_pipeline` standardization (desalt, neutralize, tautomer normalize). Stereochemistry preserved.
- **InChIKey.** Used as primary cross-source molecule join key.
- **Time.** All timestamps UTC, ISO 8601.
- **Naming.** snake_case for files and Python identifiers. CapCase for Pydantic model classes.

---

## Log entries (newest first)

### 2026-05-10 (latest) — Phase B-0 + A-0 shipped; GitHub remote configured
**Type:** result + reference
**Phase:** B-0, A-0
**Context:** User said "go ahead, just let me know when you want the compute and I can spin up the pod" plus "keep the code pushed here: https://github.com/ansschh/wolverine."

**B-0 deliverables (scaffold + frozen schemas + tests):**
- Repo tree at `rasyn/` matching PLAN.md §13.
- `pyproject.toml` with pinned dep groups: base (pydantic, pyyaml, rdkit, numpy, pandas, pyarrow), `dev` (pytest, ruff, mypy), `chem` (chembl_structure_pipeline, datamol), `ml` (torch, torch-geometric, transformers), `data` (PyTDC, deepchem, faiss-cpu).
- 11 schema modules under `rasyn/schemas/`:
  - `hashing.py` — canonical_json + sha256_hex + hash_model
  - `molecule.py` — MoleculeRef (canonical_smiles + inchi_key now Optional with `is_populated` property)
  - `registry.py` — SealedCaseRegistry, SealedCase, ForbiddenIdentifiers/Documents/Assays, QuarantineConfig, Canary
  - `challenge.py` — ADMETChallengePacket, ActivityContext, LiabilityContext, RescueContextPacket
  - `proposer.py` — ProposerRequest, ProposerOutput, CandidateAnnotation, TransformationDescriptor
  - `evidence.py` — CandidateEvidencePacket + 7 sub-blocks (structural, descriptors, deltas, retention, liability, risk, structured rationale)
  - `ranker.py` — RankerInput, RankerOutput, ConfidenceBlock; 7 RescueLabel literals + 6 FailureMode literals
  - `locked.py` — LockedPrediction (frozen, hash-stable)
  - `config.py` — DecontaminationConfig, BaselineConfig (8 baselines), ProposerConfig (6 channels), RankerConfig
  - `manifest.py` — DatasetManifest, TrainingManifest, FileEntry
- All schemas use `frozen=True, extra="forbid"` for strict validation + immutability.
- Tests in `tests/`:
  - `test_hashing.py` — canonical JSON determinism, ordering invariance, unicode passthrough, length checks
  - `test_schemas_roundtrip.py` — round-trip + hash-stability for every schema (MoleculeRef incl. populated/unpopulated, ADMETChallengePacket, ProposerRequest/Output, CandidateEvidencePacket, RankerInput/Output, all 4 configs)
  - `test_sealed_case_registry.py` — loads YAML, validates 3 case IDs present, per-case liability/mode lockdown, OXS DOI quarantine, default decontam thresholds, known synonyms, hash stability across reloads, no duplicate IDs
- `rasyn/data/registry/loader.py` — `load_sealed_case_registry()` returns validated `SealedCaseRegistry`.

**A-0 deliverables (sealed-case registry stub):**
- `rasyn/rasyn/data/registry/sealed_case_registry.yaml` — 3 ADMET cases:
  - ADMET-001: terfenadine → fexofenadine, hERG, active_metabolite_safety_rescue
  - ADMET-002: acyclovir → valacyclovir, oral_exposure, prodrug_exposure_rescue
  - ADMET-003: OXS007570 → OXS008474, solubility, polarity_solubility_rescue
- Per-case forbidden synonyms locked (e.g., Allegra/Telfast/Seldane for ADMET-001; Zovirax/Valtrex/Zelitrex for ADMET-002; OXS variants for ADMET-003).
- Per-case forbidden title fragments locked (catches paper titles via fuzzy match).
- ADMET-003 forbidden_documents.dois includes `10.1039/d4md00275j` (the RSC Med Chem 2024 lead-optimization paper that contains the answer).
- Default decontamination thresholds (Tanimoto ≥0.85 to answer; ≥0.65 with same Murcko + same target).
- **Identifiers intentionally null:** canonical_smiles, inchi_key, ChEMBL IDs, PubChem CIDs, CAS numbers all left empty in the YAML stub. The populator script (Phase A-0 task to be written next) will hit PubChem PUG REST + ChEMBL API + RDKit canonicalisation to fill them, then re-freeze the YAML with bumped version. Rationale: hand-typed SMILES are the most common silent corruption source in chemistry projects; rather than guess, leave null and let the canonicalisation pipeline populate.
- `rasyn/rasyn/data/registry/canaries.yaml` — empty stub; canary generator script (next Phase A-0 task) will populate with ~30 canaries per case across 8 layers (smiles/inchi_key/synonym/doi/pmid/chembl_id/patent/title_text).

**Known schema decisions worth flagging in MEMORY:**
- MoleculeRef.canonical_smiles + inchi_key are Optional (not required) so the registry stub can hold not-yet-populated molecules. `is_populated` property checks both are non-None for runtime code that needs them.
- All schemas freeze + forbid extra fields. Adding a field requires a schema-version bump.
- Hash function: `hash_model(model)` = sha256(canonical_json(model.model_dump(mode="json"))). All artifacts (configs, manifests, locked predictions) hash the same way.
- 7-label RescueLabel matches spec §4 exactly: strong_success, weak_success, failed_activity_loss, failed_no_liability_improvement, failed_wrong_liability, failed_new_liability, uncertain.

**Git / GitHub:**
- `git init -b main` at `A:/rasyn-case-studies/`.
- Top-level .gitignore covers Python build artifacts, raw/clean data dirs (manifests kept), checkpoints, editor cruft, `.env`, and `.claude/`.
- Initial commit (35 files) covering all 5 specs + PLAN.md + MEMORY.md + full B-0 + A-0 scaffold.
- Pushed to `https://github.com/ansschh/wolverine` (branch `main`); remote was empty → fresh push, no merge.
- User git config already set: anshtiwari9899@gmail.com / ansschh.

**Outcome / next:**
- B-0 ✅ done. A-0 ✅ scaffolded (YAML + loader + tests; populator script + canary generator are the next A-0 tasks).
- Next phases (no human gates): **A-0 populator script** (PubChem/ChEMBL ID lookup + RDKit canonicalisation), **A-0 canary generator**, **B-1 synthetic fixture**, **B-2 deterministic proposers (analog retrieval + MMP + liability rules)**, **B-4 8 baselines**, **B-5 eval harness skeleton**.
- Tests have NOT been run yet (no Python env in this session) — first task next session is `pip install -e ".[dev,chem]"` then `pytest`. Any failures get logged here.
- When user is ready to spin up the pod: that gates Layer-2 onward. Until then, all the above is CPU-only work.

**Refs:** PLAN.md §5 (Track A), §6 (Track B), §13 (layout); commit hash visible in `git log` once pulled.

---

### 2026-05-10 (later) — Compute resolved: 32 GPUs (16× H100 + 16× A100); strategy escalated
**Type:** decision
**Phase:** planning
**Context:** User confirmed full compute access in two messages: first "I can do 16 H100s", then "in addition to those 16H100s, I can also give you 16 A100s in total 32 GPUs to finish everything." Budget gating: "no cap, ask before each major run."

**What this changes:**
- G2 (GPU access) ✅ resolved.
- G4 (compute budget cap) ✅ resolved as "no cap, per-launch gating."
- L17 superseded by L18.
- Total wall-clock from "go" → locked predictions: **~1.5–2.5 days**, preprocessing-bound. Compute is no longer the bottleneck.

**Architecture upgrades enabled:**
- Pre-commit ~500M–1B-param transformer ranker backbone (was: concat-MLP starting point).
- Train our own molecular encoder from scratch on decontaminated ChEMBL — eliminates public-encoder leakage risk (Uni-Mol/MolFormer/ChemBERTa all saw fexofenadine + valacyclovir).
- Multi-seed (3–5 seeds) for variance estimates — investor-credibility multiplier.
- Wide parallel sweep (~16 configs simultaneously) instead of sequential 5–10.
- Multi-pretrain-variant selection (run 2–3 pretrain variants in parallel, keep best at L3).
- Per-mode dedicated heads jointly trained instead of finetuned later.

**New critical path:** **Track A (data)** is now the long pole, not training. Front-load preprocessing the moment B-0 scaffold lands. GPU-accelerate descriptors/fingerprints/FAISS where possible (cuML, FAISS-GPU, RDKit-CUDA paths) to compress preprocessing wall-clock from 24–72 hr → ~12–36 hr.

**Cluster partition (proposed default, PLAN.md §17.B):**
- 16× H100 → main pretrain + Stage-2 main training
- 4× A100 parallel → auxiliary predictor training (TDC's 22 endpoints)
- 4× A100 parallel → preprocessing acceleration
- 8 parallel 2× H100 jobs → Stage-2 sweep
- 3 parallel 4–8 GPU jobs → Stage-3 per-mode finetune
- Spare 1–4 A100s for Layer-1, Layer-2, ablations, debugging

**What does NOT change:**
- Verification discipline (Layers 1–3) still mandatory. 32-GPU-hours on a buggy pipeline still wastes a day.
- Decontamination still must be perfect.
- Sealed-case lock still must be human-signed before reveal.
- Per-launch user OK still required before each major training run.

**Outcome / next:** PLAN.md updated with §17.A + §17.B. MEMORY.md L18 + G2 + G4 updated. Awaiting user OK on plan as-is to proceed with Phase B-0 + A-0 in next conversation.

**Refs:** PLAN.md §11, §17.A, §17.B; MEMORY.md L18, G2, G4.

---

### 2026-05-10 (later) — Papers deferred + compute estimate added (PLAN.md → v0.2)
**Type:** decision + reference
**Phase:** planning
**Context:** User reviewed PLAN.md v0.1 lines 0–473 and approved with two notes: (a) defer paper data until safe extraction methodology is designed, (b) wants concrete compute numbers.

**Changes made to PLAN.md (v0.1 → v0.2):**
- §1 In scope: removed "non-sealed papers" from public sources list.
- §1 Out of scope: added ⏸ paper SAR extraction bullet (deferred, not cancelled).
- §5 Track A-1: papers entry replaced with deferred note.
- §5 Track A-4: pass 4 (curated paper rows) marked skipped at v1.
- §10 phase table: A-4 estimate trimmed (12 passes instead of 13), reviewer gate removed.
- §11 gates table: G5 reframed from "reviewer needed" to "methodology workstream deferred."
- **NEW §16:** "Deferred — Paper SAR extraction methodology workstream" with 5-phase plan (P-0 source whitelist → P-1 structured extraction → P-2 dual-reviewer adjudication → P-3 provenance hashing → P-4 validation dry-run → P-5 promotion to gold).
- **NEW §17:** "Compute & infrastructure estimate" — storage, RAM, CPU, GPU options table (recommended/faster/owned), cost/time totals, sensitivity caveats, hard ask for compute provider + budget pick.
- §15 "What's next" renumbered to §18.

**Compute estimate headline (full PLAN.md §17 for detail):**
- Storage: ~150–300 GB SSD
- Preprocessing: 16–32 cores, 64–128 GB RAM, ~24–72 hr wall-clock
- Training: ~120–350 single-A100-equivalent GPU hours
- Cloud cost (A100 @ ~$2/hr): **~$240–700** mid-estimate, suggest **$1000 cap**
- Wall-clock: ~1–2 weeks cloud, ~4–6 weeks owned 4090
- Recommended provider: **Lambda Labs** or **RunPod**

**Why papers are dangerous to ingest naively (now in §16):** (a) hallucinated SMILES from PDF → corrupted gold labels; (b) papers carry author names + synonyms + drawings that can slip past sealed-case quarantine, esp. for OXS where related medchem papers from the same group exist.

**Outcome / next:** PLAN.md v0.2 reflects scope. Awaiting user OK on (a) compute provider pick (G2) and (b) budget cap (G4) before Layer-2 — neither is blocking for B-0 / A-0 / A-1 / A-2 / A-3 / A-4 / A-5 / B-1 / B-2 / B-3 / B-4 / B-5 / Layer-1.

**Refs:** PLAN.md §1, §5, §10, §11, §16, §17, §18.

---

### 2026-05-10 — Project intake, scope locked, parallel-track plan agreed
**Type:** decision
**Phase:** planning
**Context:** First working session. Ansh asked Claude to deploy 25 agents (5 per file × 5 files) to deeply audit the spec stack and produce a long-term plan with model-execution estimates separated from human gates.

**What happened:**
- Spawned 25 Explore agents in parallel, 5 lenses per file (thesis / architecture / deliverables / dependencies / risks). All 25 returned reports.
- Synthesis posted as user-facing message: 5 files form one project (contamination-controlled blind benchmark for ADMET rescue with 3 sealed cases).
- Asked user 2 follow-up questions; got these locks:
  - **Antibiotic + NMR/spectra:** out of scope for now.
  - **Internal data:** none available, public-only.
  - **Wet-lab:** never.
  - **Decontamination thresholds:** defaults OK, audit-and-fix later if needed.
  - **Time crunch:** extreme. Optimize parallelism.
  - **Working style:** plan-first, then go. Maintain `PLAN.md` and `MEMORY.md`.
  - **Code location:** `A:/rasyn-case-studies/rasyn/` subdir; specs at root untouched.

**Key insight from agents:**
- The 5 specs are tightly coupled. `heldout_discovery_demo_context.md` is the constraint layer (defines what "discovered" means). `admet_rescue_architecture_context.md` defines the task (parent-candidate ranking). `admet_conditioning_architecture_benchmark_spec.md` is the implementation contract (frozen schemas). `proposer_system_test_cases.md` locks the 3 cases + 6 proposer channels. `curating_the_dataset.md` is the data layer (4-table, 13-pass).
- The OXS case (ADMET-003) requires hard quarantine of DOI `10.1039/d4md00275j` — this is the lead-optimization paper that contains the answer.
- Hard negatives are mandatory across the dataset (5 types) — not optional.
- Prodrug rescue (ADMET-002) cannot be scored on direct potency — must use delivery-probability scoring.

**Notable open ambiguities surfaced by risk agents (not blocking, logged for later):**
- Threshold ambiguity for "compatible assay" / "compatible standard_type" (curation spec).
- Stereochemistry handling not specified.
- Hard-negative-to-positive ratio not specified — use a sensible default (~1:1 to 2:1) and tune at L2.
- Forward-reward proposer reward-hacking mitigation underspecified — use validity + uncertainty penalties at first.
- Functional-recovery scoring for prodrug case open — start with rule-based delivery proxy (ester/amide hydrolysis motifs).

**Outcome / next:** wrote `PLAN.md` v0.1 and this `MEMORY.md`. Awaiting user OK before starting Phase B-0 (scaffold + schemas) and A-0 (sealed-case registry).

**Refs:** all 5 spec files at repo root; PLAN.md (full); 25-agent reports captured inline in conversation transcript (not persisted to disk — main synthesis is in PLAN.md §3, §5, §6, §10).

---

## Things to write down EVERY time
(checklist for future entries)

- [ ] Date + short title
- [ ] Phase tag
- [ ] What I tried / observed / decided
- [ ] Why
- [ ] What broke (if anything)
- [ ] What's next
- [ ] References to PLAN.md sections + spec line numbers
- [ ] If a config / dataset / model was frozen: hash + filename
- [ ] If a gate flipped status: update §"Open questions / pending gates"
- [ ] If a decision was locked: update §"Locked decisions snapshot"

---

— end MEMORY.md v0.1 —
