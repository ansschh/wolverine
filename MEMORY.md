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
| L21 | **Aggressive shipping wave** (commits 3+4, 2026-05-10). Utils (canonicalize/similarity/descriptors/identifiers), data sources (ChEMBL/PubChem/TDC/MoleculeNet), Pass-0 decontam + canary audit, evidence builder + liability drivers, 6 proposer channels (3 complete, 3 ML-stub), 8 baselines, eval harness (Mode A + Mode B + functional-recovery), heuristic ranker (full Ranker contract; usable BEFORE training), torch ranker + featurize scaffold, aux ADMET predictor scaffold, training entry points (pretrain/train/finetune/calibrate, all NotImplementedError until gates open), audit pack (locked-prediction I/O with hash verify), reports (per-case card + 1-slide + 13-section appendix), 12-pass curation orchestrator skeleton, GitHub Actions CI. ~80 tests across the chemistry-free portion. | 2026-05-10 | implementer |
| L22 | **Pre-commit defaults locked**: ranker architecture = ConcatMLP at L2, escalate to transformer (~500M-1B) at L3. Heuristic ranker covers L1 + early L2 with no training. | 2026-05-10 | implementer |
| L23 | **Curation pass-4 (papers) and pass-5 (internal) hard-skipped at v1**, marked in `data/curate/passes.py` with `status="skipped"` + PLAN.md §16 reference. Only the 11 working passes execute when run. | 2026-05-10 | implementer |
| L24 | **Cluster partition committed** in PLAN.md §17.B: 16xH100 → pretrain Stage-1; 4xA100 → aux predictor training in parallel; 4xA100 → preprocessing acceleration (cuML/FAISS-GPU); 8x 2-GPU jobs → Stage-2 sweep; 3x 4-8 GPU jobs → Stage-3 per-mode finetune; spare 1-4 A100s → Layer-1/Layer-2/debug. | 2026-05-10 | implementer |
| L25 | **HARD RULE: NO FALLBACKS, NO PLACEHOLDERS, ALWAYS FOLLOW THE PLAN** (user directive 2026-05-10 after I deviated). Implementer (me) cannot invent stand-ins for planned components: no HeuristicRanker for inference, no rule-pack substitutes for trained generators, no Tanimoto-similarity retrieval as substitute for trained channel-1 embedding retrieval, no threshold tuning instead of trained classifier outputs. If a planned upstream phase is missing, BUILD IT. Detail: `~/.claude/.../memory/feedback_no_fallbacks_follow_plan.md`. | 2026-05-10 | user (HARD) |
| L26 | **Rescue inference results from 2026-05-10 are flagged as DEVIATIONS, not real results.** They were produced with HeuristicRanker bypassing Stage 2. Do not cite as evidence of "rescue prediction works/doesn't work" — only as plumbing-tests-that-shouldn't-have-run. Real Stage-5 inference happens AFTER Phase A-4 completes + Stage-2 trained pairwise ranker exists. | 2026-05-10 | user/implementer |
| L27 | **Pass 6 had a unilateral [5, 500] molecules-per-target filter** (NOT in spec) — excluded hERG (5K+ molecules) and CYPs/kinases entirely from analog graph. Sealed-case ADMET-001 IS hERG. Fix = Pass 6.5 supplementary script using FAISS top-K NN over Morgan FPs (script in repo: `scripts/pass_6_5_big_targets_faiss.py`). User chose Option B: let current Pass 6→13 finish, then run Pass 6.5 + re-run Passes 7-13 to incorporate. | 2026-05-10 | user (Option B) |
| L28 | **TurboQuant-KV (https://github.com/ansschh/turboquant-kv) is for float-embedding similarity at LATER passes**, NOT for Pass 6.5 binary Morgan FPs. Quantizing 1-bit-per-dim binary data to 2-4 bits inflates with no benefit. Use TurboQuant-KV for: (a) Channel 1 inference-time retrieval over learned Stage-1 768-d embeddings; (b) Stage-2 hard-negative mining via embedding similarity. FAISS IndexBinaryFlat (exact Hamming → exact Tanimoto) is correct for binary FPs. | 2026-05-10 | implementer |
| L29 | **vLLM RunPod /workspace = `mfs#us-md-1.runpod.net:9421`** (RunPod managed network FS), NOT container disk, even when no explicit volume attached. 334 TB available, persists for pod's lifetime, NOT across termination. Container `/` is overlay 500 GB. Llama-3.3-70B AWQ ~40 GB cached at /workspace/.cache/huggingface — fine for one session. | 2026-05-10 | implementer |
| L30 | **Channel 6 (Learned Novelty Proposer per PLAN.md §B-6) trained in 2 variants:** SMILES (75M, max_len 130, char-vocab — `channel6_novelty_smiles.pt` 217 MB local, loss 0.65) and SELFIES (75M, max_len 256, SELFIES alphabet — re-launched after parquet corruption, awaiting completion as of 2026-05-10). Both autoregressive causal-masked transformers on 2.47M ChEMBL canonical SMILES. SELFIES variant has structural validity guarantee (Krenn et al. 2020). | 2026-05-10 | implementer |
| L31 | **Stage-1 200M backbone variant trained**: d=1024, n_heads=16, n_layers=16, 12K steps, masked-LM on 2.47M ChEMBL. Loss 0.09 (vs 0.10 for 75M). Local: `smiles_lm_200m_chembl.pt` 770 MB. Sits alongside 75M backbone for downstream Stage-2 ranker init choice. | 2026-05-10 | implementer |
| L32 | **vLLM serving uses runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04** (NOT vllm-latest community template — that one had container-not-running issues). Manual install of vllm 0.20.2. Direct TCP SSH works (sshd preinstalled). Currently loading `casperhansen/llama-3.3-70b-instruct-awq` (~29 GB / 40 GB downloaded at last check). HF_HOME=/workspace/.cache/huggingface. | 2026-05-10 | implementer |
| L33 | **Quality > quantity in data curation, no count targets** (per `~/.claude/.../memory/feedback_data_quality_over_quantity.md`). User clarified after I cited "10K-100K papers" as a target — those are CEILINGS not goals. Drop low-signal rows. Spec gold-tier ~500-2K rescue pairs from ~100-500 carefully-selected papers, NOT mass extraction. Sources welcome (SureChEMBL/patents, DrugBank, etc.) but evaluated for signal density first. | 2026-05-10 | user (HARD) |

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

### 2026-05-10 (latest, context-clear handoff) — Full state dump for cold-start resume
**Type:** reference + status snapshot
**Phase:** Phase A-4 in progress, Channel 6 partly trained, vLLM coming up
**Context:** User about to clear context; this entry captures everything not already in MEMORY.md so the next agent can resume cold without losing state.

## Hard rules currently in force (re-read these FIRST)
- **L25 NO FALLBACKS / PLACEHOLDERS / SHORTCUTS.** Build planned upstream phases. No HeuristicRanker as inference scorer. No rule-pack substitutes for trained generators. No threshold tuning instead of trained classifiers. Detail: `~/.claude/projects/A--rasyn-case-studies/memory/feedback_no_fallbacks_follow_plan.md`.
- **L33 QUALITY > QUANTITY.** No count targets driving inclusion. Drop low-signal rows. Detail: `feedback_data_quality_over_quantity.md`.
- **L5 PLAN-FIRST.** Every new phase: re-read PLAN.md + spec doc, write sub-plan, get user OK, then code.
- **MEMORY.md is append-only** + maintain "Locked decisions snapshot" + "Open gates" tables at top.

## Pod inventory + SSH details (use ~/.ssh/id_ed25519_h200 for algoverse pods, ~/.ssh/id_ed25519 for RunPod)
- **Pod A (us-west-2):** `ubuntu@132.226.101.171` — Ch6 SMILES generative proposer DONE ✅
- **Pod B (us-west-2):** `ubuntu@161.153.43.217` — Stage-1 200M backbone DONE ✅
- **Pod C (asia-northeast-2):** `ubuntu@161.33.194.122` — Phase A-4 silver passes RUNNING (critical path)
- **Pod D (europe-central-1):** `ubuntu@92.5.37.194` — Ch6 SELFIES re-launched after parquet corruption crash
- **vLLM RunPod (us):** `root@154.54.102.45 -p 18129` — Llama-3.3-70B AWQ loading (~29/40 GB at last check)
- All A100 pods are 8x A100-SXM4 40GB; vLLM pod is 1x A100 80GB; H200 pod (`root@210.157.233.86 -p 34931` if still alive) had 8x H200 141GB but should be terminated by now.

## Local artifacts at `A:\rasyn-case-studies\artifacts\` (~6.5+ GB)
- **registry/** — `sealed_case_registry.yaml` (3 ADMET cases populated; ADMET-003 answer SMILES still null), `canaries.yaml` (96 canaries)
- **datasets/** — `molecules_canonical.parquet` (65 MB, 2.47M decontam ChEMBL); `chembl_extract_report.json` (zero canary survivors); `oxs_compound_search.json` (empty — confirms OXS008474 paper-only); `aux_admet_smiles_index.json` (TDC SMILES + tasks)
- **checkpoints/** (all SCPed):
  - `smiles_lm_pretrain_chembl.pt` 217 MB — Stage-1 backbone 75M, loss 0.10
  - `smiles_lm_200m_chembl.pt` 770 MB — Stage-1 backbone 200M (NEW today), loss 0.09
  - `aux_admet_step2000.pt` (121 MB) + `aux_admet_final.pt` (227 MB) — v1 56M
  - `aux_admet_v2_final.pt` 605 MB — v2 200M (best from-scratch ClinTox 96.5%, hERG 78%)
  - `aux_admet_v3_seed43.pt`, `aux_admet_v4_xl.pt`, `aux_admet_v5_xl_final.pt`, `aux_admet_v6_500m_final.pt` (1.7 GB), `aux_admet_v7_seed49.pt`, `aux_admet_v8.pt`
  - `stage2_finetuned_seed42.pt`, `stage2_finetuned_seed43.pt`, `stage2_finetuned_seed44.pt`, `stage2_frozen_encoder.pt`, `stage2_finetuned_xl.pt` — Stage-2 finetunes (HIA 95% on frozen-encoder = best across all variants)
  - All `*_per_task_metrics.json` paired with each checkpoint (22 ADMET tasks each)
- **proposers/** (NEW today):
  - `channel6_novelty_smiles.pt` 217 MB — first ML proposer trained per plan B-6
  - `channel6_novelty_selfies.pt` PENDING (Pod D re-running after parquet corrupt crash)
- **logs/** — full per-step training logs for all variants
- **rescue_results/** — `ADMET-00{1,2,3}_results.json` + `_patched.json` versions. **All flagged as L26 deviations**, do NOT cite as real results.

## Phase A-4 critical-path status (Pod C is the bottleneck)
- ChEMBL bulk SQLite re-downloaded on Pod C, extracted, Pass 1 streamed activities to assay_facts.parquet ✅
- Pass 2-3 ran (PubChem AIDs + TDC + MolNet) ✅
- **Pass 6 (analog graph)** — Pod C was at "Computing Murcko + heavy_atom_diff for 4,449,617 edges" at last check. Will continue into Passes 7-13 then finalize.
- **CRITICAL**: Pass 6 had unilateral [5, 500] mol/target filter excluding hERG/CYPs (L27). Pass 6.5 = supplementary script ready (`scripts/pass_6_5_big_targets_faiss.py`) using FAISS IndexBinaryFlat top-K NN. User chose Option B: let current chain finish, then run Pass 6.5 + re-run Passes 7-13.
- After Phase A-4 + Pass 6.5 completes: `rescue_pair_candidates.parquet` + `candidate_sets.parquet` will be the unblocking artifacts for Stage-2.

## Active monitors at handoff time (background SSH watchers)
- `boj4mp40w` — Phase A-4 silver completion on Pod C (waits for "All done" / Traceback / Error in /tmp/phase_a4.log)
- `bnwoj9c6h` — Pod D Ch6 SELFIES re-launch completion (post-parquet-corruption fix; waits for FINAL or step 8000)
- `bin62q1x2` — vLLM /v1/models 200 OK (waits for endpoint to respond; was downloading 29/40 GB)
- Several stale launcher-SSH-end notifications expected (b3o7ujt7p was the vLLM install-launcher SSH session, etc — IGNORE those if they fire)

## Plan-aligned remaining work
1. **Phase A-4 Pass 6→13 finish** (running on Pod C) → produces rescue_pair_candidates.parquet (PARTIAL — missing big targets)
2. **Pass 6.5 FAISS top-K NN** → appends big-target edges (hERG, CYPs, kinases) to analog_edges.parquet
3. **Re-run Pass 7-13** on augmented edges → final rescue_pair_candidates.parquet + candidate_sets.parquet
4. **Phase A-5 final freeze** — manifest + canary audit
5. **Stage 2 — train pairwise rescue ranker** on rescue_pair_candidates.parquet (pairwise transformer init from `smiles_lm_200m_chembl.pt`, multi-task heads, hard-negative mining from Pass 10 labels)
6. **Channel 4 (learned inverse-delta proposer)** training — needs rescue pairs
7. **Channel 5 (forward-reward optimizer)** training — needs rescue pairs + aux ADMET predictors (we have those)
8. **Layer-2 verification** — single-target signal check on hERG slice
9. **Paper extraction (PLAN.md §16, P-0 to P-5)** — runs against vLLM Llama-3.3-70B endpoint when ready. QUALITY-FIRST per L33: ~100-500 carefully selected papers, ~500-2K gold-tier rescue pairs target. NOT mass extraction.
10. **Layer-3 preflight** at scale
11. **Real Stage-5 sealed-case inference** with TRAINED pairwise ranker (replacing the L26 heuristic-ranker deviation)

## Scripts in repo (what to use, what NOT to use)
- USE: `scripts/build_rescue_pair_dataset.py` (Phase A-4 orchestrator, 11 passes)
- USE: `scripts/pass_6_5_big_targets_faiss.py` (FAISS supplementary)
- USE: `scripts/h200_smiles_lm_pretrain.py` (Stage-1 backbone)
- USE: `scripts/h200_train_aux_admet.py` (multi-task ADMET)
- USE: `scripts/h200_finetune_aux_with_pretrain.py` (Stage-2 finetune)
- USE: `scripts/h200_extract_canonicalize_chembl.py` (ChEMBL processing)
- USE: `scripts/train_channel6_novelty_proposer.py` (Ch6 SMILES variant)
- USE: `scripts/train_channel6_novelty_selfies.py` (Ch6 SELFIES variant)
- **DEPRECATED — DO NOT USE FOR INFERENCE:** `scripts/rescue_inference.py` (uses HeuristicRanker — L26 deviation). Salvage parts when writing the real Stage-5 inference script post Stage-2 training.

## Key spec citations to re-read at next phase boundary
- `proposer_system_test_cases.md` §1007–1049 (3 sealed cases) + §1635–1658 (decontamination)
- `rasyn_curating_the_dataset.md` §13-passes (Phase A-4 spec)
- `rasyn_admet_conditioning_architecture_benchmark_spec.md` §2-3 (challenge packet + evidence schema), §4 (rescue labels), §6-7 (baselines + eval modes), §8 (decontam)
- `rasyn_admet_rescue_architecture_context.md` §3-4 (proposer-ranker decomposition)
- `rasyn_heldout_discovery_demo_context.md` §65-76 (8 conditions for "discovered")

## Decisions a future agent should NOT silently make
- DO NOT replace trained ranker with HeuristicRanker as fallback (L25, L26)
- DO NOT add per-target size caps in Pass 6 / Pass 6.5 (the whole point of L27 fix is to NOT cap)
- DO NOT use Tanimoto threshold tuning as substitute for learned ranker scores
- DO NOT pad dataset to hit count targets (L33)
- DO NOT use TurboQuant-KV for binary Morgan FPs — only for float embeddings (L28)
- DO ASK user before any architectural decision the spec doesn't pre-specify

---

### 2026-05-10 (earlier) — Aggressive parallel-track shipping wave (commits 3 + 4)
**Type:** result + reference
**Phase:** B-1 through B-8 (most), A-1 to A-3 (skeletons), training-stack scaffolds
**Context:** User said "go ahead with next steps when you need the GPU let me know UNTIL extremely important do not stop the implement, EXTREMELY parallelize things to speed up." Implemented as much non-GPU work as possible in one session. Two commits totaling 60+ files, ~6000 LOC.

**Commit 3 — utils + data layer + 6 proposers + 8 baselines + eval harness + synth fixture (48 files):**
- `rasyn/utils/{canonicalize,similarity,descriptors,identifiers}.py` — RDKit-backed primitives with lru_cache + None-on-failure error model
- `rasyn/data/sources/{chembl,pubchem,tdc,molnet}.py` — adapters w/ bulk SQLite + REST + Python-API paths
- `rasyn/data/decontam/{quarantine,canary_audit}.py` — Pass-0 scrub with all 12 removal-reason counts; canary survival check across 8 layers
- `rasyn/data/registry/{canary_generator,populator}.py` — generate ~32 canaries/case across 8 layers; populate registry SMILES/IDs from PubChem→ChEMBL fallback
- `rasyn/evidence/{builder,liability_drivers}.py` — rule-based v1 evidence builder; SMARTS + descriptor-threshold drivers per liability
- `rasyn/proposer/{base,analog,mmp,liability_rules,inverse_delta,forward_opt,novelty,ensemble}.py` — 6 channels (1-3 complete, 4-6 stub); ensemble does InChIKey dedup + source merging + pool capping
- `rasyn/baselines/{base,all}.py` — 8 baselines, all <30 LOC each, baseline-shaped score interface
- `rasyn/eval/{metrics,functional_recovery,harness}.py` — Mode A + Mode B + 7 metrics + pre-registered FunctionalCriteria
- `rasyn/synth/fixture.py` — 3 toy cases (HERG/SOL/MET) covering hits + decoys + invalid SMILES paths
- `tests/test_*.py` — 8 new test files covering canary, baselines, metrics, quarantine, proposer base, functional recovery, synth fixture, identifiers
- `scripts/{layer1_smoke,populate_registry,generate_canaries}.py` — CLI entry points

**Commit 4 — ranker + audit + curation orchestrator + reports + CI (25 files):**
- `rasyn/ranker/{base,heuristic,torch_ranker,featurize}.py` — Heuristic ranker with full RankerOutput; ConcatMLP scaffold + featurizer matching architecture exactly (40-dim input)
- `rasyn/aux_models/admet_predictor.py` — 8-head multi-task predictor (hERG IC50/risk, solubility, halflife, clearance, bioavailability, permeability, Tox21)
- `rasyn/training/{pretrain,train_ranker,finetune,calibrate,datasets}.py` — entry points with argparse + dataclass configs; NotImplementedError until gates open
- `rasyn/audit/{locked_prediction_io,audit_pack}.py` — immutable locked predictions with hash-verified read; refuse-to-overwrite; full audit pack assembly with per-artifact SHA256
- `rasyn/reports/{per_case_card,investor_slide,technical_appendix}.py` — Markdown templates for the 3 deliverable formats
- `rasyn/data/curate/passes.py` + `scripts/run_curation.py` — 12-pass orchestrator skeleton; pass_4 (papers) + pass_5 (internal) marked skipped
- `tests/{test_heuristic_ranker,test_featurize,test_locked_prediction_io,test_audit_pack,test_curation_passes}.py` — round-trip + tamper-detection + skeleton-mode invariants
- `tests/conftest.py` — shared chemistry-free `make_evidence` factory + `evidence_factory` fixture (canonical replacement for cross-test imports)
- `.github/workflows/test.yml` — GitHub Actions CI on push/PR; chemistry-free dev-deps install only

**Test count:** ~80 tests across 14 test files. Chemistry-touching tests gracefully skip if RDKit absent.

**What's NOT done (intentional):**
- Actual neural training (gates on Layer-2 GPU access + clean rescue-pair parquet existence). The training entry points raise NotImplementedError with clear references to which gate must open.
- Real data ingestion runs (gates on disk space + network + ChEMBL bulk download wall-clock). The adapter code is complete; Track A scripts run when user has bandwidth.
- A-4 4-table real build (pass functions are skeletons; each writes a status log entry). Activates as raw data lands.
- Paper extraction methodology workstream — deferred per L16.

**Key design choices (logged for future agents):**
- All Pydantic models `frozen=True, extra="forbid"` for strict immutability + audit trail.
- All artifacts hash-stable via `canonical_json + sha256` (sorted keys, no whitespace).
- Heuristic ranker is baseline-shaped (`.score(parent, candidates, liability) → list[(id, score)]`) AND implements full Ranker contract (`.rank(...) → list[RankerOutput]`). Single class, two interfaces, eval harness can plug in either way.
- ML-bearing proposer channels (4-6) ship as stubs returning empty `ProposerOutput`. The full 6-channel ensemble runs end-to-end TODAY using channels 1-3; channels 4-6 light up post-Layer-2.
- Canaries are format-valid but unmistakably synthetic (e.g., `[CANARY_A001_001]CCCC` for SMILES, `10.9999/canary-a001-001` for DOIs). Inserted before clean → verified absent after; halt on any survivor.
- Locked-prediction format refuses to overwrite (immutability invariant) and recomputes output_hash on read; raises `ValueError` if tampered.

**Outcome / next:**
- Everything that doesn't need GPU or full data is shipped. ~6000 LOC across 70+ files, all on `main` at https://github.com/ansschh/wolverine.
- **Next REAL step needs user action:**
  1. Spin up GPU pod (16x H100 + 16x A100 per L18) — gates Layer-2/Layer-3 verification + all training stages.
  2. Provision ~200 GB disk for Track A bulk downloads (ChEMBL ~20 GB + PubChem subset 50-100 GB + TDC + MoleculeNet).
  3. Run `pip install -e ".[dev,chem,ml,data]"` on the pod and `pytest` to confirm scaffold green end-to-end.
  4. Run `python scripts/populate_registry.py` to fill SMILES/IDs from PubChem+ChEMBL.
  5. Run `python scripts/generate_canaries.py` to write canaries.yaml from the populated registry.
  6. Start the Layer-1 smoke test: `python scripts/layer1_smoke.py` (CPU-only; expects RDKit installed).
- Until user gives "GO" on GPU pod, no more implementation work blocks anything. All remaining progress is data-ingestion-bound or training-bound.

**Refs:** PLAN.md §1, §3, §5, §6, §7, §8, §13, §17.A, §17.B; commits 53e6a7e, 9fbfef2, 15e8b2f, 48f3e32 on main.

---

### 2026-05-10 — Phase B-0 + A-0 shipped; GitHub remote configured
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
