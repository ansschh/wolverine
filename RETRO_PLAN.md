# RETRO_PLAN.md — Rasyn-Retro v1 Implementation Plan

**Status:** locked 2026-05-12. Plan-first per [[feedback_plan_first]]; no implementation begins until this plan is approved.

**Spec parent:** [`RETRO.md`](RETRO.md) (the architectural spec — Locks 1–8).
**Build parent:** [`MEMORY.md`](MEMORY.md) L40–L49 (ABX/ADMET shipped state we are reusing).

---

## 0. North star (3 lines)

Rasyn-Retro takes any target SMILES (typically a Rasyn-designed ADMET-rescue or antibiotic candidate) and returns a ranked, forward-validated, condition-annotated route tree to commercially purchasable starting materials, plus structured rationale and risk flags. It is a **planning system** (Lock 1), not a one-step reactant predictor. Headline demo: a Rasyn-designed molecule receives an auditable route packet that an expert chemist would judge "plausible to attempt at 10–100 mg scale."

---

## 1. Locked decisions (approved by user 2026-05-12)

| # | Decision |
|---|---|
| L1 | **Architecture: bespoke full build.** No Syntheseus, no AiZynthFinder dependency. Every component is Rasyn code. Matches the ABX/ADMET build philosophy and the [[feedback_no_fallbacks_follow_plan]] rule. |
| L2 | **Sealed cases: keep all three.** RETRO-001 oseltamivir (literature recovery), RETRO-002 nirmatrelvir/Paxlovid (novel-route), RETRO-003 Rasyn-designed ABX molecule (no literature answer). |
| L3 | **v1 deferrals accepted:** yield/impurity/selectivity prediction, stereo-aware disconnection, wet-lab demo, RL/PPO on proposers, internal/proprietary data, multi-objective route-score weight sweeps. **Diffusion proposer is IN v1** (added as R-2 Channel 5 per user override 2026-05-12, no longer deferred). |
| L4 | **Buyables source:** ZINC-22 in-stock + Enamine REAL Building Blocks (free SMILES tier) + eMolecules free monthly snapshot. Frozen at first download; cost-tier ≤ ~$10/g flagged. |
| L5 | **Reaction-data tier mix:** USPTO-full (Lowe, bronze, ~1.8M atom-mapped) + USPTO-50K (silver benchmark) + ORD (silver, conditions-enriched) + USPTO-LLM (2025, 247K, conditions). Gold paper-curated routes skipped for v1 per quality-over-quantity. |
| L6 | **Route-score formula (fixed for v1, no sweeps):** `route_score = 0.4·∏(step_plausibility) + 0.3·forward_pass_rate − 0.1·step_count_norm − 0.1·cost_norm − 0.1·risk_flags`. Documented, never tuned in v1. |
| L7 | **Encoder reuse:** all retro models (template proposer, graph-edit, seq2seq retro, forward, conditions, value) initialise their SMILES encoder from `checkpoints/smiles_lm_200m/` (200M MLM, ChEMBL-pretrained) — saves ~40 GPU-h of cold-start pretraining. |
| L8 | **Search algorithm:** Retro\* (neural-guided A\* over AND-OR tree). MCTS only appears as a baseline in Phase R-6. |
| L9 | **Honesty-verdict policy:** A `missed` verdict on any sealed case is acceptable per [[feedback_no_fallbacks_follow_plan]] and ABX honesty principle (MEMORY L43, L46) — provided the system produces structured rationale and ranks the literature route in the top decile of its candidate pool. Do not chase headlines at the cost of honesty. |

---

## 2. Scope locks (in vs out for v1)

**IN scope for v1:**
- Single-step proposer ensemble (**5 channels**): template, graph-edit, Transformer/seq2seq, retrieval, **diffusion reactant-completion**. (Diffusion = Lock 4 use case only: completes reactants given a fixed disconnection + reaction class. Never the planner.)
- Forward reaction validator (Lock 5 of RETRO.md — non-negotiable).
- Coarse condition predictor (reagent class / solvent class / temperature bin / catalyst class).
- Retro\*-style AND-OR tree search with neural value model.
- Frozen buyables index (InChIKey lookup, cost-tier flagged).
- 3 sealed retro cases, locked-prediction protocol identical to ADMET/ABX.
- 10 baselines + ~10 ablations following `run_abx_baselines.py` / `run_abx_ablations.py` patterns.
- Three-layer verification (smoke / slice / preflight) before any full-scale run.

**OUT for v1 (deferred to v2):**
- Yield, impurity, selectivity simulators.
- Stereo-aware disconnections.
- PPO/RL fine-tuning of any proposer.
- Wet-lab execution (project-level scope lock, see [[project_rasyn_scope]]).
- Internal/proprietary reaction data.
- Route-score weight sweeps.
- Closed-loop "design → plan → execute → verify" (Demo 4 of RETRO.md §14).

---

## 3. Phases (R-0 through R-7, plus deferred R-8)

### Phase R-0 — Plan & schema lock
**Goal:** freeze data contracts and registry before touching data.
**Artifacts:**
- This `RETRO_PLAN.md` (committed).
- `rasyn/rasyn/schemas/retro.py` — Pydantic v2: `Molecule`, `Reaction`, `RetroStep`, `RouteTree`, `CandidateRoute`, `RouteRationale`, `ProposerOutput`, `ForwardValidationResult`, `ConditionPrediction`, `BuyabilityRecord`.
- `rasyn/rasyn/synth/retro/registry.py` — sealed-case registry stub for the three cases.
- `tests/test_retro_schemas.py` — round-trip + invariant tests (canonical SMILES idempotency, atom-mapped reaction SMILES validation, route-tree leaf=buyable invariant).
**Cost:** 0 GPU-h. ~6 h wall.
**Done when:** schemas frozen, registry committed, tests green.

### Phase R-1 — Reaction data curation
**Goal:** Build tiered reaction fact tables with hard sealed-case decontamination.
**Artifacts:**
- `rasyn/rasyn/data/sources/uspto.py` — Lowe USPTO-full ingestion + atom mapping via RXNMapper.
- `rasyn/rasyn/data/sources/ord.py` — ORD protobuf parser + ORDerly-cleaned subset.
- `rasyn/rasyn/data/sources/buyables.py` — ZINC-22 in-stock + Enamine REAL BB + eMolecules free → unified buyables InChIKey index.
- `rasyn/data/clean/retro/reactions_bronze.parquet` (~3–4M rows: USPTO-full + ORD curated).
- `rasyn/data/clean/retro/reactions_silver.parquet` (~480K USPTO-MIT + USPTO-LLM 247K with conditions).
- `rasyn/data/clean/retro/buyables.parquet` (~5–10M canonical SMILES, cost-tier flag).
- `rasyn/data/clean/retro/templates.pkl` — RDChiral-extracted templates with frequency ≥ N.
- `rasyn/scripts/run_retro_curation.py` — multi-pass orchestrator (modelled on `run_curation.py`).
- `artifacts/retro_decontam_audit/` — canary report.
**Decontamination (RETRO §13):** quarantine sealed target + all reactions producing it + product-Tanimoto ≥ 0.85 neighbours + intermediate-Tanimoto ≥ 0.85 + DOI/patent quarantine for known route papers (Karpf/Trussardi/Federspiel for oseltamivir; Pfizer process patents + WO2021250648 for nirmatrelvir). Crucially, **template-level decontamination**: remove every extracted template whose source-reaction set hits any sealed target/intermediate at Tanimoto ≥ 0.85, otherwise a sealed disconnection leaks via a template hash.
**Cost:** 4–8 GPU-h (cuML FAISS dedup + RXNMapper at scale). 1–2 days wall.
**Done when:** all three tiers exist, buyables index resolves > 99% of known commercial BBs in smoke set, and 0 sealed-case canary survivors across product-Tanimoto, precursor-Tanimoto, template-source, and DOI quarantine.

### Phase R-2 — Single-step proposer ensemble (Track P — parallelisable, **5 channels**)
**Goal:** five trained proposers, each emitting unified `ProposerOutput` with top-K precursor sets + confidence.
**Artifacts:**
- `rasyn/scripts/train_retro_proposer_template.py` — RDChiral template applier + neural template classifier (200M-MLM encoder, FiLM-conditioned on reaction class). → `checkpoints/retro_template_v1/`.
- `rasyn/scripts/train_retro_proposer_graphedit.py` — bond-edit classifier on product graph (~30M-param graph transformer; from-scratch, no MLM init since input is graph not SMILES). → `checkpoints/retro_graphedit_v1/`.
- `rasyn/scripts/train_retro_proposer_seq2seq.py` — SMILES product→reactants encoder-decoder, encoder init from 200M MLM, decoder init from AR LM. → `checkpoints/retro_seq2seq_v1/`.
- `rasyn/scripts/build_retro_retrieval_index.py` — FAISS `IndexFlatIP` over product Morgan FP (2048-bit) + RXNFP-class metadata.
- **`rasyn/scripts/train_retro_proposer_diffusion.py` — DiGress-style discrete graph diffusion for reactant completion. Input: product graph + disconnection mask (which bonds are broken) + reaction class. Output: completed reactant subgraph. Reuses `rasyn/antibiotic/graph_diffusion.py` machinery from the ABX module (DiGress + FiLM conditioning). Scales up from the 5.4M-param ABX run (MEMORY L47, undertrained) to ~30–50M params + 200K–500K steps. → `checkpoints/retro_diffusion_v1/`.**
- `rasyn/rasyn/synth/retro/proposers/{template,graphedit,seq2seq,retrieval,diffusion}.py` — unified interface returning `ProposerOutput`.
- Per-proposer top-K reactant-recovery report on USPTO-50K silver test split.
**Cost:** template 4–8 GPU-h; graph-edit 12–24 GPU-h; seq2seq 24–48 GPU-h; retrieval 1 GPU-h; **diffusion 48–96 GPU-h (5×A100, scaled up from ABX-size to overcome the undertrained ceiling in MEMORY L47)**. **Total 90–180 GPU-h**, ~48 GPU-h wall on a 4-pod fork.
**Done when:** template top-10 ≥ 55%, graph-edit top-10 ≥ 50%, seq2seq top-10 ≥ 50%, retrieval top-10 ≥ 45%, **diffusion validity ≥ 40% on synthon-completion smoke + top-10 reactant recovery ≥ 35% on USPTO-50K silver split**; all five expose the unified interface.

**Honesty floor for the diffusion channel:** if R-6 ablation shows the 5-proposer ensemble does not beat the 4-proposer ensemble (template + graphedit + seq2seq + retrieval) on `route_found_to_buyables_rate` AND `forward_validated_route_rate` by ≥ 2 pts each, log the negative result and drop diffusion from the v1 production stack (parallels MEMORY L47 honesty principle). Do not paper over a weak channel.

### Phase R-3 — Forward validator + condition predictor (Track V — parallelisable with R-2)
**Goal:** Two models. Forward: `reactants + reagents → product`. Conditions: `reactants + product → reagent_class / solvent_class / temperature_bin / catalyst_class`.
**Artifacts:**
- `rasyn/scripts/train_retro_forward.py` — SMILES seq2seq, reactants→product direction, encoder init from 200M MLM. → `checkpoints/retro_forward_v1/`.
- `rasyn/scripts/train_retro_conditions.py` — multi-head classifier: solvent (~30 classes), catalyst (~20), temperature (4 bins: rt / reflux / cryo / high-T), reagent (~50). Encoder init from 200M MLM. → `checkpoints/retro_conditions_v1/`.
- `rasyn/rasyn/synth/retro/validator.py` — round-trip checker: given proposed retro step, run forward, accept iff `canonical(forward(reactants, conditions)) == canonical(target)` or Morgan-Tan ≥ 0.95 fallback.
- Forward held-out top-1 / top-5 on USPTO-MIT.
**Cost:** forward 24–48 GPU-h; conditions 8–16 GPU-h. **Total 40–60 GPU-h**, ~24 GPU-h wall.
**Done when:** forward top-1 ≥ 80% on USPTO-MIT test; conditions exact-class per head ≥ 60%; `round_trip_pass_rate` on USPTO-50K held-out ≥ 70%.

### Phase R-4 — Route-level value model + buyability index
**Goal:** neural `V(node)` predicting cost-to-go to buyables from a partial route node; frozen InChIKey buyability lookup.
**Artifacts:**
- `rasyn/scripts/train_retro_value_model.py` — Retro\*-style cost-to-go regressor, encoder init from 200M MLM. Offline supervision: expand USPTO-full products with the R-2 template proposer to depth 5, label nodes by realised cost. → `checkpoints/retro_value_v1/`.
- `rasyn/rasyn/synth/retro/buyability.py` — pure index, no model.
- `rasyn/rasyn/synth/retro/route_score.py` — implements L6's fixed-weight scalarisation.
**Cost:** 16–32 GPU-h. ~12 GPU-h wall.
**Done when:** value MAE on held-out offline-generated trees ≥ 25% better than depth-baseline; buyability lookup resolves > 99% of smoke-set known BBs.

### Phase R-5 — Retro\* planner integration
**Goal:** wire proposers + validator + value + buyability into a working AND-OR tree search.
**Artifacts:**
- `rasyn/rasyn/synth/retro/planner.py` — Retro\* / A\* over AND-OR tree. Termination: all leaves buyable OR depth ≥ `max_steps` (default 8) OR time budget (default 60 s smoke / 300 s eval).
- `rasyn/rasyn/synth/retro/orchestrator.py` — given target SMILES + constraints YAML packet (per RETRO.md §7), returns top-K `CandidateRoute`.
- `rasyn/scripts/run_retro_smoke.py` — three-layer verification (smoke 1 / slice 10 / preflight 50). Pattern from [[feedback_three_layer_verification]].
- `artifacts/retro_smoke/` — smoke report.
**Cost:** 4–8 GPU-h. No new training.
**Done when:** smoke passes (route found in < 60 s for a known-easy target — e.g., ibuprofen, ranitidine); slice ≥ 80% targets get a route to buyables in ≤ 8 steps; preflight numerically stable across 50 targets.

### Phase R-6 — Baselines + ablations
**Goal:** 10 baselines + ~10 ablations, mirroring ABX pattern.
**Artifacts:**
- `rasyn/scripts/run_retro_baselines.py` — (1) random disconnection, (2) template-only no-search, (3) seq2seq-only no-search, (4) retrieval-only no-search, (5) BFS no-value-model, (6) MCTS instead of A\*, (7) no-forward-validator, (8) no-buyability-pruning, (9) frozen unanimous-vote-only ensemble, (10) value-model = depth-heuristic.
- `rasyn/scripts/run_retro_ablations.py` — remove each proposer / validator / value model / FiLM conditioning / hard-negative training / multi-seed in turn.
- `artifacts/retro_baselines/` + `artifacts/retro_ablations/`.
**Cost:** 8–16 GPU-h.
**Done when:** v1 beats best baseline on `route_found_to_buyables_rate` by ≥ 15 pts AND on `forward_validated_route_rate` by ≥ 20 pts across the slice set.

### Phase R-7 — Sealed-case lock + locked-prediction reveal
**Goal:** run the three sealed cases under the same locked-prediction protocol as ADMET/ABX.
**Artifacts:**
- `rasyn/scripts/run_retro_sealed_cases.py` — orchestrator mirroring `run_abx_sealed_cases_v4.py`.
- `artifacts/retro_stage5_results/` — locked top-5 routes per case, SHA256-hashed BEFORE reveal.
- `rasyn/scripts/evaluate_retro_sealed_cases.py` — auto-judge: forward-validation pass, buyables coverage, step-count vs literature, route-class similarity (Levenshtein on canonical reaction-class sequence + Tanimoto on intermediates), verdict bucket (`literature_optimal` / `literature_competitive` / `novel_valid` / `missed`).
- 13-section appendix per case (reuse ADMET report builder pattern).
**Cost:** 4–8 GPU-h.
**Done when:** verdicts locked, hashed, signed off; honesty floor met (L9).

### Phase R-8 — FOLDED INTO R-2 (2026-05-12 override)
Diffusion reactant-completion is no longer a deferred fork. It is R-2 Channel 5 from the start. R-8 slot is reserved for a future v2 enhancement (e.g., yield-prediction layer or stereo-aware extension).

---

## 4. Critical path

The shortest chain to a working end-to-end demo (excludes diffusion since it is the most expensive R-2 channel):

`R-0 → R-1 → R-2-light (template + seq2seq only) → R-3-forward (skip conditions) → R-4 → R-5 smoke → R-7 single-case (RETRO-003 only, since it has no literature canary)`.

**Estimate:** ~80–120 GPU-h, ~5–7 days wall on a single 5×A100 pod.
This produces a runnable system on RETRO-003 (Rasyn-designed) before investing in the full benchmark. Use only if compute or time pressure demands a fast first demo; default plan is the full 5-channel path.

**Full v1 path including diffusion:** ~200–330 GPU-h, ~10–14 days wall on 4 parallel pods.

---

## 5. Parallelisable forks ([[feedback_parallel_tracks]])

- **Pod 1 — Track D (data):** R-1. CPU-heavy, 1×A100 sufficient.
- **Pod 2 — Track P-fast (proposers, fast channels):** R-2 template + graph-edit + seq2seq in parallel sub-jobs on 5×A100. Retrieval index runs CPU-side.
- **Pod 3 — Track P-diff (diffusion):** R-2 Channel 5 (diffusion) on its own 5×A100 pod since it is the longest-running channel (48–96 GPU-h). Joins the ensemble when ready.
- **Pod 4 — Track V (validator/conditions):** R-3 forward + conditions in parallel on 5×A100.
- **Pod 5 — Track S (search):** R-4 value model starts as soon as any proposer in R-2 produces offline trees; R-5 integration is CPU / single-GPU.
- **Convergence:** Pod 2 joins R-5 first; the planner runs initially with the 4-channel ensemble; Pod 3 (diffusion) joins the ensemble when its training completes, and R-5/R-6/R-7 re-run with the 5-channel ensemble. This lets us produce a usable system earlier rather than gating everything on the slowest channel.

Total wall with 5 parallel pods: ~10–14 days for R-0 through R-7 with full 5-channel ensemble. (4-channel checkpoint demo possible at ~7–10 days.)

---

## 6. Three-layer verification ritual ([[feedback_three_layer_verification]])

Every training script and every full-scale run must pass, in order:
1. **Smoke** — 1 minibatch / 1 example, asserts forward pass + loss finite + saves to disk.
2. **Slice** — ~1% of data, ~10 min, asserts metrics move in the right direction and no nan/inf.
3. **Preflight** — ~10% of data, ~1 h, full pipeline including eval, asserts all downstream artifacts produced and schema-valid.

No full run launches without smoke + slice + preflight all green. This rule applies to R-1, R-2, R-3, R-4 individually, and to the R-5 planner integration.

---

## 7. Risks (top 5, ranked) + mitigations

1. **Forward validator weak on out-of-distribution intermediates** → plausible-looking but wrong routes.
   *Mitigation:* OOD split by Murcko + reaction-class jointly during R-3 training; require round-trip Tanimoto ≥ 0.95 not exact match; report OOD metrics separately in R-6.
2. **Search blows up / never terminates on hard targets.**
   *Mitigation:* hard step-budget (default 8) + wall-time budget; iterative deepening; log "no-route-found" as honest verdict (mirrors MEMORY L43).
3. **Buyables list inflates "route found" rate artificially.**
   *Mitigation:* freeze buyables to a single dated snapshot, restrict to cost-tier ≤ ~$10/g, and report `tier-1-only` rates separately from `any-buyable` rates.
4. **Sealed-case leakage via templates.** A template extracted from USPTO may directly encode a sealed disconnection.
   *Mitigation:* template-level decontamination (R-1) — drop every template whose source-reaction set contains the sealed target or any of its named intermediates within Tanimoto ≥ 0.85.
5. **Single-step proposers memorise USPTO** → great in-distribution numbers, poor real-world generalisation (parallels ABX classical-scaffold ceiling, MEMORY L43, L46).
   *Mitigation:* OOD test split by patent-year + Murcko cluster; never headline in-distribution numbers without OOD alongside.

---

## 8. Sealed cases (locked-prediction protocol, identical to ADMET/ABX)

### RETRO-001 — Literature-precedent recovery: **oseltamivir (Tamiflu)**
Multi-step natural-product-derived synthesis with multiple published industrial routes (shikimic-acid Roche route, Karpf/Trussardi 2001 route, Federspiel 1999 route). Tests whether the system rediscovers a *known canonical* route to a marketed drug.
- **Auto-judge:** route-class match (carbohydrate-derived vs. de-novo) + step-count within ±2 of published shortest + forward-validation across all steps.
- **Decontamination:** quarantine Karpf/Trussardi, Federspiel, and Roche-route DOIs/patents; remove all reactions whose product Tanimoto to oseltamivir or any of its 6 named intermediates ≥ 0.85.
- **Verdict buckets:** `literature_optimal` (matches shortest published), `literature_competitive` (within 2 steps of shortest, forward-validates), `novel_valid` (different route, forward-validates), `missed`.

### RETRO-002 — Novel-route discovery: **nirmatrelvir (PF-07321332, Paxlovid)**
Recent target with a published Pfizer route using bespoke chemistry. Several short alternatives have been proposed post-approval. Tests whether the system finds a *novel valid* alternative the Pfizer route did not use.
- **Auto-judge:** route differs from Pfizer at ≥ 2 disconnection sites **AND** forward-validates **AND** lands on tier-1 buyables **AND** ≤ 8 steps. Verdict `novel_valid` if alternative is internally consistent; `literature_competitive` if it recovers Pfizer.
- **Decontamination:** quarantine Pfizer process papers + WO2021250648 + immediate intermediates.

### RETRO-003 — Rasyn-designed molecule synthesis: **a top-K hit from `artifacts/abx_round3_external_summary.json` or the de-novo AR-LM pool (MEMORY L48/L49)**
Candidate selection: pick a single Rasyn-generated SMILES with (a) composite_score ≥ 0.85, (b) novelty (1 − max Tanimoto to training actives) ≥ 0.55, (c) drug-like (8 ≤ heavy ≤ 60, ≥ 1 ring). The chosen target is hashed into the registry before R-7 starts.
- **No literature answer exists.** Tests Demo 1 of RETRO.md §14: "Rasyn designs a molecule and tells you how to make it."
- **Auto-judge:** route reaches buyables (binary) + forward-validation pass rate ≥ 80% + structured rationale populated + ≥ 1 route in top-3 flagged "would attempt" by expert review.
- **Verdict bucket:** `route_proposed_no_literature_baseline` (analogous to ABX-003 `not_measurable`).
- **No decontamination needed** (no literature answer to leak).

---

## 9. Honesty floor (per L9)

A `missed` verdict on any sealed case is acceptable provided:
- structured rationale is populated,
- the literature route appears in the top decile of the candidate pool, and
- the report explicitly identifies the data-distribution / coverage gap that drove the miss.

Do not paper over a miss with rule-pack heuristics or threshold tuning (per [[feedback_no_fallbacks_follow_plan]]). If a phase reveals an upstream gap, build the missing upstream phase rather than substitute a heuristic.

---

## 10. Approval checklist (sign off here before R-0 starts)

- [ ] L1–L9 confirmed.
- [ ] §2 scope locks confirmed.
- [ ] §3 phase definitions accepted.
- [ ] §8 sealed-case targets confirmed (or RETRO-003 candidate selection deferred until after R-2 ranks candidates).
- [ ] Compute budget: estimate ~200–330 GPU-h for full v1 path including diffusion, ~80–120 GPU-h for critical path (4-channel). Approve a pod plan (default: 5 parallel pods).
- [ ] Next action when approved: execute Phase R-0 (schemas + registry + tests). Estimated wall: ~6 h, 0 GPU-h.
