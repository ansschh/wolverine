# Rasyn ADMET Rescue System: Conditioning Schema, Architecture, Ablations, Baselines, Benchmark Protocol, and Decontamination Report

**Document purpose:** This document is the ADMET-only design specification for the Rasyn held-out discovery program. It is intended as a context file for coding agents, research agents, data-curation agents, benchmark agents, and evaluation agents. It turns the current decisions into a concrete spec without overcommitting to one exact neural architecture.

**Scope:** This document covers only the ADMET rescue system. It does **not** cover antibiotic discovery or NMR/spectra work. Those will have separate specs.

**Core claim we are building toward:**

> Rasyn was trained from scratch on a contamination-controlled chemistry corpus. ADMET rescue case studies were sealed before training. Direct molecules, case-family analogs, assay records, documents, synonyms, and leakage neighborhoods were removed. After training, Rasyn was given only the starting chemistry problem and candidate structures or candidate-generation task. It discovered successful held-out ADMET rescue solutions by proposing and ranking candidates that improve the specified liability while preserving the desired pharmacology.

---

## 0. Locked high-level decisions

The following decisions are now locked for the ADMET rescue track unless explicitly revised later.

### 0.1 The central task

ADMET rescue is not molecule-only ADMET prediction.

The central task is:

```text
Given:
  parent molecule P
  candidate molecule C
  desired activity/pharmacology context A
  liability L
  rescue mode R

Predict:
  whether C is a successful rescue of P for liability L while preserving A.
```

The main training and inference object is therefore:

```text
(P, C, A, L, R) -> rescue score / rescue label / failure mode
```

not:

```text
C -> ADMET property
```

Absolute ADMET prediction is useful, but only as auxiliary evidence, teacher signal, pretraining, or side module. The north-star objective is parent-candidate rescue ranking.

### 0.2 System shape

The ADMET rescue system has two major parts:

```text
candidate proposer system -> candidate ranker / rescue scorer
```

The proposer optimizes for **recall**:

> Did the system create or retrieve the true or functionally correct rescue candidate?

The ranker optimizes for **precision**:

> Did the system move the best rescue candidates into top-k?

### 0.3 Version to build first

Build the hybrid rescue model first:

```text
Version 3: pairwise rescue model + absolute property evidence + delta evidence + activity-retention evidence + structured rationale evidence
```

Then run ablations later:

```text
Version 1: pairwise-only rescue model
Version 2: absolute-property-first rescue model
Version 3: hybrid rescue model
```

### 0.4 Evidence mode

For held-out discovery cases, we use **Mode 1: true rescue discovery**.

Allowed:

```text
parent measured facts
candidate structure
computed candidate descriptors
internally predicted candidate properties from clean models
structured mechanism/rationale evidence generated without hidden labels
```

Forbidden:

```text
candidate measured held-out ADMET outcome
candidate measured held-out activity outcome
candidate known historical success label
papers or assay records describing the sealed solution
answer names/synonyms or direct hints
```

### 0.5 Candidate universe

Use a hybrid multi-proposer system:

```text
analog retrieval proposer
+ matched molecular pair transformation proposer
+ liability-specific rule proposer
+ learned inverse-delta proposer
+ forward-model optimization proposer
+ pure learned novelty proposer
+ prodrug/exposure proposer where relevant
```

The pure learned proposer is important for novelty and possible headline generation, but the official v1 system should not rely on it alone.

### 0.6 Sealed ADMET case structure

The sealed ADMET benchmark should include three ADMET rescue modes:

```text
1. hERG / cardiac safety rescue
2. oral exposure / prodrug rescue
3. direct analog solubility or metabolic-stability rescue
```

Primary case candidates:

```text
Case 1: Terfenadine -> fexofenadine
Case 2: Acyclovir -> valacyclovir
Case 3: Direct analog solubility/metabolic-stability rescue from curated SAR or internal data; exact case TBD after audit
```

Backup:

```text
Enalaprilat -> enalapril
```

Use the backup only if Case 3 cannot be prepared in time.

---

# 1. Definitions

## 1.1 Parent molecule

The parent molecule is the starting molecule that has useful pharmacology but an ADMET liability.

Example:

```text
Parent: terfenadine
Desired activity: H1 antihistamine activity
Liability: hERG/QT/cardiac risk
```

The parent’s measured facts are allowed inputs because the rescue task starts from a known problem.

Allowed parent measured facts may include:

```text
parent potency or activity bucket
parent solubility
parent hERG liability
parent metabolic stability
parent microsomal clearance
parent oral bioavailability
parent permeability
parent known toxicity/cytotoxicity flags
```

## 1.2 Candidate molecule

The candidate molecule is a proposed rescue analog/prodrug/metabolite/replacement generated or retrieved by the system.

The candidate structure is allowed input.

The candidate’s hidden measured outcome is not allowed during held-out evaluation.

## 1.3 Desired activity context

The desired activity context specifies what must be preserved.

Examples:

```text
H1 antagonism
antiviral active species delivery
ACE inhibition
kinase inhibition
GPCR antagonism
```

The context may include:

```text
target name / target family
assay family
activity type
parent potency
acceptable potency loss
pharmacophore features to preserve
```

It must not include the answer molecule or historical rescue description.

## 1.4 Liability

The liability is the ADMET problem to fix.

Supported v1 liability families:

```text
hERG / cardiac liability
poor solubility
poor metabolic stability / high clearance
poor oral exposure / poor bioavailability
```

Later liability families may include:

```text
CYP inhibition
cytotoxicity
permeability
efflux
plasma protein binding
reactive metabolite risk
```

but these are not v1 priorities.

## 1.5 Rescue mode

The rescue mode specifies the kind of medicinal chemistry operation.

Supported v1 rescue modes:

```text
direct_analog_safety_rescue
polarity_solubility_rescue
metabolic_soft_spot_rescue
prodrug_exposure_rescue
active_metabolite_safety_rescue
```

The rescue mode matters because the same transformation can be good or bad depending on context.

Example:

```text
Adding polarity may help hERG or solubility but hurt permeability.
A prodrug may not directly retain target potency but can still preserve pharmacology by converting to the active species.
```

## 1.6 Acceptable tradeoff

The acceptable tradeoff defines how much activity loss or other risk is allowed.

Default activity-retention buckets:

```text
strong_retention: candidate potency within 3x parent potency
acceptable_retention: candidate potency within 10x parent potency
weak_retention: candidate potency 10x-100x worse
failed_retention: candidate potency >100x worse
unknown_retention: data insufficient or incomparable
```

For prodrug/exposure rescue, this is different:

```text
candidate does not need direct potency retention if it plausibly delivers the active species.
```

For prodrug cases, activity retention is represented as:

```text
active_species_delivery_retained
```

not:

```text
candidate_direct_potency_retained
```

---

# 2. Conditioning schema for ADMET rescue

The conditioning schema defines exactly what the system receives as input and what each module is allowed to use.

This is not just a prompt format. It defines the scientific task.

## 2.1 Conditioning principles

### Principle 1: Tell the model the problem, not the answer

Allowed:

```text
This parent has hERG liability. Preserve H1 activity. Reduce hERG risk.
```

Forbidden:

```text
Find the carboxylated active metabolite fexofenadine.
```

### Principle 2: Parent measured facts are allowed

The system is rescuing a known flawed parent. It is fair to tell it what is wrong with the parent.

Allowed:

```text
Parent hERG IC50 indicates high risk.
Parent has poor solubility.
Parent has poor oral bioavailability.
Parent has high microsomal clearance.
```

### Principle 3: Candidate hidden outcomes are forbidden

During held-out evaluation, the system cannot receive:

```text
candidate measured potency
candidate measured ADMET outcome
candidate known success label
historical paper text describing candidate
```

### Principle 4: Computed and internally predicted evidence is allowed

Allowed:

```text
computed descriptors
molecular graph
predicted ADMET from clean auxiliary model
predicted activity retention from clean model
pharmacophore similarity
shape similarity
soft-spot predictions
substructure alerts
```

But any internally predicted model used for candidate evidence must itself be trained under the same decontamination protocol.

### Principle 5: Inputs must be structured

The system should not rely on free-form explanations as hidden reasoning.

Use structured fields:

```text
liability_type
rescue_mode
activity_context
computed_delta_descriptors
transformation_class
preserved_activity_features
liability_driver_features
possible_failure_modes
```

The final natural-language explanation can be generated from structured fields after ranking.

---

## 2.2 Standard input packet: ADMETChallengePacket

The challenge packet is the case-level object given to the proposer and ranker.

```yaml
schema_version: "admet_challenge_packet_v1"
case_id: "ADMET_CASE_001"
mode: "heldout_discovery"

parent:
  molecule_id: "PARENT_001"
  canonical_smiles: "..."
  inchi_key: "..."
  known_aliases_allowed: []
  known_aliases_forbidden: []

activity_context:
  activity_context_id: "ACTCTX_001"
  desired_pharmacology: "H1 antagonism"
  target_name: "Histamine H1 receptor"
  target_family: "GPCR"
  organism: "human"
  assay_context_available: true
  activity_preservation_goal: "retain H1 antagonist activity"
  parent_activity:
    value: 40
    unit: "nM"
    endpoint_type: "Ki_or_IC50"
    confidence: "approximate"
  acceptable_activity_tradeoff:
    strong_retention_fold: 3
    acceptable_retention_fold: 10
    failed_retention_fold: 100

liability_context:
  liability_type: "hERG_cardiac_risk"
  liability_description: "parent has cardiac potassium-channel / QT risk"
  parent_liability:
    endpoint_type: "hERG_IC50_or_risk_category"
    value: "high_risk"
    confidence: "known"
  liability_reduction_goal: "reduce hERG-like cardiac risk"

rescue_context:
  rescue_mode: "direct_analog_safety_rescue"
  objective: "reduce liability while preserving desired activity"
  allowed_strategy_types:
    - "polarity_tuning"
    - "basicity_tuning"
    - "active_metabolite_like_rescue"
  forbidden_clues:
    - "answer_name"
    - "historical_solution_description"
    - "paper_text"

constraints:
  max_candidate_count_after_filter: 2000
  final_top_k_values: [5, 10, 20]
  allow_prodrug: false
  require_candidate_direct_activity: true
  require_basic_druglike_filters: true
```

### Notes

For prodrug/exposure rescue, the challenge packet changes:

```yaml
rescue_context:
  rescue_mode: "prodrug_exposure_rescue"
  objective: "improve oral exposure while delivering active species"

constraints:
  allow_prodrug: true
  require_candidate_direct_activity: false
  require_active_species_delivery_rationale: true
```

---

## 2.3 Proposer input schema

The proposer receives the challenge packet but no candidate labels.

```yaml
schema_version: "admet_proposer_request_v1"
challenge_packet_ref: "ADMET_CASE_001"
parent_structure: "..."
activity_context: {...}
liability_context: {...}
rescue_context: {...}
constraints:
  raw_candidate_budget: 20000
  filtered_candidate_budget: 2000
  desired_candidate_diversity: "high"
  include_learned_novelty_channel: true
  include_retrieval_channel: true
  include_mmp_channel: true
  include_liability_rule_channel: true
  include_forward_optimization_channel: true
```

The proposer output is:

```yaml
schema_version: "admet_proposer_output_v1"
challenge_id: "ADMET_CASE_001"
raw_candidates:
  - candidate_id: "CAND_RAW_000001"
    candidate_smiles: "..."
    proposer_sources:
      - "mmp_transform"
      - "hERG_rule"
    initial_transformation_class:
      - "polarity_increase"
      - "basicity_tuning"
  - candidate_id: "CAND_RAW_000002"
    candidate_smiles: "..."
    proposer_sources:
      - "learned_inverse_delta"
```

---

## 2.4 Candidate evidence packet

After candidate generation and filtering, every candidate gets a candidate evidence packet.

```yaml
schema_version: "admet_candidate_evidence_packet_v1"
challenge_id: "ADMET_CASE_001"
parent_id: "PARENT_001"
candidate_id: "CAND_000123"
candidate_smiles: "..."

provenance:
  proposer_sources:
    - "mmp_transform"
    - "learned_inverse_delta"
  generated_after_training: true
  measured_hidden_outcomes_included: false

structure:
  canonical_smiles: "..."
  inchi_key: "..."
  valid_structure: true
  same_as_parent: false
  heavy_atom_count: 38

computed_parent_candidate_evidence:
  tanimoto_similarity: 0.74
  same_murcko_scaffold: true
  max_common_substructure_fraction: 0.81
  transformation_distance: "small_to_moderate"
  changed_atoms_or_substructures:
    - "terminal_alkyl_group_modified"

computed_descriptors_parent:
  molecular_weight: 471.7
  logP_estimate: 5.2
  TPSA: 42.1
  HBD: 1
  HBA: 4
  rotatable_bonds: 10
  formal_charge: 0
  aromatic_ring_count: 3
  fsp3: 0.32

computed_descriptors_candidate:
  molecular_weight: 501.6
  logP_estimate: 3.6
  TPSA: 78.4
  HBD: 2
  HBA: 6
  rotatable_bonds: 11
  formal_charge: 0
  aromatic_ring_count: 3
  fsp3: 0.35

computed_delta_descriptors:
  delta_molecular_weight: 29.9
  delta_logP: -1.6
  delta_TPSA: 36.3
  delta_HBD: 1
  delta_HBA: 2
  delta_rotatable_bonds: 1

activity_retention_evidence:
  target_family: "GPCR"
  pharmacophore_similarity: 0.84
  shape_similarity: 0.78
  preserved_activity_features:
    - "aryl_core"
    - "key_HBA_geometry"
    - "basic_center_spacing_or_replacement"
  predicted_activity_retention:
    category: "acceptable_retention"
    confidence: 0.71
  predicted_potency_delta:
    bucket: "within_10x"
    confidence: 0.64

liability_evidence:
  liability_type: "hERG_cardiac_risk"
  suspected_parent_liability_drivers:
    - "high_lipophilicity"
    - "lipophilic_basic_amine_character"
  candidate_changes_affecting_liability:
    - "reduced_logP"
    - "increased_polar_surface_area"
    - "basicity_tuning"
  predicted_liability_delta:
    category: "strong_improvement"
    confidence: 0.73

risk_evidence:
  new_liability_flags:
    - "possible_permeability_loss"
  severe_alerts:
    - null
  synthesizability_or_plausibility:
    category: "plausible"
    confidence: 0.68
```

---

## 2.5 Ranker input schema

The ranker receives the challenge packet plus candidate evidence packets.

```yaml
schema_version: "admet_ranker_input_v1"
challenge_packet: {...}
candidates:
  - candidate_evidence_packet: {...}
  - candidate_evidence_packet: {...}
ranking_context:
  compare_candidates_within_case: true
  top_k_values: [5, 10, 20]
  liability_priority: "primary"
  activity_retention_priority: "must_not_fail"
  absolute_risk_priority: "must_not_remain_high_risk"
```

---

## 2.6 Ranker output schema

```yaml
schema_version: "admet_ranker_output_v1"
challenge_id: "ADMET_CASE_001"
model_checkpoint_hash: "..."
dataset_hash: "..."
inference_timestamp: "..."

ranked_candidates:
  - rank: 1
    candidate_id: "CAND_000123"
    canonical_smiles: "..."
    rescue_score: 0.91
    rescue_label_prediction: "strong_success"
    predicted_failure_mode_probabilities:
      failed_activity_loss: 0.08
      failed_no_liability_improvement: 0.05
      failed_new_liability: 0.12
      uncertain: 0.07
    predicted_activity_retention:
      category: "acceptable_retention"
      confidence: 0.71
    predicted_liability_improvement:
      category: "strong_improvement"
      confidence: 0.73
    structured_rationale:
      liability_driver_tuned:
        - "lipophilic_basic_amine_character"
      activity_features_preserved:
        - "aryl_core"
        - "key_HBA_geometry"
      transformation_class:
        - "polarity_increase"
        - "basicity_tuning"
      main_risk:
        - "permeability_loss"
    proposer_sources:
      - "mmp_transform"
      - "learned_inverse_delta"

  - rank: 2
    candidate_id: "CAND_000456"
    ...

summary:
  exact_historical_answer_detected_by_system: "unknown_until_reveal"
  final_top_k_locked: true
  hidden_measured_candidate_outcomes_used: false
```

---

# 3. ADMET architecture

This section describes the system architecture at the level needed for implementation planning. It does not lock a specific neural model family.

## 3.1 High-level architecture

```text
ADMETChallengePacket
        |
        v
Candidate Proposer Ensemble
        |
        v
Candidate Canonicalization + Deduplication + Filters
        |
        v
Candidate Evidence Builder
        |
        v
Auxiliary Evidence Modules
        |
        v
Pairwise Rescue Ranker
        |
        v
Top-k Locked Predictions + Structured Rationales
        |
        v
Benchmark Scoring + Audit Report
```

## 3.2 Component 1: Challenge packet loader

Responsibilities:

```text
load challenge packets
validate allowed fields
check forbidden fields are absent
attach parent measured facts
attach activity/liability/rescue contexts
```

Important invariant:

```text
The challenge packet must not include answer names, answer structures, hidden measured candidate outcomes, or paper text describing the solution.
```

## 3.3 Component 2: Candidate proposer ensemble

The proposer ensemble creates candidate molecules.

### 3.3.1 Analog retrieval proposer

Retrieves analogs from clean candidate databases.

Inputs:

```text
parent structure
activity context
liability
rescue mode
```

Retrieval signals:

```text
structural similarity
same or related target context
same scaffold or pharmacophore
same medicinal chemistry series outside sealed cases
purchasability or makeability if known
```

### 3.3.2 Matched molecular pair transformation proposer

Applies learned transformations mined from non-held-out rescue pairs and analog series.

Examples:

```text
phenyl -> pyridyl
methyl -> fluorine
tertiary amine -> less basic amine
lipophilic group -> polar bioisostere
benzylic H -> steric blocker
```

### 3.3.3 Liability-specific rule proposer

Uses curated transformation libraries by liability type.

For hERG/cardiac risk:

```text
reduce lipophilic basic amine character
reduce excessive logD/logP
increase polarity carefully
reduce aromatic hydrophobic surface
basicity tuning
```

For solubility:

```text
add solubilizing heteroatoms
reduce planarity
replace phenyl with heteroaryl
add ionizable handle if compatible
reduce excessive lipophilicity
```

For metabolic stability:

```text
block metabolic soft spots
replace labile motifs
reduce lipophilicity
add steric shielding
replace oxidation-prone groups
```

For prodrug/exposure rescue:

```text
ester prodrug
amino-acid ester prodrug
temporary masking of polar group
transporter-aware promoiety
```

### 3.3.4 Learned inverse-delta proposer

Learns:

```text
parent P + desired delta + activity context + liability + rescue mode -> candidate C
```

This proposer is expected to generate strong candidates that are not limited to known hand-coded rules.

### 3.3.5 Forward-model optimization proposer

Generates candidates and optimizes them using clean forward models:

```text
predicted liability improvement
predicted activity retention
absolute risk category
uncertainty
synthetic plausibility
```

This proposer is useful but dangerous because reward optimization can exploit model weaknesses. It must be constrained by validity, parent-relative plausibility, and uncertainty penalties.

### 3.3.6 Pure learned novelty proposer

A more unconstrained learned generator that tries to propose novel rescue candidates.

Use as a novelty channel, not the only official path.

Outputs from this channel should be heavily filtered and ranked.

---

## 3.4 Component 3: Candidate filters

After raw generation, candidates must pass filters.

### Hard structure filters

```text
valid valence
canonicalizable structure
no impossible atoms/groups
reasonable molecular weight
reasonable heavy atom count
no disconnected nonsense unless expected salt/form
```

### Parent-relative filters

```text
not identical to parent
reasonable similarity unless novelty channel
reasonable transformation distance
preserves core pharmacophore if direct analog rescue
```

### Rescue-mode filters

Direct analog rescue:

```text
candidate itself must plausibly retain activity
```

Prodrug rescue:

```text
candidate must plausibly convert to active species
```

### Safety/toxicophore filters

```text
severe reactive alerts removed or flagged
PAINS-like or assay-interference alerts flagged
unstable motifs removed or flagged
```

Do not automatically remove every risky motif. Some case studies may legitimately involve unusual chemistry. But severe invalid structures should be removed.

---

## 3.5 Component 4: Candidate evidence builder

Builds structured candidate evidence packets.

Responsibilities:

```text
compute descriptors
compute parent-candidate deltas
compute scaffold and pharmacophore similarity
identify changed fragments
classify transformation type
run clean auxiliary predictors
create structured rationale fields
identify possible failure modes
```

No hidden measured candidate outcomes are allowed during evaluation.

---

## 3.6 Component 5: Auxiliary evidence modules

Auxiliary evidence modules are not the main task. They support the ranker.

### Absolute property evidence

Predicts or estimates:

```text
solubility
hERG risk
metabolic stability
logD/logP
permeability proxy
toxicity flags
```

### Delta property evidence

Predicts:

```text
candidate improves/worsens liability relative to parent
expected delta category
```

### Activity-retention evidence

Predicts:

```text
activity-retention probability
potency delta bucket
pharmacophore preservation
shape/electrostatic similarity
activity-cliff risk
```

### Mechanism-aware evidence

Generates structured fields:

```text
liability driver features
modified features
preserved activity features
transformation class
possible failure modes
```

### Synthetic/plausibility evidence

Predicts or flags:

```text
chemical plausibility
synthetic plausibility
stability alerts
candidate novelty
```

---

## 3.7 Component 6: Pairwise rescue ranker

The ranker is the core model.

Input:

```text
challenge packet
candidate evidence packet
local candidate set context
```

Output:

```text
rescue score
predicted rescue label
activity-retention prediction
liability-improvement prediction
failure-mode probabilities
confidence / uncertainty
structured rationale
```

Main ranker objective:

```text
Rank candidates that improve the specified liability while preserving desired activity above candidates that fail one or both requirements.
```

The ranker must learn the difference between:

```text
ADMET improved but activity lost
activity retained but liability not fixed
wrong liability fixed
new liability introduced
true rescue
```

---

## 3.8 Component 7: Locked prediction ledger

Every final benchmark run must write:

```text
challenge packet hash
candidate pool hash
model checkpoint hash
dataset hash
inference timestamp
ranked output file hash
allowed tools manifest
forbidden fields check
```

This protects the integrity of the final claim.

---

# 4. Training objectives

## 4.1 Main training objective: rescue ranking

The main objective is local ranking.

Given one parent/liability/activity context and many candidates:

```text
rank strong_success candidates above weak_success candidates
rank weak_success candidates above hard negatives
rank hard negatives above invalid/irrelevant candidates if they are still chemically plausible
```

Example ranking order:

```text
strong_success
weak_success
uncertain_but_promising
failed_no_liability_improvement
failed_activity_loss
failed_new_liability
invalid_or_irrelevant
```

## 4.2 Pairwise rescue classification

For each pair:

```text
(P, C, A, L, R) -> rescue_label
```

Labels:

```text
strong_success
weak_success
failed_activity_loss
failed_no_liability_improvement
failed_wrong_liability
failed_new_liability
uncertain
```

## 4.3 Delta prediction auxiliary objective

```text
(P, C, L) -> liability delta category or value
```

Examples:

```text
hERG risk reduced
solubility increased 10x+
metabolic stability improved 3x+
```

## 4.4 Activity-retention auxiliary objective

```text
(P, C, A) -> activity retention category / potency delta bucket
```

Use continuous potency when reliable.

Use buckets when assay data is noisy or heterogeneous.

## 4.5 Failure-mode objective

Predict why a candidate fails:

```text
activity lost
liability unchanged
wrong liability fixed
new liability introduced
chemical implausibility
uncertain / insufficient evidence
```

This is important because failure-mode supervision prevents generic shortcut behavior.

## 4.6 Structured rationale objective

Predict structured rationale tags, not free-form text.

Fields:

```text
liability_driver_features
modified_features
preserved_activity_features
transformation_class
expected_liability_effect
expected_activity_effect
possible_failure_modes
```

---

# 5. Ablation plan

Ablations answer: what is actually making the system work?

Do not run every ablation immediately. But the final technical package should include the most important ones.

## 5.1 Model objective ablations

### Ablation A: pairwise-only model

Input:

```text
parent + candidate + activity context + liability
```

No auxiliary property evidence.

Purpose:

```text
Test whether rescue comparison alone is enough.
```

### Ablation B: absolute-property-only model

Predict candidate ADMET/activity separately, then score with formula.

Purpose:

```text
Test whether normal ADMET/property prediction can solve rescue.
```

### Ablation C: hybrid model

Full version:

```text
pairwise rescue + absolute evidence + delta evidence + activity-retention evidence + structured rationale
```

Expected winner.

---

## 5.2 Conditioning ablations

Remove one conditioning field at a time.

### Remove liability type

Tests whether the model needs explicit liability conditioning.

Expected result:

```text
performance should drop because solubility/hERG/stability require different rescue logic.
```

### Remove rescue mode

Tests whether prodrug/direct analog/safety modes need to be separated.

Expected result:

```text
performance should drop especially on prodrug/exposure case.
```

### Remove activity context

Tests whether model merely optimizes ADMET without preserving pharmacology.

Expected result:

```text
more activity-loss candidates ranked highly.
```

### Remove parent measured facts

Tests whether knowing the parent’s actual liability helps.

Expected result:

```text
weaker case-specific decisions.
```

### Remove computed descriptor deltas

Tests whether explicit deltas help.

### Remove structured rationale tags

Tests whether mechanism-aware fields improve ranking and failure-mode prediction.

---

## 5.3 Proposer ablations

### Retrieval-only

Candidate pool from analog retrieval only.

### MMP-only

Candidate pool from matched molecular pair transformations only.

### Rule-only

Candidate pool from liability-specific rules only.

### Learned inverse-only

Candidate pool from learned inverse-delta proposer only.

### Forward-optimization-only

Candidate pool from forward reward optimization only.

### Pure learned novelty-only

Candidate pool from unconstrained learned proposer only.

### Full multi-proposer

Union of all proposer channels.

Metrics:

```text
exact answer recall@N
functional rescue recall@N
invalid rate
candidate diversity
top-k final rank after ranker
```

Expected result:

```text
Full multi-proposer should maximize recall. Learned channels may provide novelty but may have higher invalid/failure rates.
```

---

## 5.4 Data ablations

### No hard negatives

Tests whether the model learns shortcuts.

Expected failure:

```text
model ranks candidates that improve ADMET but lose activity.
```

### No curated gold rows

Tests value of high-confidence curated data.

### No silver rows

Tests value of scale.

### No structured mechanism annotations

Tests whether mechanism tags improve generalization.

### No prodrug-specific labels

Tests whether prodrug and direct analog rescue must be separated.

---

## 5.5 Decontamination radius ablations

Train/evaluate under different quarantine radii.

```text
Loose: exact answer and documents removed
Medium: exact + documents + close analogs removed
Strict: case-family + similarity neighborhood removed
```

The final claim should rely on strict or at least medium-strict decontamination.

The purpose of this ablation is diagnostic, not to weaken the final benchmark.

---

## 5.6 Candidate pool size ablations

Evaluate:

```text
100 candidates
500 candidates
1,000 candidates
2,000 candidates
5,000 candidates
```

Measure:

```text
proposer recall
ranker stability
top-k recovery
runtime
```

Recommended v1 production range:

```text
500-2,000 filtered candidates per case
```

---

# 6. Baselines

Baselines are necessary even if no existing system exactly matches our end-to-end task.

They answer:

> Could this have been solved by a dumb heuristic or standard property model?

## 6.1 Random baseline

Randomly rank candidates.

Purpose:

```text
establish expected top-k chance.
```

## 6.2 Parent similarity baseline

Rank candidates by structural similarity to parent.

Purpose:

```text
test whether preserving activity via closeness is enough.
```

Failure mode:

```text
may preserve activity but fail to fix liability.
```

## 6.3 Most-polar / lowest-logP baseline

For hERG and solubility cases, rank by:

```text
lowest logP / logD
highest TPSA
highest polarity increase
```

Purpose:

```text
test whether simple polarity heuristics solve the case.
```

Failure mode:

```text
may destroy activity or permeability.
```

## 6.4 Liability-only property baseline

Rank by predicted liability improvement only.

Example:

```text
lowest predicted hERG risk
highest predicted solubility
highest predicted metabolic stability
```

Purpose:

```text
test whether ADMET-only optimization is enough.
```

Expected failure:

```text
ranks inactive or implausible candidates too high.
```

## 6.5 Activity-only baseline

Rank by predicted activity retention only.

Purpose:

```text
test whether closest/potent analogs solve the case.
```

Expected failure:

```text
preserves potency but does not fix liability.
```

## 6.6 Weighted property baseline

Use a formula:

```text
score = liability_improvement_score + activity_retention_score - new_risk_score
```

where each score comes from separate property predictors.

Purpose:

```text
compare hybrid rescue ranker against standard modular scoring.
```

## 6.7 MMP frequency baseline

Rank transformations by historical frequency of successful MMP property improvements.

Purpose:

```text
test whether common medicinal chemistry transformations are enough.
```

## 6.8 Medicinal chemistry heuristic baseline

A hand-coded baseline per liability.

Examples:

hERG:

```text
reduce logP
reduce basicity
increase polarity
```

Solubility:

```text
increase polarity
add heteroatoms
reduce planarity
```

Metabolic stability:

```text
block soft spots
replace labile groups
```

Purpose:

```text
test whether expert-like simple rules solve the benchmark.
```

## 6.9 Oracle-proposer / weak-ranker diagnostic

For closed hard-ranking mode, include the known answer in the candidate set and test weak rankers.

Purpose:

```text
separate proposer failure from ranker failure.
```

## 6.10 Full system baseline comparison format

Final table:

```text
Baseline                         Case1 top-k  Case2 top-k  Case3 top-k  Mean rank  Notes
Random                           ...          ...          ...          ...        ...
Similarity-only                  ...          ...          ...          ...        ...
Liability-only                   ...          ...          ...          ...        ...
Activity-only                    ...          ...          ...          ...        ...
Weighted property                ...          ...          ...          ...        ...
MMP frequency                    ...          ...          ...          ...        ...
Medicinal chemistry heuristic    ...          ...          ...          ...        ...
Rasyn full system                ...          ...          ...          ...        ...
```

---

# 7. Final benchmark protocol

## 7.1 Benchmark modes

Each sealed ADMET case should be evaluated in two modes.

### Mode A: open proposer mode

Input:

```text
parent molecule
activity context
liability context
rescue mode
parent measured facts
```

System must:

```text
generate/retrieve candidates
filter candidates
rank candidates
lock top-k
```

Success:

```text
known historical answer or functional equivalent appears in top-k
```

This is the stronger discovery mode.

### Mode B: closed hard-ranking mode

Input:

```text
parent molecule
activity context
liability context
rescue mode
sealed hard candidate pool
```

The candidate pool includes:

```text
known answer
close-but-wrong analogs
activity-retaining non-rescues
ADMET-improving activity-loss candidates
wrong-liability candidates
simple heuristic traps
```

System must:

```text
rank candidates
lock top-k
```

Success:

```text
known answer or functional equivalent ranked near top
```

This isolates ranker ability.

---

## 7.2 Pre-registration

Before final training/evaluation, create:

```text
sealed_case_registry.json
benchmark_protocol.md
success_metrics.yaml
forbidden_entities.json
decontamination_config.yaml
baseline_config.yaml
```

These files must be frozen before final evaluation.

## 7.3 Freeze points

Freeze and hash:

```text
sealed case registry
raw data manifest
forbidden entity list
decontamination scripts
clean training dataset
training config
model checkpoint
proposer config
ranker config
candidate pool if closed mode
inference packet
ranked predictions
scoring script
```

## 7.4 Evaluation steps

```text
1. Select sealed cases.
2. Create forbidden entity lists.
3. Decontaminate raw corpus.
4. Build clean training data.
5. Train model/system from scratch.
6. Freeze model and configs.
7. Run open proposer mode for each case.
8. Run closed hard-ranking mode for each case.
9. Lock predictions before answer reveal/scoring.
10. Reveal hidden answer / scoring labels.
11. Score exact recovery and functional recovery.
12. Compare against baselines.
13. Generate decontamination report.
14. Generate final benchmark report.
```

---

## 7.5 Success definitions

### Exact recovery

The known held-out successful molecule appears in:

```text
top 5
top 10
top 20
```

Exact recovery is strongest when it occurs in open proposer mode.

### Functional recovery

A candidate is functionally recovered if it satisfies pre-registered criteria:

```text
same rescue mode
same desired activity context
same liability-improvement logic
candidate predicted/known to preserve activity sufficiently
candidate predicted/known to improve liability sufficiently
chemically plausible
not a trivial invalid or nonsensical molecule
```

Functional recovery should be judged by structured criteria, not vibes.

For final reporting, use:

```text
exact_top5
exact_top10
exact_top20
functional_top5
functional_top10
functional_top20
```

### Prodrug case scoring

For prodrug/exposure rescue:

```text
success = candidate plausibly delivers active species and improves exposure
```

not:

```text
candidate directly retains parent potency in vitro
```

### Direct analog case scoring

For direct analog rescue:

```text
success = candidate itself preserves desired activity and improves liability
```

---

## 7.6 Metrics

### Proposer metrics

```text
raw candidate count
filtered candidate count
invalid rate
exact answer recall@N
functional rescue recall@N
candidate diversity
novelty rate
proposer-source contribution
```

### Ranker metrics

```text
rank of known answer
rank of best functional candidate
top-k exact recovery
top-k functional recovery
mean reciprocal rank
failure-mode calibration
ranking over hard negatives
```

### End-to-end metrics

```text
open mode exact top-k
open mode functional top-k
closed mode exact top-k
closed mode functional top-k
baseline-relative improvement
```

---

## 7.7 Final benchmark report structure

The final report should include:

```text
1. Executive claim
2. Sealed case descriptions
3. Allowed input packets
4. What was hidden
5. Decontamination summary
6. Candidate proposer results
7. Ranker results
8. Baseline comparison
9. Ablations
10. Failure cases and near misses
11. Locked prediction hashes
12. Decontamination report appendix
13. Raw scoring tables
```

---

# 8. Decontamination report

The decontamination report is as important as the model result.

The purpose is to prove:

> The model did not see the answer, the case-family evidence, the assay labels, or the historical rescue story during training or inference.

## 8.1 Decontamination principle

Use **case-family quarantine**, not mechanism-family quarantine.

Remove:

```text
specific sealed case
answer molecule
parent-answer pair
case-family analogs
case papers
case assay records
case patents if needed
synonyms and identifiers
near-duplicate molecules
```

Do not remove:

```text
all prodrug examples
all hERG data
all solubility data
all metabolic stability data
all medicinal chemistry transformations
```

The model should be allowed to learn general chemistry, but not the sealed case.

---

## 8.2 Forbidden entity list

For each sealed case, define:

```yaml
case_id: "ADMET_CASE_001"
parent_molecules:
  - canonical_smiles: "..."
  - inchi_key: "..."
answer_molecules:
  - canonical_smiles: "..."
  - inchi_key: "..."
forbidden_synonyms:
  - "..."
forbidden_cas_numbers:
  - "..."
forbidden_pubchem_cids:
  - "..."
forbidden_chembl_ids:
  - "..."
forbidden_documents:
  - doi: "..."
  - pmid: "..."
  - chembl_doc_id: "..."
forbidden_assays:
  - chembl_assay_id: "..."
  - pubchem_aid: "..."
forbidden_patents:
  - "..."
manual_forbidden_analogs:
  - canonical_smiles: "..."
```

## 8.3 Exact molecule quarantine

Remove exact and normalized matches:

```text
canonical SMILES
InChIKey
neutralized form
desalted form
tautomer-normalized form
stereochemistry-normalized form
parent/answer variants
salts
prodrugs/metabolites if case-revealing
```

Report:

```text
number of exact records removed
number of salt/tautomer/stereo records removed
source datasets affected
```

## 8.4 Document and assay quarantine

Remove:

```text
papers describing the sealed rescue
review articles explicitly describing the sealed rescue
ChEMBL documents associated with the case
PubChem assay records associated with the answer
patents containing the parent-answer relationship
training rows derived from those documents
```

Report:

```text
removed document IDs
removed assay IDs
removed activity rows
removed ADMET rows
removed derived pairs
```

## 8.5 Neighborhood quarantine

Remove close analogs likely to reveal the case.

Suggested criteria for strict setting:

```text
exact normalized molecule match: remove
same ChEMBL document as case: remove
same PubChem assay tied to case: remove
manual case-family analog: remove
Tanimoto >= 0.85 to answer: remove
same Murcko scaffold AND same target/liability context AND Tanimoto >= 0.65: remove
same named analog series: remove
```

These thresholds should be finalized per case, then frozen.

## 8.6 Text leakage quarantine

For v1, the main discovery model should not train on free-form literature text.

If any text is used in auxiliary systems, scan and remove:

```text
answer names
parent names
synonyms
CAS numbers
InChIKeys
paper titles
phrases describing parent-answer rescue
patent text
review paragraphs
```

Report:

```text
text chunks removed
query terms used
embedding/fuzzy search results if applicable
```

## 8.7 Nearest-neighbor audit

For each held-out answer and parent, report closest remaining training molecules.

Table:

```text
heldout_molecule | nearest_train_molecule | tanimoto | same_scaffold | same_target | source | allowed_reason
```

This table answers:

> Did the model just see a near-copy?

If a close neighbor remains, explain why it is allowed or remove it.

## 8.8 Canary tests

Insert artificial canaries into raw data before decontamination.

Examples:

```text
CANARY_ADMET_CASE_FAKE_001
CANARY_FEXO_LIKE_FAKE_LABEL
CANARY_VALACYCLOVIR_FAKE_PAIR
```

The decontamination pipeline must remove them.

Report:

```text
canaries inserted
canaries removed
canary removal pass/fail
```

## 8.9 Raw-to-clean data manifest

Report:

```text
raw molecule count
raw assay fact count
raw pair count
removed molecule count
removed document count
removed assay count
removed analog-neighborhood count
final clean molecule count
final clean assay fact count
final clean pair count
```

## 8.10 Dataset hashes

Hash:

```text
raw manifest
forbidden entity list
clean training dataset
clean validation dataset
clean auxiliary datasets
sealed case registry
model checkpoint
final inference packets
final outputs
```

## 8.11 Decontamination pass/fail criteria

A sealed case passes decontamination if:

```text
answer molecule absent from training corpus
parent-answer relationship absent from training corpus
case papers absent
case assay rows absent
case-family analog leakage removed
nearest-neighbor audit acceptable
candidate hidden measured outcomes not provided at inference
```

If any condition fails, the case must be fixed or removed from headline claims.

---

# 9. Implementation-oriented file structure

Recommended repository layout:

```text
rasyn_admet/
  configs/
    admet_liabilities.yaml
    rescue_modes.yaml
    candidate_filters.yaml
    proposer_config.yaml
    ranker_config.yaml
    baseline_config.yaml
    decontamination_config.yaml

  sealed_cases/
    sealed_case_registry.json
    forbidden_entities.json
    case_001_terfenadine_fexofenadine.yaml
    case_002_acyclovir_valacyclovir.yaml
    case_003_direct_analog_tbd.yaml

  data_raw/
    chembl/
    pubchem/
    tdc/
    moleculenet/
    papers_curated/
    internal/

  data_clean/
    molecule_table.parquet
    assay_fact_table.parquet
    rescue_pair_table.parquet
    ranking_task_table.parquet
    candidate_universe_table.parquet

  decontamination/
    canonicalize.py
    expand_forbidden_entities.py
    remove_exact_matches.py
    remove_documents_assays.py
    remove_similarity_neighborhood.py
    text_leakage_scan.py
    nearest_neighbor_audit.py
    canary_tests.py
    reports/

  proposer/
    analog_retrieval.py
    mmp_transformer.py
    liability_rule_proposer.py
    inverse_delta_proposer.py
    forward_optimization_proposer.py
    novelty_proposer.py
    prodrug_proposer.py
    candidate_filter.py
    candidate_annotator.py

  evidence/
    descriptors.py
    pharmacophore_similarity.py
    activity_retention.py
    admet_property_predictors.py
    delta_predictors.py
    mechanism_annotations.py
    plausibility.py

  ranker/
    train_ranker.py
    infer_ranker.py
    scoring.py
    failure_mode_heads.py

  baselines/
    random.py
    similarity.py
    polarity.py
    liability_only.py
    activity_only.py
    weighted_property.py
    mmp_frequency.py
    heuristic.py

  benchmark/
    build_closed_candidate_sets.py
    run_open_mode.py
    run_closed_mode.py
    lock_predictions.py
    reveal_and_score.py
    report_generator.py

  outputs/
    locked_predictions/
    benchmark_reports/
    decontamination_reports/
```

---

# 10. Coding-agent priorities

## Priority 1: schemas

Implement schema validation for:

```text
ADMETChallengePacket
ProposerRequest
ProposerOutput
CandidateEvidencePacket
RankerInput
RankerOutput
DecontaminationReport
```

## Priority 2: decontamination-first data loading

No training data should be built before decontamination is applied.

## Priority 3: candidate proposer scaffolding

Implement proposer interfaces before optimizing any one proposer.

All proposers should output the same candidate format.

## Priority 4: candidate evidence builder

This is the glue between proposers and ranker.

## Priority 5: ranker training and inference

Train hybrid ranker first.

## Priority 6: baselines and benchmark runner

Baselines should run on the same candidate sets as Rasyn.

## Priority 7: reports

Generate benchmark and decontamination reports automatically.

---

# 11. Final locked ADMET plan

The ADMET rescue system will use:

```text
1. Structured conditioning packets
2. Hybrid multi-proposer candidate generation
3. Candidate evidence packets with computed/predicted evidence
4. Pairwise rescue ranker as the central model
5. Auxiliary ADMET/activity/delta/mechanism evidence modules
6. Structured rationale, not free-form hidden reasoning
7. Open proposer mode + closed hard-ranking mode
8. Strict case-family decontamination
9. Baselines that test dumb shortcuts
10. Locked prediction ledger and full audit trail
```

The key final design sentence:

> Rasyn ADMET rescue is a parent-candidate, liability-conditioned, activity-aware discovery system. It proposes chemically plausible rescue candidates, ranks them by whether they improve the specified ADMET liability while preserving the desired pharmacology, and validates success under sealed, contamination-controlled benchmark cases.

