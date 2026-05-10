# Rasyn ADMET Rescue Architecture Context

**Purpose of this document:** This is a working architecture-context file for coding agents and research agents who will help build the ADMET-rescue part of the Rasyn held-out discovery system. It captures the currently locked design decisions, the rationale behind them, the expected system behavior, input/output formats, and the conceptual boundaries of what the system is and is not allowed to use.

This document intentionally avoids committing to a specific neural architecture, specific model family, specific chemistry toolkit, or specific training-data implementation. Those decisions will be made later. The goal here is to define the **shape of the task**, the **system decomposition**, the **allowed evidence**, the **forbidden evidence**, the **proposer/ranker interaction**, and the **structured output format** so future coding agents do not accidentally build the wrong problem.

The core system described here is for **ADMET rescue** only. Antibiotic discovery and NMR/spectra modules will be specified separately.

---

## 0. Executive Summary

The ADMET-rescue system is not a normal ADMET property predictor.

A normal ADMET predictor answers questions like:

```text
Given molecule X, predict its solubility.
Given molecule X, predict hERG risk.
Given molecule X, predict microsomal stability.
```

The Rasyn ADMET-rescue system must answer a harder question:

```text
Given a parent molecule P that already has useful biological activity but suffers from an ADMET liability L,
and given a candidate molecule C,
does C rescue P by improving liability L while preserving the desired activity A?
```

The central object is therefore not a molecule alone. It is:

```text
(parent molecule P,
candidate molecule C,
desired pharmacology / activity context A,
known parent liability L,
rescue mode R,
acceptable tradeoff T)
```

The central prediction is:

```text
rescue_score(P, C, A, L, R, T)
```

The system’s north-star behavior is:

```text
Rank candidate molecules by likelihood of being successful ADMET rescues of the parent.
```

A successful ADMET rescue is not merely a candidate with better ADMET. A successful rescue is a candidate that:

1. Improves the requested liability.
2. Preserves the desired activity or pharmacological mechanism.
3. Avoids creating a new fatal liability.
4. Remains chemically plausible and useful.
5. Fits the requested rescue mode.

The model should not collapse into simple heuristics like:

```text
make everything more polar
choose the most similar analog
choose the safest-looking molecule
choose the most drug-like molecule
optimize one property while ignoring potency
```

The system must be explicitly designed to punish these shortcuts.

---

## 1. Locked Design Decisions

These decisions are considered locked for the current architecture context.

### 1.1 Central training signal

The central training signal is:

```text
parent-candidate rescue comparison
```

The primary model should learn:

```text
Given P, C, A, L, R, and T, predict whether C is a good rescue of P.
```

It should not be built as a molecule-only ADMET predictor.

Molecule-only ADMET prediction can be useful, but only as auxiliary evidence, pretraining, teacher signal, or side information.

### 1.2 Final decision behavior

The final system behavior is candidate ranking.

The system receives or creates a candidate set and returns ranked candidates:

```text
parent problem -> proposer(s) -> candidate set -> rescue ranker -> top-k locked predictions
```

### 1.3 Hybrid rescue system first

Given time constraints, the first full build should be the hybrid system:

```text
Version 3: hybrid rescue model
```

This means the rescue ranker uses:

- parent-candidate comparison,
- computed descriptors,
- predicted absolute properties,
- predicted property deltas,
- predicted activity-retention evidence,
- mechanism-aware evidence,
- structured rationale features.

Later, ablations can compare:

```text
Version 1: pairwise-only rescue model
Version 2: property-prediction-first model
Version 3: hybrid rescue model
```

But the first implementation should build Version 3 because it is likely to be strongest under time pressure.

### 1.4 Allowed and forbidden evidence

The system will use **Mode 1: true rescue discovery**.

Allowed:

- parent measured facts,
- parent molecule,
- candidate molecule,
- desired activity context,
- known parent liability,
- rescue mode,
- acceptable tradeoff,
- computed descriptors,
- internally predicted candidate properties from clean models,
- internally predicted activity-retention evidence,
- mechanism-aware computed evidence.

Forbidden:

- candidate’s measured held-out ADMET outcome,
- candidate’s measured held-out potency outcome,
- candidate’s historical success label,
- paper text saying the candidate solved the problem,
- assay rows from the sealed case,
- leaked names/synonyms/patents/articles that reveal the answer.

The parent’s measured liability is allowed because the whole problem is “this parent has a known problem.”

The candidate’s hidden measured success is forbidden because giving it would turn discovery into answer lookup.

### 1.5 Absolute ADMET prediction is auxiliary, not central

Absolute ADMET prediction should not be the north-star task.

It may be used as:

- pretraining,
- auxiliary loss,
- side predictor,
- teacher signal,
- feature generator,
- ensemble component.

The final objective remains:

```text
rank candidates by rescue likelihood
```

### 1.6 Activity retention should be hierarchical

The system should not rely only on exact potency regression.

Activity-retention evidence should support multiple levels:

1. Continuous potency or potency delta when data quality supports it.
2. Pairwise activity retention relative to parent.
3. Retention buckets when data is noisy or heterogeneous.
4. Mechanism-preservation evidence when target/mechanism information is available.

The final rescue ranker should consume activity-retention evidence rather than rely on one fragile potency number.

### 1.7 Mechanism-aware evidence is required

The system should not only predict:

```text
ADMET improved and activity retained.
```

It should also represent structured reasons such as:

```text
liability-causing feature changed
activity-relevant pharmacophore preserved
expected liability delta
expected activity-retention rationale
new-risk flags
```

These rationales should be structured, not free-form chain-of-thought.

Free-form rationales are allowed only after prediction as user-facing explanations generated from structured evidence.

### 1.8 Structured rationale only

The model/system should not depend on unstructured free-text reasoning as the primary internal representation.

Instead, it should use structured rationale fields such as:

```text
liability_driver_features
modified_features
preserved_activity_features
transformation_class
expected_delta_direction
failure_mode_risks
confidence_by_subclaim
```

### 1.9 Multi-proposer system

The candidate proposer should be hybrid.

It should include multiple proposal sources, potentially including:

- learned inverse proposer,
- forward-model reward optimization proposer,
- matched molecular pair proposer,
- bioisostere proposer,
- prodrug proposer,
- solubilizing-group proposer,
- metabolic-soft-spot proposer,
- purchasable analog retriever,
- scaffold-preserving analog retriever,
- pharmacophore-preserving proposer.

The first job of the proposer is **candidate recall**.

The ranker’s job is **candidate precision**.

The system should log which proposer produced each candidate.

### 1.10 Try both inverse proposer and forward-reward optimization

Two proposal approaches should be explored:

#### Approach A: explicit inverse proposer

```text
parent + desired rescue delta -> candidate proposal
```

#### Approach B: forward model as reward

```text
generate candidate -> score expected delta/activity/rescue -> optimize/filter
```

Neither is locked as the final winner. Both should be tried if practical.

The first implementation can support both at the interface level even if one is initially stubbed.

### 1.11 Pure learned proposer should be tried

A pure learned proposer may produce novel chemically plausible rescues that rule-based transformations miss.

This is strategically important because if the system proposes a truly novel plausible rescue candidate, the demo becomes far more compelling.

However, pure learned proposal should not be the only proposal mechanism in v1, because it may miss boring but correct medicinal chemistry moves.

---

## 2. The ADMET Rescue Task

### 2.1 What the system is trying to do

The system is given a flawed but useful parent molecule.

The parent has:

- known desired activity,
- known ADMET liability,
- known measured facts,
- a rescue objective.

The system must propose and rank candidate molecules that may fix the liability without destroying the useful behavior.

Example:

```text
Parent molecule P is active as an H1 antagonist.
Parent molecule P has hERG/QT liability.
Goal: find a candidate C that preserves H1 antagonism while reducing hERG/QT risk.
```

The system must decide whether candidate C is a good rescue.

It should not merely ask:

```text
Is C safe?
```

It should ask:

```text
Is C a better version of P for this specific rescue objective?
```

### 2.2 Primary scoring concept

The conceptual rescue score is:

```text
rescue_score = liability_improvement
             + activity_retention
             - new_liability_risk
             - implausibility
             - uncertainty_penalty
```

This formula is not a committed implementation. It is conceptual.

The actual scoring model may be learned, heuristic-assisted, ensemble-based, or otherwise designed later.

But every implementation must preserve this conceptual shape.

### 2.3 Why molecule-only ADMET prediction is insufficient

A molecule-only ADMET predictor might correctly say:

```text
Candidate C is more soluble than parent P.
```

But it might miss:

```text
Candidate C lost potency.
Candidate C broke the pharmacophore.
Candidate C is too polar for oral exposure.
Candidate C improved the wrong liability.
Candidate C introduced a new risk.
```

The rescue problem is multi-objective and parent-conditioned.

The final system must learn tradeoffs, not isolated properties.

---

## 3. Allowed Context

Allowed context is information the model/system is allowed to receive at inference time for a true rescue discovery task.

### 3.1 Parent molecule

The system receives the parent structure.

Potential representations:

```text
canonical SMILES
molecular graph
InChI / InChIKey
3D conformer ensemble if generated internally
standardized neutralized/salt-stripped structure
stereochemistry where relevant
```

### 3.2 Candidate molecule

The system receives candidate structures from one or more proposers.

Potential representations are the same as parent molecule representations.

### 3.3 Desired pharmacology / activity context

The system should know what must be preserved.

Examples:

```text
H1 antagonism
ACE inhibition
antiviral active-species delivery
kinase inhibition
GPCR agonism/antagonism
enzyme inhibition
```

This context can be represented as:

```text
target ID
target family
assay type
activity label
pharmacological mechanism
therapeutic class
known parent activity value or bucket
```

### 3.4 Known parent liability

The system should know what must be fixed.

Examples:

```text
hERG/QT risk
poor aqueous solubility
poor microsomal stability
poor oral bioavailability
poor permeability
CYP inhibition
cytotoxicity
reactive/toxic motif
fast clearance
```

### 3.5 Parent measured facts

The system may know measured facts for the parent.

Examples:

```text
parent potency: known or approximate
parent hERG liability: known
parent solubility: known
parent microsomal stability: known
parent permeability: known
parent CYP risk: known
parent oral exposure: known or approximate
```

This is allowed because a real rescue problem begins with knowledge that the parent has a flaw.

### 3.6 Rescue mode

The system should know the rescue mode.

Potential rescue modes:

```text
direct analog rescue
prodrug / exposure rescue
metabolic stability rescue
solubility rescue
safety/off-target rescue
permeability rescue
selectivity rescue
```

The rescue mode matters because the same structural change can be good or bad depending on the mode.

Example:

```text
Adding polarity may help hERG or solubility rescue.
Adding too much polarity may hurt permeability or CNS exposure.
```

### 3.7 Acceptable tradeoff

The system should receive an acceptable tradeoff if available.

Examples:

```text
retain activity within 3x of parent
retain activity within 10x of parent
improve solubility by at least 5x
reduce hERG risk category
improve microsomal half-life by at least 3x
avoid large increase in molecular weight
avoid new reactive/toxic alerts
```

Tradeoffs can be specified as thresholds, preferences, or qualitative constraints.

---

## 4. Computed Evidence

Computed evidence is information derived from parent and candidate structures or from clean internal models. It is allowed if it does not include hidden measured outcomes for the candidate from sealed cases.

### 4.1 Basic molecular descriptors

For parent and candidate:

```text
molecular weight
logP estimate
logD estimate
pKa estimate
topological polar surface area
hydrogen-bond donor count
hydrogen-bond acceptor count
rotatable bond count
formal charge
aromatic ring count
aliphatic ring count
Fsp3
heteroatom counts
ionization state estimates
ring system descriptors
lipophilic efficiency proxies
```

### 4.2 Delta descriptors

For candidate relative to parent:

```text
delta molecular weight
delta logP / logD
delta pKa
delta TPSA
delta HBD/HBA
delta rotatable bonds
delta formal charge
delta aromatic ring count
delta Fsp3
delta substructure alerts
delta pharmacophore features
```

Delta descriptors are critical because rescue is comparative.

### 4.3 Structural similarity and transformation descriptors

Possible evidence:

```text
fingerprint similarity
scaffold similarity
Bemis-Murcko scaffold match
maximum common substructure features
transformation class
number of atoms changed
number of bonds changed
substituent replacement description
bioisostere class
prodrug transformation class
pharmacophore distance preservation
shape similarity
```

### 4.4 Liability-related evidence

Examples:

```text
suspected hERG-risk motifs
lipophilic basic amine flags
high aromatic hydrophobicity flags
metabolic soft-spot predictions
reactive/toxicophore alerts
solubility-risk features
permeability-risk features
CYP-risk features
chemical instability motifs
```

### 4.5 Activity-retention evidence

Examples:

```text
target family
assay type
parent pharmacophore map
candidate pharmacophore map
pharmacophore preservation score
shape/electrostatic similarity
similarity to known actives
nearest known active analogs
protein-ligand interaction fingerprint if target structure exists
docking evidence if trustworthy
activity-cliff risk
predicted activity-retention probability
predicted potency delta
retention bucket
```

### 4.6 Expected property deltas

Internally predicted evidence may include:

```text
predicted delta solubility
predicted delta hERG risk
predicted delta microsomal stability
predicted delta permeability
predicted delta CYP inhibition
predicted delta cytotoxicity risk
predicted delta oral exposure proxy
```

These predictions must come from models trained on decontaminated data.

### 4.7 Mechanism-aware evidence

The system should create structured evidence like:

```text
liability driver likely changed?
activity pharmacophore likely preserved?
new liability likely introduced?
transformation class matches rescue mode?
expected property-delta direction matches objective?
```

Example:

```text
Candidate reduced predicted basicity and lipophilicity.
Candidate preserved parent aromatic spacing and H-bond acceptor arrangement.
Candidate may reduce hERG risk while retaining H1-like pharmacophore.
```

This evidence should be structured as fields, not unstructured reasoning.

---

## 5. Forbidden Evidence

The following must not be used for sealed held-out discovery cases.

### 5.1 Candidate measured hidden outcome

Do not give the system:

```text
candidate measured hERG from sealed case
candidate measured solubility from sealed case
candidate measured potency from sealed case
candidate measured microsomal stability from sealed case
candidate success/failure label from sealed case
```

### 5.2 Historical answer leakage

Do not give the system:

```text
paper text identifying the rescue molecule
patent text identifying the rescue molecule
review text describing the case solution
compound names/synonyms that reveal the answer
assay rows that directly label the hidden candidate as successful
```

### 5.3 Candidate rescue label leakage

Do not give the system:

```text
this molecule is the approved safer analog
this molecule is the active metabolite that solved the liability
this analog was selected by the original researchers
```

### 5.4 Why this matters

If the model receives candidate hidden measured outcomes, the task changes from discovery to aggregation.

Aggregation can be useful in practice, but it does not support the claim:

```text
The model discovered the held-out rescue.
```

For the discovery benchmark, candidate outcomes must be predicted or inferred, not looked up.

---

## 6. Core System Decomposition

The current architecture concept is a proposer-ranker system.

```text
Problem packet
  -> candidate proposers
  -> candidate normalization/deduplication
  -> evidence builder
  -> rescue ranker
  -> structured rationale builder
  -> top-k locked predictions
```

### 6.1 Problem packet

The problem packet defines one ADMET rescue task.

It contains:

```text
parent molecule
activity context
known parent liability
parent measured facts
rescue mode
acceptable tradeoffs
allowed candidate sources
forbidden leakage metadata
```

### 6.2 Candidate proposers

Candidate proposers generate or retrieve candidate molecules.

The system should support multiple proposers, including learned and rule-based proposers.

### 6.3 Candidate normalization/deduplication

All candidates should be standardized.

Tasks:

```text
canonicalize structure
normalize salts/charges when appropriate
handle stereochemistry
remove invalid molecules
deduplicate candidates
record proposer provenance
record parent-candidate transformation
```

### 6.4 Evidence builder

For each parent-candidate pair, build allowed computed evidence.

The evidence builder should produce a structured object.

### 6.5 Rescue ranker

The rescue ranker scores candidate molecules for the specific parent, activity context, liability, rescue mode, and tradeoff.

The ranker’s central output is:

```text
rescue_score
```

### 6.6 Structured rationale builder

The structured rationale builder converts evidence and ranker outputs into auditable structured rationale.

It should not invent unsupported free-text explanations.

### 6.7 Top-k locked predictions

Final predictions should be locked before answer reveal.

Each top-k prediction should include:

```text
candidate structure
rescue score
activity-retention evidence
liability-improvement evidence
new-risk evidence
structured rationale
proposer provenance
confidence fields
```

---

## 7. Candidate Proposer System

### 7.1 Purpose of proposers

The proposer’s job is recall.

It must generate a broad enough candidate set so that true or functionally equivalent rescue candidates are present.

The ranker cannot select a candidate that was never proposed.

### 7.2 Multi-proposer design

The system should support multiple proposal sources.

Potential proposers:

```text
learned inverse proposer
forward-reward optimization proposer
matched molecular pair proposer
bioisostere proposer
prodrug proposer
solubilizing-group proposer
metabolic-soft-spot proposer
hERG-liability-tuning proposer
purchasable analog retriever
scaffold-preserving analog retriever
pharmacophore-preserving proposer
manual/case-specific seed proposer if allowed by protocol
```

Each proposer must tag candidate provenance.

Example:

```json
{
  "candidate_id": "cand_000123",
  "source_proposer": "metabolic_soft_spot_proposer",
  "parent_id": "parent_001",
  "transformation_summary": "benzylic soft spot blocked by fluorination",
  "proposal_confidence": 0.72
}
```

### 7.3 Learned inverse proposer

Input concept:

```text
parent molecule
liability type
desired delta
activity context
rescue mode
```

Output concept:

```text
candidate molecules likely to achieve desired rescue
```

Example:

```text
Parent has poor metabolic stability due to suspected benzylic oxidation.
Desired delta: improve microsomal stability while preserving target binding.
Proposer suggests analogs that block or tune the soft spot.
```

### 7.4 Forward-reward optimization proposer

Input concept:

```text
parent molecule
candidate generation space
forward models / ranker reward
```

Process concept:

```text
generate candidate
score predicted property deltas and activity retention
filter by plausibility and uncertainty
return high-scoring candidates
```

Risk:

```text
The generator may exploit the forward model and propose unrealistic molecules.
```

Therefore this proposer must include constraints:

```text
chemical validity
synthetic plausibility
parent similarity/transformation distance
activity-retention risk
uncertainty penalty
reactive/toxic motif filters
```

### 7.5 Pure learned proposer

The system should support a pure learned proposer because it may produce novel rescue candidates.

This is strategically important.

However, it should not be the only proposer in early versions.

Reason:

```text
A learned proposer may miss standard medicinal chemistry moves that rule-based/MMP proposers capture.
```

### 7.6 Rule/MMP/transformation proposers

These proposers provide medicinal-chemistry plausibility and coverage.

They can propose transformations such as:

```text
add solubilizing group
replace lipophilic aryl with heteroaryl
reduce basicity
block metabolic soft spot
replace labile motif
introduce prodrug handle
reduce planarity
remove reactive alert
bioisosteric replacement
```

These proposers should be used as candidate generators, not as final decision-makers.

### 7.7 Proposer evaluation fields

Every run should record:

```text
number of candidates proposed by each proposer
number of unique valid candidates after normalization
candidate overlap between proposers
whether known held-out answer was present in candidate set if applicable
ranker movement from unordered candidate set to top-k
```

This will help distinguish proposer failure from ranker failure.

---

## 8. Rescue Ranker

### 8.1 Purpose

The rescue ranker is the central decision component.

It receives:

```text
parent molecule
candidate molecule
activity context
liability type
rescue mode
acceptable tradeoff
computed evidence
```

It returns:

```text
rescue score
activity-retention score
liability-improvement score
new-risk score
confidence fields
structured rationale fields
```

### 8.2 What the ranker must learn

The ranker must learn that successful rescue requires multiple conditions:

```text
candidate improves requested liability
candidate preserves useful activity
candidate does not create worse new liability
candidate remains chemically plausible
candidate matches rescue mode
candidate satisfies acceptable tradeoff
```

### 8.3 What the ranker must avoid

The ranker must avoid:

```text
choosing safest molecule regardless of activity
choosing closest analog regardless of liability
choosing most polar molecule regardless of permeability/activity
choosing predicted property winner with impossible chemistry
choosing molecules outside applicability domain with false confidence
```

### 8.4 Conceptual ranker input schema

```json
{
  "case_id": "ADMET_CASE_001",
  "parent": {
    "molecule_id": "parent_001",
    "canonical_smiles": "...",
    "measured_facts": {
      "desired_activity": {
        "target": "H1 receptor",
        "activity_value": "known_or_bucketed",
        "endpoint": "IC50/Ki/etc"
      },
      "liabilities": {
        "hERG_QT_risk": "high",
        "solubility": "optional",
        "microsomal_stability": "optional"
      }
    }
  },
  "candidate": {
    "molecule_id": "cand_001",
    "canonical_smiles": "...",
    "source_proposers": ["learned_inverse", "bioisostere"],
    "transformation_from_parent": {
      "class": "basicity_tuning",
      "atoms_changed": ["..."],
      "summary": "reduced basicity while preserving core scaffold"
    }
  },
  "objective": {
    "activity_context": "H1 antagonism",
    "liability_type": "hERG_QT_risk",
    "rescue_mode": "direct_analog_rescue",
    "acceptable_tradeoff": {
      "activity_retention": "within_10x_parent",
      "liability_improvement": "risk_category_reduction"
    }
  },
  "computed_evidence": {
    "descriptor_deltas": {},
    "activity_retention_evidence": {},
    "liability_evidence": {},
    "new_risk_flags": [],
    "uncertainty_evidence": {}
  }
}
```

### 8.5 Conceptual ranker output schema

```json
{
  "case_id": "ADMET_CASE_001",
  "candidate_id": "cand_001",
  "rescue_score": 0.87,
  "rank": 1,
  "subscores": {
    "liability_improvement_score": 0.91,
    "activity_retention_score": 0.83,
    "new_risk_penalty": 0.12,
    "chemical_plausibility_score": 0.76,
    "tradeoff_fit_score": 0.88
  },
  "confidence": {
    "overall": 0.74,
    "activity_retention": 0.68,
    "liability_improvement": 0.81,
    "new_risk": 0.61
  },
  "structured_rationale": {
    "liability_driver_features": [],
    "modified_features": [],
    "preserved_activity_features": [],
    "transformation_class": "basicity_tuning",
    "expected_delta_direction": {
      "hERG_QT_risk": "decrease",
      "activity": "retain",
      "permeability": "uncertain"
    },
    "failure_mode_risks": []
  }
}
```

---

## 9. Activity-Retention Evidence

### 9.1 Why activity retention matters

A candidate that fixes ADMET but destroys activity is not a rescue.

Example:

```text
Candidate improves solubility 100x but loses target potency 1000x.
This is not a successful rescue.
```

The system must model whether the candidate likely preserves the desired function.

### 9.2 Hierarchical activity-retention modeling

Activity retention should be represented at multiple levels.

#### Level 1: continuous potency when reliable

Use when:

```text
same assay or highly comparable assay
same target
same endpoint type
same or related series
data quality is high
```

Output example:

```text
predicted potency delta: 2.4x worse than parent
```

#### Level 2: pairwise retention

Output example:

```text
candidate likely retains activity within 10x of parent
```

#### Level 3: retention bucket

Buckets may include:

```text
strong_retention
moderate_loss
severe_loss
unknown
```

#### Level 4: mechanism-preservation evidence

Evidence may include:

```text
pharmacophore preservation
shape similarity
binding motif preservation
target-family SAR support
nearest active analog support
activity-cliff risk
```

### 9.3 Why exact potency alone is risky

Exact potency regression can be noisy because public bioactivity data often mixes different assay types, labs, endpoint definitions, and conditions.

The rescue system should not pretend to know false precision when the underlying data is noisy.

The final question is often:

```text
Is this candidate likely active enough to be worth testing?
```

not:

```text
Is the IC50 exactly 37.2 nM?
```

### 9.4 Mechanism-aware activity evidence

Where possible, activity evidence should include target/mechanism information.

Examples:

```text
candidate preserves H-bond donor/acceptor arrangement
candidate preserves key aromatic spacing
candidate preserves ionizable center relevant to target binding
candidate maintains shape/electrostatic similarity to parent
candidate does not disrupt known pharmacophore
candidate is near known actives in clean training space
candidate has high activity-cliff risk due to sensitive modification
```

This evidence should be fed to the rescue ranker.

---

## 10. ADMET and Liability Evidence

### 10.1 Absolute candidate risk

Even if candidate improves relative to parent, it may still be unacceptable.

Example:

```text
Parent hERG risk is extremely high.
Candidate improves hERG risk 10x but remains high-risk.
```

The system should evaluate both:

```text
relative improvement
absolute acceptability
```

### 10.2 Liability-conditioned interpretation

The same structural change may help one liability and hurt another.

Example:

```text
Adding polarity may help solubility.
Adding polarity may hurt permeability.
Adding polarity may reduce hERG risk.
Adding polarity may hurt CNS exposure if CNS activity is desired.
```

Therefore the ranker must be conditioned on liability type and rescue mode.

### 10.3 Examples of liability-specific reasoning

#### hERG / QT risk rescue

Possible evidence:

```text
reduced basicity
reduced lipophilicity
reduced lipophilic cationic character
reduced aromatic hydrophobic surface
preserved target-binding pharmacophore
```

#### Solubility rescue

Possible evidence:

```text
increased aqueous compatibility
reduced crystal-packing risk
added controlled ionizable/polar feature
reduced excessive lipophilicity
activity pharmacophore preserved
```

#### Metabolic stability rescue

Possible evidence:

```text
blocked metabolic soft spot
removed labile motif
reduced lipophilicity
introduced steric/electronic protection
preserved binding core
```

#### Exposure/prodrug rescue

Possible evidence:

```text
prodrug handle likely improves absorption/delivery
candidate can plausibly convert to active species
active species preserved
promoiety does not destroy deliverability
```

---

## 11. Structured Rationale

### 11.1 Purpose

Structured rationale should help the model/system make better predictions and help humans audit the predictions.

It should not be a decorative explanation added after the fact.

### 11.2 Rationale format

Use structured fields.

Example:

```json
{
  "liability_driver_features": [
    {
      "feature_type": "lipophilic_basic_amine",
      "evidence_strength": "medium",
      "parent_atoms": [12, 13, 14]
    }
  ],
  "modified_features": [
    {
      "feature_type": "basicity_reduction",
      "candidate_atoms": [12, 13, 14],
      "expected_effect": "lower_hERG_risk"
    }
  ],
  "preserved_activity_features": [
    {
      "feature_type": "aryl_spacing",
      "parent_atoms": [1, 2, 3, 4, 5, 6],
      "candidate_atoms": [1, 2, 3, 4, 5, 6],
      "evidence_strength": "high"
    },
    {
      "feature_type": "hydrogen_bond_acceptor",
      "evidence_strength": "medium"
    }
  ],
  "transformation_class": "basicity_tuning",
  "expected_delta_direction": {
    "hERG_QT_risk": "decrease",
    "target_activity": "retain",
    "permeability": "uncertain"
  },
  "failure_mode_risks": [
    {
      "risk": "activity_loss_due_to_changed_pKa",
      "severity": "medium"
    }
  ]
}
```

### 11.3 Rationale is not free-form chain-of-thought

Do not require or store long free-form hidden reasoning traces.

Do not rely on free-form rationales as the scoring basis.

Use structured fields that can be checked, scored, and audited.

### 11.4 Rationale should inform prediction

The structured rationale should not be merely post-hoc.

Potential uses:

```text
as input features to ranker
as auxiliary prediction target
as consistency check
as human audit artifact
as failure-mode diagnosis
```

### 11.5 Examples of rationale-target fields

Possible labels/fields:

```text
liability_driver_identified: yes/no/uncertain
liability_driver_removed_or_tuned: yes/no/uncertain
activity_pharmacophore_preserved: yes/no/uncertain
requested_liability_improved: predicted yes/no/uncertain
wrong_liability_improved: predicted yes/no/uncertain
new_liability_created: predicted yes/no/uncertain
transformation_matches_rescue_mode: yes/no/uncertain
```

---

## 12. Training Signal Stack

The ADMET-rescue system should be trained with a stack of signals.

### 12.1 Main signal: rescue comparison

Input:

```text
parent P
candidate C
activity context A
liability L
rescue mode R
acceptable tradeoff T
```

Output:

```text
successful rescue?
rescue score
failure mode
```

This is the primary training signal.

### 12.2 Supporting signal: pairwise property delta

Input:

```text
parent P
candidate C
endpoint/liability L
```

Output:

```text
predicted direction and magnitude of property change
```

Examples:

```text
candidate improves solubility relative to parent
candidate reduces hERG risk relative to parent
candidate worsens permeability relative to parent
candidate improves microsomal stability relative to parent
```

### 12.3 Supporting signal: activity retention

Input:

```text
parent P
candidate C
activity context A
```

Output:

```text
activity retained?
predicted potency delta?
retention bucket?
mechanism-preservation evidence?
```

### 12.4 Supporting signal: absolute candidate risk

Input:

```text
candidate C
endpoint L
```

Output:

```text
absolute risk or acceptability
```

This prevents relative-improvement-only errors.

### 12.5 Supporting signal: mechanism/rationale annotation

Input:

```text
parent P
candidate C
activity context A
liability L
```

Output:

```text
liability driver
modified feature
preserved activity feature
transformation class
failure mode risks
```

### 12.6 Supporting signal: candidate proposal

Input:

```text
parent P
desired rescue delta
activity context A
liability L
rescue mode R
```

Output:

```text
candidate molecules
```

---

## 13. Rescue Pair Dataset Concept

The core dataset object should be an ADMET rescue pair table.

Each row represents a parent-candidate-objective relationship.

### 13.1 Conceptual table fields

```text
parent_id
candidate_id
activity_context
liability_type
rescue_mode
acceptable_tradeoff
parent_activity_value
candidate_activity_value_or_label
parent_liability_value
candidate_liability_value_or_label
delta_activity
delta_liability
new_risk_flags
transformation_class
liability_driver_features
preserved_activity_features
rescue_label
failure_mode_label
source_assay_ids
quality_score
contamination_status
```

### 13.2 Example positive row

```json
{
  "parent_id": "P001",
  "candidate_id": "C017",
  "activity_context": "target_T_inhibition",
  "liability_type": "poor_solubility",
  "rescue_mode": "direct_analog_rescue",
  "parent_activity": "20 nM",
  "candidate_activity": "35 nM",
  "parent_liability": "solubility = 1 uM",
  "candidate_liability": "solubility = 25 uM",
  "delta_activity": "1.75x worse",
  "delta_liability": "25x better",
  "rescue_label": "strong_success",
  "failure_mode_label": null,
  "transformation_class": "solubilizing_group_addition"
}
```

### 13.3 Example hard negative row: ADMET improved, activity lost

```json
{
  "parent_id": "P001",
  "candidate_id": "C021",
  "activity_context": "target_T_inhibition",
  "liability_type": "poor_solubility",
  "rescue_mode": "direct_analog_rescue",
  "parent_activity": "20 nM",
  "candidate_activity": "8000 nM",
  "parent_liability": "solubility = 1 uM",
  "candidate_liability": "solubility = 50 uM",
  "delta_activity": "400x worse",
  "delta_liability": "50x better",
  "rescue_label": "failed_rescue",
  "failure_mode_label": "activity_lost",
  "transformation_class": "excessive_polarity_increase"
}
```

### 13.4 Example hard negative row: activity retained, liability not fixed

```json
{
  "parent_id": "P001",
  "candidate_id": "C034",
  "activity_context": "target_T_inhibition",
  "liability_type": "poor_solubility",
  "rescue_mode": "direct_analog_rescue",
  "parent_activity": "20 nM",
  "candidate_activity": "30 nM",
  "parent_liability": "solubility = 1 uM",
  "candidate_liability": "solubility = 1.3 uM",
  "delta_activity": "1.5x worse",
  "delta_liability": "1.3x better",
  "rescue_label": "failed_rescue",
  "failure_mode_label": "liability_not_fixed",
  "transformation_class": "minor_substituent_change"
}
```

### 13.5 Example hard negative row: new liability introduced

```json
{
  "parent_id": "P001",
  "candidate_id": "C055",
  "activity_context": "target_T_inhibition",
  "liability_type": "poor_solubility",
  "rescue_mode": "direct_analog_rescue",
  "parent_activity": "20 nM",
  "candidate_activity": "45 nM",
  "parent_liability": "solubility = 1 uM",
  "candidate_liability": "solubility = 20 uM",
  "delta_activity": "2.25x worse",
  "delta_liability": "20x better",
  "new_risk_flags": ["reactive_alert", "CYP_inhibition_risk"],
  "rescue_label": "partial_or_failed_rescue",
  "failure_mode_label": "new_liability_introduced",
  "transformation_class": "solubilizing_group_addition"
}
```

---

## 14. Label Types

The rescue system should not be limited to binary labels.

### 14.1 Rescue labels

Possible labels:

```text
strong_success
weak_success
partial_success
failed_rescue
uncertain
```

### 14.2 Failure-mode labels

Possible labels:

```text
activity_lost
liability_not_fixed
wrong_liability_improved
new_liability_introduced
chemically_implausible
outside_applicability_domain
insufficient_data
```

### 14.3 Activity-retention labels

Possible labels:

```text
strong_retention
acceptable_retention
moderate_loss
severe_loss
unknown
```

### 14.4 Liability-improvement labels

Possible labels:

```text
large_improvement
moderate_improvement
small_improvement
no_improvement
worse
unknown
```

### 14.5 Mechanistic labels

Possible labels:

```text
liability_driver_removed
liability_driver_tuned
liability_driver_unchanged
pharmacophore_preserved
pharmacophore_disrupted
soft_spot_blocked
soft_spot_unchanged
basicity_reduced
polarity_increased
prodrug_handle_added
bioisostere_replacement
```

---

## 15. Hard Negatives

Hard negatives are essential.

They prevent the system from learning shortcuts.

### 15.1 ADMET improved, activity lost

This teaches:

```text
Do not optimize ADMET alone.
```

Example:

```text
Candidate improves solubility but loses potency.
```

### 15.2 Activity retained, liability not fixed

This teaches:

```text
Do not choose close analogs that fail to solve the problem.
```

Example:

```text
Candidate remains potent but still has hERG risk.
```

### 15.3 Wrong liability improved

This teaches:

```text
Fix the requested liability, not a random property.
```

Example:

```text
Candidate improves solubility, but the task was metabolic stability.
```

### 15.4 New liability introduced

This teaches:

```text
Do not trade one fatal flaw for another.
```

Example:

```text
Candidate reduces hERG risk but introduces reactive/toxic motif.
```

### 15.5 Heuristic trap negatives

These punish simplistic strategies.

Examples:

```text
most polar analog
closest analog
lowest predicted toxicity analog
most drug-like analog
highest predicted ADMET score but inactive
```

### 15.6 Over-transformation negatives

These punish modifications that are too disruptive.

Example:

```text
Candidate changes too much of the parent scaffold and likely loses target-specific activity.
```

### 15.7 Under-transformation negatives

These punish trivial analogs that preserve activity but do not fix the liability.

Example:

```text
Candidate differs by a methyl shift but parent liability remains.
```

---

## 16. Rescue Modes

The system should explicitly know the rescue mode.

### 16.1 Direct analog rescue

Candidate itself is expected to be active.

Goal:

```text
candidate directly preserves desired pharmacology and improves liability
```

### 16.2 Prodrug / exposure rescue

Candidate may convert into active species.

Goal:

```text
candidate improves delivery/exposure while preserving active species after conversion
```

Important:

```text
The prodrug may not have the same in vitro potency before conversion.
```

So activity retention must be interpreted differently.

### 16.3 Solubility rescue

Goal:

```text
increase usable aqueous solubility while preserving activity and exposure potential
```

### 16.4 hERG / safety rescue

Goal:

```text
reduce off-target cardiac-channel risk while preserving target activity
```

### 16.5 Metabolic stability rescue

Goal:

```text
reduce clearance / improve stability while preserving activity
```

### 16.6 Permeability/exposure rescue

Goal:

```text
improve exposure/permeability without destroying activity or introducing new liabilities
```

---

## 17. Example Problem Packets

### 17.1 hERG / cardiac safety rescue

```json
{
  "case_id": "ADMET_HERG_001",
  "parent": {
    "canonical_smiles": "...",
    "known_activity": {
      "activity_context": "H1 antagonism",
      "activity_strength": "active",
      "approx_potency": "known_or_bucketed"
    },
    "known_liabilities": {
      "hERG_QT_risk": "high"
    }
  },
  "objective": {
    "rescue_mode": "direct_analog_rescue",
    "liability_to_fix": "hERG_QT_risk",
    "activity_to_preserve": "H1 antagonism",
    "acceptable_tradeoff": {
      "activity_retention": "within_10x_parent_or_better",
      "liability_improvement": "risk_category_reduction"
    }
  },
  "allowed_evidence": [
    "computed_descriptors",
    "predicted_property_deltas",
    "activity_retention_evidence",
    "mechanistic_structured_rationale"
  ],
  "forbidden_evidence": [
    "candidate_measured_hidden_hERG",
    "candidate_measured_hidden_potency",
    "historical_success_label",
    "case_paper_text"
  ]
}
```

### 17.2 Solubility rescue

```json
{
  "case_id": "ADMET_SOL_001",
  "parent": {
    "canonical_smiles": "...",
    "known_activity": {
      "activity_context": "target_T_inhibition",
      "activity_strength": "active"
    },
    "known_liabilities": {
      "aqueous_solubility": "poor"
    }
  },
  "objective": {
    "rescue_mode": "direct_analog_rescue",
    "liability_to_fix": "poor_aqueous_solubility",
    "activity_to_preserve": "target_T_inhibition",
    "acceptable_tradeoff": {
      "activity_retention": "within_10x_parent_or_better",
      "solubility_improvement": "at_least_5x_or_category_improvement",
      "avoid": ["severe_permeability_loss", "reactive_alerts"]
    }
  }
}
```

### 17.3 Metabolic stability rescue

```json
{
  "case_id": "ADMET_METSTAB_001",
  "parent": {
    "canonical_smiles": "...",
    "known_activity": {
      "activity_context": "target_T_inhibition",
      "activity_strength": "active"
    },
    "known_liabilities": {
      "microsomal_stability": "poor",
      "suspected_soft_spots": ["benzylic_position", "labile_motif"]
    }
  },
  "objective": {
    "rescue_mode": "metabolic_stability_rescue",
    "liability_to_fix": "poor_microsomal_stability",
    "activity_to_preserve": "target_T_inhibition",
    "acceptable_tradeoff": {
      "activity_retention": "within_10x_parent_or_better",
      "stability_improvement": "at_least_3x_or_category_improvement"
    }
  }
}
```

### 17.4 Prodrug / exposure rescue

```json
{
  "case_id": "ADMET_PRODRUG_001",
  "parent": {
    "canonical_smiles": "...",
    "known_activity": {
      "activity_context": "active_species_has_target_activity",
      "activity_strength": "active"
    },
    "known_liabilities": {
      "oral_exposure": "poor",
      "permeability_or_absorption": "poor"
    }
  },
  "objective": {
    "rescue_mode": "prodrug_exposure_rescue",
    "liability_to_fix": "poor_oral_exposure",
    "activity_to_preserve": "delivery_of_active_species",
    "acceptable_tradeoff": {
      "active_species_preserved": true,
      "conversion_plausible": true,
      "exposure_improvement": "category_improvement"
    }
  }
}
```

---

## 18. Example Candidate Output

A candidate output should include both score and structured rationale.

```json
{
  "case_id": "ADMET_HERG_001",
  "parent_id": "P001",
  "candidate_id": "C042",
  "rank": 1,
  "canonical_smiles": "...",
  "source_proposers": ["learned_inverse", "basicity_tuning_proposer"],
  "scores": {
    "rescue_score": 0.91,
    "liability_improvement_score": 0.88,
    "activity_retention_score": 0.82,
    "absolute_candidate_risk_score": 0.76,
    "new_liability_penalty": 0.08,
    "chemical_plausibility_score": 0.84
  },
  "predicted_deltas": {
    "hERG_QT_risk": "decrease",
    "activity": "retain_or_mild_loss",
    "logD": "decrease",
    "basicity": "decrease",
    "permeability": "uncertain"
  },
  "activity_retention": {
    "retention_bucket": "acceptable_retention",
    "expected_potency_delta_bucket": "within_10x_parent",
    "pharmacophore_preservation_score": 0.79,
    "activity_cliff_risk": "medium"
  },
  "structured_rationale": {
    "liability_driver_features": [
      "lipophilic_basic_amine",
      "high_aromatic_hydrophobicity"
    ],
    "modified_features": [
      "basicity_reduced",
      "polarity_tuned"
    ],
    "preserved_activity_features": [
      "core_scaffold",
      "key_hydrogen_bond_acceptor",
      "aryl_spacing"
    ],
    "transformation_class": "basicity_and_lipophilicity_tuning",
    "failure_mode_risks": [
      "possible_activity_loss_due_to_pKa_shift"
    ]
  },
  "confidence": {
    "overall": 0.74,
    "liability_improvement": 0.82,
    "activity_retention": 0.67,
    "new_risk_assessment": 0.58
  }
}
```

---

## 19. Internal Consistency Checks

The system should include checks that prevent contradictory outputs.

Examples:

### 19.1 High rescue score but low activity retention

If:

```text
rescue_score is high
activity_retention_score is low
```

Then flag:

```text
inconsistent_rescue_score
```

Because a rescue cannot be strong if activity is likely lost.

### 19.2 High liability improvement but high new liability

If:

```text
requested liability improves
new severe liability appears
```

Then rescue score should be penalized.

### 19.3 Prodrug mode mismatch

If:

```text
rescue_mode = prodrug_exposure_rescue
candidate does not plausibly convert to active species
```

Then penalize.

### 19.4 Direct analog mode mismatch

If:

```text
rescue_mode = direct_analog_rescue
candidate likely requires conversion to active species
```

Then penalize unless conversion is explicitly allowed.

### 19.5 Wrong-liability improvement

If:

```text
requested liability = hERG
candidate only improves solubility
```

Then penalize.

---

## 20. Implementation Expectations for Coding Agents

This section tells coding agents how to treat this architecture context.

### 20.1 Do not prematurely choose model architecture

Do not assume:

```text
transformer only
GNN only
fingerprint model only
LLM only
```

The model architecture will be decided later.

This document defines the task and data interfaces.

### 20.2 Build interfaces that support multiple model types

Design schemas so that different models can plug into:

```text
proposer
feature/evidence builder
activity-retention estimator
property-delta estimator
rescue ranker
structured rationale generator
```

### 20.3 Keep evidence and labels separate

Do not mix:

```text
allowed computed evidence
hidden measured outcomes
training labels
inference-time features
```

This is critical for decontamination.

### 20.4 Always preserve provenance

Every candidate should track:

```text
which proposer generated it
what parent it came from
what transformation was applied
what evidence was computed
what model scored it
what data sources informed the score
```

### 20.5 Avoid free-text dependence

Do not make the main prediction pipeline depend on free-text rationales.

Use structured rationale fields.

Free text can be generated later from structured fields for UI/investor presentation.

### 20.6 Design for ablation

Even though the first build should be hybrid, the code should allow ablations later:

```text
pairwise-only rescue
property-only rescue
hybrid rescue
with/without activity evidence
with/without mechanistic evidence
with/without hard negatives
with/without learned proposer
with/without rule/MMP proposers
```

### 20.7 Separate proposer recall from ranker precision

Log candidate set before ranking.

This allows diagnosis:

```text
Did the proposer fail to include the right answer?
Did the ranker fail to rank it highly?
```

---

## 21. Current Open Questions

These are not locked yet.

### 21.1 Exact training objective

Possible components:

```text
binary rescue classification
multiclass failure-mode classification
pairwise/listwise candidate ranking
continuous rescue score regression
property-delta prediction
activity-retention prediction
structured-rationale prediction
```

Current preference:

```text
Use rescue ranking as final objective, with auxiliary outputs.
```

### 21.2 Exact activity-retention thresholds

Possible thresholds:

```text
within 3x parent potency
within 10x parent potency
within 100x parent potency
endpoint-specific threshold
```

Current preference:

```text
Use endpoint/context-specific thresholds where possible.
Use within-10x as a general early default for acceptable retention.
```

### 21.3 Exact liability-improvement thresholds

Need endpoint-specific thresholds.

Examples:

```text
solubility may require 5x or 10x improvement
microsomal stability may use half-life/category improvement
hERG may use risk-category reduction
CYP may use inhibition-category reduction
```

### 21.4 How to build structured rationale labels

Need to decide which rationale labels can be:

```text
manually curated
rule-derived
model-predicted
weakly supervised
```

### 21.5 How much mechanistic evidence to include

Need to decide when to include:

```text
docking
pharmacophore maps
protein-ligand interaction fingerprints
shape/electrostatics
metabolic soft-spot predictions
```

These should be included only when trustworthy and non-leaky.

### 21.6 How to balance learned proposer and rule-based proposers

Need to compare:

```text
pure learned proposer
hybrid proposer
rule/MMP-only proposer
forward-reward proposer
explicit inverse proposer
```

### 21.7 How to represent prodrug rescues

Need special handling because prodrug candidates may not directly preserve in vitro potency before conversion.

---

## 22. Non-Goals for This Document

This document does not specify:

```text
specific neural architecture
specific model sizes
specific loss functions
specific datasets
specific decontamination thresholds
specific chemistry software
specific training pipeline
specific evaluation metrics
specific benchmark cases
```

Those will be handled later.

This document specifies:

```text
the ADMET-rescue task shape
allowed and forbidden evidence
proposer-ranker system decomposition
structured rationale requirements
input/output schemas
locked conceptual decisions
implementation constraints for coding agents
```

---

## 23. One-Sentence Definition

The ADMET-rescue system is:

```text
A parent-conditioned, liability-conditioned, activity-aware proposer-ranker system that evaluates whether candidate molecules rescue a flawed but useful parent by improving a specified ADMET liability while preserving the parent’s desired pharmacological function, using structured mechanistic evidence and strict prevention of held-out outcome leakage.
```

---

## 24. Short Checklist for Future Agents

Before building anything, verify:

```text
[ ] Are we treating ADMET rescue as parent-candidate comparison?
[ ] Are we avoiding molecule-only ADMET prediction as the final task?
[ ] Are parent measured facts allowed but candidate hidden outcomes forbidden?
[ ] Are computed features allowed only if non-leaky?
[ ] Does the system know desired activity, liability type, rescue mode, and tradeoff?
[ ] Does the proposer generate candidates before the ranker scores them?
[ ] Do candidates track proposer provenance?
[ ] Does the ranker score activity retention, liability improvement, new risk, and plausibility?
[ ] Are rationales structured, not free-form?
[ ] Can the system handle direct analog and prodrug/exposure rescue differently?
[ ] Can we later ablate pairwise-only, property-only, and hybrid versions?
[ ] Are hard negatives planned to punish shortcut strategies?
[ ] Are hidden held-out candidate outcomes excluded from inference?
```

---

## 25. Final Practical Interpretation

When a coding agent receives this file, the agent should understand that the task is not:

```text
Build an ADMET predictor.
```

The task is:

```text
Build the scaffolding for an ADMET-rescue discovery system.
```

The system should ask:

```text
Given this useful but flawed parent molecule, and this candidate modification, does the candidate actually rescue the parent for the specified liability without breaking the desired activity?
```

That is the core of the architecture.

Everything else is supporting evidence.
