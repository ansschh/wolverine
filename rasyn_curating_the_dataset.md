I would create the ADMET rescue-pair dataset in **three layers**, not one.

The mistake would be trying to mine “successful ADMET rescues” directly from raw databases as if they are already labeled. They are not. ChEMBL, PubChem, TDC, MoleculeNet, papers, and internal data each contain fragments of the answer. Our job is to convert those fragments into a derived table:

> **parent P + candidate C + activity context A + liability L → rescue outcome**

That table is the real asset.

# The recommendation

Build the ADMET rescue-pair dataset as:

1. **Gold rescue pairs**
   Small, high-confidence, measured, curated examples. These are the highest-value rows.

2. **Silver rescue pairs**
   Automatically mined parent-candidate pairs with measured activity and measured ADMET/property endpoints, but some assay heterogeneity.

3. **Auxiliary weak data**
   Molecule-only ADMET, activity, toxicity, and property data used to train supporting predictors/evidence modules, not directly used as trusted rescue labels.

In other words:

> **Do not treat all public data equally. Build a hierarchy of trust.**

---

# 1. What one row should mean

Every rescue-pair row should represent a specific medicinal chemistry question:

```text
Given parent P, candidate C, desired activity A, and liability L:
does C rescue P by improving L while preserving A?
```

A row should look like this:

```yaml
row_id: ADMET_PAIR_000001

parent:
  molecule_id: MOL_P
  canonical_smiles: "..."
  parent_measured_activity:
    target: "H1 receptor"
    endpoint: "IC50"
    value: 35
    unit: "nM"
    source: "ChEMBL"
  parent_measured_liability:
    liability_type: "hERG"
    endpoint: "hERG IC50"
    value: 0.08
    unit: "uM"
    source: "curated_literature"

candidate:
  molecule_id: MOL_C
  canonical_smiles: "..."
  candidate_measured_activity:
    target: "H1 receptor"
    endpoint: "IC50"
    value: 60
    unit: "nM"
    source: "ChEMBL"
  candidate_measured_liability:
    liability_type: "hERG"
    endpoint: "hERG IC50"
    value: 25
    unit: "uM"
    source: "curated_literature"

context:
  desired_activity: "H1 antagonism"
  liability_type: "hERG / QT risk"
  rescue_mode: "direct analog safety rescue"
  acceptable_tradeoff: "retain activity within 10x parent potency"

computed_evidence:
  tanimoto_parent_candidate: 0.72
  same_murcko_scaffold: true
  delta_logP: -1.4
  delta_TPSA: +45.2
  parent_basic_center_count: 1
  candidate_basic_center_count: 0
  transformation_class: "polarity/basicity tuning"

label:
  activity_retention: "strong_retention"
  liability_improvement: "strong_improvement"
  rescue_label: "strong_success"
  failure_mode: null

quality:
  tier: "gold"
  measured_activity: true
  measured_liability: true
  same_assay_activity: true
  same_assay_liability: true
  curator_reviewed: true
```

The important point is that the row is not just “candidate is good.” It says:

> **Candidate C is a good rescue of parent P for this specific liability and activity context.**

---

# 2. What each data source is for

## ChEMBL

ChEMBL should be the main source for **target activity context** and a significant source of ADMET/physicochemical/toxicity measurements. ChEMBL describes itself as a manually curated database of bioactive drug-like molecules that combines chemical, bioactivity, and genomic data; its activity fields include standardized activity values and pChEMBL, which is defined as the negative log of molar IC50, EC50, Ki, Kd, potency, and related half-maximal endpoints under specific validity conditions. ([EMBL-EBI][1])

Use ChEMBL for:

```text
molecule structures
target IDs
assay IDs
document IDs
activity values
pChEMBL values
target activity retention
some physicochemical/ADMET/toxicity assays
document-level grouping
case-family quarantine
```

ChEMBL also classifies assay types, including physicochemical assays measuring properties such as solubility or chemical stability, which is useful for finding liability measurements. ([ChEMBL][2])

ChEMBL is especially valuable because it gives us:

```text
molecule ↔ target ↔ assay ↔ activity ↔ document
```

That lets us build analog series and activity-retention labels.

## PubChem BioAssay

PubChem BioAssay should be used mostly for **large-scale screening labels, toxicity/counter-screen data, and weak or silver activity/liability evidence**. PubChem BioAssay contains small-molecule and RNAi screening data with assay annotations from contributing organizations, and BioAssay records include assay descriptions and detailed test results; PubChem also records activity outcomes such as active/inactive. ([PubChem][3])

Use PubChem for:

```text
high-throughput active/inactive outcomes
toxicity/cytotoxicity screens
hERG or safety-related assay records if available
broad phenotype screens
large negative sets
counter-screen labels
weak auxiliary labels
```

But do not blindly treat PubChem as gold. It is heterogeneous. Many assays are noisy, context-specific, or screening-only. It is excellent for scale, less ideal for clean rescue labels unless the assay metadata is strong.

## TDC

TDC should be used for **auxiliary ADMET property modeling**, not as the primary source of rescue-pair labels. TDC’s ADMET benchmark group is explicitly built from 22 ADMET datasets, making it useful for endpoint predictors such as solubility, permeability, bioavailability, toxicity, CYP, hERG-like, and related property heads depending on task availability. ([TDC][4])

Use TDC for:

```text
auxiliary ADMET property heads
endpoint-specific pretraining
weak candidate evidence
calibration of property predictors
comparison against standard ADMET tasks
```

Do not expect TDC to directly tell us:

```text
parent P was rescued by candidate C
```

It generally gives molecule-level labels, not parent-candidate rescue relationships.

## MoleculeNet

MoleculeNet should also be used mainly for **auxiliary property learning and sanity-check benchmarking**. MoleculeNet curates multiple public molecular datasets, evaluation metrics, featurizations, and benchmark implementations for molecular machine learning. ([RSC Publishing][5])

Use MoleculeNet for:

```text
property pretraining
toxicity/property auxiliary heads
model sanity checks
baseline comparisons
low-level molecular representation learning
```

Again, MoleculeNet is not a rescue-pair dataset by itself.

## Papers

Papers are for **gold rescue labels and mechanistic rationale**, but only outside the sealed holdout cases.

Use papers for:

```text
curated parent-candidate rescue pairs
SAR tables
measured activity + measured ADMET in same study
mechanistic explanations
liability-driving feature annotations
pharmacophore-preservation annotations
hard negative examples
```

But for v1, I would **not train the discovery model on free-form paper text**. Instead, curators should extract structured rows:

```text
parent
candidate
activity value
liability value
transformation
rationale
source DOI/document
```

That prevents text leakage and makes the data auditable.

## Internal data

Internal data is likely your highest-quality source if you have it.

Use internal data for:

```text
true gold rescue pairs
private analog series
consistent assay protocols
measured negatives
failed analogs
mechanistic annotations
prospective validation
```

Internal data is especially useful because it can include failures. Public data often over-represents successful molecules. A rescue model needs failed attempts badly.

---

# 3. The actual dataset should have four derived tables

Do not start with one giant flat table. Build four derived assets.

## Table 1: Molecule table

One canonical row per molecule.

```yaml
molecule_id: MOL_123
canonical_smiles: "..."
inchi_key: "..."
parent_inchi_key: "..."
neutralized_smiles: "..."
tautomer_hash: "..."
murcko_scaffold: "..."
source_ids:
  chembl_id: "CHEMBL..."
  pubchem_cid: "..."
computed_descriptors:
  molecular_weight: 412.5
  logP: 3.7
  TPSA: 67.2
  HBD: 1
  HBA: 6
  rotatable_bonds: 8
  formal_charge: 0
  aromatic_ring_count: 3
  fsp3: 0.28
  substructure_alerts: [...]
forbidden_status:
  is_forbidden: false
  nearest_sealed_case_similarity: 0.42
```

This table handles canonicalization, de-duplication, decontamination, and descriptor computation.

## Table 2: Assay fact table

One row per measured fact.

```yaml
fact_id: FACT_123
molecule_id: MOL_123
source: "ChEMBL"
source_record_id: "CHEMBL_ACTIVITY_..."
assay_id: "CHEMBL..."
document_id: "CHEMBL_DOC..."
endpoint_family: "target_activity"
target: "EGFR"
organism: "human"
assay_type: "binding"
standard_type: "IC50"
standard_relation: "="
standard_value: 25
standard_units: "nM"
pchembl_value: 7.6
quality_flags:
  valid_units: true
  exact_relation: true
  data_validity_comment: null
```

This table stores raw-but-standardized facts. It is not yet rescue data.

## Table 3: ADMET rescue-pair table

This is the main training table.

```yaml
pair_id: PAIR_123
parent_id: MOL_P
candidate_id: MOL_C
activity_context_id: ACTCTX_001
liability_type: "solubility"
rescue_mode: "direct analog polarity rescue"

activity_evidence:
  parent_activity_fact_ids: [...]
  candidate_activity_fact_ids: [...]
  activity_delta_log10: -0.2
  activity_retention_label: "strong_retention"

liability_evidence:
  parent_liability_fact_ids: [...]
  candidate_liability_fact_ids: [...]
  liability_delta_log10: +1.1
  liability_improvement_label: "strong_improvement"

computed_delta_evidence:
  delta_logP: -1.2
  delta_TPSA: +31.4
  same_scaffold: true
  mmp_transform: "aryl-H -> aryl-piperazine"
  pharmacophore_similarity: 0.84

label:
  rescue_label: "strong_success"
  failure_mode: null
  label_confidence: 0.94

quality:
  tier: "gold"
  source_type: "curated_paper"
  curator_reviewed: true
```

This is what the rescue ranker learns from.

## Table 4: Candidate-set table

This table defines local ranking tasks.

```yaml
ranking_task_id: RANK_001
parent_id: MOL_P
activity_context_id: ACTCTX_001
liability_type: "hERG"
rescue_mode: "direct analog safety rescue"

candidate_ids:
  - MOL_C1
  - MOL_C2
  - MOL_C3
  - ...

labels:
  MOL_C1: "strong_success"
  MOL_C2: "failed_activity_loss"
  MOL_C3: "failed_no_liability_improvement"

hard_negative_types:
  MOL_C2: "ADMET_improved_but_activity_lost"
  MOL_C3: "activity_retained_but_liability_not_fixed"

source_pair_ids:
  - PAIR_...
```

This is what teaches the model to rank candidates, not merely classify one pair at a time.

---

# 4. The mining strategy

The dataset should be built in passes.

## Pass 0: seal and decontaminate before mining

Before pair generation, remove sealed cases from all raw sources.

Do not generate pairs first and decontaminate later. That risks derived leakage.

The order should be:

```text
raw data
→ sealed-case quarantine
→ molecule/document/assay decontamination
→ canonicalized clean corpus
→ pair generation
```

This matters because a pair row can leak information even if the final molecule row looks clean.

---

# 5. Pass 1: build activity contexts from ChEMBL

The first major object is an `activity_context`.

An activity context defines what we are trying to preserve.

Example:

```yaml
activity_context_id: ACTCTX_H1_001
target: "Histamine H1 receptor"
target_chembl_id: "..."
organism: "human"
activity_type: "antagonism / binding / functional activity"
accepted_standard_types:
  - "IC50"
  - "Ki"
  - "EC50"
```

For ChEMBL mining, group activity data by increasing strictness:

## Strict grouping

```text
same assay_chembl_id
same target_chembl_id
same standard_type
same document_chembl_id
```

This is the cleanest but sparse.

## Medium grouping

```text
same target_chembl_id
same standard_type
same document_chembl_id
compatible assay_type
```

This is useful for analog series.

## Loose grouping

```text
same target_chembl_id
compatible standard_type
different document
```

This gives more data but lower confidence.

Use ChEMBL’s pChEMBL where valid because it standardizes several potency/affinity endpoints onto a negative-log molar scale under specific conditions. ([ChEMBL][2])

For each molecule in an activity context, aggregate potency carefully:

```text
if same assay: median pChEMBL
if same document: median pChEMBL with lower confidence
if multiple conflicting values: flag conflict
if censored values like >10 uM: use bucket/interval, not exact regression
```

---

# 6. Pass 2: build liability fact tables

For v1, I recommend only these ADMET liability families:

```text
1. hERG / cardiac off-target risk
2. solubility
3. metabolic stability / clearance
4. oral exposure / prodrug rescue, but curated only
```

Do not include every ADMET endpoint at first.

## Liability family 1: hERG / cardiac risk

Data types:

```text
hERG IC50
hERG inhibition %
QT/cardiac safety proxy assays
ion-channel counterscreens
```

Useful label direction:

```text
higher hERG IC50 = lower hERG inhibition risk
lower hERG pIC50 = lower hERG inhibition risk
```

For rescue:

```text
candidate should reduce hERG risk while retaining desired target activity
```

## Liability family 2: solubility

Data types:

```text
aqueous solubility
kinetic solubility
thermodynamic solubility
logS
```

Useful label direction:

```text
higher solubility = better
```

For rescue:

```text
candidate should increase solubility while retaining activity
```

## Liability family 3: metabolic stability / clearance

Data types:

```text
microsomal half-life
intrinsic clearance
hepatocyte clearance
parent remaining after incubation
```

Useful label direction depends on endpoint:

```text
higher half-life = better
lower intrinsic clearance = better
higher percent remaining = better
```

For rescue:

```text
candidate should survive metabolism longer while retaining activity
```

## Liability family 4: oral exposure / prodrug rescue

This one is harder to mine automatically.

Use curated rows only for v1.

Why? Because prodrug rescue is not the same as direct analog rescue. The candidate may not directly retain target potency; instead, it converts into the active species.

For prodrug rows, label:

```text
rescue_mode: prodrug_exposure_rescue
activity_retention_type: active_species_delivery
```

not:

```text
candidate_direct_potency_retained
```

---

# 7. Pass 3: create analog graphs

Once you have molecules with activity and liability facts, build analog relationships.

Create edges between molecules if they are plausible parent-candidate pairs.

Use multiple signals:

```text
same or compatible activity context
same target or same assay family
same document or same project if available
same/similar scaffold
ECFP/Morgan similarity above threshold
matched molecular pair transformation
maximum common substructure overlap
reasonable heavy-atom difference
```

A good edge is not just “similar molecules.” It should be a plausible medicinal chemistry transformation.

Matched molecular pair analysis is useful here because it extracts medicinal chemistry design rules from structural changes and property changes; reviews describe MMPA as a tool for multiple-parameter optimization and design-rule generation, while also noting the importance of context-specific rules. ([ScienceDirect][6])

For each edge:

```text
P ↔ C
```

compute:

```text
activity delta
liability delta
descriptor deltas
pharmacophore similarity
scaffold relation
transformation class
```

Then orient the edge into rescue pairs.

Example:

```text
If P has bad solubility and C has better solubility,
then P → C is a possible solubility rescue pair.
```

---

# 8. Pass 4: label rescue outcomes

For each parent-candidate pair, label the outcome.

## Activity retention labels

Use hierarchical labeling.

```text
strong_retention:
  candidate potency within 3x parent potency

acceptable_retention:
  candidate potency within 10x parent potency

weak_retention:
  candidate potency 10x-100x worse

failed_retention:
  candidate potency >100x worse

unknown_retention:
  insufficient or incomparable data
```

For noisy cross-assay data, do not pretend exact potency is reliable. Use retention buckets.

## Liability improvement labels

Make endpoint-specific thresholds.

### Solubility

```text
strong_improvement:
  ≥10x improvement

moderate_improvement:
  3x-10x improvement

no_improvement:
  <3x improvement

worse:
  candidate worse than parent
```

### hERG

Use risk categories, not only fold-change.

Example:

```text
strong_improvement:
  candidate moves from high-risk hERG inhibition to low/moderate risk

moderate_improvement:
  meaningful IC50 increase but still potentially risky

no_improvement:
  hERG risk unchanged

worse:
  hERG risk increased
```

### Metabolic stability

```text
strong_improvement:
  ≥3x or ≥5x half-life improvement
  or major intrinsic-clearance reduction

moderate_improvement:
  measurable but smaller improvement

no_improvement:
  unchanged

worse:
  stability decreases
```

These thresholds should be configurable by endpoint. Do not hard-code one universal threshold.

## Final rescue labels

Use multi-class labels:

```text
strong_success
weak_success
failed_activity_loss
failed_no_liability_improvement
failed_wrong_liability
failed_new_liability
uncertain
```

I would define them like this:

```text
strong_success:
  activity strong/acceptable retention
  liability strong improvement
  no major new risk

weak_success:
  activity acceptable or weak retention
  liability moderate/strong improvement
  no fatal new risk

failed_activity_loss:
  liability improves
  activity fails

failed_no_liability_improvement:
  activity retained
  liability does not improve

failed_wrong_liability:
  candidate improves something else but not requested L

failed_new_liability:
  requested liability improves, but another major risk appears

uncertain:
  missing, conflicting, or low-quality evidence
```

Do not throw away uncertain rows. Store them, but do not train the core rescue objective on them initially.

---

# 9. Pass 5: construct hard negatives deliberately

This is not optional. Hard negatives are what prevent the model from learning dumb shortcuts.

For every strong positive, try to create hard negatives in the same parent/task context.

## Negative 1: ADMET improved but activity lost

```text
Candidate improves solubility/hERG/stability
but loses target potency.
```

This punishes ADMET-only optimization.

## Negative 2: activity retained but liability not fixed

```text
Candidate is still potent
but hERG/solubility/stability remains bad.
```

This punishes nearest-neighbor or potency-only selection.

## Negative 3: wrong liability improved

```text
Candidate improves solubility
but the requested task is hERG rescue.
```

This teaches liability conditioning.

## Negative 4: new liability introduced

```text
Candidate fixes hERG
but becomes too insoluble or cytotoxic.
```

This teaches multi-objective caution.

## Negative 5: heuristic trap

These are candidates that simple heuristics would choose:

```text
most polar candidate
closest candidate
most drug-like candidate
lowest predicted toxicity candidate
largest descriptor shift
```

But they should fail for some reason.

This is critical.

If the model can beat these hard negatives, the result is meaningful.

---

# 10. Gold, silver, bronze tiers

Not all pairs should have equal training weight.

## Gold pairs

Use for main training and evaluation-style ablations.

Requirements:

```text
measured parent activity
measured candidate activity
measured parent liability
measured candidate liability
same or highly compatible activity context
same or highly compatible liability assay
clear analog relationship
source traceable
decontaminated
ideally curator-reviewed
```

Sources:

```text
internal data
curated papers
high-quality ChEMBL same-document series
```

## Silver pairs

Use for training, but with lower weight.

Requirements:

```text
measured activity and measured liability
some assay heterogeneity
analog relationship plausible
endpoint direction reliable
```

Sources:

```text
ChEMBL + PubChem
ChEMBL + TDC-overlap
PubChem screens with clear metadata
```

## Bronze pairs

Use as weak supervision or auxiliary training only.

Characteristics:

```text
predicted liability values
activity from loose target context
ADMET from different sources
weak analog relationship
no curator review
```

Do not let bronze labels dominate the model.

## Auxiliary molecule-only data

Use for supporting predictors, not rescue labels.

Examples:

```text
TDC ADMET tasks
MoleculeNet property tasks
PubChem activity outcomes
single-molecule toxicity datasets
```

---

# 11. How to mine from each source in practice

## ChEMBL mining workflow

Use ChEMBL bulk data if possible rather than scraping the interface.

Pipeline:

```text
1. Load molecules, structures, activities, assays, targets, documents.
2. Canonicalize all molecules.
3. Remove sealed-case molecules, documents, assays, and analog neighborhoods.
4. Select valid activity rows:
   - standard_relation = "=" where possible
   - standard_units convertible
   - pChEMBL available for potency-like endpoints where possible
   - no serious data validity flags
5. Build activity contexts by target/assay/document.
6. Identify analog series within each activity context.
7. Search for ADMET/physicochemical/toxicity assays for those molecules.
8. Compute parent-candidate deltas.
9. Label rescue outcomes.
10. Store source IDs for audit.
```

ChEMBL should be your best general source for target activity retention because it is organized around molecules, targets, assays, documents, standardized values, and pChEMBL. ([EMBL-EBI][1])

## PubChem BioAssay mining workflow

Pipeline:

```text
1. Load BioAssay records and tested compound outcomes.
2. Map PubChem CIDs to canonical molecules.
3. Identify assays relevant to:
   - cytotoxicity
   - hERG / ion channels
   - solubility / stability if present
   - safety counterscreens
4. Filter assays by metadata quality.
5. Convert active/inactive and numeric readouts into endpoint facts.
6. Use PubChem mostly for:
   - negatives
   - counterscreens
   - toxicity flags
   - weak auxiliary labels
7. Only promote to silver/gold when assay context is very clear.
```

PubChem BioAssay is especially useful because it stores assay descriptions and detailed results, and assay tables can be partitioned by activity outcomes such as active and inactive. ([OUP Academic][7])

## TDC workflow

Pipeline:

```text
1. Load TDC ADMET datasets.
2. Canonicalize molecules.
3. Remove sealed-case molecules and neighborhoods.
4. Train/support endpoint predictors.
5. Use TDC labels as auxiliary molecule-level facts.
6. Use only as rescue-pair labels if paired with independent activity context and analog relationship.
```

TDC should not be the main rescue-pair source. It should support:

```text
absolute ADMET evidence
weak predicted features
endpoint calibration
auxiliary losses
```

## MoleculeNet workflow

Pipeline:

```text
1. Load relevant MoleculeNet datasets.
2. Canonicalize and decontaminate.
3. Use for auxiliary molecular property/toxicity learning.
4. Do not treat as direct rescue pairs unless paired with activity/analog context elsewhere.
```

MoleculeNet is useful because it was explicitly designed as a benchmark suite for molecular property learning, not because it directly encodes medicinal-chemistry rescue decisions. ([RSC Publishing][5])

## Papers workflow

Pipeline:

```text
1. Choose non-held-out papers with SAR + ADMET/property data.
2. Extract tables into structured format.
3. Identify parent compounds with liabilities.
4. Identify analogs that fix liabilities and analogs that fail.
5. Curate mechanistic rationale.
6. Store DOI/document and table references.
7. Mark rows as gold if measured and consistent.
```

For papers, I would not train on full free text initially. Use them as structured curation sources.

## Internal data workflow

Pipeline:

```text
1. Map internal molecules to canonical IDs.
2. Load assay tables from LIMS/ELN/CRO reports.
3. Standardize endpoints.
4. Create project/series-specific activity contexts.
5. Build parent-candidate pairs.
6. Include failed analogs, not just winners.
7. Assign highest quality tier if assays are consistent.
```

Internal data can become the cleanest gold source because assay protocols are often more consistent than public data.

---

# 12. How to handle measured vs predicted evidence

This is a key rule:

> **Measured candidate outcomes can be used as training labels. They should not be provided as inference inputs for held-out discovery cases.**

During training row construction:

```text
candidate measured activity = label evidence
candidate measured ADMET = label evidence
```

At inference:

```text
candidate structure = allowed
candidate computed descriptors = allowed
candidate predicted ADMET from clean auxiliary model = allowed
candidate measured hidden ADMET = forbidden
candidate measured hidden activity = forbidden
```

This distinction preserves the discovery claim.

---

# 13. Structured rationale annotations

For each gold/silver row, add structured rationale if possible.

Do not rely on free-form text.

Use fields like:

```yaml
structured_rationale:
  liability_driver:
    - "high_logD"
    - "basic_amine"
  preserved_activity_features:
    - "aryl_core"
    - "hydrogen_bond_acceptor"
    - "basic_center_distance_to_arene"
  transformation_class:
    - "basicity_tuning"
    - "polarity_increase"
  expected_mechanism:
    liability_improvement: "reduced hERG-like liability due to lower lipophilic basic character"
    activity_retention: "core pharmacophore preserved"
  evidence_strength: "rule_based_plus_measured_delta"
```

For solubility:

```yaml
liability_driver:
  - "high_lipophilicity"
  - "flat_aromatic_core"
transformation_class:
  - "solubilizing_group_addition"
preserved_activity_features:
  - "binding_core"
  - "key_hbond_acceptor"
```

For metabolic stability:

```yaml
liability_driver:
  - "benzylic_soft_spot"
transformation_class:
  - "soft_spot_blocking"
preserved_activity_features:
  - "scaffold_core"
```

These rationales can be auto-generated using rules and then curator-reviewed for gold rows.

---

# 14. The minimum viable v1 dataset

I would not try to solve every ADMET endpoint.

For v1, build:

## Direct analog rescue

```text
hERG / safety rescue
solubility rescue
metabolic stability rescue
```

## Separate curated prodrug/exposure rescue

```text
oral exposure / prodrug rescue
```

But keep prodrug rows as a distinct rescue mode.

The minimum viable target size:

```text
Gold pairs: 500-2,000 if possible
Silver pairs: 10,000-100,000
Bronze/auxiliary molecule facts: millions if available
Ranking tasks: 1,000-10,000 local parent-candidate tasks
```

If gold pairs are much fewer, that is okay. Gold is for high-quality signal and validation-style fine-tuning. Silver/bronze provide scale.

---

# 15. The most important design choice

I would make the training examples **local ranking tasks**, not only independent pair rows.

Instead of only:

```text
P, C1 → success
P, C2 → failure
P, C3 → failure
```

also create:

```text
For parent P and liability L:
rank C1, C2, C3, C4, C5...
```

This directly matches what the model must do.

A local ranking task might look like:

```yaml
task_id: ADMET_RANK_0042
parent: MOL_P
activity_context: "EGFR inhibition"
liability: "solubility"
rescue_mode: "direct analog rescue"

candidates:
  - id: MOL_C1
    label: strong_success
  - id: MOL_C2
    label: failed_activity_loss
  - id: MOL_C3
    label: failed_no_liability_improvement
  - id: MOL_C4
    label: heuristic_trap_most_polar
  - id: MOL_C5
    label: weak_success
```

This teaches the model:

> **Choose the best rescue among plausible alternatives.**

That is much closer to the actual demo than molecule-level prediction.

---

# 16. What I would lock down now

I would lock these decisions:

## Decision 1

The main dataset is:

```text
ADMET rescue-pair + local ranking dataset
```

not molecule-only ADMET data.

## Decision 2

The central row is:

```text
parent P + candidate C + desired activity A + liability L + rescue mode R
```

## Decision 3

Candidate measured outcomes are labels, not inference inputs.

## Decision 4

Use three quality tiers:

```text
gold = measured, curated, high-confidence
silver = measured but heterogeneous
bronze = weak/predicted/auxiliary
```

## Decision 5

Use these v1 liability families:

```text
hERG / safety
solubility
metabolic stability
oral exposure / prodrug, curated separately
```

## Decision 6

Use multi-class rescue labels:

```text
strong_success
weak_success
failed_activity_loss
failed_no_liability_improvement
failed_wrong_liability
failed_new_liability
uncertain
```

## Decision 7

Construct hard negatives deliberately.

Do not rely only on whatever negatives happen to appear.

## Decision 8

Use ChEMBL for activity contexts, PubChem for large screening/counter-screen evidence, TDC/MoleculeNet for auxiliary property learning, papers/internal data for gold rescue labels and rationale.

---

# 17. The build order

The practical build order should be:

```text
1. Define sealed cases and forbidden entities.
2. Build molecule canonicalization pipeline.
3. Load ChEMBL and build activity contexts.
4. Load ADMET/property/liability facts from ChEMBL, TDC, MoleculeNet, PubChem.
5. Load curated paper/internal gold rows.
6. Build analog graph using similarity + scaffold + MMP.
7. Generate parent-candidate rescue pairs.
8. Label activity retention and liability improvement.
9. Create hard negatives.
10. Create local ranking tasks.
11. Assign quality tiers.
12. Generate structured rationales.
13. Run leakage/decontamination audit.
14. Freeze dataset manifest and hashes.
```

The main thing is:

> **Do not start by training. Start by building the rescue-pair table.**

That table is the foundation of the ADMET rescue system.

[1]: https://www.ebi.ac.uk/chembl/?utm_source=chatgpt.com "ChEMBL"
[2]: https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions?utm_source=chatgpt.com "Assay and Activity Questions | ChEMBL Interface Documentation"
[3]: https://pubchem.ncbi.nlm.nih.gov/docs/bioassays?utm_source=chatgpt.com "BioAssays"
[4]: https://tdcommons.ai/benchmark/admet_group/overview/?utm_source=chatgpt.com "ADMET Benchmark Group - TDC"
[5]: https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a?utm_source=chatgpt.com "MoleculeNet: a benchmark for molecular machine learning"
[6]: https://www.sciencedirect.com/science/article/abs/pii/S1359644613000937?utm_source=chatgpt.com "Matched Molecular Pair Analysis in drug discovery"
[7]: https://academic.oup.com/nar/article/38/suppl_1/D255/3112310?utm_source=chatgpt.com "overview of the PubChem BioAssay resource - Oxford Academic"
