Yes. Retrosynthesis should become its own serious module, but we need to be careful not to define it as:

> “product SMILES → reactant SMILES.”

That is the naive version.

For our system, retrosynthesis should mean:

> **Given a target molecule, produce a ranked, feasible, auditable synthetic route to purchasable or available starting materials, with predicted conditions, forward validation, impurity/yield risk, and route-level confidence.**

That is much more useful than a one-step retrosynthesis predictor.

The key distinction:

```text
Naive retrosynthesis model:
target molecule → possible precursors

Rasyn retrosynthesis system:
target molecule → complete route → conditions → forward feasibility → cost/purchasability → risk analysis → top ranked route
```

This matters because many retrosynthesis papers optimize one-step prediction on benchmarks, but route-level usefulness is a different problem. Recent evaluation work argues that USPTO-50K is insufficient for judging real synthesis-planning performance and that single-step model choice can change multi-step planning success substantially. ([RSC Publishing][1])

---

# 1. What are we trying to build?

The retrosynthesis system should support the bigger Rasyn story:

> **Rasyn can design molecules and then tell you how to make them.**

For ADMET rescue and antibiotic discovery, this is essential. A molecule that cannot be synthesized is not a real candidate.

So the retrosynthesis module should answer:

```text
Can we make this molecule?
From what starting materials?
In how many steps?
With what reactions?
Under what conditions?
What are the risks?
What impurities or failure modes are likely?
How confident are we?
```

The headline demo should not be:

> “Our retrosynthesis model got top-1 accuracy on USPTO-50K.”

The headline should be:

> **“Rasyn designed a novel compound and autonomously generated a synthesis route that was executed in the lab.”**

Or:

> **“Given a target molecule, Rasyn found a route experts missed, predicted the conditions, and the compound was made and verified.”**

---

# 2. What counts as success?

For retrosynthesis, success has multiple levels.

## Level 1: one-step exact recovery

The model predicts the same reactants as the historical reaction.

Example:

```text
Product P → historical reactants R1 + R2
```

This is useful for diagnostics, but not sufficient.

## Level 2: one-step chemically valid alternative

The model proposes different reactants that could plausibly make the product.

This is more useful than exact recovery because many targets have multiple possible disconnections.

## Level 3: complete route to purchasable materials

The system finds a full route:

```text
target → intermediate 1 → intermediate 2 → purchasable starting materials
```

This is the first real product-level success.

## Level 4: route passes forward validation

Every retrosynthetic step is checked by a forward reaction predictor or reaction feasibility model.

This matters because a retrosynthetic step can look plausible backwards but fail forwards.

## Level 5: route includes conditions and risk analysis

The route includes:

```text
reagents
catalysts
solvent
temperature
expected side products
yield/risk estimate
purification concerns
```

ASKCOS is a good precedent for this integrated view: the newest ASKCOS system includes retrosynthetic planning plus complementary modules for condition prediction, reaction outcome prediction, pathway evaluation, and other capabilities. ([American Chemical Society Publications][2])

## Level 6: route is experimentally executed

This is the real headline.

> **The model proposed the route, the lab ran it, and the target compound was made and verified.**

That should be our north star.

---

# 3. What is the central training signal?

Retrosynthesis has two different learning problems:

## Problem A: single-step retrosynthesis

Given product:

```text
product → precursor set
```

This is the standard ML task.

Existing model families include:

```text
template-based methods
semi-template graph-edit methods
template-free sequence models
synthon/reactant completion methods
diffusion graph-to-graph models
```

Graph2Edits is an example of a graph-edit style system: it predicts edits on the product graph autoregressively to generate transformation intermediates and final reactants, aiming for more interpretable predictions. ([Nature][3]) RetroDiff frames retrosynthesis as a conditional graph-to-graph generative task and uses a multi-stage diffusion process. ([Proceedings of Machine Learning Research][4]) GDiffRetro similarly uses reaction-center identification plus conditional diffusion to complete reactants. ([arXiv][5])

## Problem B: route planning

Given target:

```text
target → complete synthetic tree
```

This is not solved by one-step prediction alone. It requires search.

Retro* is a key precedent: it frames retrosynthetic planning as an AND-OR tree search and uses neural guidance with an A*-like algorithm to find synthetic routes efficiently. ([Proceedings of Machine Learning Research][6])

So our central training signal should not be only:

```text
product → reactants
```

It should be:

```text
target molecule + route state + constraints → next disconnection / route value / feasibility
```

The retrosynthesis system needs both:

1. **one-step proposer**: suggests disconnections/reactants;
2. **route-level planner/value model**: decides which route tree is worth pursuing.

---

# 4. The biggest mistake to avoid

Do not build only a one-step retrosynthesis predictor.

A one-step predictor can be impressive on paper but fail in real use because:

```text
the reactants are not purchasable
the route dead-ends
the forward reaction is unlikely
the conditions are missing
the route is too expensive
the step has bad selectivity
the product is impossible to purify
the model proposes a known memorized reaction
```

The correct architecture is:

```text
single-step proposer
+ forward reaction validator
+ condition predictor
+ route search algorithm
+ route value/cost model
+ inventory/purchasability model
+ risk/impurity/yield model
```

This is why I would treat retrosynthesis as a **planning system**, not one model.

---

# 5. The model should have three layers

## Layer 1: single-step retrosynthesis proposers

This layer answers:

> “What are plausible precursors for this molecule?”

Use a **multi-proposer ensemble**, just like ADMET and antibiotic.

### Proposer 1: template/rule-based proposer

Useful because:

```text
high precision
chemically interpretable
good for common transformations
easy to audit
```

Weakness:

```text
limited novelty
template coverage problem
```

### Proposer 2: graph-edit proposer

This predicts reaction centers and graph edits.

Useful because retrosynthesis often consists of local bond changes, and graph-edit models like Graph2Edits and LocalRetro-style approaches exploit that structure. Graph2Edits explicitly frames retrosynthesis as predicting product graph edits inspired by arrow-pushing. ([Nature][3])

### Proposer 3: sequence/Transformer proposer

This treats retrosynthesis like translation:

```text
product SMILES → reactant SMILES
```

This can be strong but may hallucinate invalid or memorized outputs. Transformer-based reaction prediction/retrosynthesis has a substantial literature; Tetko et al. showed data augmentation can improve NLP-style transformer models for synthesis and retrosynthesis prediction. ([Nature][7])

### Proposer 4: diffusion reactant-completion proposer

This is the non-naive generative piece.

Use diffusion for:

```text
synthon completion
reactant generation
alternate precursor generation
inpainting missing fragments
scaffold-preserving disconnections
```

Not necessarily for full route planning.

RetroDiff and GDiffRetro show that diffusion can be applied to retrosynthesis as graph-to-graph/reactant-generation machinery. RetroDiff describes retrosynthesis as conditional graph-to-graph generation and uses multi-stage diffusion, while GDiffRetro uses reaction-center identification plus conditional diffusion for reactant generation. ([Proceedings of Machine Learning Research][4])

### Proposer 5: precedent retrieval proposer

Given target/product, retrieve similar reactions from clean reaction data.

Useful because chemists often reason by precedent:

```text
This bond was made in similar molecules under these conditions.
```

This also helps produce explainable routes.

---

## Layer 2: step validator/ranker

This layer answers:

> “Which proposed retrosynthetic step is actually plausible?”

For each proposed step:

```text
product P
precursors R
reaction class
```

score:

```text
forward reaction plausibility
reaction class match
selectivity risk
condition availability
yield likelihood
literature precedent similarity
reactant availability
step novelty/risk
```

This layer should include a forward model:

```text
precursors + conditions → predicted product
```

If the forward model does not recover the target product, the retrosynthetic step is suspicious.

This is crucial. Retrosynthesis without forward validation is too easy to fool.

---

## Layer 3: multi-step route planner

This layer answers:

> “What complete route should we choose?”

Search algorithms:

```text
A* / Retro*-style search
MCTS
beam search
bidirectional search
constraint-aware search
```

Retro* is especially relevant because it uses a neural-guided A*-like search over an AND-OR tree for multi-step synthesis planning. ([Proceedings of Machine Learning Research][6])

The planner should optimize route-level score:

```text
route_score =
  step plausibility
  purchasability
  cost
  step count
  convergence
  condition availability
  yield/risk
  selectivity risk
  impurity risk
  safety/green chemistry
  novelty
  uncertainty
```

This route score is more important than one-step top-1 accuracy.

---

# 6. Where diffusion belongs

Diffusion should not be the whole retrosynthesis system.

It should be a **single-step proposer**, especially for reactant completion and alternative precursor generation.

## Good use of diffusion

```text
target product + disconnection site → complete reactants
product graph with broken bonds → fill missing precursor fragments
synthon + reaction class → reactant completion
route state + constraints → diverse precursor proposals
```

## Bad use of diffusion

```text
diffusion generates entire route tree from scratch with no validation
diffusion outputs reactants without forward check
diffusion replaces route search
```

The reason is simple:

> Retrosynthesis is a planning problem. Diffusion is a proposal mechanism.

The right design:

```text
graph-edit / template / retrieval proposer identifies disconnections
diffusion completes/reacts around synthons
forward model validates
route planner searches
route ranker chooses
```

This is analogous to our antibiotic system:

```text
diffusion proposes
ranker/validator decides
```

---

# 7. What should the retrosynthesis model be conditioned on?

Conditioning is very important.

The model should not just see:

```text
target molecule
```

It should see:

```text
target molecule
route constraints
available inventory
desired step count
forbidden chemistry
preferred chemistry
scale
green chemistry constraints
functional group tolerance
candidate starting-material library
```

A standard input packet might look like:

```yaml
task_id: RETRO_001

target:
  canonical_smiles: "..."
  structure: molecular_graph

constraints:
  max_steps: 5
  must_terminate_in: "commercially_available_building_blocks"
  inventory_database: "clean_commercial_inventory_v1"
  avoid_reagents:
    - "highly hazardous"
    - "expensive_precious_metal_if_possible"
  preferred_route_style:
    - "short"
    - "robust"
    - "medicinal_chemistry_scale"
  scale: "10-100 mg"

context:
  target_use_case: "candidate synthesis"
  allowed_route_novelty: "known_precedent_or_low_risk"
  require_condition_prediction: true
  require_forward_validation: true
```

The proposer sees the target and constraints.

The route planner sees the current route state.

The ranker sees each step and the full route.

---

# 8. Training data

Retrosynthesis needs several kinds of data.

## A. Reaction examples

Rows like:

```text
reactants + reagents + product + conditions + yield
```

Sources:

```text
USPTO-style patent reactions
ORD
curated literature reactions
internal ELN/LIMS data
commercial route examples if available
```

ORD is valuable because it is explicitly designed as an open-access structured reaction database to support reaction prediction, synthesis planning, and experiment design. ([Open Reaction Database][8])

## B. One-step retrosynthesis pairs

Rows like:

```text
product → reactants
```

These train the one-step proposer.

## C. Forward reaction data

Rows like:

```text
reactants + reagents/conditions → product
```

These train the validator.

## D. Conditions data

Rows like:

```text
reactants + product/reaction class → reagents, catalysts, solvent, temperature
```

This lets route output be lab-actionable.

## E. Route-level data

Rows like:

```text
target → route tree → starting materials
```

Harder to get, but extremely valuable.

Public reaction data often gives individual reactions, not full route trees. Internal synthetic campaigns or curated total-synthesis/medicinal chemistry routes are much better.

## F. Negative/failure data

This is hugely important and often missing.

Examples:

```text
reaction attempted but failed
low yield
wrong regioisomer
decomposition
poor selectivity
unavailable starting material
route dead-end
```

A model trained only on successful literature reactions will be too optimistic.

---

# 9. Derived data tables

Just like ADMET/antibiotic, we should not train directly from raw data. Build derived tables.

## Molecule table

```yaml
molecule_id: MOL_123
canonical_smiles: "..."
inchi_key: "..."
commercial_availability: true
cost_per_g: ...
inventory_source: ...
computed_descriptors: {...}
forbidden_status: ...
```

## Reaction fact table

```yaml
reaction_id: RXN_123
reactants: [...]
reagents: [...]
catalysts: [...]
solvents: [...]
product: MOL_P
conditions:
  temperature: ...
  time: ...
yield: ...
source: "ORD / USPTO / internal / paper"
reaction_class: "Suzuki coupling"
atom_mapping_available: true
quality_flags: {...}
```

## Retrosynthesis step table

```yaml
retro_step_id: STEP_123
product: MOL_P
precursors: [MOL_A, MOL_B]
reaction_class: "amide coupling"
source_reaction_id: RXN_123
conditions_available: true
yield: ...
quality_tier: "gold/silver/bronze"
```

## Route table

```yaml
route_id: ROUTE_123
target: MOL_T
steps:
  - STEP_1
  - STEP_2
  - STEP_3
starting_materials:
  - MOL_SM1
  - MOL_SM2
route_metrics:
  step_count: 3
  longest_linear_sequence: 3
  total_estimated_yield: ...
  purchasable_fraction: 1.0
  cost_score: ...
  risk_score: ...
```

## Candidate route table

For model outputs:

```yaml
candidate_route_id: CAND_ROUTE_001
target: MOL_T
route_tree: ...
step_predictions: [...]
forward_validation_results: [...]
condition_predictions: [...]
route_score: ...
uncertainty: ...
```

---

# 10. Hard negatives

Retrosynthesis needs hard negatives too.

## Step-level hard negatives

```text
plausible-looking disconnection but wrong reaction class
precursors that forward-predict to wrong product
reactants incompatible under conditions
wrong regioselectivity
missing protecting group issue
unavailable precursor
requires impossible selectivity
```

## Route-level hard negatives

```text
route dead-ends before purchasable materials
route uses unavailable starting material
route too long
route has one very risky step
route contains incompatible functional groups
route ignores stereochemistry
route depends on impossible condition
```

## Heuristic traps

```text
shortest route but chemically risky
most precedent-similar route but unavailable starting material
cheapest route but bad selectivity
high one-step score route that dead-ends
```

This is essential because a route planner can game simple cost functions.

---

# 11. Architecture recommendation

I would define the Rasyn retrosynthesis system as:

```text
Rasyn-Retro =
  multi-proposer one-step retrosynthesis engine
  + forward reaction validator
  + condition predictor
  + route planner/search
  + route-level value model
  + inventory/cost/purchasability layer
  + uncertainty/risk/explanation layer
```

## Component 1: one-step proposer ensemble

```text
template proposer
graph-edit proposer
SMILES/Transformer proposer
diffusion reactant-completion proposer
precedent retrieval proposer
```

## Component 2: step validator

```text
forward reaction model
reaction class checker
condition plausibility model
yield/risk estimator
selectivity/impurity risk model
```

## Component 3: planner

```text
Retro*/A*-style route search
MCTS backup
beam search diagnostic
inventory-aware termination
```

## Component 4: route ranker

```text
route-level score
cost
step count
yield
risk
green chemistry
condition availability
synthesis confidence
```

## Component 5: explanation/audit layer

Structured, not free-form:

```yaml
route_rationale:
  key_disconnections:
    - "amide bond disconnection"
    - "Suzuki coupling"
  precedent_support:
    - reaction_id: RXN_...
  risk_flags:
    - "possible regioselectivity issue"
    - "protecting group may be required"
  validation:
    forward_model_recovered_target: true
    condition_prediction_available: true
```

---

# 12. Benchmark design

Do not only use USPTO-50K.

Use four levels.

## Benchmark 1: single-step held-out prediction

Tests one-step models.

Metrics:

```text
top-1/top-5/top-10 reactant recovery
round-trip forward validation
reaction-class accuracy
```

Useful, but not sufficient.

## Benchmark 2: multi-step route completion

Given target, find route to purchasable materials.

Metrics:

```text
route found?
step count
route cost
forward validation of all steps
starting material availability
```

## Benchmark 3: historical route reconstruction

Hold out full target/route families.

The model gets target only and must recover:

```text
historical route
or plausible route of equal/better quality
```

This is analogous to ADMET sealed cases.

## Benchmark 4: prospective lab route

Given a Rasyn-designed molecule, the system proposes route, conditions, and risk analysis. Lab attempts route.

This is the headline benchmark.

---

# 13. Decontamination

Retrosynthesis decontamination is more complex than molecule decontamination.

For a sealed route case, remove:

```text
target molecule
known intermediates
route papers/patents
exact reactions in the route
near-duplicate reaction examples
same product/reaction records
same route family
starting material-to-target text
commercial synthesis pages that reveal route
```

But be careful:

> The model should still learn general reaction chemistry.

So we remove the case family, not the entire reaction class.

For example, if holding out a Suzuki route, do not remove all Suzuki reactions. Remove that specific target/intermediate/route family.

---

# 14. What should be headline-worthy?

Here are the real retrosynthesis demos that matter.

## Demo 1: “AI makes its own designed molecule”

Rasyn designs an ADMET-rescued or antibiotic candidate.

Retrosynthesis module proposes a route.

Lab makes it.

This is the best integrated demo.

## Demo 2: “AI finds route to a molecule experts struggled with”

Give a target molecule that is hard but not impossible.

The model proposes a shorter/cheaper/greener route.

Expert chemists compare.

## Demo 3: “AI fixes failed route”

Input:

```text
target molecule
failed route or failed step
```

Model proposes alternate disconnection or condition.

Lab verifies.

## Demo 4: “AI closed loop”

```text
design molecule → plan route → predict conditions → run reaction → verify NMR/LC-MS
```

This is the strongest.

---

# 15. What I would lock now

I would lock these decisions.

## Lock 1

Retrosynthesis is a **route-planning system**, not a one-step model.

## Lock 2

The central output is:

```text
ranked complete routes to purchasable starting materials
```

not just reactant SMILES.

## Lock 3

Use a multi-proposer single-step engine:

```text
template
graph-edit
Transformer
diffusion
retrieval
```

## Lock 4

Use diffusion as a **reactant/synthon completion proposer**, not the whole planner.

## Lock 5

Every proposed retrosynthetic step must be forward-validated.

## Lock 6

Route planner should use A*/Retro*/MCTS-style search, with route-level value/cost functions.

## Lock 7

Condition prediction, purchasability, cost, and risk are first-class route-ranking features.

## Lock 8

Benchmark must include route-level and prospective/lab validation, not only USPTO-style single-step accuracy.

---

# 16. Initial architecture in one line

> **Rasyn-Retro should be an inventory-aware, forward-validated, multi-proposer retrosynthetic planning system where graph-edit/template/retrieval/diffusion models generate one-step disconnections, neural validators and condition models score feasibility, and a Retro*/A*-style planner searches for complete routes to purchasable starting materials.**

That is the version that fits the overall Rasyn ambition.

The next thing to decide is the same kind of question we handled for ADMET:

> **What training signal teaches route-quality retrosynthesis, not just one-step reactant prediction?**

[1]: https://pubs.rsc.org/en/content/articlehtml/2024/dd/d3dd00252g?utm_source=chatgpt.com "Models Matter: the impact of single-step retrosynthesis on ..."
[2]: https://pubs.acs.org/doi/abs/10.1021/acs.accounts.5c00155?utm_source=chatgpt.com "ASKCOS: Open-Source, Data-Driven Synthesis Planning"
[3]: https://www.nature.com/articles/s41467-023-38851-5?utm_source=chatgpt.com "Retrosynthesis prediction using an end-to-end graph ..."
[4]: https://proceedings.mlr.press/v258/wang25e.html?utm_source=chatgpt.com "RetroDiff: Retrosynthesis as Multi-stage Distribution Interpolation"
[5]: https://arxiv.org/html/2501.08001v1?utm_source=chatgpt.com "GDiffRetro: Retrosynthesis Prediction with Dual Graph ..."
[6]: https://proceedings.mlr.press/v119/chen20k.html?utm_source=chatgpt.com "Learning Retrosynthetic Planning with Neural Guided A* Search"
[7]: https://www.nature.com/articles/s41467-020-19266-y?utm_source=chatgpt.com "State-of-the-art augmented NLP transformer models for ..."
[8]: https://docs.open-reaction-database.org/?utm_source=chatgpt.com "the Open Reaction Database!"
