Yes. Let’s lock this in ADMET-only.

The next two decisions are tightly coupled:

1. **How do we build the ADMET candidate universe and proposer system?**
2. **How do we choose sealed ADMET cases that are solvable, fair, and headline-worthy?**

My recommendation is:

> **Use a hybrid multi-proposer system that maximizes recall, then let the rescue ranker maximize precision. For sealed cases, choose one hERG/safety rescue, one exposure/prodrug rescue, and one direct analog solubility/metabolic-stability rescue.**

Do **not** rely on one pure generator as the only proposer yet. We should absolutely include a pure learned proposer as a novelty track, but the benchmark pass/fail should not depend only on it.

---

# Part 1: ADMET candidate universe and proposer system

## The core idea

The ADMET rescue system should look like this:

```text
Parent molecule P
+ desired activity A
+ liability L
+ rescue mode R
        ↓
candidate proposer system
        ↓
500–2,000 plausible rescue candidates
        ↓
rescue ranker
        ↓
top-k locked predictions
```

The proposer’s job is **recall**:

> “Does the candidate pool contain the true or functionally correct rescue?”

The ranker’s job is **precision**:

> “Can the system move the best rescue candidates into the top 5 / top 10?”

This separation is important. If the proposer never creates the right molecule, the ranker cannot save us.

Matched molecular pair analysis is a strong conceptual fit here because it uses structural transformations between related compounds to infer how molecular changes affect biological or physicochemical properties; reviews emphasize its value for medicinal chemistry design-rule generation and multi-parameter optimization. ([PubMed][1]) Pairwise/delta modeling is also directly relevant: DeepDelta was designed to process two molecules simultaneously and predict ADMET property differences between derivatives. ([PMC][2])

So our proposer should not be one thing. It should be a **multi-proposer ensemble**.

---

# 1.1 The proposer stack

I would build six proposer channels.

## Proposer 1: analog retrieval proposer

This proposer searches the clean, decontaminated molecule universe for plausible analogs.

Input:

```text
parent P
activity context A
liability L
```

It retrieves molecules that are:

```text
same or related activity context
similar scaffold
similar pharmacophore
reasonable structural distance from parent
not forbidden by sealed-case decontamination
not measured-label leakage for the evaluation case
```

Sources:

```text
ChEMBL
internal analog series
curated papers outside sealed cases
clean purchasable compound libraries
```

ChEMBL is especially useful here because it brings together chemical structures, bioactivity, targets, assays, and document metadata for drug-like molecules. ([EMBL-EBI][3])

Purpose:

> Find boring-but-plausible analogs that medicinal chemists would actually consider.

This is the conservative recall engine.

---

## Proposer 2: matched molecular pair transformation proposer

This proposer learns common transformations from training data:

```text
parent fragment → candidate fragment
```

Examples:

```text
phenyl → pyridyl
methyl → trifluoromethyl
alkyl chain → morpholine side chain
tertiary amine → less basic amine
amide → bioisostere
benzylic H → F / steric blocker
ester → amide / carbamate
```

It applies transformations observed in clean training pairs.

Input:

```text
parent P
liability L
desired delta
```

Output:

```text
candidate analogs one or few transformations away
```

Purpose:

> Use real medicinal chemistry transformations, not unconstrained molecule hallucination.

This is likely one of the most valuable proposer channels.

---

## Proposer 3: liability-specific rule proposer

This proposer uses structured medicinal chemistry transformation libraries by liability type.

It is not the final decision-maker. It just creates plausible candidates.

### hERG / cardiac liability proposer

Possible transformations:

```text
reduce basicity
reduce lipophilicity
reduce lipophilic basic amine character
add polarity carefully
reduce aromatic hydrophobic surface
move or mask cationic center
replace high-risk amine motifs
reduce excessive molecular size / hydrophobicity
```

The point is not “always make it polar.” The point is:

> Generate candidates that might reduce hERG-like liability while preserving target pharmacophore.

hERG risk mitigation is a real medicinal chemistry problem; reviews describe avoiding hERG interactions as a recurring optimization task and discuss strategies/case studies for rational design. ([ScienceDirect][4])

### Solubility rescue proposer

Possible transformations:

```text
add solubilizing heteroatoms
add ionizable handle if compatible
reduce lipophilicity
reduce planarity / crystal packing tendency
replace phenyl with heteroaryl
add small polar side chain
reduce excessive aromaticity
```

### Metabolic stability proposer

Possible transformations:

```text
block metabolic soft spots
replace labile groups
reduce lipophilicity
add steric shielding
replace oxidizable benzylic/allylic sites
heteroaryl or fluorine substitution where reasonable
remove obvious hydrolysis-prone motifs
```

Metabolic stability is a major early ADME optimization goal, often investigated with liver microsomes or hepatocytes, and medicinal chemistry programs commonly use structural changes to reduce clearance and improve exposure. ([PubMed][5])

### Prodrug / exposure proposer

Possible transformations:

```text
ester prodrug
amino-acid ester prodrug
phosphate/phosphoramidate-style delivery motif where appropriate
temporary masking of polar group
transporter-aware promoiety
```

This must be its own rescue mode.

A prodrug candidate does **not** necessarily need to have the same direct in vitro potency as the active parent. It needs to deliver the active species.

---

## Proposer 4: learned inverse-delta proposer

This is the learned proposer you want.

Training pattern:

```text
parent P
desired activity A
liability L
desired delta:
  improve L
  preserve A
        ↓
candidate C
```

This proposer learns from successful rescue pairs.

Input:

```text
parent molecule
liability type
desired property change
activity context
rescue mode
```

Output:

```text
candidate molecules or edits
```

This is the first serious “AI designs the rescue” component.

It can produce non-obvious transformations that rule-based proposers may miss.

The risk is that it may generate invalid, unstable, or unrealistic molecules. That is why it should feed into the ranker rather than become the final output directly.

---

## Proposer 5: forward-model optimization proposer

This proposer uses forward models as rewards.

Process:

```text
generate/edit candidates
score with:
  predicted liability improvement
  predicted activity retention
  predicted absolute risk
  uncertainty
  synthesizability/plausibility
optimize candidates
```

You were right to be cautious here. A forward-reward optimizer can exploit model weaknesses and miss real candidates if the reward is too narrow. But it is still useful as a parallel proposer because it may explore different regions than the inverse proposer.

So we should try both:

```text
explicit inverse proposer
forward-reward optimizer
```

but keep them as separate candidate sources.

---

## Proposer 6: pure learned novelty proposer

This is the moonshot channel.

It is not constrained to known MMP transformations.

Input:

```text
parent P
activity A
liability L
rescue mode R
```

Output:

```text
novel rescue candidates
```

This is the channel that may produce the headline:

> **“The model did not just recover the known solution. It proposed novel chemically plausible alternatives.”**

However, I would not make the entire system depend on this channel early.

Use it as:

```text
novelty track
extra candidate source
potential headline amplifier
```

not the only route to success.

---

# 1.2 The final v1 proposer system

I would lock this:

```text
Candidate universe = union of:
  analog retrieval candidates
  MMP transformation candidates
  liability-specific transformation candidates
  learned inverse-delta candidates
  forward-model optimization candidates
  pure learned novelty candidates
```

Then deduplicate and annotate.

The ranker receives:

```text
parent P
candidate C
activity context A
liability L
rescue mode R
computed evidence
proposer source
structured rationale
```

The output is:

```text
rescue score
failure-mode probabilities
confidence
rank
```

---

# 1.3 Candidate pool size

For v1, I would set:

```text
raw generated candidates per case: 5,000–20,000
after filters: 500–2,000
ranker output: top 5 / top 10 / top 20
```

Why this range?

* Below 100 candidates, recall may be too low.
* Above 10,000 filtered candidates, ranking may become noisy and expensive.
* 500–2,000 is enough to include diverse transformations, hard decoys, and novel candidates.

For investor-facing claims, top-k matters:

```text
top 5 = strongest
top 10 = strong
top 20 = acceptable for hard open-generation tasks
```

---

# 1.4 Candidate filters

Before ranking, every candidate should pass basic filters.

## Hard validity filters

```text
valid valence
valid canonicalization
no disconnected nonsense unless salt/form allowed
no impossible atoms/groups
no severe instability alert
no obvious pan-assay/reactive alert unless allowed
reasonable molecular weight range
reasonable heavy atom count
```

## Parent-relative filters

```text
not identical to parent
not too far unless novelty channel
maintains core pharmacophore if direct analog rescue
reasonable transformation distance
```

## Rescue-mode filters

For direct analog rescue:

```text
candidate itself should plausibly retain activity
```

For prodrug rescue:

```text
candidate should plausibly convert to active parent/species
```

For hERG rescue:

```text
candidate should not simply destroy all activity-relevant features
```

## Decontamination filters

For training:

```text
remove sealed case molecules
remove sealed case analog neighborhoods
remove sealed case documents/assays
```

For evaluation:

```text
candidate may match hidden answer only if generated/ranked after model training
candidate must not carry measured hidden labels
```

This distinction matters. If the model generates the known answer after training, that is allowed. If it saw its measured outcome during training, not allowed.

---

# 1.5 Candidate annotations

Every candidate should carry structured metadata.

Example:

```yaml
candidate_id: CAND_000123
parent_id: MOL_PARENT
candidate_smiles: "..."
proposer_sources:
  - "mmp_transform"
  - "hERG_rule"
  - "learned_inverse"

transformation:
  changed_atoms: [...]
  transformation_class:
    - "basicity_tuning"
    - "polarity_increase"
  mmp_rule_id: "MMP_0421"

computed_evidence:
  delta_logP: -1.1
  delta_TPSA: +38.2
  delta_pKa_basic_center: -1.8
  pharmacophore_similarity: 0.82
  shape_similarity: 0.77
  scaffold_similarity: 0.69
  predicted_liability_delta: "hERG risk down"
  predicted_activity_retention: "medium_high"

structured_rationale:
  liability_driver_removed:
    - "lipophilic_basic_amine_character"
  activity_features_preserved:
    - "aryl_core"
    - "HBA_position"
    - "basic_center_spacing"
  possible_failure_modes:
    - "permeability_loss"
    - "target_affinity_loss"
```

This is not just for explanations. It is also how we make the ranker’s input chemically meaningful.

---

# 1.6 Proposer success metrics

Before scoring the ranker, score proposer recall.

For each sealed case:

```text
Did proposer generate exact historical answer?
Did proposer generate functional equivalents?
How many?
From which channels?
At what candidate-pool size?
```

Metrics:

```text
exact_answer_recall@N
functional_rescue_recall@N
diversity of candidates
invalid rate
novelty rate
synthetic plausibility rate
```

Example:

```text
Generated 12,000 raw candidates
Filtered to 1,200
Exact historical answer present: yes
Functional equivalents present: 37
Pure learned proposer contributed: 12 candidates
MMP proposer contributed: exact answer
Ranker placed exact answer at #3
```

This would be extremely strong.

---

# 1.7 The key lock

For ADMET v1, I would lock this:

> **The official candidate universe is produced by a hybrid multi-proposer system. The pure learned proposer is included as a novelty channel, but the system is allowed to use retrieval, MMP transformations, liability-specific transformations, inverse-delta generation, and forward-model optimization. The ranker is responsible for final selection.**

That is the best balance of:

```text
high recall
chemical plausibility
novelty potential
auditability
benchmark pass probability
```

---

# Part 2: ADMET solvable-case criteria and sealed case selection

Now we need to choose the sealed ADMET cases.

The key is:

> **A solvable case is not merely a famous drug story. It is a case where the correct rescue is chemically learnable from general training data, but the specific answer is held out.**

We need cases that are:

```text
fair
decontaminatable
not trivial
not impossible
headline-worthy
chemically actioned
scorable
```

---

# 2.1 Solvable-case criteria

A sealed ADMET case must pass these gates.

## Gate 1: clear parent molecule

There must be a known parent molecule P.

The model input can include:

```text
parent structure
known desired activity
known liability
parent measured facts
```

## Gate 2: clear desired activity

The case must define what we are preserving.

Good:

```text
preserve H1 antagonism
preserve antiviral acyclovir active species
preserve ACE inhibition
preserve kinase target activity
```

Bad:

```text
make this drug better
improve clinical success
reduce side effects generally
```

The model needs an activity context.

## Gate 3: structure-actionable liability

The liability must be fixable by chemistry.

Good:

```text
hERG/QT risk
poor oral bioavailability
poor solubility
poor metabolic stability
CYP inhibition
cytotoxicity/off-target risk
```

Bad:

```text
failed Phase III efficacy
wrong target biology
dosing schedule issue
formulation-only issue
idiosyncratic clinical toxicity with no chemical signal
```

## Gate 4: known successful rescue

There must be a known answer or successful analog/prodrug:

```text
historical answer molecule
validated improved analog
validated prodrug
validated active metabolite
```

This is necessary for the sealed historical discovery benchmark.

## Gate 5: answer can be withheld cleanly

We must be able to remove:

```text
answer molecule
parent-answer pair
papers describing the rescue
assay records
synonyms
patents if needed
close analog series
```

If the case is everywhere in training data and impossible to quarantine, it is risky.

## Gate 6: model can learn the general mechanism elsewhere

We should hold out the specific case family, not the entire concept.

For example:

* If holding out valacyclovir, the model may still learn prodrug logic from other prodrugs.
* If holding out fexofenadine, the model may still learn hERG-reduction logic from other hERG data.
* If holding out a solubility case, the model may still learn solubility rescue from other analog series.

This is fair.

## Gate 7: candidate reachability

The answer or a functional equivalent must be reachable by our proposer.

Examples:

```text
one to three transformations from parent
recognizable prodrug transformation
MMP-like analog transformation
pharmacophore-preserving polarity/basicity change
```

If the answer requires a huge scaffold hop with little clue, it may be too hard for v1.

## Gate 8: not too trivial

The rescue should not be:

```text
parent plus one obvious methyl group
parent plus most common prodrug from textbook with no hard decoys
```

We need hard alternatives.

## Gate 9: evidence package exists

For the final writeup, we need:

```text
parent liability evidence
answer rescue evidence
activity-retention evidence
ADMET improvement evidence
source traceability
```

---

# 2.2 The sealed ADMET case types

We want three ADMET cases, each testing a different rescue mode.

I recommend:

```text
Case 1: hERG / cardiac safety rescue
Case 2: oral exposure / prodrug rescue
Case 3: direct analog solubility or metabolic-stability rescue
```

This gives breadth.

Do **not** use three prodrug cases. That would make the benchmark look narrow.

Do **not** use three hERG cases. That would make it look like a single-trick model.

---

# 2.3 Recommended sealed case 1: terfenadine → fexofenadine

## Case type

```text
hERG / QT / cardiac safety rescue
active metabolite / polar analog rescue
```

## Why it is strong

Terfenadine is associated with QT prolongation due to cardiac potassium channel blockage; FDA review materials describe fexofenadine hydrochloride as the major active metabolite of terfenadine and discuss terfenadine’s cardiac risk. ([FDA Access Data][6]) Fexofenadine lacks the hERG/IKr liability associated with terfenadine in electrophysiology work; one paper states that terfenadine blocks HERG channels and can cause QT prolongation/torsades, whereas its carboxylate fexofenadine lacks HERG activity. ([PMC][7])

## What model sees

```text
Parent: terfenadine structure
Desired activity: H1 antihistamine activity
Known liability: hERG/QT/cardiac potassium-channel risk; CYP3A4 exposure risk
Rescue mode: direct analog / active metabolite safety rescue
Goal: reduce cardiac liability while preserving H1 activity
```

## What is hidden

```text
fexofenadine structure
fexofenadine name/synonyms
Allegra references
terfenadine→fexofenadine papers
assay records linking fexofenadine to this rescue
close case-family analogs if they reveal answer
```

## Success

Strong success:

```text
fexofenadine exact structure in top-k
```

Functional success:

```text
carboxylated/polar H1-active analog that plausibly reduces hERG risk while preserving H1 pharmacophore
```

## Why this is investor-friendly

> “The model took an allergy drug that worked but carried heart-risk liability and discovered the safer active metabolite/rescue direction.”

This is the most public-comprehensible ADMET case.

## Risk

It is famous. Decontamination must be brutal.

Because you train from scratch, this is manageable, but you need a very strong holdout manifest.

---

# 2.4 Recommended sealed case 2: acyclovir → valacyclovir

## Case type

```text
oral bioavailability / prodrug exposure rescue
```

## Why it is strong

Valacyclovir is the L-valyl ester prodrug of acyclovir, and published pharmacokinetic literature reports that valacyclovir improves acyclovir oral bioavailability; one source states absolute bioavailability after oral valacyclovir is about 54.5%, while other clinical pharmacology literature describes valacyclovir as producing roughly threefold to fivefold higher acyclovir bioavailability than oral acyclovir. ([PubMed][8])

## What model sees

```text
Parent: acyclovir structure
Desired activity: preserve acyclovir antiviral active species
Known liability: poor oral bioavailability / poor absorption
Rescue mode: prodrug exposure rescue
Goal: improve oral exposure while delivering active acyclovir
```

## What is hidden

```text
valacyclovir structure
valacyclovir name/synonyms
Valtrex references
papers and assay records linking valacyclovir to acyclovir bioavailability rescue
specific L-valyl ester answer
```

## Success

Strong success:

```text
valacyclovir exact structure in top-k
```

Functional success:

```text
amino-acid ester or transporter-aware prodrug predicted to deliver acyclovir and improve oral exposure
```

## Important labeling rule

Do **not** score this as:

```text
candidate direct potency must equal acyclovir
```

Score it as:

```text
candidate delivers active acyclovir after conversion
```

This is an exposure rescue, not a direct analog potency rescue.

## Why this is investor-friendly

> “The model rescued a drug that worked but was poorly absorbed by discovering the prodrug strategy that made it clinically useful.”

---

# 2.5 Recommended sealed case 3: direct analog solubility/metabolic-stability rescue

For case 3, I would **not** immediately lock a famous approved-drug pair.

I would lock the **case type** now and choose the exact molecule after data audit.

## Case type

```text
direct analog rescue
liability: poor solubility or poor metabolic stability
activity retained directly by candidate
```

## Why we need this case

Terfenadine/fexofenadine and acyclovir/valacyclovir are both somewhat special:

* one is an active metabolite/safety rescue,
* one is a prodrug/exposure rescue.

We need a case where the model does ordinary medicinal chemistry:

> “Change the molecule itself so it keeps target activity and improves solubility/stability.”

This is the clearest test of parent-candidate rescue learning.

## What the case should look like

Input:

```text
Parent P
Desired activity: target T
Known liability: poor solubility OR poor metabolic stability
Rescue mode: direct analog rescue
Goal: retain potency while improving liability
```

Hidden answer:

```text
candidate C from a SAR series with measured potency and measured solubility/stability improvement
```

Success:

```text
exact candidate in top-k
or functional analog with same rescue mechanism in top-k
```

## Where to find it

Best sources:

```text
internal medicinal chemistry data
curated SAR papers with ADMET tables
ChEMBL same-document analog series
recent open-access lead-optimization papers
```

A public example of the kind of paper we want is a lead-optimization study that reports improvements in potency, solubility, metabolic stability, and off-target toxicity around a compound series; those are exactly the kinds of tables that can yield direct analog rescue pairs. ([RSC Publishing][9])

Another example style is hit-to-lead work where the authors explicitly report analogs that improve clearance/metabolic stability without completely compromising potency; this kind of table is useful because it contains both successes and hard negatives. ([RSC Publishing][10])

## How to choose the exact case

Create a candidate list of 10–20 direct analog case studies, then score them.

Scoring rubric:

| Criterion                             | Weight |
| ------------------------------------- | -----: |
| Same target/assay potency data        |    20% |
| Measured solubility or stability data |    20% |
| Clear successful analog               |    20% |
| Hard negatives in same series         |    15% |
| Decontamination manageable            |    15% |
| Investor/comms clarity                |    10% |

Pick the highest-scoring case.

## My recommendation

Lock this as:

> **ADMET Case 3 = direct analog solubility/metabolic-stability rescue from a curated SAR series, exact molecule selected after audit.**

Do not make it another prodrug case unless no suitable direct analog case is available.

---

# 2.6 Backup case: enalaprilat → enalapril

Use this only if case 3 cannot be found in time.

## Case type

```text
oral absorption / prodrug rescue
```

Enalaprilat is the active ACE inhibitor but is poorly absorbed orally; enalapril is the oral prodrug hydrolyzed to enalaprilat. ([NCBI][11])

This is a good case, but it overlaps conceptually with acyclovir/valacyclovir.

So:

```text
primary backup: enalaprilat → enalapril
not preferred as one of the three if acyclovir → valacyclovir is already included
```

---

# 2.7 Final sealed ADMET case recommendation

I would lock the sealed case structure like this:

## Case 1: safety rescue

```text
Terfenadine → fexofenadine
Liability: hERG/QT/cardiac safety
Rescue mode: direct analog / active metabolite safety rescue
```

## Case 2: exposure rescue

```text
Acyclovir → valacyclovir
Liability: poor oral bioavailability
Rescue mode: prodrug exposure rescue
```

## Case 3: direct analog rescue

```text
Exact molecule TBD after audit
Liability: poor solubility OR poor metabolic stability
Rescue mode: direct analog rescue
Source: internal data or curated SAR paper with measured potency + measured ADMET
```

Backup:

```text
Enalaprilat → enalapril
only if no direct analog case is ready
```

This gives the benchmark breadth:

| Case                                    | Rescue mode                       | What it tests                                                       |
| --------------------------------------- | --------------------------------- | ------------------------------------------------------------------- |
| Terfenadine → fexofenadine              | Safety / active-metabolite rescue | Can model reduce off-target cardiac risk while preserving function? |
| Acyclovir → valacyclovir                | Prodrug exposure rescue           | Can model improve exposure by delivering active species?            |
| Direct analog solubility/stability case | Ordinary med-chem rescue          | Can model tune a molecule while preserving potency?                 |

That is the right shape.

---

# Part 3: How the candidate universe interacts with the sealed cases

For each sealed case, run two evaluation modes.

## Mode A: open proposer mode

The system is not given the answer candidate set.

It receives:

```text
parent
activity context
liability
rescue mode
```

Then the proposer must generate/retrieve candidates.

Success:

```text
known answer appears in generated pool and/or top-k
functional alternatives appear in top-k
```

This is the more impressive mode.

## Mode B: closed hard-ranking mode

The model receives a sealed candidate set:

```text
known answer
hard decoys
close analogs
wrong-liability candidates
activity-loss candidates
simple heuristic traps
```

Labels are hidden.

Success:

```text
ranker puts the known rescue or functional rescue near top
```

This is useful because it isolates ranker ability.

My recommendation:

> **Use both.**

Why?

* Open proposer mode proves generative/retrieval discovery.
* Closed hard-ranking mode proves the ranker can choose correctly when the answer is present among hard alternatives.

This also helps debugging:

```text
If open mode fails but closed ranking works:
  proposer recall problem

If open mode includes answer but ranker misses it:
  ranker problem

If both work:
  strong discovery system
```

---

# Part 4: What to lock now

I would lock these decisions.

## Lock 1: Hybrid multi-proposer

The ADMET proposer is not one model.

It is:

```text
analog retrieval
+ MMP transformations
+ liability-specific transformations
+ learned inverse-delta proposer
+ forward-model optimization proposer
+ pure learned novelty proposer
```

## Lock 2: Proposer/ranker separation

The proposer optimizes for recall.

The ranker optimizes for precision.

Track both separately.

## Lock 3: Candidate pool size

For each ADMET rescue case:

```text
raw candidates: 5,000–20,000
filtered candidates: 500–2,000
locked final outputs: top 5 / top 10 / top 20
```

## Lock 4: Structured candidate annotations

Every candidate gets:

```text
proposer source
transformation class
delta descriptors
pharmacophore similarity
liability-driver changes
activity-feature preservation
possible failure modes
```

## Lock 5: Three ADMET sealed case types

Use:

```text
hERG/safety rescue
oral exposure/prodrug rescue
direct analog solubility/metabolic-stability rescue
```

## Lock 6: Specific case picks

Primary:

```text
1. Terfenadine → fexofenadine
2. Acyclovir → valacyclovir
3. TBD direct analog solubility/metabolic-stability rescue from curated SAR/internal data
```

Backup:

```text
Enalaprilat → enalapril
```

## Lock 7: Evaluate open + closed modes

For each sealed case:

```text
open proposer mode
closed hard-ranking mode
```

This gives both a headline result and a diagnostic result.

---

# Part 5: The next step after this

Now that this is decided, the next ADMET-only topic should be:

> **Conditioning schema for ADMET rescue.**

Meaning:

```text
What exactly does the system receive as input?
What is the standard input packet?
What fields are allowed?
What fields are forbidden?
How do we represent activity context, liability, rescue mode, and acceptable tradeoff?
What does the proposer see vs the ranker see?
```

That should come before detailed architecture.

Once the conditioning schema is locked, architecture becomes much easier.

[1]: https://pubmed.ncbi.nlm.nih.gov/23557664/?utm_source=chatgpt.com "Matched molecular pair analysis in drug discovery"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10605784/?utm_source=chatgpt.com "DeepDelta: predicting ADMET improvements of molecular ..."
[3]: https://www.ebi.ac.uk/chembl/?utm_source=chatgpt.com "ChEMBL"
[4]: https://www.sciencedirect.com/science/article/abs/pii/S0223523420302592?utm_source=chatgpt.com "hERG toxicity assessment: Useful guidelines for drug design"
[5]: https://pubmed.ncbi.nlm.nih.gov/11579441/?utm_source=chatgpt.com "Optimization of metabolic stability as a goal of modern drug ..."
[6]: https://www.accessdata.fda.gov/drugsatfda_docs/nda/2011/201373Orig1s000MedR.pdf?utm_source=chatgpt.com "201373Orig1s000 - accessdata.fda.gov"
[7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC1573545/?utm_source=chatgpt.com "The antihistamine fexofenadine does not affect IKr currents in ..."
[8]: https://pubmed.ncbi.nlm.nih.gov/16759825/?utm_source=chatgpt.com "Stability of valacyclovir: implications for its oral bioavailability"
[9]: https://pubs.rsc.org/en/content/articlehtml/2024/md/d4md00275j?utm_source=chatgpt.com "Lead optimisation of OXS007417: in vivo PK profile and hERG ..."
[10]: https://pubs.rsc.org/en/content/articlehtml/2020/md/d0md00165a?utm_source=chatgpt.com "Hit-to-lead optimization of a benzene sulfonamide series for ..."
[11]: https://www.ncbi.nlm.nih.gov/books/NBK534299/?utm_source=chatgpt.com "Enalaprilat - StatPearls - NCBI Bookshelf"

Lock **ADMET Case 3** as:

# ADMET-003: OXS007570 → OXS008474 direct analog rescue

This is the best third ADMET case because it is not another famous prodrug or marketed-drug metabolite story. It tests the exact thing we wanted:

> **Can the model take a potent parent molecule with poor physicochemical/metabolic properties and propose a direct analog that improves solubility and metabolic stability while retaining activity?**

## Final decision

Use the 2024 RSC Medicinal Chemistry lead-optimization paper around **OXS007417/OXS007570** as the third sealed ADMET case.

The paper explicitly says the campaign around OXS007417 led to improved potency, solubility, metabolic stability, and off-target toxicity, and that hERG liability was alleviated through nitrogen insertion at key positions. It also reports that **OXS008255** and **OXS008474** had improved murine PK profiles versus OXS007417. 

For the benchmark scoring pair, use:

```text
Parent: OXS007570, compound 6
Primary hidden rescue answer: OXS008474, compound 8
Task type: direct analog solubility/metabolic-stability rescue
```

Why not use OXS007417 as the parent? Because **OXS007570 → OXS008474** is a cleaner direct analog rescue task. OXS007570 already has strong activity, but has weak solubility/metabolic-stability profile, so the model’s job is not “invent a whole new series,” but:

> **modify a working parent so it keeps activity and fixes ADMET liabilities.**

That exactly matches our ADMET-rescue benchmark.

---

# Why this is the right third case

## 1. It is a true direct analog case

Our first two ADMET cases are:

```text
ADMET-001: terfenadine → fexofenadine
Type: hERG / cardiac safety / active-metabolite rescue

ADMET-002: acyclovir → valacyclovir
Type: oral exposure / prodrug rescue
```

So the third case should not be another prodrug.

OXS007570 → OXS008474 gives us the missing category:

```text
ADMET-003: direct analog rescue
Type: solubility + metabolic stability improvement while preserving activity
```

This makes the three-case ADMET benchmark much stronger.

---

## 2. The data are unusually clean

For **OXS007570, compound 6**, the paper reports:

```text
EC50: 2 nM
ER in mouse S9: 0.57
Solubility: 15 μM
hERG IC50: 7.4 μM
clogP: 3.3
LLE: 5.4
```

For **OXS008474, compound 8**, the paper reports:

```text
EC50: 15 nM
ER in mouse S9: 0.11
Solubility: >200 μM
hERG: 46% inhibition at 30 μM
clogP: 2.6
LLE: 5.2
```

So the candidate has:

```text
activity retained within roughly one order of magnitude
metabolic stability improved
solubility improved from 15 μM to >200 μM
lower clogP
acceptable hERG profile
```

The table also gives many failed or partial analogs, which is perfect for hard negatives. 

---

## 3. It has a clear medicinal-chemistry mechanism

The paper says the team explored C-6 ring systems with the goal of improving metabolic stability and solubility while maintaining high AML differentiation activity and low hERG affinity. It specifically notes that introducing nitrogen atoms into the aromatic ring reduced clogP and changed ring electronics, and that the 6-fluoro-4-methylpyridin-3-yl derivative 8 improved metabolic stability and hERG activity. 

So the structured rationale is clean:

```text
Liability drivers:
  poor solubility
  intermediate/poor metabolic stability
  lipophilic aromatic character

Transformation:
  introduce site-selective nitrogen into attached aromatic ring
  reduce clogP / change electronics
  preserve differentiation-active pharmacophore

Rescue result:
  solubility improved
  metabolic stability improved
  activity retained
```

This is exactly the kind of “mechanistic rescue” we wanted.

---

## 4. It has excellent hard negatives

This paper is not just one parent and one winner. It contains many analogs that teach why the problem is nontrivial.

Examples:

```text
Compound 7:
  improves metabolic stability but only modest solubility and limited hERG benefit

Compound 9:
  good activity and solubility but poor metabolic stability

Compounds 10 and 11:
  severe potency loss

Compound 25:
  hERG reduced but potency reduced and metabolic stability not improved

Compounds 29 and 30:
  nitrogen placement causes potency loss

Compound 34:
  potency loss despite related heteroaromatic design
```

The paper explicitly says that lowering clogP alone was not enough and that there was not a clear correlation between clogP and hERG activity. That is very useful because it prevents the benchmark from becoming “just make it more polar.” 

This is the biggest reason I prefer this case over a simple textbook solubility example.

---

# Exact ADMET-003 sealed-case definition

## Case ID

```text
ADMET_003_OXS_DIRECT_ANALOG_RESCUE
```

## Parent molecule

```text
OXS007570
Compound 6
```

## Primary hidden successful rescue

```text
OXS008474
Compound 8
```

## Secondary hidden successful / functional rescues

Include these as “functional success” candidates, not necessarily the primary exact answer:

```text
OXS008203, compound 27
OXS008255, compound 20
Compound 31
Compound 32
```

Reason: these compounds also show useful combinations of activity and ADMET improvement. For example, compound 27 has EC50 23 nM, ER 0.23, solubility >200 μM, and hERG IC50 18 μM; compounds 31 and 32 also show low ER and moderate solubility with reduced hERG inhibition. 

But the official exact-recovery target should be:

```text
OXS008474 / compound 8
```

because it is the cleanest direct rescue from compound 6.

---

# Model input packet

The model should see:

```yaml
case_id: ADMET_003_OXS_DIRECT_ANALOG_RESCUE

parent:
  name_for_inference: "Parent P"
  structure: OXS007570 / compound 6 structure
  measured_activity:
    assay: HL-60 CD11b differentiation assay
    EC50: approximately 2-3 nM
  measured_liabilities:
    metabolic_stability:
      endpoint: extraction ratio in mouse S9
      value: 0.57
      interpretation: intermediate / poor
    solubility:
      value: 15 μM
      interpretation: low
    hERG:
      value: IC50 7.4 μM
      interpretation: monitor / avoid worsening

activity_context:
  desired_activity: AML differentiation activity in HL-60 CD11b assay
  preserve_activity_threshold: within 10x parent EC50

liability_context:
  primary_liability: improve solubility and metabolic stability
  secondary_constraint: do not worsen hERG liability

rescue_mode:
  direct_analog_rescue

goal:
  propose direct analogs that improve solubility and metabolic stability while retaining differentiation activity
```

The model should **not** see:

```text
OXS008474 name
OXS008474 structure
OXS008255 name/structure
OXS008203 name/structure
the 2024 OXS paper
compound table outcomes
SAR text explaining nitrogen insertion
the statement that compound 8 was successful
any assay rows for this OXS family
```

---

# Success criteria

## Exact success

```text
OXS008474 appears in top 10
```

Strongest version:

```text
OXS008474 appears in top 5
```

Holy-shit version:

```text
OXS008474 appears top 1-3 and the model also proposes structurally plausible alternatives
```

## Functional success

A candidate counts as a functional rescue if it satisfies:

```text
activity retained:
  EC50 within 10x of parent, approximately ≤20-30 nM

solubility improved:
  at least 5x better, ideally ≥10x better

metabolic stability improved:
  ER moves from intermediate/poor toward low, ideally <0.3

hERG not worsened:
  no clear worsening of hERG profile

chemistry:
  direct analog, not unrelated scaffold
```

This lets the model win even if it proposes a different but valid rescue analog.

---

# Evaluation modes

Use both modes.

## Mode A: open proposer mode

The model gets only the parent and task packet.

It must generate candidates.

Success questions:

```text
Did the proposer generate OXS008474?
Did it generate functional equivalents?
Did the ranker put them in top-k?
```

This is the headline mode.

## Mode B: closed hard-ranking mode

Give the ranker a candidate pool containing:

```text
OXS008474
OXS008203
OXS008255
compound 31
compound 32
hard negatives from compounds 7-35
additional generated decoys
```

Labels are hidden.

Success question:

```text
Can the ranker place the true/functional rescue candidates above hard negatives?
```

This mode is important because it tells us whether failure is due to candidate generation or ranking.

---

# Hard negatives to include

From the same paper, include:

```text
Compound 7:
  metabolic stability improves, but solubility only modest and no hERG breakthrough

Compound 9:
  good activity and high solubility, but poor metabolic stability

Compounds 10 and 11:
  potency collapses

Compound 15:
  activity still decent, but solubility/metabolic stability not rescued

Compound 25:
  hERG improves, but potency and metabolic stability are not good enough

Compounds 29 and 30:
  nitrogen placement causes potency loss

Compound 34:
  related heteroaromatic design but potency loss
```

These are perfect because they punish simplistic logic like:

```text
just add nitrogen
just lower clogP
just increase solubility
just preserve scaffold
```

---

# Decontamination plan for ADMET-003

This case needs strict case-family quarantine.

Remove from training:

```text
DOI: 10.1039/d4md00275j
the entire OXS007417/OXS007570/OXS008xxx paper
all OXS compound identifiers
all structures from compounds 1 and 6-35
all ESI structures and assay tables
all ChEMBL/PubChem entries for these molecules if present
all text mentioning OXS007417, OXS007570, OXS008474, OXS008255, OXS008203
all close analogs from the same series
prior OXS differentiation-agent papers from the same group if they reveal the case family
```

At inference, the parent structure and parent measured facts are allowed because the task is:

> “Here is a working parent with a known problem. Rescue it.”

But all candidate outcomes and historical solution information are forbidden.

---

# Final ADMET sealed-case set

Lock the ADMET benchmark as:

## ADMET-001

```text
Terfenadine → fexofenadine
Type: hERG / cardiac safety rescue
```

## ADMET-002

```text
Acyclovir → valacyclovir
Type: oral exposure / prodrug rescue
```

## ADMET-003

```text
OXS007570 → OXS008474
Type: direct analog solubility/metabolic-stability rescue
```

This is a strong trio:

| Case                       | What it tests                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| Terfenadine → fexofenadine | Can the model reduce off-target cardiac/safety liability while preserving desired activity? |
| Acyclovir → valacyclovir   | Can the model discover prodrug/exposure rescue logic?                                       |
| OXS007570 → OXS008474      | Can the model do ordinary direct analog medicinal chemistry optimization?                   |

That gives breadth, credibility, and a real nontrivial direct-analog case.

# Decision

Use **OXS007570 → OXS008474** as the third ADMET case.

Next step: put these three cases into the sealed-case registry and begin implementation with the decontamination/canonicalization pipeline.
