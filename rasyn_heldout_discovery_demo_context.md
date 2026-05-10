# Rasyn Held-Out Discovery Demonstration Context

## Purpose of this document

This document is the high-level context file for coding agents, research assistants, evaluation builders, data-cleaning agents, and internal contributors who will work on the Rasyn held-out discovery demonstration.

This is not a model architecture document.  
This is not a training-recipe document.  
This is not a final dataset-selection document.  
This is not a pitch deck.

This document explains the project we are building, why it exists, what the demonstration must prove, what claims it must support, what constraints must be respected, what kind of blind studies we are designing, what counts as success, what counts as failure, and what evidence must exist for the final result to be credible.

The architecture, model family, training recipe, exact datasets, scaling plan, and implementation details will be decided later. Any coding agent reading this should avoid making hidden assumptions about those things. The immediate goal is to preserve the intent of the project and prevent future work from drifting into something weaker, easier, less honest, or less headline-worthy.

The whole project exists to support one strong claim:

> Rasyn can train a chemistry discovery system from scratch, with strict contamination controls, and the system can discover successful held-out solutions across multiple chemistry discovery tasks that it did not see during training or inference.

The demonstration is intentionally designed to be hard to dismiss as cherry-picked. It should feel closer to a research-paper-quality blind study than to a marketing demo.

---

## One-sentence project summary

We are building a contamination-controlled, pre-registered, auditable blind discovery benchmark in which a chemistry system trained from scratch is evaluated on sealed held-out case studies across ADMET rescue, antibiotic discovery, and spectral structure elucidation, with all predictions locked before reveal and all leakage controls documented.

---

## The strategic motivation

Rasyn is building chemistry foundation systems for real laboratory and pharmaceutical workflows. The major risk in this area is that demos can look impressive while proving very little. A system can generate attractive molecules, predict properties on a benchmark, or reproduce known drug stories, but investors, chemists, and technical diligence teams can still attack the result as:

- cherry-picked,
- memorized,
- benchmark-overfit,
- not experimentally meaningful,
- not truly blind,
- not connected to real chemistry,
- lacking proper controls,
- or too vague to distinguish from ordinary cheminformatics.

The purpose of this project is to avoid that failure mode.

Instead of showing one isolated “miracle molecule,” we want to show a reproducible experimental framework:

> Before training, we seal multiple historical or privately defined chemistry discovery cases. We remove those cases and their leakage neighborhoods from the corpus. We train from scratch. At evaluation time, the system receives only the allowed problem statement. It makes locked predictions. Only then do we reveal the hidden successful outcomes. The system succeeds across multiple task families.

This should create a stronger investor reaction than a normal demo because the audience sees that the result was not hand-picked after the fact. The evaluation was structured before the model saw the answer.

The emotional reaction we want is:

> “They did not just find one example. They built a test where the answers were hidden, the system had to solve several distinct chemistry problems, and the predictions were auditable.”

---

## The core demo claim

The intended headline-level claim is:

> Rasyn discovered all six held-out solutions across ADMET rescue and antibiotic discovery in a contamination-controlled blind benchmark.

This sentence is ambitious and should only be used if the protocol supports it. The entire project must be designed so that the statement is not misleading.

The word **discovered** is allowed only under strict conditions:

1. The relevant solution was excluded from training data.
2. The relevant solution was excluded from inference context.
3. Direct references to the case were excluded.
4. Close leakage neighborhoods were removed or documented.
5. The system’s prediction was locked before reveal.
6. The success criteria were pre-registered.
7. The hidden answer or experimental outcome was not available to the model-running process.
8. The final evaluation includes failures, false positives, and baselines where applicable.

If any of these conditions are not met for a case, the wording must change for that case. It may become “rediscovered,” “reconstructed,” “ranked,” or “recovered,” but the main project is designed specifically so that the stronger word “discovered” can be defensible.

---

## What this project is not

This project is not primarily a benchmark leaderboard exercise.

This project is not a generic MoleculeNet-style evaluation.

This project is not “generate a million molecules.”

This project is not “show pretty structures.”

This project is not “dock molecules and claim drug discovery.”

This project is not “train on all public data and then recover famous examples.”

This project is not “use an uncontrolled pretrained language model and hope nobody asks about memorization.”

This project is not “claim clinical drug discovery from an early in vitro signal.”

This project is not “pretend a historical public example is blind if the system may have memorized it.”

This project is not “prove Rasyn can solve any chemistry task.”

This project is a rigorously controlled demonstration of held-out discovery capability in a narrow but meaningful set of chemistry tasks.

---

## Why training from scratch matters

A key assumption of this project is that Rasyn can train from scratch and control the training corpus.

This matters because many famous chemistry examples are public. If an uncontrolled pretrained model recovers a public example, skeptics can reasonably ask whether it memorized the answer. That would weaken the demonstration.

Training from scratch changes the situation.

If we define the sealed case studies before training and remove all relevant records from the corpus, then a public historical example can become a valid held-out blind discovery case. The example is public in the world, but it is not present in the model’s training environment.

Therefore, the central evidence burden becomes:

> Prove that the held-out examples, their direct documents, their labels, their synonyms, and their close leakage neighborhoods were not included in training or inference.

This is why decontamination is not a minor data-cleaning task. It is one of the core products of the demo.

The decontamination proof must be as strong as possible while remaining fair. We should not remove all general chemistry knowledge related to a concept, because then the task becomes artificially impossible. We should remove the specific case and leakage neighborhood, not the entire scientific principle.

---

## The three evaluation pillars

The full demonstration has three pillars.

### Pillar 1: ADMET rescue

The system is given a drug-like molecule or active chemical series with a known problem such as poor exposure, poor solubility, metabolic instability, toxicity, hERG/cardiac risk, CYP interaction, permeability limitation, or some other measurable developability issue.

The system must propose or rank a solution that preserves the desired pharmacological function while improving the liability.

The key idea:

> The system rescues a molecule that works but has a problem.

This is commercially powerful because many real drug-discovery failures are not due to a complete lack of potency. They are due to tradeoffs: potency versus exposure, activity versus toxicity, stability versus permeability, solubility versus binding, and so on.

### Pillar 2: Antibiotic discovery

The system is given a blinded candidate set, target organism, assay context, or discovery objective related to antibacterial activity.

The system must rank, identify, or propose compounds that show antibacterial activity under the defined conditions.

The key idea:

> The system discovers antibacterial hits from a hidden or held-out search space.

This is public-facing and headline-friendly, but the wording must be disciplined. An antibacterial hit is not automatically a drug. The strongest fair claim is that the system discovers experimentally or historically validated antibacterial candidates in a held-out setting.

### Pillar 3: Spectral structure elucidation and impurity reasoning

The system is given NMR, LC-MS, MS/MS, HPLC, or related spectral/analytical evidence, possibly with reaction context, and must infer structures, products, impurities, or candidate identities.

The key idea:

> The system can read messy real chemistry evidence and determine what happened.

This pillar supports trust in the other two pillars. If the system proposes molecules or reactions, it must also be able to help verify what was made. Spectra and impurity reasoning are the bridge from computational suggestion to laboratory reality.

The NMR/spectra pillar is already locked in as an important part of the larger Rasyn demo. It may be implemented and evaluated separately from the ADMET and antibiotic discovery pillars.

---

## Why the six-case design exists

The main investor-facing and paper-like study should not rely on a single example.

A single success can be dismissed as luck, leakage, cherry-picking, or hidden prior knowledge.

Six carefully designed held-out studies are much harder to dismiss, especially if they cover multiple task families and each one follows the same pre-registered blind protocol.

The planned structure is:

- Three ADMET rescue studies.
- Three antibiotic discovery studies.
- A separate NMR/spectra blind-test suite.

The six headline cases are the ADMET and antibiotic cases. The NMR/spectra suite is a broader trust and verification layer.

The reason for three plus three:

- It is small enough to finish and document thoroughly.
- It is large enough to show reproducibility.
- It shows the system can handle at least two distinct chemistry-discovery modes.
- It prevents the story from being overly dependent on one famous example.

---

## The intended final narrative

The final narrative should sound like this:

> We trained Rasyn from scratch on a contamination-controlled chemistry corpus. Before training, we sealed six discovery case studies: three ADMET rescue challenges and three antibiotic discovery challenges. We removed all direct examples, answer molecules, relevant assay records, documents, synonyms, identifiers, and close leakage neighborhoods from the training corpus. After training, the system was given only the allowed problem statements or blinded candidate sets. Predictions were locked before reveal. Rasyn discovered the successful solution in all six held-out studies. In parallel, Rasyn solved blinded NMR/LC-MS structure-identification and impurity-reasoning tasks, showing that the system can also interpret experimental evidence.

That is the story all implementation work should support.

---

## Claim discipline

The project is ambitious, but the language must be controlled. The following distinctions matter.

### “Discovered”

Use when:

- the answer was not in training;
- the answer was not in inference context;
- leakage controls were applied;
- prediction was locked before reveal;
- success criteria were pre-registered.

Example:

> The system discovered the held-out antibacterial hit from the blinded library.

### “Recovered”

Use when:

- the system found a known answer, but there is some uncertainty about possible indirect exposure or leakage;
- the benchmark is still useful but not strong enough for “discovered.”

Example:

> The system recovered the known rescue molecule in the answer-redacted historical test.

### “Rediscovered”

Use when:

- the case is public and may have been seen by a pretrained or uncontrolled model;
- not ideal for this project, but useful for controls.

Example:

> The system rediscovered a famous historical ADMET rescue.

### “Ranked”

Use when:

- the task is library prioritization;
- the hidden hit was included in a candidate set;
- the system’s achievement is putting it near the top.

Example:

> The system ranked the hidden antibacterial hit in the top 1% of the blinded library.

### “Generated”

Use when:

- the molecule was not provided in the candidate list;
- the system proposed it or a close functional equivalent.

Example:

> The system generated a candidate matching the held-out rescue strategy.

### “Validated”

Use when:

- there is experimental or external evidence confirming the result.

Example:

> The model-selected compound was validated in an antibacterial growth-inhibition assay.

---

## What counts as a successful ADMET rescue

An ADMET rescue case involves a starting molecule with a useful activity and a developability liability.

The system succeeds if it proposes or ranks a candidate that satisfies the pre-registered rescue objective.

A rescue objective usually has two sides:

1. Preserve the desired function.
2. Improve the liability.

For example:

- preserve target activity while improving solubility;
- preserve target activity while reducing hERG risk;
- preserve pharmacology while improving oral exposure;
- preserve active exposure while reducing metabolic clearance;
- preserve biological effect while lowering cytotoxicity;
- preserve potency while improving permeability.

A case should not be considered a rescue if the model merely improves one property while destroying the desired function.

A case should not be considered a rescue if the solution is not chemical in nature, unless the case is explicitly designed around formulation or delivery. The current project is focused on chemical rescue: analogs, prodrugs, modifications, scaffold changes, matched molecular pair logic, or related molecular changes.

A case should not be considered a rescue if the starting problem is vague or not measurable.

A case should not be considered a rescue if the “solution” requires clinical data, animal studies, or long-term outcomes that are impossible to evaluate in the current benchmark.

---

## What counts as a successful antibiotic discovery

An antibiotic discovery case involves a hidden or held-out antibacterial signal.

The system succeeds if it identifies, ranks, or proposes compounds that meet the pre-registered antibacterial criterion.

Possible success modes:

- ranking a hidden active compound near the top of a blinded library;
- generating a compound from a held-out active family;
- selecting candidates that show higher hit rate than baseline in a prospective screen;
- identifying organism-specific activity rather than generic toxicity;
- discovering a hit that passes an early selectivity screen.

The benchmark must not count known positive controls as discoveries. Positive controls are useful to verify the assay or scoring, but they are not discoveries.

The benchmark must avoid counting generic cytotoxic molecules as antibiotic discoveries. A molecule that simply kills everything is not the same as an antibacterial candidate. If possible, antibiotic cases should include a selectivity or cytotoxicity-aware criterion.

The benchmark must avoid the phrase “new drug” unless the evidence is far beyond early discovery. In the current project, the stronger and more accurate phrase is:

> discovered antibacterial hits or antibiotic candidates.

---

## What counts as a successful NMR/spectra blind test

The NMR/spectra suite should evaluate whether the system can infer structures and impurities from experimental evidence.

Possible success modes:

- correct pure-compound structure ranked first;
- correct pure-compound structure in top-k;
- correct molecular formula;
- correct scaffold;
- correct regioisomer;
- correct impurity candidate in top-k;
- correct major product in a crude reaction mixture;
- correctly flags ambiguity or uncertainty;
- proposes the right confirmatory experiment.

The spectra suite should include multiple difficulty tiers:

1. Clean pure compounds.
2. Public benchmark-style examples.
3. Crude reaction mixtures.
4. Impurity cases.
5. Ambiguous or adversarial cases with close isomers.

A system that only works on clean spectra is not enough for the long-term Rasyn story. The powerful claim is that Rasyn can operate in messy real chemistry.

---

## The fairness philosophy

The benchmark must be hard, but it must not be unfair.

Fairness means the system gets the information a competent chemist or discovery model would reasonably need to solve the problem.

Unfairness means withholding essential problem definition, making the answer impossible, or removing all general knowledge needed for the task.

For example:

- If the task is ADMET rescue, the model should be told the desired activity and the liability.
- If the task is antibiotic ranking, the model should be told the organism or assay context.
- If the task is spectra identification, the model should get appropriate spectra and any allowed constraints such as formula, reaction context, or candidate set.
- If the task is prodrug-style exposure rescue, the model should know the objective is to preserve active pharmacology while improving exposure.
- If the task is not solvable without knowing the target, the target should not be hidden.

The goal is not to create a magic test where the system guesses an answer from nothing. The goal is to create a fair held-out discovery task where the system can use learned chemical principles but cannot access the specific answer.

---

## The decontamination philosophy

The decontamination process must remove case-specific leakage while preserving general learnable chemistry.

This is subtle.

If the held-out case is an oral-bioavailability rescue, we should remove the specific parent-answer pair, the direct papers, assay records, and close analogs. We should not remove every example of oral bioavailability, every prodrug, or every absorption-related molecule. The system needs general principles.

If the held-out case is a hERG liability rescue, we should remove the specific case and its close analog neighborhood. We should not remove all hERG data, because then the system cannot learn what hERG liability means.

If the held-out case is an antibiotic hit, we should remove the specific hit, its discovery records, and close family. We should not remove all antibacterial data.

This is the key balance:

> Remove the answer and its leakage neighborhood. Preserve the general domain.

The decontamination proof must be strong enough to convince a skeptical technical reviewer that the system did not see the answer, while not making the task artificially impossible.

---

## Case-level quarantine

Each sealed case should have a quarantine zone.

The quarantine zone includes all information that would make the hidden solution directly or indirectly available.

The quarantine zone should include at least:

- answer molecule;
- parent molecule if needed;
- direct analogs from the same study;
- molecule synonyms;
- molecule IDs;
- salts;
- stereoisomers;
- tautomers;
- metabolites where relevant;
- prodrugs where relevant;
- canonical and non-canonical structures;
- assay records;
- target annotations directly revealing the answer;
- papers;
- patents;
- review articles;
- datasets derived from the original study;
- close molecular neighbors if they make the answer trivial;
- spectra records for the answer if spectra are part of the task;
- vendor pages if they include the relevant activity/rescue description;
- any document containing the exact case story.

The quarantine zone is not just a list of molecule names. It is a case-level information boundary.

---

## Decontamination evidence requirements

For every sealed case, we should be able to produce an audit report showing:

1. The case was registered before training.
2. The forbidden entities were defined before training.
3. Exact molecules were removed.
4. Synonyms and identifiers were removed.
5. Direct assay records were removed.
6. Direct documents were removed.
7. Close analog neighborhoods were removed according to pre-registered rules.
8. Remaining nearest neighbors were audited.
9. Inference context did not contain the answer.
10. The prediction was locked before reveal.

The audit report should be understandable by both technical diligence teams and serious chemists.

The strongest form of audit includes:

- human-readable summary;
- machine-readable manifest;
- hashes of training data;
- hashes of model checkpoints if applicable;
- hashes of inference packets;
- hashes of locked predictions;
- removed-record lists;
- nearest-neighbor tables;
- leakage-scan logs;
- canary tests;
- baseline results;
- final scoring scripts.

---

## Canary tests

A canary test is a fake or synthetic leakage item inserted into raw data before cleaning to verify that the decontamination pipeline catches it.

For example, if we define a fake forbidden molecule name or fake discovery sentence and place it in the raw corpus, the decontamination pipeline should remove it.

Canaries help prove that the pipeline works mechanically rather than relying on trust.

Potential canaries:

- fake molecule synonym;
- fake case title;
- fake assay entry;
- fake structure duplicate;
- fake near-neighbor analog;
- fake paper abstract containing the answer;
- fake text paragraph linking parent and answer.

The final audit can show:

> All seeded canaries were successfully removed before training.

This is not a substitute for real decontamination, but it strengthens the evidence.

---

## Nearest-neighbor audit

The nearest-neighbor audit is one of the most important parts of the decontamination proof.

For each held-out answer molecule, we should identify the closest remaining molecules in the training corpus under pre-defined similarity metrics.

The audit should answer:

- What is the closest remaining molecule?
- How similar is it?
- Does it share the same scaffold?
- Does it share the same target?
- Does it come from the same paper or assay family?
- Could it reveal the answer?
- Was it removed or allowed?
- Why?

This prevents the criticism:

> The model did not see the exact answer, but it saw a near-copy.

The audit does not need to remove every chemically related molecule. It needs to show that the remaining neighbors do not trivially reveal the hidden solution.

---

## Pre-registration

Before any evaluation, each case must have a pre-registered protocol.

The protocol should include:

- case ID;
- task type;
- allowed inputs;
- hidden answer;
- forbidden information;
- success criteria;
- top-k threshold;
- property thresholds;
- scoring rules;
- allowed baselines;
- failure criteria;
- output format;
- reveal procedure;
- who controls the answer;
- when predictions are locked.

This prevents moving the goalposts after seeing results.

Pre-registration is essential because if the model outputs something unexpected, there will be temptation to reinterpret success. That must be avoided.

---

## Locked predictions

A locked prediction is a prediction that cannot be changed after answer reveal.

For each evaluation run, save:

- exact input packet;
- exact system version;
- exact output;
- ranked candidates;
- reasoning or rationale if produced;
- predicted properties;
- confidence values;
- timestamp;
- hash;
- run logs;
- allowed tool state.

The locked prediction should be saved before any scoring or reveal.

If multiple runs are allowed, the protocol must say so in advance. Otherwise, a single run should count.

Do not silently rerun until the output looks good.

If reruns are part of the product design, they must be part of the protocol and counted honestly.

---

## Baseline philosophy

Baselines are not optional. Without baselines, critics can claim the tasks were obvious.

The benchmark should include simple baselines that represent plausible alternative explanations.

For ADMET, possible baselines include:

- random candidate selection;
- nearest-neighbor similarity;
- simple property heuristics;
- single-property optimization;
- known medicinal chemistry rules;
- standard off-the-shelf predictors where allowed;
- candidate ranking by drug-likeness only.

For antibiotics, possible baselines include:

- random ranking;
- similarity to known antibiotics;
- generic toxicity ranking;
- single-task antibacterial classifier;
- ranking by lipophilicity or other simple physicochemical features;
- nearest-neighbor activity transfer.

For spectra, possible baselines include:

- database search;
- formula-only filtering;
- shift-matching;
- candidate ranking without reaction context;
- simple similarity to known spectra.

The goal is not to beat every possible baseline in a fully exhaustive academic way in v1. The goal is to show that the result is not explained by trivial random selection, exact similarity, or naive heuristics.

---

## Failure reporting

The demo should not hide all failures.

If the system generates ten candidates and one succeeds, that is still a success if the protocol says the objective is top-k discovery. But the other nine should not vanish.

For each case, report:

- top-ranked candidates;
- which candidates succeeded;
- which candidates failed;
- false positives;
- near misses;
- confidence calibration;
- whether the true answer was top-1, top-3, top-5, or lower;
- whether the system proposed a functional equivalent rather than exact answer.

This makes the result more credible, not less.

Investors and scientists trust systems that can show the full trail.

---

## Functional recovery versus exact recovery

For ADMET rescue, exact recovery means the model proposes the known successful molecule.

Functional recovery means the model proposes a different molecule or strategy that satisfies the same objective.

Both are valuable, but they imply different things.

Exact recovery is easy to explain:

> The model discovered the same molecule that solved the historical case.

Functional recovery may be more scientifically interesting:

> The model found a different chemical solution to the same problem.

For v1, define success in a way that allows both, but separate them in reporting.

A good report should say:

- exact answer recovered: yes/no;
- functional rescue recovered: yes/no;
- key design move recovered: yes/no;
- measured or known liability improvement: yes/no;
- desired activity preserved: yes/no.

This prevents ambiguity.

---

## Candidate-universe clarity

A critical distinction:

### Ranking task

The answer is included in a blinded candidate set. The system must rank it highly.

This is fair and useful when the search space is known.

Claim:

> The system discovered the hidden hit from a blinded library.

### Generative task

The answer is not provided. The system must generate it or a functional equivalent.

This is harder and more open-ended.

Claim:

> The system designed a rescue candidate or antibacterial candidate.

Both are legitimate. The protocol must clearly state which one each case uses.

Do not blur the difference.

If a case is a ranking task, do not pretend the system generated the molecule from nothing. If a case is generative, do not quietly include the answer in a candidate list.

---

## The ADMET case-study design

The three ADMET cases should ideally cover different rescue types.

A strong set would include:

1. Safety/off-target liability rescue.
2. Exposure or oral-bioavailability rescue.
3. Metabolic stability, solubility, or permeability rescue.

Each case should be chosen so that the success criterion is measurable and meaningful.

Each case should include:

- starting molecule;
- desired activity;
- liability;
- allowed candidate space or generation constraints;
- hidden successful solution;
- withheld documents and assay records;
- success threshold;
- evaluation outputs.

The system should not be rewarded merely for making molecules more polar, less lipophilic, or more drug-like. It must preserve the desired function.

Potential ADMET liability categories:

- hERG/cardiac liability;
- CYP interaction;
- poor oral bioavailability;
- poor solubility;
- poor permeability;
- rapid metabolic clearance;
- poor microsomal stability;
- high cytotoxicity;
- reactive metabolite risk;
- poor selectivity;
- excessive CNS penetration when peripheral activity is desired;
- insufficient CNS penetration when brain exposure is desired.

The final choices should be based on feasibility, cleanliness of ground truth, and decontamination controllability.

---

## The antibiotic case-study design

The three antibiotic cases should ideally cover different antibacterial discovery modes.

A strong set would include:

1. Broad-spectrum or general antibacterial hit discovery.
2. Pathogen-specific or narrow-spectrum hit discovery.
3. Selectivity-aware or de novo/scaffold-hopping antibacterial discovery.

Each case should include:

- organism or assay context;
- candidate library or generation constraints;
- hidden active compounds;
- negative examples;
- cytotoxicity or selectivity context where possible;
- success threshold;
- withheld documents and assay records;
- evaluation outputs.

The system should not be rewarded for simply finding known antibiotics unless the task is explicitly a positive-control task.

The system should not be rewarded for selecting generic cytotoxins.

The system should be evaluated against random selection and simple similarity-based baselines.

Potential antibacterial task types:

- rank held-out active in a blinded repurposing library;
- rank pathogen-specific active among decoys;
- identify non-antibiotic molecule with antibacterial activity;
- propose candidates from a restricted purchasable set;
- recover active family from held-out study;
- distinguish antibacterial activity from mammalian cytotoxicity.

The final choices should be based on data quality, decontamination feasibility, and clarity of scoring.

---

## The NMR/spectra suite design

The NMR/spectra suite should provide a separate proof that Rasyn can understand experimental chemistry data.

It should include more than six examples because structure elucidation benefits from statistical evaluation.

A good initial suite might include:

- clean pure-compound cases;
- spectra with close isomer decoys;
- crude reaction mixtures;
- impurity cases;
- cases with incomplete spectra;
- cases with LC-MS plus NMR;
- cases where uncertainty is appropriate.

The suite should evaluate:

- exact structure;
- top-k structure;
- molecular formula;
- scaffold;
- impurity;
- major product;
- confidence calibration;
- suggested confirmatory experiment.

The most valuable cases are not clean textbook spectra. They are messy, real, and experimentally relevant.

The NMR suite should also be subject to leakage controls if any spectra or structures are held out from training.

---

## Investor-demo philosophy

The investor demo should feel like a stress test, not a magic trick.

The strongest framing:

> We built this so you can tear it apart.

Show:

- sealed cases;
- decontamination evidence;
- locked predictions;
- full candidate rankings;
- baselines;
- raw or source evidence;
- scoring script;
- failures and false positives;
- NMR blind tests;
- one or more external validations if available.

Do not show only the winner. Show the process.

The investor should walk away believing:

1. The system did not see the answer.
2. The evaluation was defined before the result.
3. The predictions were locked.
4. The result happened across multiple cases.
5. The system is not just a molecule generator.
6. The team understands the difference between early discovery and clinical proof.
7. The system is ready for prospective validation.

---

## What the final demo should not overclaim

Even if all six cases succeed, the demo does not prove:

- Rasyn can rescue any drug.
- Rasyn can discover approved antibiotics.
- Rasyn has solved clinical translation.
- Rasyn can replace chemists.
- Rasyn can handle every ADMET liability.
- Rasyn can discover molecules without experimental validation.
- Rasyn is guaranteed to work on private customer programs.

The demo does prove, if executed correctly:

- Rasyn can learn general chemistry from a controlled corpus.
- Rasyn can solve held-out discovery tasks it did not see.
- Rasyn can recover successful chemistry decisions across multiple task types.
- Rasyn can operate under audit.
- Rasyn can support real design-make-test-analyze workflows.
- Rasyn can be evaluated honestly rather than through cherry-picked screenshots.

---

## Recommended final language

Use language like:

> We trained Rasyn from scratch under strict data-contamination controls. Before training, six discovery cases were sealed and removed from the corpus along with their leakage neighborhoods. After training, Rasyn was given only the allowed blinded problem statements. It discovered the successful solutions across all six held-out ADMET and antibiotic studies. Separately, Rasyn solved a suite of blinded NMR/LC-MS structure-identification tasks, showing that the same system can reason over experimental evidence.

More technical version:

> We present a contamination-controlled held-out discovery benchmark. Six case studies were pre-registered before training. Direct molecules, identifiers, assay records, source documents, analog neighborhoods, and leakage routes were excluded. Predictions were locked before reveal. Evaluation measured exact recovery, functional recovery, top-k ranking, property-objective satisfaction, and baseline enrichment across ADMET rescue and antibacterial discovery tasks.

Short investor version:

> We did not ask the AI to remember famous drug stories. We hid the answers before training, trained the system from scratch, and asked it to solve them. It discovered all six.

---

## Coding-agent instructions

Any coding agent working on this project should follow these principles:

1. Preserve auditability over convenience.
2. Never silently change evaluation rules after seeing results.
3. Never mix hidden answers into training or inference context.
4. Never assume a molecule is safe to include just because its name is absent.
5. Treat synonyms, salts, tautomers, stereoisomers, metabolites, and identifiers as potential leakage.
6. Treat documents, assays, patents, review articles, and derived datasets as potential leakage.
7. Every generated artifact should be reproducible.
8. Every evaluation run should be logged.
9. Every prediction should be lockable.
10. Every success should be traceable to a pre-registered criterion.
11. Every case should have a human-readable and machine-readable record.
12. Avoid adding architecture assumptions to benchmark code.
13. Avoid embedding dataset choices into the concept layer.
14. Separate challenge definition, decontamination, inference, scoring, and reporting.
15. Prefer explicit manifests over implicit folder conventions.
16. Keep raw data, cleaned data, held-out data, and evaluation packets clearly separated.
17. Make failure states obvious.
18. Make leakage impossible to ignore.
19. Make it easy for a skeptical reviewer to reproduce the audit trail.
20. If unsure whether something is leakage, flag it rather than silently include it.

---

## Suggested repository concept map

This is not a required final repository architecture. It is a conceptual map of concerns that should remain separate.

```text
project_context/
  high_level_context.md
  claim_policy.md
  terminology.md

sealed_cases/
  case_registry/
  forbidden_entities/
  pre_registration/
  reveal_materials/

decontamination/
  entity_expansion/
  exact_match_checks/
  similarity_neighborhood_checks/
  document_quarantine/
  assay_quarantine/
  leakage_reports/
  canary_tests/

evaluation/
  inference_packets/
  locked_predictions/
  scoring_rules/
  baselines/
  final_scores/
  run_logs/

reports/
  investor_summary/
  technical_appendix/
  audit_report/
  case_study_cards/
  spectra_blind_suite_report/
```

Again, this is a separation-of-concerns guide, not a final implementation mandate.

---

## Case record requirements

Every case should eventually have a record containing:

- case ID;
- task family;
- task subtype;
- hidden answer;
- allowed input;
- forbidden information;
- success criterion;
- scoring metric;
- baseline list;
- decontamination rules;
- evaluator notes;
- reveal notes;
- final outcome;
- locked prediction references;
- audit report references.

A case should not be run if its record is incomplete.

---

## Pre-registration template

Each case should have a pre-registration document like:

```markdown
# Case ID

## Task family

ADMET rescue / antibiotic discovery / spectra identification

## Task objective

What the system must do.

## Allowed input

Exactly what the system receives.

## Hidden answer

What is withheld until reveal.

## Forbidden information

Molecules, documents, assays, identifiers, analog families, text references.

## Decontamination level

Exact / case-family / neighborhood / document / assay / spectra.

## Candidate universe

Ranking library or generative constraints.

## Success criteria

Exact top-k, functional recovery, property thresholds, hit thresholds, etc.

## Baselines

Random, similarity, heuristic, simple predictor, etc.

## Prediction lock procedure

How outputs are saved before reveal.

## Reveal procedure

Who reveals and when.

## Failure criteria

What counts as failure.

## Notes

Any caveats.
```

---

## Locked prediction template

Each locked prediction should eventually include:

```markdown
# Locked Prediction

## Case ID

## Run ID

## Timestamp

## System version

## Input packet hash

## Output hash

## Candidate rankings

## Proposed structures

## Predicted properties

## Confidence

## Rationale

## Any warnings or uncertainty

## Allowed tools

## Notes
```

The locked prediction should be immutable after reveal.

---

## Technical appendix requirements

The final technical appendix should include:

1. Study overview.
2. Case-selection rationale.
3. Decontamination protocol.
4. Forbidden entity expansion.
5. Similarity-neighborhood rules.
6. Document and assay quarantine rules.
7. Training corpus manifest, at least at a high level.
8. Held-out case registry.
9. Pre-registration documents.
10. Locked prediction hashes.
11. Scoring rules.
12. Baselines.
13. Per-case results.
14. NMR/spectra suite results.
15. Failure analysis.
16. Limitations.
17. Prospective validation plan.

This appendix is what makes the demo serious.

---

## Limitations we should state openly

The demo should openly state limitations.

Possible limitations:

- Six cases do not cover all chemistry.
- Historical held-out discovery is not the same as fully prospective customer discovery.
- Decontamination can be extremely strong but cannot prove philosophical absence of all indirect general knowledge.
- Functional recovery may require expert interpretation if not exact.
- Antibiotic hits are early-stage, not clinical drugs.
- ADMET rescue depends on the quality and relevance of known or measured endpoints.
- Spectra identification depends on available spectral modalities and sample quality.
- The system should be viewed as decision support and discovery acceleration, not autonomous replacement of laboratory validation.

These limitations do not weaken the demo if framed correctly. They make the team look honest.

---

## What success looks like

The ideal final result:

- Three ADMET rescue cases succeed.
- Three antibiotic discovery cases succeed.
- NMR/spectra blind suite shows strong top-k performance and impurity reasoning.
- Decontamination audit is clean.
- Baselines are weaker.
- Predictions are locked.
- Failure modes are disclosed.
- At least one result has fresh external or wet-lab validation, if feasible.
- The final report is strong enough to survive technical investor diligence.

The strongest single slide:

| Pillar | Number of cases | Blind? | Held out before training? | Locked predictions? | Result |
|---|---:|---|---|---|---|
| ADMET rescue | 3 | Yes | Yes | Yes | 3/3 discovered |
| Antibiotic discovery | 3 | Yes | Yes | Yes | 3/3 discovered |
| NMR/spectra | many | Yes | Yes where applicable | Yes | strong top-k / impurity results |

The strongest spoken line:

> We built the benchmark so skeptics could attack it. The answers were hidden before training, the system was trained from scratch, the predictions were locked before reveal, and the same system succeeded across six discovery studies.

---

## Open decisions intentionally left unresolved

This document intentionally does not decide:

- the model architecture;
- the exact training method;
- the exact datasets;
- the exact six final cases;
- the final similarity thresholds;
- the exact candidate-generation method;
- the exact baselines;
- the final experimental partners;
- the final user interface;
- the final repo layout;
- the final deployment format.

Those decisions should be made later with this document as context.

The important thing is that later decisions must preserve the central idea:

> contamination-controlled held-out discovery across multiple chemistry tasks, with locked predictions and auditable evidence.

---

## Final reminder for all agents

Do not optimize for a demo that merely looks good.

Optimize for a demo that survives attack.

The final system must be able to answer:

- What was hidden?
- When was it hidden?
- How do we know it was hidden?
- What did the system see?
- What did the system output?
- When was the output locked?
- How was success defined before reveal?
- What baselines were used?
- What failed?
- Why is this discovery and not memorization?
- Why is this meaningful chemistry and not a toy benchmark?

If a future implementation choice makes any of those questions harder to answer, it is probably the wrong choice.

The project succeeds when a skeptical investor, chemist, or technical reviewer can tear through the audit trail and still conclude:

> This was not cherry-picked. This was not memorized. This system actually solved held-out chemistry discovery tasks.

