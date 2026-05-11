"""Rasyn antibiotic discovery system.

Parallel to rasyn/ ADMET infrastructure but specialized for organism-conditioned
antibacterial discovery per `rasyn_antibiotic_discovery_architecture_benchmark_spec.md`.

Sealed cases (per spec §14):
- ABX-001 (halicin-style broad-spectrum repurposing)
- ABX-002 (abaucin-style A. baumannii-specific)
- ABX-003 (de novo / scaffold-hopping active-family recovery)

Five core data tables (per spec §8):
- molecule
- antibacterial_assay_fact
- counter_screen_fact (cytotoxicity, hemolysis, aggregation, artifact)
- antibiotic_ranking_task (organism-conditioned listwise ranking targets)
- generative_training_example (fragment + edit pretraining)

Seven proposer channels (per spec §10):
- A: repurposing retriever
- B: screened-library retriever
- C: organism-specific analog
- D: scaffold-hopping
- E: fragment-conditioned diffusion (learnable)
- F: phenotype-conditioned edit diffusion (learnable)
- G: diversity/novelty selector

Ranker (per spec §12): multi-head with organism-conditioned antibacterial, selectivity,
cytotoxicity/hemolysis/artifact risks, known-antibiotic + training-active penalties,
novelty, synthesizability, uncertainty, and failure-mode probabilities.

Per L25: no fallbacks/placeholders for trained components.
Per L33: quality > quantity in data curation.
"""
