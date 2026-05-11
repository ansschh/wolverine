"""Deterministic structured-rationale generator (spec §7.2, §7.3).

Builds the allowed rationale fields from numeric evidence ONLY — no LLM,
no free-form chain-of-thought, no mechanistic claims (which are explicitly
forbidden by §7.3).

Allowed fields (from spec §7.3):
  predicted_activity_basis
  selectivity_basis
  novelty_basis
  synthesizability_basis
  risk_flags
  nearest_training_neighbors
  possible_failure_modes

Each rationale entry is a short string drawn from a fixed template set,
populated with measured scores. The rule generator is deterministic so
identical inputs produce identical rationales.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StructuredRationale:
    predicted_activity_basis: list[str] = field(default_factory=list)
    selectivity_basis: list[str] = field(default_factory=list)
    novelty_basis: list[str] = field(default_factory=list)
    synthesizability_basis: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    nearest_training_neighbors: list[str] = field(default_factory=list)
    possible_failure_modes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "predicted_activity_basis":      self.predicted_activity_basis,
            "selectivity_basis":             self.selectivity_basis,
            "novelty_basis":                 self.novelty_basis,
            "synthesizability_basis":        self.synthesizability_basis,
            "risk_flags":                    self.risk_flags,
            "nearest_training_neighbors":    self.nearest_training_neighbors,
            "possible_failure_modes":        self.possible_failure_modes,
        }


def build_rationale(
    *,
    organism: str,
    antibacterial_score: float,
    cytotox_risk: float,
    artifact_risk: float,
    failure_mode_probs: dict[str, float] | None = None,
    nearest_known_antibiotic_similarity: float | None = None,
    nearest_training_active_similarity: float | None = None,
    synthesizability_score: float | None = None,
    novelty_score: float | None = None,
    uncertainty_score: float | None = None,
    nearest_training_neighbor_ids: list[str] | None = None,
) -> StructuredRationale:
    r = StructuredRationale()

    # Predicted activity basis — only fact-based statements, no mechanism.
    if antibacterial_score >= 0.7:
        r.predicted_activity_basis.append(
            f"high organism-conditioned antibacterial score for {organism}: {antibacterial_score:.2f}"
        )
    elif antibacterial_score >= 0.4:
        r.predicted_activity_basis.append(
            f"moderate antibacterial score for {organism}: {antibacterial_score:.2f}"
        )
    else:
        r.predicted_activity_basis.append(
            f"low antibacterial score for {organism}: {antibacterial_score:.2f}"
        )

    if nearest_known_antibiotic_similarity is not None:
        if nearest_known_antibiotic_similarity < 0.35:
            r.novelty_basis.append(
                f"low Tanimoto similarity to known antibiotics: {nearest_known_antibiotic_similarity:.2f}"
            )
            r.predicted_activity_basis.append("non-obvious to a known-antibiotic kNN baseline")
        elif nearest_known_antibiotic_similarity >= 0.7:
            r.risk_flags.append(
                f"close to known antibiotic (Tanimoto={nearest_known_antibiotic_similarity:.2f}) — novelty penalty"
            )

    if nearest_training_active_similarity is not None and nearest_training_active_similarity >= 0.85:
        r.risk_flags.append(
            f"near-duplicate of training active (Tanimoto={nearest_training_active_similarity:.2f})"
        )

    # Selectivity basis
    if cytotox_risk <= 0.2:
        r.selectivity_basis.append(f"low predicted mammalian cytotoxicity: {cytotox_risk:.2f}")
    elif cytotox_risk >= 0.5:
        r.risk_flags.append(f"elevated predicted cytotoxicity: {cytotox_risk:.2f}")
    if artifact_risk <= 0.2:
        r.selectivity_basis.append(f"low predicted assay artifact / reactive risk: {artifact_risk:.2f}")
    elif artifact_risk >= 0.5:
        r.risk_flags.append(f"elevated predicted artifact / reactive risk: {artifact_risk:.2f}")

    # Synthesizability
    if synthesizability_score is not None:
        if synthesizability_score >= 0.65:
            r.synthesizability_basis.append(f"plausible synthesizability score: {synthesizability_score:.2f}")
        elif synthesizability_score <= 0.3:
            r.risk_flags.append(f"low synthesizability score: {synthesizability_score:.2f}")

    # Novelty score (if separate from kNN similarity)
    if novelty_score is not None and novelty_score >= 0.7:
        r.novelty_basis.append(f"high novelty score: {novelty_score:.2f}")

    # Uncertainty
    if uncertainty_score is not None and uncertainty_score >= 0.3:
        r.risk_flags.append(f"moderate-to-high ensemble uncertainty: {uncertainty_score:.2f}")

    # Failure-mode probabilities → possible_failure_modes (per §16.7 categories)
    if failure_mode_probs:
        for mode, p in failure_mode_probs.items():
            if p >= 0.2:  # surfaced if non-trivial probability
                r.possible_failure_modes.append(f"{mode}: probability {p:.2f}")

    if nearest_training_neighbor_ids:
        r.nearest_training_neighbors.extend(nearest_training_neighbor_ids[:5])

    return r
