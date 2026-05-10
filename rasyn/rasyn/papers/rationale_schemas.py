"""Pydantic schemas for ChEMBL-pair rationale enrichment (P-1e).

For each (parent, candidate) pair from rescue_pair_candidates.parquet, the
LLM produces a `PairRationale` describing the medicinal-chemistry
transformation. NO PAPER TEXT REQUIRED — input is purely SMILES + metrics.

Output structure follows the spec rasyn_curating_the_dataset.md §13:

    structured_rationale:
      liability_driver: ["high_logD", "basic_amine"]
      preserved_activity_features: ["aryl_core", "h_bond_acceptor"]
      transformation_class: ["basicity_tuning", "polarity_increase"]
      expected_mechanism:
        liability_improvement: "..."
        activity_retention: "..."
      evidence_strength: "rule_based_plus_measured_delta"

Per L25: schema is strict; LLM output failing the schema is dropped.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Open enums — LLM may emit any string; downstream stats track frequency.
# We don't constrain to a fixed Literal because new transformations get
# discovered (the whole point of mining).

EvidenceStrength = Literal[
    "structure_only",                # Only SMILES + descriptors
    "structure_plus_measured_delta", # + measured ADMET + activity values
    "structure_plus_known_motif",    # + recognized canonical motif (e.g. valyl ester)
    "uncertain",
]


class ExpectedMechanism(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    liability_improvement: str = Field(
        description="One-sentence mechanistic explanation of why the candidate "
                    "should improve the liability."
    )
    activity_retention: str = Field(
        description="One-sentence explanation of why activity should be preserved."
    )


class PairRationale(BaseModel):
    """LLM-derived structured rationale for one (parent, candidate) rescue pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str = Field(description="Echoed from input.")

    transformation_class: list[str] = Field(
        description="Medicinal-chemistry transformation labels. Examples: "
                    "'phenyl_to_pyridyl_bioisostere', 'fluoro_shielding', "
                    "'soft_spot_block', 'basicity_reduction', 'polar_addition', "
                    "'prodrug_ester', 'ring_closure', 'methyl_to_trifluoromethyl'. "
                    "Use snake_case. Multi-label allowed."
    )

    liability_driver: list[str] = Field(
        default_factory=list,
        description="Structural / physicochemical features in the PARENT that "
                    "drive the liability. Examples: 'high_lipophilicity', "
                    "'basic_amine', 'aromatic_ring_count_high', 'low_TPSA', "
                    "'metabolically_labile_methyl', 'phosphate_charge_at_pH7'."
    )

    preserved_activity_features: list[str] = Field(
        default_factory=list,
        description="Structural features conserved between parent and candidate "
                    "that explain retained target activity. Examples: 'aryl_core', "
                    "'key_hbond_acceptor', 'pharmacophore_shape', 'basic_center_distance'."
    )

    expected_mechanism: ExpectedMechanism = Field(
        description="Mechanistic narrative in two sentences."
    )

    evidence_strength: EvidenceStrength = Field(
        description="How confident the rationale is, given the input signals."
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="LLM-flagged concerns: e.g. 'transformation also adds metabolic "
                    "soft spot', 'measured liability_improvement is borderline', "
                    "'parent and candidate differ in chiral center stereochemistry'."
    )

    @field_validator("transformation_class")
    @classmethod
    def _at_least_one_transformation(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("transformation_class must contain at least one label.")
        # Normalise to snake_case-ish: lowercase + replace spaces with underscores.
        return [t.strip().lower().replace(" ", "_") for t in v if t.strip()]


class PairRationaleBatch(BaseModel):
    """Wrapper for batch outputs (one row per pair processed)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rationale: PairRationale
    extraction_timestamp_utc: str
    model_id: str
    prompt_sha256: str
    extraction_runtime_ms: int | None = None
