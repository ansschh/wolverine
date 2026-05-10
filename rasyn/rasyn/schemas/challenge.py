"""ADMETChallengePacket — the per-case input the system receives at inference.

This is the ONLY thing the system gets at inference for a sealed case. It must
not contain: the answer molecule, hidden measured candidate outcomes, paper
text, or anything in the case's `forbidden_*` lists. The challenge packet is
hashed and the hash goes into the locked prediction.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rasyn.schemas.registry import LiabilityType, RescueMode

PotencyUnit = Literal["nM", "uM", "mM", "pIC50", "pKi", "pEC50", "pKd"]

LiabilityCategory = Literal["low", "moderate", "high", "critical", "unknown"]


class ActivityContext(BaseModel):
    """What target / pharmacology the candidate must preserve."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_name: str
    target_chembl_id: str | None = None
    desired_pharmacology: str
    parent_potency_value: float
    parent_potency_unit: PotencyUnit
    parent_potency_endpoint: str = Field(..., description="IC50, Ki, EC50, Kd, etc.")
    assay_type: str | None = None

    # Activity-retention tradeoff thresholds (folds vs. parent potency).
    # Defaults from `rasyn_admet_conditioning_architecture_benchmark_spec.md`.
    strong_retention_fold: float = 3.0
    acceptable_retention_fold: float = 10.0
    failed_retention_fold: float = 100.0


class LiabilityContext(BaseModel):
    """The liability the candidate must improve."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    liability_type: LiabilityType
    measurement_endpoint: str = Field(..., description="e.g. 'hERG IC50', 'aqueous solubility', 'human microsomal half-life'.")
    parent_value: float | None = None
    parent_unit: str | None = None
    parent_category: LiabilityCategory = "unknown"
    target_improvement_fold: float | None = Field(default=None, description="Numeric target (e.g. 5.0 for 5x solubility).")
    target_improvement_category: LiabilityCategory | None = Field(default=None, description="Categorical target (e.g. 'low risk').")


class RescueContextPacket(BaseModel):
    """The rescue mode + any constraints / forbidden strategy clues.

    `forbidden_strategy_clues` lists strings that would leak the answer
    (e.g. 'try a carboxylic acid version' for terfenadine). These must be
    omitted from any free-form context handed to the system.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rescue_mode: RescueMode
    forbidden_strategy_clues: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    goal_description: str | None = None


class ADMETChallengePacket(BaseModel):
    """The per-case task input. Hashed and referenced by the locked prediction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    parent_canonical_smiles: str
    parent_inchi_key: str
    activity_context: ActivityContext
    liability_context: LiabilityContext
    rescue_context: RescueContextPacket
    schema_version: str = "0.1.0"
