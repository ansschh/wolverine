"""Run-time configurations: decontamination, baselines, proposer, ranker.

These are frozen + hashed so that any artifact (dataset, model, locked
prediction) can reference the exact configs that produced it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DecontaminationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strictness: Literal["loose", "medium", "strict"] = "strict"
    tanimoto_to_answer: float = Field(default=0.85, ge=0.0, le=1.0)
    tanimoto_with_context: float = Field(default=0.65, ge=0.0, le=1.0)
    require_same_murcko_for_context: bool = True
    require_same_target_for_context: bool = True
    apply_synonym_scrub: bool = True
    apply_document_quarantine: bool = True
    apply_assay_quarantine: bool = True
    canary_enforcement: Literal["halt_on_survivor", "warn_on_survivor"] = "halt_on_survivor"


class BaselineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: list[
        Literal[
            "random",
            "similarity_only",
            "most_polar",
            "liability_only_property",
            "activity_only",
            "weighted_property",
            "mmp_frequency",
            "medchem_heuristic",
        ]
    ] = Field(
        default_factory=lambda: [
            "random",
            "similarity_only",
            "most_polar",
            "liability_only_property",
            "activity_only",
            "weighted_property",
            "mmp_frequency",
            "medchem_heuristic",
        ]
    )
    seed: int = 42


class ProposerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target_pool_size_min: int = 5_000
    target_pool_size_max: int = 20_000
    filtered_pool_size_max: int = 2_000
    enable_analog_retrieval: bool = True
    enable_mmp_transformer: bool = True
    enable_liability_rules: bool = True
    enable_learned_inverse_delta: bool = True
    enable_forward_reward_optimizer: bool = True
    enable_learned_novelty: bool = True
    invalid_rate_warn_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class RankerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backbone: Literal["concat_mlp", "gnn", "transformer"] = "concat_mlp"
    hidden_dim: int = 512
    n_layers: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 64
    hard_negative_ratio: float = Field(default=1.5, description="Hard negatives per positive.")
    auxiliary_loss_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "rescue_label": 1.0,
            "activity_retention": 0.5,
            "liability_improvement": 0.5,
            "failure_mode": 0.3,
            "structured_rationale": 0.2,
        }
    )
