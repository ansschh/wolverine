"""LockedPrediction — the immutable per-case prediction record.

Created BEFORE the answer is revealed. Hash-stable. Once written, never
modified. The audit pack reads these to score the system after reveal.

Locked predictions are the canonical evidence that the system "discovered"
the answer rather than memorised it. See spec §10.4 (locked prediction format)
and `rasyn_heldout_discovery_demo_context.md` §65-76 (the 8 conditions).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rasyn.schemas.ranker import RankerOutput


class LockedPrediction(BaseModel):
    """One immutable per-case prediction record. Hash-stable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    locked_at_utc: str = Field(..., description="ISO 8601 UTC timestamp of lock.")
    system_version: str = Field(..., description="rasyn package semver at lock time.")
    sealed_case_registry_hash: str
    challenge_packet_hash: str
    dataset_manifest_hash: str
    training_manifest_hash: str
    model_checkpoint_hash: str
    predictions: list[RankerOutput] = Field(..., description="Full ranking, not just top-k.")
    top_k_locked: dict[str, list[str]] = Field(
        ..., description="Maps '5'/'10'/'20' to ordered lists of candidate_ids."
    )
    output_hash: str = Field(..., description="SHA256 of canonical JSON of `predictions`.")
    notes: str | None = None
    warnings: list[str] = Field(default_factory=list)
