"""Frozen Pydantic v2 schemas — the project contract.

Every cross-component data shape is defined here. Schemas are immutable
(`frozen=True`) and hashed via canonical-JSON SHA256 for audit trails.

Imports are organised by concern:

- `hashing` — canonical_json + sha256 helpers used to hash any schema instance
- `molecule` — MoleculeRef and shared molecule-identifier types
- `registry` — SealedCaseRegistry, ForbiddenEntities (Phase A-0)
- `challenge` — ADMETChallengePacket (the per-case task input)
- `proposer` — ProposerRequest, ProposerOutput, CandidateAnnotation
- `evidence` — CandidateEvidencePacket
- `ranker` — RankerInput, RankerOutput
- `locked` — LockedPrediction (immutable per-case prediction record)
- `config` — DecontaminationConfig, BaselineConfig, ProposerConfig, RankerConfig
- `manifest` — DatasetManifest, TrainingManifest
"""

from rasyn.schemas.hashing import canonical_json, hash_model, sha256_hex

__all__ = ["canonical_json", "hash_model", "sha256_hex"]
