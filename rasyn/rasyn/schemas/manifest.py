"""Dataset + training manifests. Hashed and referenced by every downstream artifact."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FileEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sha256: str
    size_bytes: int
    rows: int | None = None


class DatasetManifest(BaseModel):
    """Frozen manifest of the clean 4-table dataset + decontamination audits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    frozen_at_utc: str
    sealed_case_registry_hash: str
    decontamination_config_hash: str
    sources: dict[str, str] = Field(
        ..., description="Source name -> source-specific version (e.g. {'chembl': '35', 'tdc': '0.4.1'})"
    )
    tables: dict[str, FileEntry] = Field(..., description="Logical table name -> file entry.")
    audits: dict[str, FileEntry] = Field(..., description="e.g. 'decontam_pre', 'decontam_post', 'canary_report'.")
    notes: str | None = None


class TrainingManifest(BaseModel):
    """Frozen record of what produced a model checkpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    frozen_at_utc: str
    stage: str = Field(..., description="'pretrain' | 'main_train' | 'finetune_<mode>' | 'calibrate'")
    sealed_case_registry_hash: str
    dataset_manifest_hash: str
    decontamination_config_hash: str
    proposer_config_hash: str | None = None
    ranker_config_hash: str
    code_git_sha: str | None = None
    seed: int
    checkpoint: FileEntry
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: str | None = None
