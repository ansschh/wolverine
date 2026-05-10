"""PyTorch datasets for the rescue ranker + auxiliary predictors.

Imports torch lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from rasyn.ranker.featurize import featurize
from rasyn.schemas.evidence import CandidateEvidencePacket

_RESCUE_LABELS = (
    "strong_success",
    "weak_success",
    "failed_activity_loss",
    "failed_no_liability_improvement",
    "failed_wrong_liability",
    "failed_new_liability",
    "uncertain",
)


def label_to_index(label: str) -> int:
    return _RESCUE_LABELS.index(label)


def index_to_label(idx: int) -> str:
    return _RESCUE_LABELS[idx]


class RescuePairDataset:
    """In-memory rescue-pair dataset.

    Each item:
      x: feature tensor (float32, length TOTAL_INPUT_DIM)
      y_rescue_score: scalar in [0, 1]
      y_rescue_label_idx: int in [0, 7)
      y_failure_mode_mask: 6-vector of {0, 1}
      y_retention_bucket_idx: int in [0, 5)
      y_improvement_idx: int in [0, 6)

    Wrap with `torch_dataset(...)` to get a real torch Dataset.
    """

    def __init__(
        self,
        evidence_packets: list[CandidateEvidencePacket],
        labels: list[dict],
    ):
        if len(evidence_packets) != len(labels):
            raise ValueError("evidence_packets and labels length mismatch")
        self._features = np.stack([featurize(ev) for ev in evidence_packets])
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def __iter__(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self._row(i)

    def _row(self, i: int) -> dict:
        lab = self._labels[i]
        return {
            "x": self._features[i],
            "y_rescue_score": float(lab.get("rescue_score", 0.0)),
            "y_rescue_label_idx": label_to_index(lab.get("rescue_label", "uncertain")),
            "y_failure_mode_mask": np.asarray(lab.get("failure_mode_mask", [0, 0, 0, 0, 0, 0]), dtype=np.float32),
            "y_retention_bucket_idx": int(lab.get("retention_bucket_idx", 4)),
            "y_improvement_idx": int(lab.get("improvement_idx", 5)),
        }


def torch_dataset(rpd: RescuePairDataset):
    """Return a torch Dataset wrapping the in-memory RescuePairDataset."""
    import torch
    from torch.utils.data import Dataset

    class _TorchRPD(Dataset):
        def __len__(self):
            return len(rpd)

        def __getitem__(self, idx):
            row = rpd._row(idx)
            return {
                "x": torch.from_numpy(row["x"]),
                "y_rescue_score": torch.tensor(row["y_rescue_score"], dtype=torch.float32),
                "y_rescue_label_idx": torch.tensor(row["y_rescue_label_idx"], dtype=torch.long),
                "y_failure_mode_mask": torch.from_numpy(row["y_failure_mode_mask"]),
                "y_retention_bucket_idx": torch.tensor(row["y_retention_bucket_idx"], dtype=torch.long),
                "y_improvement_idx": torch.tensor(row["y_improvement_idx"], dtype=torch.long),
            }

    return _TorchRPD()


def load_parquet(path: Path | str):
    """Load a clean rescue-pair parquet into the in-memory dataset.

    Expected columns include the parent + candidate evidence packet fields
    plus label columns. Loader unpacks via Pydantic for validation.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    return df  # caller does the unpack; this is a thin wrapper for now
