"""Stage-4 calibration: failure-mode + confidence calibration on held-out pairs.

Isotonic regression (sklearn) on each prob head independently. Output is a
calibrator pickle keyed by head; loaded at inference and applied to logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CalibrateConfig:
    checkpoint: Path
    held_out_parquet: Path
    out_calibrators: Path = Path("rasyn/data/clean/checkpoints/calibrators.pkl")


def calibrate(cfg: CalibrateConfig):
    raise NotImplementedError(
        "Stage-4 calibration. Activates after Stage-2/3 produce predictions "
        "on held-out (non-sealed) pairs."
    )
