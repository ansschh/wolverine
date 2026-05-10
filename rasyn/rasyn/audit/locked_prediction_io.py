"""Write + verify immutable LockedPrediction JSON files.

Locked predictions ARE the audit trail for the discovered-vs-memorised claim.
Once written they must never be modified. Reading verifies the recorded
output_hash matches the canonical-JSON hash of `predictions`.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Sequence

from rasyn.schemas.hashing import canonical_json, hash_model, sha256_hex
from rasyn.schemas.locked import LockedPrediction
from rasyn.schemas.ranker import RankerOutput


def build_locked_prediction(
    *,
    case_id: str,
    system_version: str,
    sealed_case_registry_hash: str,
    challenge_packet_hash: str,
    dataset_manifest_hash: str,
    training_manifest_hash: str,
    model_checkpoint_hash: str,
    predictions: Sequence[RankerOutput],
    notes: str | None = None,
    warnings: list[str] | None = None,
) -> LockedPrediction:
    """Compose + finalise a LockedPrediction from a ranking. Stamps timestamp + output_hash."""
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    payload = canonical_json([p.model_dump(mode="json") for p in predictions])
    output_hash = sha256_hex(payload)

    top_k_locked: dict[str, list[str]] = {}
    sorted_preds = sorted(predictions, key=lambda p: p.rank or 10**9)
    for k in (5, 10, 20):
        top_k_locked[str(k)] = [p.candidate_id for p in sorted_preds[:k]]

    return LockedPrediction(
        case_id=case_id,
        locked_at_utc=timestamp,
        system_version=system_version,
        sealed_case_registry_hash=sealed_case_registry_hash,
        challenge_packet_hash=challenge_packet_hash,
        dataset_manifest_hash=dataset_manifest_hash,
        training_manifest_hash=training_manifest_hash,
        model_checkpoint_hash=model_checkpoint_hash,
        predictions=list(predictions),
        top_k_locked=top_k_locked,
        output_hash=output_hash,
        notes=notes,
        warnings=warnings or [],
    )


def write_locked_prediction(lp: LockedPrediction, path: Path | str) -> str:
    """Write the locked prediction to disk and return its model-level SHA256."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(f"Locked prediction already exists at {p}; refusing to overwrite.")
    p.write_text(canonical_json(lp.model_dump(mode="json")) + "\n", encoding="utf-8")
    return hash_model(lp)


def read_locked_prediction(path: Path | str) -> LockedPrediction:
    """Load + verify a locked prediction. Raises if output_hash mismatches."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    lp = LockedPrediction.model_validate(raw)
    payload = canonical_json([p for p in raw["predictions"]])
    expected = sha256_hex(payload)
    if expected != lp.output_hash:
        raise ValueError(
            f"Locked prediction output_hash mismatch at {path}: "
            f"recorded={lp.output_hash[:16]}...; computed={expected[:16]}..."
        )
    return lp
