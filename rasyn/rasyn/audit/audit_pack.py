"""Audit pack assembler: bundles every artifact into one JSON for review.

Inputs (all already on disk from earlier phases):
  - sealed_case_registry.yaml
  - decontam_audit_pre.json + decontam_audit_post.json
  - canary_report.json
  - nearest_neighbor_table.csv (path only, not inlined)
  - dataset_manifest.json
  - training_manifest.json (per stage)
  - locked_predictions/{admet_001,002,003}.json

Output:
  - audit_pack.json containing every hash + a flat summary row per case
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from rasyn.audit.locked_prediction_io import read_locked_prediction
from rasyn.schemas.hashing import sha256_hex


def _file_sha256(path: Path) -> str:
    return sha256_hex(path.read_bytes())


def assemble_audit_pack(
    *,
    sealed_case_registry: Path,
    decontam_pre: Path,
    decontam_post: Path,
    canary_report: Path,
    nearest_neighbor_table: Path | None,
    dataset_manifest: Path,
    training_manifests: list[Path],
    locked_predictions: list[Path],
    output_path: Path,
) -> dict:
    """Assemble + write audit_pack.json. Returns the dict that was written."""
    pack: dict = {
        "assembled_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "artifact_hashes": {},
        "cases": {},
    }

    for label, path in [
        ("sealed_case_registry", sealed_case_registry),
        ("decontam_pre", decontam_pre),
        ("decontam_post", decontam_post),
        ("canary_report", canary_report),
        ("dataset_manifest", dataset_manifest),
    ]:
        pack["artifact_hashes"][label] = {"path": str(path), "sha256": _file_sha256(path)}

    if nearest_neighbor_table is not None:
        pack["artifact_hashes"]["nearest_neighbor_table"] = {
            "path": str(nearest_neighbor_table),
            "sha256": _file_sha256(nearest_neighbor_table),
        }

    pack["artifact_hashes"]["training_manifests"] = [
        {"path": str(p), "sha256": _file_sha256(p)} for p in training_manifests
    ]

    for lp_path in locked_predictions:
        lp = read_locked_prediction(lp_path)  # raises on hash mismatch
        pack["cases"][lp.case_id] = {
            "locked_at_utc": lp.locked_at_utc,
            "system_version": lp.system_version,
            "model_checkpoint_hash": lp.model_checkpoint_hash,
            "challenge_packet_hash": lp.challenge_packet_hash,
            "output_hash": lp.output_hash,
            "top_5": lp.top_k_locked.get("5", []),
            "top_10": lp.top_k_locked.get("10", []),
            "top_20": lp.top_k_locked.get("20", []),
            "predictions_count": len(lp.predictions),
            "warnings": lp.warnings,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pack, indent=2, sort_keys=True), encoding="utf-8")
    return pack
