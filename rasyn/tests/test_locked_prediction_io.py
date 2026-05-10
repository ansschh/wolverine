"""Locked-prediction I/O: write-then-read round trip + hash verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rasyn.audit.locked_prediction_io import (
    build_locked_prediction,
    read_locked_prediction,
    write_locked_prediction,
)
from rasyn.schemas.evidence import (
    ActivityRetentionEvidence,
    LiabilityEvidence,
    StructuredRationale,
)
from rasyn.schemas.ranker import ConfidenceBlock, RankerOutput


def _mk_ranker_output(cid: str, score: float, rank: int) -> RankerOutput:
    return RankerOutput(
        case_id="ADMET-001",
        candidate_id=cid,
        rescue_score=score,
        rescue_label_probs={
            "strong_success": 0.4, "weak_success": 0.3, "failed_activity_loss": 0.05,
            "failed_no_liability_improvement": 0.05, "failed_wrong_liability": 0.05,
            "failed_new_liability": 0.05, "uncertain": 0.1,
        },
        failure_mode_probs={
            "activity_lost": 0.1, "liability_not_fixed": 0.1, "wrong_liability_improved": 0.05,
            "new_liability_introduced": 0.05, "implausible_chemistry": 0.05, "uncertain": 0.1,
        },
        activity_retention_pred=ActivityRetentionEvidence(
            predicted_retention_bucket="acceptable", predicted_retention_confidence=0.6,
        ),
        liability_improvement_pred=LiabilityEvidence(
            liability_drivers_in_parent=[],
            candidate_changes_affecting_liability=[],
            predicted_improvement_category="moderate",
        ),
        structured_rationale=StructuredRationale(),
        confidence=ConfidenceBlock(
            overall=0.6, activity_retention=0.5, liability_improvement=0.5,
            new_risk=0.6, chemical_plausibility=0.7,
        ),
        rank=rank,
    )


def test_round_trip_with_hash_verification(tmp_path: Path):
    preds = [_mk_ranker_output(f"c{i}", 1.0 - i / 10, i + 1) for i in range(3)]
    lp = build_locked_prediction(
        case_id="ADMET-001",
        system_version="0.1.0",
        sealed_case_registry_hash="0" * 64,
        challenge_packet_hash="1" * 64,
        dataset_manifest_hash="2" * 64,
        training_manifest_hash="3" * 64,
        model_checkpoint_hash="4" * 64,
        predictions=preds,
    )
    p = tmp_path / "locked.json"
    h = write_locked_prediction(lp, p)
    assert len(h) == 64

    rebuilt = read_locked_prediction(p)
    assert rebuilt.case_id == "ADMET-001"
    assert len(rebuilt.predictions) == 3
    assert rebuilt.top_k_locked["5"] == ["c0", "c1", "c2"]


def test_corrupted_predictions_fails_hash_verification(tmp_path: Path):
    preds = [_mk_ranker_output("c1", 0.9, 1)]
    lp = build_locked_prediction(
        case_id="ADMET-001",
        system_version="0.1.0",
        sealed_case_registry_hash="0" * 64,
        challenge_packet_hash="1" * 64,
        dataset_manifest_hash="2" * 64,
        training_manifest_hash="3" * 64,
        model_checkpoint_hash="4" * 64,
        predictions=preds,
    )
    p = tmp_path / "locked.json"
    write_locked_prediction(lp, p)
    raw = json.loads(p.read_text())
    raw["predictions"][0]["candidate_id"] = "tampered"
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="output_hash mismatch"):
        read_locked_prediction(p)


def test_refuses_to_overwrite(tmp_path: Path):
    preds = [_mk_ranker_output("c1", 0.5, 1)]
    lp = build_locked_prediction(
        case_id="ADMET-001",
        system_version="0.1.0",
        sealed_case_registry_hash="0" * 64,
        challenge_packet_hash="1" * 64,
        dataset_manifest_hash="2" * 64,
        training_manifest_hash="3" * 64,
        model_checkpoint_hash="4" * 64,
        predictions=preds,
    )
    p = tmp_path / "locked.json"
    write_locked_prediction(lp, p)
    with pytest.raises(FileExistsError):
        write_locked_prediction(lp, p)
