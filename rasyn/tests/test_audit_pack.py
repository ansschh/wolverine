"""Audit pack assembler tests."""

from __future__ import annotations

import json
from pathlib import Path

from rasyn.audit.audit_pack import assemble_audit_pack
from rasyn.audit.locked_prediction_io import build_locked_prediction, write_locked_prediction
from rasyn.schemas.evidence import (
    ActivityRetentionEvidence,
    LiabilityEvidence,
    StructuredRationale,
)
from rasyn.schemas.ranker import ConfidenceBlock, RankerOutput


def _mk_pred(case_id: str) -> RankerOutput:
    return RankerOutput(
        case_id=case_id,
        candidate_id=f"{case_id}-cand-1",
        rescue_score=0.7,
        rescue_label_probs={
            "strong_success": 0.3, "weak_success": 0.3, "failed_activity_loss": 0.1,
            "failed_no_liability_improvement": 0.1, "failed_wrong_liability": 0.05,
            "failed_new_liability": 0.05, "uncertain": 0.1,
        },
        failure_mode_probs={
            "activity_lost": 0.1, "liability_not_fixed": 0.1, "wrong_liability_improved": 0.05,
            "new_liability_introduced": 0.05, "implausible_chemistry": 0.05, "uncertain": 0.1,
        },
        activity_retention_pred=ActivityRetentionEvidence(predicted_retention_bucket="acceptable"),
        liability_improvement_pred=LiabilityEvidence(
            liability_drivers_in_parent=[],
            candidate_changes_affecting_liability=[],
            predicted_improvement_category="moderate",
        ),
        structured_rationale=StructuredRationale(),
        confidence=ConfidenceBlock(
            overall=0.5, activity_retention=0.5, liability_improvement=0.5, new_risk=0.5, chemical_plausibility=0.5
        ),
        rank=1,
    )


def test_assemble_audit_pack(tmp_path: Path):
    # Stand up artifact files.
    reg = tmp_path / "registry.yaml"
    reg.write_text("version: '0.0.0'\n")
    pre = tmp_path / "pre.json"
    pre.write_text("{}")
    post = tmp_path / "post.json"
    post.write_text("{}")
    canary = tmp_path / "canary.json"
    canary.write_text("{}")
    nn = tmp_path / "nn.csv"
    nn.write_text("a,b\n")
    dsm = tmp_path / "dataset.json"
    dsm.write_text("{}")
    tm = tmp_path / "train.json"
    tm.write_text("{}")

    lp_dir = tmp_path / "locked"
    lp_dir.mkdir()
    locked_paths = []
    for case_id in ("ADMET-001", "ADMET-002", "ADMET-003"):
        lp = build_locked_prediction(
            case_id=case_id,
            system_version="0.1.0",
            sealed_case_registry_hash="0" * 64,
            challenge_packet_hash="1" * 64,
            dataset_manifest_hash="2" * 64,
            training_manifest_hash="3" * 64,
            model_checkpoint_hash="4" * 64,
            predictions=[_mk_pred(case_id)],
        )
        p = lp_dir / f"{case_id.lower()}.json"
        write_locked_prediction(lp, p)
        locked_paths.append(p)

    out = tmp_path / "audit_pack.json"
    pack = assemble_audit_pack(
        sealed_case_registry=reg,
        decontam_pre=pre,
        decontam_post=post,
        canary_report=canary,
        nearest_neighbor_table=nn,
        dataset_manifest=dsm,
        training_manifests=[tm],
        locked_predictions=locked_paths,
        output_path=out,
    )
    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert set(on_disk["cases"].keys()) == {"ADMET-001", "ADMET-002", "ADMET-003"}
    assert "sealed_case_registry" in on_disk["artifact_hashes"]
    assert pack == on_disk
