"""Featurize a CandidateEvidencePacket into a flat tensor for ConcatMLPRanker.

Layout exactly matches `EVIDENCE_FEATURE_DIMS` in `torch_ranker`. Numpy by
default; torch tensor on demand.
"""

from __future__ import annotations

import numpy as np

from rasyn.ranker.torch_ranker import EVIDENCE_FEATURE_DIMS, TOTAL_INPUT_DIM
from rasyn.schemas.evidence import CandidateEvidencePacket

_RETENTION_BUCKETS = ("strong", "acceptable", "weak", "failed", "unknown")
_IMPROVEMENT_CATEGORIES = ("large", "moderate", "minor", "none", "worse", "unknown")


def _one_hot(value: str, classes: tuple[str, ...]) -> list[float]:
    return [1.0 if value == c else 0.0 for c in classes]


def featurize(ev: CandidateEvidencePacket) -> np.ndarray:
    """Returns a 1D numpy array of length `TOTAL_INPUT_DIM`."""
    s = ev.structural
    structural = [
        float(s.tanimoto_to_parent),
        1.0 if s.murcko_scaffold_match else 0.0,
        float(s.transformation_distance),
        float(s.pharmacophore_similarity or 0.0),
        float(s.shape_similarity or 0.0),
    ]

    d = ev.descriptors
    descriptors = [d.mw, d.log_p, d.tpsa, d.hbd, d.hba, d.rotatable_bonds, d.aromatic_rings, d.fsp3, d.formal_charge]
    descriptors = [float(x) for x in descriptors]

    dd = ev.descriptor_deltas
    deltas = [
        dd.delta_mw, dd.delta_log_p, dd.delta_tpsa, dd.delta_hbd, dd.delta_hba,
        dd.delta_rotatable_bonds, dd.delta_aromatic_rings, dd.delta_fsp3, dd.delta_formal_charge,
    ]
    deltas = [float(x) for x in deltas]

    retention_oh = _one_hot(ev.activity_retention.predicted_retention_bucket, _RETENTION_BUCKETS)
    improvement_oh = _one_hot(ev.liability.predicted_improvement_category, _IMPROVEMENT_CATEGORIES)

    sr = ev.structured_rationale
    rationale_counts = [
        float(len(sr.liability_driver_features)),
        float(len(sr.modified_features)),
        float(len(sr.preserved_activity_features)),
    ]

    risk_flags = [
        float(len(ev.risk.new_liability_flags)),
        float(len(ev.risk.reactive_alert_flags)),
        1.0 if ev.risk.synthesizability_score is not None else 0.0,
    ]

    flat = (
        structural
        + descriptors
        + deltas
        + retention_oh
        + improvement_oh
        + rationale_counts
        + risk_flags
    )
    assert len(flat) == TOTAL_INPUT_DIM, (len(flat), TOTAL_INPUT_DIM, EVIDENCE_FEATURE_DIMS)
    return np.asarray(flat, dtype=np.float32)
