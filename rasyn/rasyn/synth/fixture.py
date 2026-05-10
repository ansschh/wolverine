"""Synthetic Layer-1 fixture: small, fully self-contained, no chemistry deps required.

Builds a tiny ADMETChallengePacket per "fake case" plus a candidate pool that
exercises every code path: hits, hard negatives, invalid molecules, decoys.

This fixture is the smoke-test input. Real chemistry happens later.
"""

from __future__ import annotations

from rasyn.schemas.challenge import (
    ActivityContext,
    ADMETChallengePacket,
    LiabilityContext,
    RescueContextPacket,
)

# Tiny set of toy SMILES that RDKit can parse. None of these correspond to
# the sealed-case answers — synthetic-only.
SYNTHETIC_PARENTS = {
    "SYNTH-HERG-001": {
        "smiles": "CCN(CC)CCOc1ccc(C(=O)Cl)cc1",
        "liability": "hERG",
        "rescue_mode": "active_metabolite_safety_rescue",
        "target": "synth_target_1",
    },
    "SYNTH-SOL-001": {
        "smiles": "c1ccc2ccccc2c1",
        "liability": "solubility",
        "rescue_mode": "polarity_solubility_rescue",
        "target": "synth_target_2",
    },
    "SYNTH-MET-001": {
        "smiles": "CC(C)(C)c1ccc(O)cc1",
        "liability": "metabolic_stability",
        "rescue_mode": "metabolic_soft_spot_rescue",
        "target": "synth_target_3",
    },
}

SYNTHETIC_CANDIDATE_POOL = [
    "CCN(CC)CCOc1ccc(C(=O)O)cc1",  # hERG candidate (acid version)
    "CCN(CC)CCOc1ccc(C(=O)NC)cc1",
    "c1ccc2ncccc2c1",  # solubility candidate (N inserted)
    "c1ccc2ccccc2c1F",
    "CC(C)(C)c1ccc(F)cc1",  # metstab candidate (F instead of OH)
    "CC(C)(F)c1ccc(O)cc1",
    "CCCCCCCCCCCC",  # irrelevant
    "c1ccccc1",  # too simple
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin (decoy)
    "CC(C)C(N)C(=O)OCCOCN1C=NC2=C1N=C(N)NC2=O",  # an L-valyl-ester pattern
]


def _packet_for(case_id: str) -> ADMETChallengePacket:
    spec = SYNTHETIC_PARENTS[case_id]
    return ADMETChallengePacket(
        case_id=case_id,
        parent_canonical_smiles=spec["smiles"],
        parent_inchi_key="AAAAAAAAAAAAAA-BBBBBBBBBB-N",  # placeholder; real fixture path computes via RDKit
        activity_context=ActivityContext(
            target_name=spec["target"],
            desired_pharmacology="synthetic-pharma",
            parent_potency_value=10.0,
            parent_potency_unit="nM",
            parent_potency_endpoint="IC50",
        ),
        liability_context=LiabilityContext(
            liability_type=spec["liability"],
            measurement_endpoint=f"synthetic-{spec['liability']}",
            parent_value=1.0,
            parent_unit="uM",
            parent_category="high",
            target_improvement_category="low",
        ),
        rescue_context=RescueContextPacket(
            rescue_mode=spec["rescue_mode"],
            constraints=["preserve activity within 10x"],
        ),
    )


def build_synthetic_fixture():
    """Return (packets_by_case_id, candidate_pool)."""
    packets = {cid: _packet_for(cid) for cid in SYNTHETIC_PARENTS}
    return packets, list(SYNTHETIC_CANDIDATE_POOL)
