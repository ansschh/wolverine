"""Proposer interface tests (RDKit-light: stub-channel paths only)."""

from __future__ import annotations

from rasyn.proposer.base import Proposer, ProposerContext
from rasyn.proposer.forward_opt import ForwardRewardOptimizerProposer
from rasyn.proposer.inverse_delta import LearnedInverseDeltaProposer
from rasyn.proposer.novelty import LearnedNoveltyProposer
from rasyn.synth.fixture import build_synthetic_fixture


def test_subclass_must_define_channel():
    import pytest

    with pytest.raises(TypeError):
        class _Bad(Proposer):  # type: ignore[abstract]
            pass


def test_ml_proposers_are_empty_stubs_at_v1():
    packets, pool = build_synthetic_fixture()
    packet = next(iter(packets.values()))
    ctx = ProposerContext(candidate_smiles_pool=pool)
    for proposer in [
        LearnedInverseDeltaProposer(),
        ForwardRewardOptimizerProposer(),
        LearnedNoveltyProposer(),
    ]:
        out = proposer.propose(packet, ctx)
        assert out.case_id == packet.case_id
        assert out.candidates == []
        assert out.channel == proposer.channel


def test_proposer_output_round_trips_through_pydantic():
    from rasyn.schemas.proposer import CandidateAnnotation, ProposerOutput

    out = ProposerOutput(
        case_id="SYNTH-X",
        channel="learned_inverse_delta",
        candidates=[
            CandidateAnnotation(
                candidate_id="x",
                canonical_smiles="CCO",
                inchi_key="AAAAAAAAAAAAAA-BBBBBBBBBB-N",
                parent_inchi_key="AAAAAAAAAAAAAA-BBBBBBBBBB-N",
                proposer_sources=["learned_inverse_delta"],
            )
        ],
        raw_count=1,
        invalid_count=0,
        deduplicated_count=1,
    )
    rebuilt = ProposerOutput.model_validate_json(out.model_dump_json())
    assert rebuilt == out
