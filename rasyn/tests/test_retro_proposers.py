"""Tests for retro proposer runtime modules.

Verifies each proposer:
  - subclasses RetroProposer
  - exposes the unified ProposerOutput interface
  - degrades gracefully when no checkpoint is provided (empty candidates,
    not a crash)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from rasyn.synth.retro.proposers import (
    DiffusionProposer, DiffusionProposerConfig,
    GraphEditProposer, GraphEditProposerConfig,
    RetrievalProposer, RetrievalProposerConfig,
    RetroProposer,
    Seq2SeqProposer, Seq2SeqProposerConfig,
    TemplateProposer, TemplateProposerConfig,
)
from rasyn.synth.retro.schemas import ProposerOutput


# ===== Interface conformance =====

@pytest.mark.parametrize("proposer_factory", [
    lambda: TemplateProposer(TemplateProposerConfig()),
    lambda: RetrievalProposer(RetrievalProposerConfig()),
    lambda: Seq2SeqProposer(Seq2SeqProposerConfig()),
    lambda: GraphEditProposer(GraphEditProposerConfig()),
    lambda: DiffusionProposer(DiffusionProposerConfig()),
])
def test_proposer_is_subclass_of_base(proposer_factory):
    p = proposer_factory()
    assert isinstance(p, RetroProposer)
    assert hasattr(p, "channel")
    assert hasattr(p, "propose")


@pytest.mark.parametrize("proposer_factory,expected_channel", [
    (lambda: TemplateProposer(TemplateProposerConfig()), "template"),
    (lambda: RetrievalProposer(RetrievalProposerConfig()), "retrieval"),
    (lambda: Seq2SeqProposer(Seq2SeqProposerConfig()), "seq2seq"),
    (lambda: GraphEditProposer(GraphEditProposerConfig()), "graphedit"),
    (lambda: DiffusionProposer(DiffusionProposerConfig()), "diffusion"),
])
def test_proposer_channel_name(proposer_factory, expected_channel):
    p = proposer_factory()
    assert p.channel == expected_channel


@pytest.mark.parametrize("proposer_factory", [
    lambda: TemplateProposer(TemplateProposerConfig()),
    lambda: RetrievalProposer(RetrievalProposerConfig()),
    lambda: Seq2SeqProposer(Seq2SeqProposerConfig()),
    lambda: GraphEditProposer(GraphEditProposerConfig()),
    lambda: DiffusionProposer(DiffusionProposerConfig()),
])
def test_proposer_propose_returns_proposer_output(proposer_factory):
    """Without a checkpoint, every proposer should return an empty but valid ProposerOutput."""
    p = proposer_factory()
    out = p.propose(
        target_smiles="CCOC(C)=O",
        target_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        top_k=5,
    )
    assert isinstance(out, ProposerOutput)
    assert out.channel == p.channel
    assert out.target_smiles == "CCOC(C)=O"
    # Same length invariant
    assert len(out.candidates) == len(out.candidate_smiles)
    assert len(out.candidates) == len(out.confidences)
    assert len(out.candidates) == len(out.reaction_class_predictions)


# ===== Retrieval-proposer-specific: synthetic in-memory index =====

def test_retrieval_proposer_returns_candidates_with_metadata(tmp_path):
    metadata = [
        {
            "product_smiles": "CCOC(C)=O",
            "product_inchi_key": "XEKOWRVHYACXOJ-UHFFFAOYSA-N",
            "reactant_smiles": ["CCO", "CC(=O)Cl"],
            "reaction_class": "amide_coupling",
            "source_id": "RXN1",
        },
        {
            "product_smiles": "c1ccccc1-c1ccccc1",
            "product_inchi_key": "ZUOUZKKEUPVFJK-UHFFFAOYSA-N",
            "reactant_smiles": ["c1ccccc1Br", "c1ccccc1B(O)O"],
            "reaction_class": "suzuki_coupling",
            "source_id": "RXN2",
        },
    ]
    metadata_path = tmp_path / "metadata.pkl"
    with open(metadata_path, "wb") as fh:
        pickle.dump(metadata, fh)
    cfg = RetrievalProposerConfig(metadata_path=metadata_path, top_k=10)
    p = RetrievalProposer(cfg)

    # Query close to the first reference -> expect that one back
    out = p.propose(
        target_smiles="CCOC(C)=O",
        target_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        top_k=2,
    )
    assert isinstance(out, ProposerOutput)
    # If RDKit is installed, we should retrieve at least the exact match.
    try:
        import rdkit  # noqa: F401
        assert len(out.candidates) >= 1
        assert out.candidate_smiles[0] == ["CCO", "CC(=O)Cl"] or out.candidates[0] == [
            "LFQSCWFLJHTTHZ-UHFFFAOYSA-N", "WETWJCDKMRHUPV-UHFFFAOYSA-N",
        ]
    except ImportError:
        pytest.skip("rdkit not installed; skipping fingerprint search")


def test_retrieval_proposer_class_filter(tmp_path):
    metadata = [
        {
            "product_smiles": "CCOC(C)=O",
            "reactant_smiles": ["CCO", "CC(=O)Cl"],
            "reaction_class": "amide_coupling",
            "source_id": "RXN1",
        },
        {
            "product_smiles": "CCOC(C)=O",  # same product, different class
            "reactant_smiles": ["CCO", "CC(=O)O"],
            "reaction_class": "sn2",
            "source_id": "RXN2",
        },
    ]
    metadata_path = tmp_path / "metadata.pkl"
    with open(metadata_path, "wb") as fh:
        pickle.dump(metadata, fh)
    cfg = RetrievalProposerConfig(metadata_path=metadata_path, top_k=10)
    p = RetrievalProposer(cfg)

    try:
        import rdkit  # noqa: F401
    except ImportError:
        pytest.skip("rdkit not installed")

    out = p.propose(
        target_smiles="CCOC(C)=O",
        target_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        top_k=5,
        reaction_class_hint="amide_coupling",
    )
    # Only amide_coupling hits should come back.
    for rc in out.reaction_class_predictions:
        assert rc == "amide_coupling"


# ===== Diffusion-proposer-specific: kwargs threading =====

def test_diffusion_proposer_accepts_disconnection_mask():
    """The diffusion proposer should accept disconnection_mask via kwargs without crashing."""
    p = DiffusionProposer(DiffusionProposerConfig())
    out = p.propose(
        target_smiles="CCOC(C)=O",
        target_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        top_k=5,
        disconnection_mask=[1, 0, 1, 0],
    )
    assert isinstance(out, ProposerOutput)
    # No model loaded -> empty candidate list, no crash.
    assert len(out.candidates) == 0
