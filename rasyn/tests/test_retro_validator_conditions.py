"""Tests for forward validator + conditions predictor runtime."""
from __future__ import annotations

import pytest

from rasyn.synth.retro.conditions import ConditionsPredictor, ConditionsPredictorConfig
from rasyn.synth.retro.schemas import (
    ConditionPrediction,
    ForwardValidationResult,
    RetroStep,
)
from rasyn.synth.retro.validator import ForwardValidator, ForwardValidatorConfig


# ===== ForwardValidator =====

def test_forward_validator_no_model_returns_fail():
    v = ForwardValidator(ForwardValidatorConfig())
    step = RetroStep(
        retro_step_id="S1",
        product_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        precursor_inchi_keys=["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"],
        reaction_class="amide_coupling",
        proposed_by_channel="template",
        proposed_by_top_k_rank=1,
        confidence=0.8,
    )
    result = v.validate_step(step, precursor_smiles=["CCO"], target_smiles="CCOC(C)=O")
    assert isinstance(result, ForwardValidationResult)
    assert result.pass_rule == "fail"
    assert result.canonical_smiles_match is False


def test_forward_validator_callable_hook_exact_match():
    """When the loaded checkpoint provides a callable that returns the exact target, pass with exact_match."""
    v = ForwardValidator(ForwardValidatorConfig())
    v._model = lambda reactants_smiles, reaction_class_hint=None: "CCOC(C)=O"  # canonical-already
    step = RetroStep(
        retro_step_id="S1",
        product_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        precursor_inchi_keys=["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"],
        reaction_class="amide_coupling",
        proposed_by_channel="template",
        proposed_by_top_k_rank=1,
        confidence=0.8,
    )
    result = v.validate_step(step, precursor_smiles=["CCO", "CC(=O)Cl"], target_smiles="CCOC(C)=O")
    try:
        import rdkit  # noqa: F401
    except ImportError:
        pytest.skip("rdkit not installed; canonical equality not testable")
    assert result.pass_rule == "exact_match"
    assert result.canonical_smiles_match is True
    assert result.tanimoto_to_target == 1.0


def test_forward_validator_tanimoto_threshold():
    """If the predicted product is close-but-not-exact, pass via tanimoto>=0.95."""
    v = ForwardValidator(ForwardValidatorConfig(tanimoto_threshold=0.5))  # lenient for test
    # Predict same molecule by different SMILES string
    v._model = lambda reactants_smiles, reaction_class_hint=None: "O=C(OCC)C"
    step = RetroStep(
        retro_step_id="S2",
        product_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        precursor_inchi_keys=["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"],
        reaction_class="amide_coupling",
        proposed_by_channel="template",
        proposed_by_top_k_rank=1,
        confidence=0.8,
    )
    result = v.validate_step(step, precursor_smiles=["CCO"], target_smiles="CCOC(C)=O")
    try:
        import rdkit  # noqa: F401
    except ImportError:
        pytest.skip("rdkit not installed")
    # The two SMILES are the same molecule, so should canonicalize to equal.
    assert result.pass_rule in {"exact_match", "tanimoto>=0.95"}


# ===== ConditionsPredictor =====

def test_conditions_predictor_returns_unknown_when_no_model():
    cp = ConditionsPredictor(ConditionsPredictorConfig())
    result = cp.predict(
        reactant_smiles_list=["CCO", "CC(=O)Cl"],
        product_smiles="CCOC(C)=O",
        reactant_inchi_keys=["LFQSCWFLJHTTHZ-UHFFFAOYSA-N", "WETWJCDKMRHUPV-UHFFFAOYSA-N"],
        product_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        reaction_class="amide_coupling",
    )
    assert isinstance(result, ConditionPrediction)
    assert result.solvent_class == "unknown"
    assert result.catalyst_class == "unknown"
    assert result.temperature_bin == "unknown"
    assert result.reagent_classes == []
    assert result.overall_confidence == 0.0


def test_conditions_predictor_callable_hook_returns_real_prediction():
    cp = ConditionsPredictor(ConditionsPredictorConfig())
    cp._predict_fn = lambda *a, **kw: ConditionPrediction(
        reactant_inchi_keys=kw["reactant_inchi_keys"],
        product_inchi_key=kw["product_inchi_key"],
        reaction_class=kw["reaction_class_hint"] or "unclassified",
        solvent_class="DMF",
        catalyst_class="none",
        temperature_bin="rt",
        reagent_classes=["HATU_HBTU_family", "base_TEA_DIPEA"],
        overall_confidence=0.78,
    )
    result = cp.predict(
        reactant_smiles_list=["CCO", "CC(=O)Cl"],
        product_smiles="CCOC(C)=O",
        reactant_inchi_keys=["LFQSCWFLJHTTHZ-UHFFFAOYSA-N", "WETWJCDKMRHUPV-UHFFFAOYSA-N"],
        product_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        reaction_class="amide_coupling",
    )
    assert result.solvent_class == "DMF"
    assert result.reagent_classes == ["HATU_HBTU_family", "base_TEA_DIPEA"]
    assert result.overall_confidence == 0.78
