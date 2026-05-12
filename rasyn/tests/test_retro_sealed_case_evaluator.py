"""Tests for the sealed-case auto-judge (RETRO_PLAN R-7)."""
from __future__ import annotations

import sys
from pathlib import Path

# Make scripts/ importable
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_retro_sealed_cases import _bucket, _levenshtein  # type: ignore[import-not-found]


# ===== Levenshtein =====

def test_levenshtein_empty_inputs():
    assert _levenshtein([], []) == 0
    assert _levenshtein([], ["a"]) == 1
    assert _levenshtein(["a"], []) == 1


def test_levenshtein_equal_sequences():
    assert _levenshtein(["amide", "suzuki"], ["amide", "suzuki"]) == 0


def test_levenshtein_one_substitution():
    assert _levenshtein(["amide", "suzuki"], ["amide", "buchwald"]) == 1


def test_levenshtein_one_insertion():
    assert _levenshtein(["amide", "suzuki"], ["amide", "suzuki", "deprotection"]) == 1


# ===== _bucket =====

def test_bucket_retro_003_always_no_literature_baseline_when_passes_floor():
    v = _bucket(
        case_id="RETRO-003",
        reference_class_seq=[],
        candidate_class_seq=["amide"],
        forward_pass_rate=0.9,
        step_count=3,
        min_fwd=0.8,
        step_tolerance=0,
        reference_step_count=None,
    )
    assert v == "route_proposed_no_literature_baseline"


def test_bucket_retro_003_missed_when_below_fwd_floor():
    v = _bucket(
        case_id="RETRO-003",
        reference_class_seq=[],
        candidate_class_seq=["amide"],
        forward_pass_rate=0.5,
        step_count=3,
        min_fwd=0.8,
        step_tolerance=0,
        reference_step_count=None,
    )
    assert v == "missed"


def test_bucket_retro_001_exact_match_is_literature_optimal():
    v = _bucket(
        case_id="RETRO-001",
        reference_class_seq=["amide", "suzuki", "amide"],
        candidate_class_seq=["amide", "suzuki", "amide"],
        forward_pass_rate=0.95,
        step_count=3,
        min_fwd=0.8,
        step_tolerance=2,
        reference_step_count=3,
    )
    assert v == "literature_optimal"


def test_bucket_retro_001_one_off_is_literature_competitive():
    v = _bucket(
        case_id="RETRO-001",
        reference_class_seq=["amide", "suzuki", "amide"],
        candidate_class_seq=["amide", "buchwald_hartwig", "amide"],
        forward_pass_rate=0.95,
        step_count=4,
        min_fwd=0.8,
        step_tolerance=2,
        reference_step_count=3,
    )
    assert v == "literature_competitive"


def test_bucket_retro_002_very_different_route_is_novel_valid():
    v = _bucket(
        case_id="RETRO-002",
        reference_class_seq=["amide_coupling", "protection_deprotection", "amide_coupling"],
        candidate_class_seq=["suzuki_coupling", "click", "sn_ar"],
        forward_pass_rate=0.85,
        step_count=3,
        min_fwd=0.8,
        step_tolerance=1,
        reference_step_count=7,
    )
    assert v == "novel_valid"


def test_bucket_missed_when_below_floor_even_with_exact_class_match():
    v = _bucket(
        case_id="RETRO-001",
        reference_class_seq=["amide", "suzuki", "amide"],
        candidate_class_seq=["amide", "suzuki", "amide"],
        forward_pass_rate=0.4,  # below floor
        step_count=3,
        min_fwd=0.8,
        step_tolerance=2,
        reference_step_count=3,
    )
    assert v == "missed"
