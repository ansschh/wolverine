"""ConcatMLPRanker feature dimensionality tests (no torch needed).

Uses the `evidence_factory` fixture defined in tests/conftest.py.
"""

from __future__ import annotations

import numpy as np

from rasyn.ranker.featurize import featurize
from rasyn.ranker.torch_ranker import TOTAL_INPUT_DIM


def test_featurize_returns_correct_length(evidence_factory):
    arr = featurize(evidence_factory("a"))
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.shape == (TOTAL_INPUT_DIM,)


def test_featurize_one_hot_retention(evidence_factory):
    a = featurize(evidence_factory("a", retention="strong"))
    b = featurize(evidence_factory("b", retention="failed"))
    assert not np.array_equal(a, b)


def test_featurize_one_hot_improvement(evidence_factory):
    a = featurize(evidence_factory("a", improvement="large"))
    b = featurize(evidence_factory("b", improvement="none"))
    assert not np.array_equal(a, b)
