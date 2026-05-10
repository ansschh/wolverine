"""ConcatMLPRanker feature dimensionality tests (no torch needed)."""

from __future__ import annotations

import numpy as np
from conftest import make_evidence

from rasyn.ranker.featurize import featurize
from rasyn.ranker.torch_ranker import TOTAL_INPUT_DIM


def test_featurize_returns_correct_length():
    ev = make_evidence("a")
    arr = featurize(ev)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.shape == (TOTAL_INPUT_DIM,)


def test_featurize_one_hot_retention():
    a = featurize(make_evidence("a", retention="strong"))
    b = featurize(make_evidence("b", retention="failed"))
    assert not np.array_equal(a, b)


def test_featurize_one_hot_improvement():
    a = featurize(make_evidence("a", improvement="large"))
    b = featurize(make_evidence("b", improvement="none"))
    assert not np.array_equal(a, b)
