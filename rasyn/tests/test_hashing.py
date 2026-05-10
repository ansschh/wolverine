"""Tests for canonical-JSON + SHA256 hashing helpers."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from rasyn.schemas.hashing import canonical_json, hash_model, sha256_hex


class _Toy(BaseModel):
    a: int
    b: str
    c: list[int]


def test_canonical_json_sorts_keys():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_canonical_json_no_whitespace():
    s = canonical_json({"a": 1, "b": [1, 2, 3]})
    assert " " not in s
    assert "\n" not in s


def test_canonical_json_unicode_passthrough():
    s = canonical_json({"k": "α-β"})
    assert "α-β" in s
    assert s == '{"k":"α-β"}'


def test_sha256_hex_string_and_bytes_match():
    assert sha256_hex("hello") == sha256_hex(b"hello")


def test_hash_model_deterministic():
    m1 = _Toy(a=1, b="x", c=[3, 1, 2])
    m2 = _Toy(a=1, b="x", c=[3, 1, 2])
    assert hash_model(m1) == hash_model(m2)


def test_hash_model_changes_with_field_change():
    m1 = _Toy(a=1, b="x", c=[1, 2, 3])
    m2 = _Toy(a=1, b="y", c=[1, 2, 3])
    assert hash_model(m1) != hash_model(m2)


def test_hash_model_invariant_under_construction_order():
    """Pydantic doesn't preserve dict order, but our hash must be stable."""
    m1 = _Toy(a=1, b="x", c=[1, 2, 3])
    m2 = _Toy(c=[1, 2, 3], a=1, b="x")
    assert hash_model(m1) == hash_model(m2)


def test_canonical_json_round_trips_for_dict():
    obj = {"a": 1, "b": "x", "c": [1, 2, 3], "d": None, "e": True}
    parsed = json.loads(canonical_json(obj))
    assert parsed == obj


@pytest.mark.parametrize("payload", ["", "a", "x" * 10_000, "α-β-γ-δ"])
def test_sha256_hex_length(payload: str):
    assert len(sha256_hex(payload)) == 64
