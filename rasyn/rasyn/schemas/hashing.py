"""Canonical JSON + SHA256 helpers used for hashing any schema instance.

Every artifact in Rasyn (configs, dataset manifests, model checkpoints, locked
predictions) gets a SHA256 hash computed over the canonical-JSON serialisation
of its public fields. "Canonical" means: sorted keys, no whitespace, UTF-8.

Two equal Pydantic models always produce equal hashes; equal hashes mean
byte-equal canonical JSON. This is the audit-trail invariant for the locked
prediction ledger and the dataset manifest.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace separators, UTF-8.

    For Pydantic models, pass `model.model_dump(mode="json")` first so that
    non-JSON-native types (datetime, Path, Enum) are coerced to strings.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: str | bytes) -> str:
    """SHA256 hex digest of a string (UTF-8 encoded) or raw bytes."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_model(model: BaseModel) -> str:
    """SHA256 of the canonical JSON of a Pydantic model.

    Always uses `mode="json"` so that all field types are JSON-coercible.
    """
    return sha256_hex(canonical_json(model.model_dump(mode="json")))
