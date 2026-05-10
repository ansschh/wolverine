"""Generate synthetic leakage canaries for the sealed-case registry.

Per case, emit ~30 canaries spanning all 8 layers (smiles, inchi_key, synonym,
doi, pmid, chembl_id, patent, title_text). Each canary is unique to its case
so a survivor immediately tells you which case's quarantine failed.

Canaries must be:
  - syntactically valid for their layer (parses via the relevant validator)
  - semantically obvious as fakes (so a human reviewer spots them)
  - case-tagged (canary_id encodes case)

Run via:
    from rasyn.data.registry.canary_generator import generate_canaries_for_registry
    from rasyn.data.registry.loader import load_sealed_case_registry
    reg = load_sealed_case_registry()
    canaries = generate_canaries_for_registry(reg, per_layer=4)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import yaml

from rasyn.schemas.registry import Canary, SealedCaseRegistry

LAYERS = ("smiles", "inchi_key", "synonym", "doi", "pmid", "chembl_id", "patent", "title_text")


def _fake_smiles(case_short: str, n: int) -> str:
    # Fake SMILES with unmistakable canary marker. RDKit will reject this
    # (square-bracket atom name), which is fine - injection happens into raw
    # text fields, not parsed molecule columns.
    return f"[CANARY_{case_short}_{n:03d}]CCCC"


def _fake_inchi_key(case_short: str, n: int) -> str:
    # Format-valid: 14 chars + - + 10 chars + - + 1 char.
    # Use 'Z' to make it obviously synthetic (real InChIKeys distribute uniformly).
    body = f"CANARY{case_short:>4}".replace(" ", "Z").upper()[:14].ljust(14, "Z")
    suffix = f"CANARY{n:04d}".upper()[:10].ljust(10, "Z")
    return f"{body}-{suffix}-N"


def _fake_synonym(case_short: str, n: int) -> str:
    return f"CANARY-{case_short}-synonym-{n:03d}"


def _fake_doi(case_short: str, n: int) -> str:
    return f"10.9999/canary-{case_short.lower()}-{n:03d}"


def _fake_pmid(case_short: str, n: int) -> str:
    # PMID range 99999900-99999999 has no real PMIDs (real max ~38M as of 2024).
    return str(99_999_000 + (hash(case_short) % 10) * 100 + n)


def _fake_chembl_id(case_short: str, n: int) -> str:
    return f"CHEMBL999{abs(hash(case_short)) % 1000:03d}{n:03d}"


def _fake_patent(case_short: str, n: int) -> str:
    return f"CANARY-{case_short}-PT{n:05d}"


def _fake_title_text(case_short: str, n: int) -> str:
    return f"<<canary-title-fragment-{case_short.lower()}-{n:03d}>>"


_GENERATORS = {
    "smiles": _fake_smiles,
    "inchi_key": _fake_inchi_key,
    "synonym": _fake_synonym,
    "doi": _fake_doi,
    "pmid": _fake_pmid,
    "chembl_id": _fake_chembl_id,
    "patent": _fake_patent,
    "title_text": _fake_title_text,
}


def _case_short(case_id: str) -> str:
    """ADMET-001 -> A001."""
    if "-" in case_id:
        prefix, num = case_id.split("-", 1)
        return f"{prefix[:1]}{num}"
    return case_id[:4]


def generate_canaries_for_registry(reg: SealedCaseRegistry, *, per_layer: int = 4) -> list[Canary]:
    """Generate `per_layer` canaries per (case, layer). Default 4 * 8 = 32 per case."""
    out: list[Canary] = []
    for case in reg.cases:
        cs = _case_short(case.case_id)
        for layer in LAYERS:
            gen = _GENERATORS[layer]
            for n in range(1, per_layer + 1):
                payload = gen(cs, n)
                out.append(
                    Canary(
                        canary_id=f"CANARY-{case.case_id}-{layer}-{n:03d}",
                        case_id=case.case_id,
                        layer=layer,  # type: ignore[arg-type]
                        payload=payload,
                        inserted_into=[],  # populator records targets when injecting
                    )
                )
    return out


def write_canaries_yaml(canaries: list[Canary], path: Path | str, *, version: str = "0.1.0") -> None:
    """Persist a canary list to YAML in the format expected by the loader."""
    data = {
        "version": version,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "canaries": [c.model_dump(mode="json") for c in canaries],
    }
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
