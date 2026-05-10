"""Populate the sealed-case registry stub with real identifiers.

Reads the YAML stub (where canonical_smiles, inchi_key, ChEMBL/PubChem IDs
are null), looks up each parent + answer molecule in PubChem (then ChEMBL
as fallback), runs RDKit canonicalisation, and writes back a populated YAML
with bumped patch version.

Usage:
    python -m rasyn.data.registry.populator
        [--input  rasyn/data/registry/sealed_case_registry.yaml]
        [--output rasyn/data/registry/sealed_case_registry.yaml]

Network required. Idempotent: if all fields are already populated, exits no-op.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import yaml

from rasyn.data.registry.loader import REGISTRY_DEFAULT_PATH, load_sealed_case_registry
from rasyn.data.sources import chembl as chembl_src
from rasyn.data.sources import pubchem as pubchem_src
from rasyn.schemas.molecule import MoleculeRef
from rasyn.utils.canonicalize import standardize_pair


def _populate_one(ref: MoleculeRef) -> tuple[MoleculeRef, dict]:
    """Look up `ref` by name and return a populated copy + diagnostic info."""
    if ref.is_populated:
        return ref, {"action": "skip", "reason": "already populated"}
    if not ref.name:
        return ref, {"action": "skip", "reason": "no name to look up"}

    info: dict = {"action": "lookup", "queries": {}, "errors": []}

    pc = pubchem_src.lookup_compound(ref.name)
    info["queries"]["pubchem"] = bool(pc)
    cs = pc.get("canonical_smiles") if pc else None
    ik = pc.get("inchi_key") if pc else None
    cid = pc.get("pubchem_cid") if pc else None
    iupac = pc.get("iupac_name") if pc else None

    chembl_data = chembl_src.lookup_molecule_by_name(ref.name)
    info["queries"]["chembl"] = bool(chembl_data)
    chembl_id = chembl_data.get("chembl_id") if chembl_data else None
    if not cs and chembl_data:
        cs = chembl_data.get("canonical_smiles")
        ik = chembl_data.get("inchi_key")

    if not cs:
        info["errors"].append("no SMILES from any source")
        return ref, info

    canonical_cs, computed_ik = standardize_pair(cs)
    if not canonical_cs:
        info["errors"].append("RDKit standardisation failed")
        return ref, info
    if computed_ik and ik and computed_ik != ik:
        info["errors"].append(f"InChIKey mismatch (source={ik}, computed={computed_ik}); using computed")
    final_ik = computed_ik or ik

    populated = MoleculeRef(
        name=ref.name,
        canonical_smiles=canonical_cs,
        inchi_key=final_ik,
        chembl_id=chembl_id or ref.chembl_id,
        pubchem_cid=cid or ref.pubchem_cid,
        cas_number=ref.cas_number,
        drugbank_id=ref.drugbank_id,
        iupac_name=iupac or ref.iupac_name,
    )
    info["result"] = {"chembl_id": populated.chembl_id, "pubchem_cid": populated.pubchem_cid}
    return populated, info


def populate_registry(input_path: Path, output_path: Path) -> dict:
    """Populate the registry. Returns a diagnostic report (per-case info)."""
    reg = load_sealed_case_registry(input_path)
    raw = yaml.safe_load(input_path.read_text(encoding="utf-8"))

    diagnostics: dict = {"per_case": {}, "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}

    for case_yaml in raw.get("cases", []):
        case_id = case_yaml["case_id"]
        case_info = {"parent": None, "answer": None}

        for role in ("parent", "answer"):
            ref_yaml = case_yaml.get(role) or {}
            ref = MoleculeRef.model_validate(ref_yaml)
            populated, info = _populate_one(ref)
            if populated is not ref:
                case_yaml[role] = populated.model_dump(mode="json", exclude_none=True)
            case_info[role] = info

        diagnostics["per_case"][case_id] = case_info

    raw["locked_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    major, minor, patch = map(int, raw["version"].split("."))
    raw["version"] = f"{major}.{minor}.{patch + 1}"

    output_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    diagnostics["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    diagnostics["new_version"] = raw["version"]
    return diagnostics


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Populate sealed-case registry SMILES/IDs.")
    p.add_argument("--input", type=Path, default=REGISTRY_DEFAULT_PATH)
    p.add_argument("--output", type=Path, default=REGISTRY_DEFAULT_PATH)
    args = p.parse_args(argv)
    diag = populate_registry(args.input, args.output)
    import json

    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
