"""Populate sealed_case_registry.yaml answer SMILES via PubChem PUG REST.

ADMET-001: terfenadine (parent) -> fexofenadine (rescue)  [hERG]
ADMET-002: acyclovir (parent)   -> valacyclovir (rescue)  [oral_exposure]
ADMET-003: OXS007570 (parent)   -> OXS008474 (rescue)     [solubility]

ADMET-001 and -002 have well-known PubChem CIDs and are populated automatically.
ADMET-003 OXS compounds are paper-only (10.1039/d4md00275j, paywalled and
quarantined). They cannot be auto-populated; they are flagged as
needs_user_input. ADMET-003 evaluation falls back to Mode B (no answer leak)
unless the user provides them out-of-band.

Run:
    python scripts/populate_sealed_case_registry.py

Output:
    Updates rasyn/data/registry/sealed_case_registry.yaml in place.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import yaml

REGISTRY_PATH = Path("rasyn/data/registry/sealed_case_registry.yaml")
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"


CASE_PUBCHEM = {
    "ADMET-001": {
        "parent":    {"name": "terfenadine",   "cid": 5405},
        "candidate": {"name": "fexofenadine",  "cid": 3348},
    },
    "ADMET-002": {
        "parent":    {"name": "acyclovir",     "cid": 2022},
        "candidate": {"name": "valacyclovir",  "cid": 60773},
    },
    "ADMET-003": {
        "parent":    {"name": "OXS007570",     "cid": None, "needs_user_input": True},
        "candidate": {"name": "OXS008474",     "cid": None, "needs_user_input": True},
    },
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def fetch_pubchem(cid: int) -> dict | None:
    url = f"{PUBCHEM_BASE}/{cid}/property/SMILES,ConnectivitySMILES,InChIKey,IUPACName/JSON"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        props = data["PropertyTable"]["Properties"][0]
        return {
            "smiles": props.get("SMILES"),
            "inchi_key": props.get("InChIKey"),
            "iupac_name": props.get("IUPACName"),
        }
    except Exception as e:
        _log(f"  PubChem fetch failed for CID {cid}: {e}")
        return None


def canonicalize_via_csp(smi: str | None) -> str | None:
    if not smi:
        return None
    try:
        from rdkit import Chem
        from chembl_structure_pipeline.standardizer import standardize_mol, get_parent_mol
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        std = standardize_mol(mol)
        parent_result = get_parent_mol(std)
        parent = parent_result[0] if isinstance(parent_result, tuple) else parent_result
        return Chem.MolToSmiles(parent, isomericSmiles=True, canonical=True)
    except Exception:
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        except Exception:
            return None


def main() -> int:
    if not REGISTRY_PATH.exists():
        _log(f"FATAL: {REGISTRY_PATH} not found")
        return 1

    reg = yaml.safe_load(REGISTRY_PATH.read_text())

    case_map = {c["case_id"]: c for c in reg["cases"]}

    for case_id, info in CASE_PUBCHEM.items():
        case = case_map.get(case_id)
        if case is None:
            _log(f"  WARN: {case_id} not in registry; skipping")
            continue
        for role in ("parent", "candidate"):
            entry = info[role]
            target_block = case.get(role) or case.get(f"{role}_compound") or {}
            if entry.get("needs_user_input"):
                target_block["smiles_populate_status"] = "needs_user_input_paper_quarantined"
                target_block["name"] = entry["name"]
                _log(f"[{case_id}/{role}] {entry['name']}: NEEDS USER INPUT (paper-only / quarantined)")
                continue
            cid = entry["cid"]
            _log(f"[{case_id}/{role}] {entry['name']} CID {cid} ...")
            resp = fetch_pubchem(cid)
            if not resp:
                target_block["smiles_populate_status"] = "pubchem_failed"
                continue
            canonical = canonicalize_via_csp(resp.get("smiles"))
            if canonical:
                target_block["name"] = entry["name"]
                target_block["pubchem_cid"] = cid
                target_block["canonical_smiles"] = canonical
                target_block["inchi_key"] = resp.get("inchi_key")
                target_block["iupac_name"] = resp.get("iupac_name")
                target_block["smiles_populate_status"] = "ok"
                _log(f"  -> {canonical[:80]}{'...' if len(canonical) > 80 else ''}")
            else:
                target_block["smiles_populate_status"] = "canonicalize_failed"
            time.sleep(0.4)

            # Write back into the original case dict (might already point there)
            if role in case:
                case[role] = target_block
            else:
                case[f"{role}_compound"] = target_block

    # Bump version
    reg["version"] = (reg.get("version", "0.0.0") + ".1")
    REGISTRY_PATH.write_text(yaml.safe_dump(reg, sort_keys=False, allow_unicode=True))
    _log("")
    _log(f"Updated {REGISTRY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
