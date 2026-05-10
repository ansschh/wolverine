"""Populate ground_truth_set.yaml SMILES from PubChem CIDs + decontam check.

For each pair × {parent, candidate}:
  - If `pubchem_cid` is set: fetch IsomericSMILES + InChIKey via PubChem PUG REST.
  - If `smiles_populate_from: paper_si`: leave SMILES None, mark `needs_manual_population: true`.
  - RDKit + chembl_structure_pipeline canonicalize every fetched SMILES.

Then run sealed-case decontamination check on each populated GT pair:
  - Compute Morgan FP for parent + candidate vs each sealed-case answer.
  - If Tanimoto >= 0.85 OR (Tanimoto >= 0.65 AND same Murcko), FLAG.
  - DO NOT silently drop — surface to user for review (per L25).

Output:
  rasyn/papers/ground_truth_set.populated.yaml

Run:
  python scripts/populate_gt_smiles.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import yaml


GT_PATH = Path("rasyn/papers/ground_truth_set.yaml")
GT_OUT = Path("rasyn/papers/ground_truth_set.populated.yaml")
DECONTAM_REPORT = Path("rasyn/papers/gt_decontam_report.json")
REGISTRY_PATH = Path("rasyn/data/registry/sealed_case_registry.yaml")

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"


def _log(msg: str) -> None:
    print(msg, flush=True)


def fetch_pubchem(cid: int) -> dict | None:
    """Fetch SMILES + InChIKey for a given PubChem CID.

    PubChem PUG REST changed property names ~2024:
      - 'SMILES' (full canonical with stereo) replaced 'CanonicalSMILES'/'IsomericSMILES'
      - 'ConnectivitySMILES' (stereo-less) is the new connectivity-only form.
    """
    url = (
        f"{PUBCHEM_BASE}/{cid}/property/"
        f"SMILES,ConnectivitySMILES,InChIKey,IUPACName/JSON"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            props = data["PropertyTable"]["Properties"][0]
            return {
                "smiles": props.get("SMILES"),
                "connectivity_smiles": props.get("ConnectivitySMILES"),
                "inchi_key": props.get("InChIKey"),
                "iupac_name": props.get("IUPACName"),
            }
    except urllib.error.HTTPError as e:
        _log(f"  PubChem HTTP {e.code} for CID {cid}: {e.reason}")
        return None
    except Exception as e:
        _log(f"  PubChem fetch failed CID {cid}: {e}")
        return None


def canonicalize_smi(smi: str | None) -> tuple[str | None, str | None]:
    """Return (canonical_smiles, error)."""
    if not smi:
        return None, "empty"
    try:
        from rdkit import Chem
    except ImportError:
        return smi, "rdkit_not_available"

    try:
        from chembl_structure_pipeline.standardizer import standardize_mol, get_parent_mol
    except ImportError:
        # Fallback: plain RDKit canonical
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None, "rdkit_parse_failed"
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True), "no_csp"

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, "rdkit_parse_failed"
    try:
        std = standardize_mol(mol)
        parent_result = get_parent_mol(std)
        if isinstance(parent_result, tuple):
            parent = parent_result[0]
        else:
            parent = parent_result
        canonical = Chem.MolToSmiles(parent, isomericSmiles=True, canonical=True)
        return canonical, None
    except Exception as e:
        # Fallback to plain RDKit
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True), f"csp_failed:{e}"


def load_sealed_answers() -> list[tuple[str, str | None]]:
    """Load sealed-case answer SMILES for decontam check.

    Returns list of (case_id, answer_smiles_or_None).
    """
    if not REGISTRY_PATH.exists():
        _log(f"  WARN: registry not at {REGISTRY_PATH}; decontam check skipped.")
        return []
    reg = yaml.safe_load(REGISTRY_PATH.read_text())
    answers: list[tuple[str, str | None]] = []
    for case in reg.get("cases", []):
        cid = case["case_id"]
        ans = case.get("answer", {}) or {}
        smi = ans.get("canonical_smiles")
        answers.append((cid, smi))
    return answers


def decontam_check(parent_smi: str, cand_smi: str, sealed: list[tuple[str, str | None]]) -> list[str]:
    """Return list of decontam violations (empty list = clean)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
    except ImportError:
        return ["rdkit_unavailable_skipped"]

    def fp(s: str):
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)

    def murcko(s: str) -> str | None:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        sf = GetScaffoldForMol(m)
        if sf is None:
            return None
        return Chem.MolToSmiles(sf)

    violations: list[str] = []
    p_fp = fp(parent_smi)
    c_fp = fp(cand_smi)
    p_murcko = murcko(parent_smi)
    c_murcko = murcko(cand_smi)

    for case_id, ans_smi in sealed:
        if not ans_smi:
            continue
        a_fp = fp(ans_smi)
        a_murcko = murcko(ans_smi)
        if a_fp is None:
            continue
        for role, role_fp, role_murcko, role_smi in (
            ("parent", p_fp, p_murcko, parent_smi),
            ("candidate", c_fp, c_murcko, cand_smi),
        ):
            if role_fp is None:
                continue
            t = DataStructs.TanimotoSimilarity(role_fp, a_fp)
            if t >= 0.85:
                violations.append(
                    f"{role} Tanimoto {t:.3f} >= 0.85 vs {case_id} answer"
                )
            elif t >= 0.65 and role_murcko and a_murcko and role_murcko == a_murcko:
                violations.append(
                    f"{role} Tanimoto {t:.3f} + same Murcko vs {case_id} answer"
                )
    return violations


def main() -> int:
    if not GT_PATH.exists():
        _log(f"FATAL: {GT_PATH} not found.")
        return 1

    gt = yaml.safe_load(GT_PATH.read_text())
    pairs = gt["pairs"]
    sealed = load_sealed_answers()
    _log(f"Loaded {len(sealed)} sealed-case answer(s); proceeding to populate {len(pairs)} pairs.")

    n_populated = 0
    n_paper_si = 0
    n_failed = 0
    decontam_report: dict[str, Any] = {"violations_per_pair": {}, "total_violations": 0}

    for pair in pairs:
        pid = pair["id"]
        for role in ("parent", "candidate"):
            entry = pair[role]
            cid = entry.get("pubchem_cid")
            if cid is not None:
                _log(f"[{pid}/{role}] PubChem CID {cid} ...")
                resp = fetch_pubchem(int(cid))
                if not resp:
                    n_failed += 1
                    entry["population_status"] = "pubchem_failed"
                    continue
                raw_smi = resp.get("smiles") or resp.get("connectivity_smiles")
                canonical, err = canonicalize_smi(raw_smi)
                if canonical:
                    entry["canonical_smiles"] = canonical
                    entry["inchi_key"] = resp.get("inchi_key")
                    entry["iupac_name_pubchem"] = resp.get("iupac_name")
                    entry["population_status"] = "ok" + (f"(warn:{err})" if err else "")
                    n_populated += 1
                    _log(f"  -> {canonical[:60]}{'...' if len(canonical) > 60 else ''}")
                else:
                    n_failed += 1
                    entry["population_status"] = f"canonicalize_failed:{err}"
                time.sleep(0.4)  # PubChem rate limit
            elif entry.get("smiles_populate_from") == "paper_si":
                entry["canonical_smiles"] = None
                entry["needs_manual_population"] = True
                entry["population_status"] = "needs_manual_paper_si_extraction"
                n_paper_si += 1

        # Decontamination check (skip if any SMILES still missing)
        p_smi = pair["parent"].get("canonical_smiles")
        c_smi = pair["candidate"].get("canonical_smiles")
        if p_smi and c_smi:
            v = decontam_check(p_smi, c_smi, sealed)
            if v:
                decontam_report["violations_per_pair"][pid] = v
                decontam_report["total_violations"] += len(v)
                _log(f"  DECONTAM VIOLATION on {pid}: {v}")
                pair["decontam_violations"] = v

    GT_OUT.write_text(yaml.safe_dump(gt, sort_keys=False, allow_unicode=True))
    DECONTAM_REPORT.write_text(json.dumps(decontam_report, indent=2))

    _log("")
    _log("=" * 60)
    _log(f"POPULATED: {n_populated} entries from PubChem")
    _log(f"PAPER_SI:  {n_paper_si} entries need manual population")
    _log(f"FAILED:    {n_failed} entries failed (see status fields)")
    _log(f"DECONTAM:  {decontam_report['total_violations']} violations across {len(decontam_report['violations_per_pair'])} pairs")
    _log(f"OUTPUT:    {GT_OUT}")
    _log(f"REPORT:    {DECONTAM_REPORT}")
    _log("=" * 60)

    if decontam_report["total_violations"] > 0:
        _log("WARN: Decontamination violations detected. Review and replace flagged GT pairs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
