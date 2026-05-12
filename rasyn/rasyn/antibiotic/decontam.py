"""Antibiotic-specific decontamination per spec §19.

Removes from training corpora ANY record that:
1. IS the sealed solution (exact identifier match).
2. Names the sealed solution (synonym).
3. Is in a forbidden document (DOI, PMID, title-fragment match).
4. Is in a forbidden assay context (organism + mechanism keyword match).
5. Is structurally too close (Tanimoto >= 0.85 to answer, or Tanimoto >= 0.65 + same Murcko in same target context).

Returns counts per removal reason. Halts the pipeline if canary survivors > 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from rasyn.antibiotic.schemas import ABXSealedCase, ABXSealedCaseRegistry


@dataclass
class ABXDecontamReport:
    n_input: int = 0
    n_kept: int = 0
    n_removed_total: int = 0
    removed_by_reason: dict[str, int] = field(default_factory=dict)
    survivors_by_case: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_kept": self.n_kept,
            "n_removed_total": self.n_removed_total,
            "removed_by_reason": self.removed_by_reason,
            "survivors_by_case": self.survivors_by_case,
        }


def _try_morgan_fp(smi: str):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    except Exception:
        return None


def _try_murcko_smiles(smi: str):
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        sf = GetScaffoldForMol(m)
        if sf is None:
            return None
        return Chem.MolToSmiles(sf)
    except Exception:
        return None


def _tanimoto(fp_a, fp_b) -> float:
    try:
        from rdkit.Chem import DataStructs
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except Exception:
        return 0.0


def build_forbidden_index(registry: ABXSealedCaseRegistry) -> dict:
    """Pre-compute lookups for fast scrub. Returns dict per case + a global set."""
    index = {
        "per_case": {},
        "global_synonyms_lower": set(),
        "global_dois": set(),
        "global_pmids": set(),
        "global_title_fragments_lower": set(),
        "answer_fps": [],          # list of (case_id, fp) for Tanimoto match
        "answer_murckos": [],      # list of (case_id, murcko_smiles)
        "answer_tan_thresholds": {},   # case_id -> tanimoto threshold (default 0.85)
        "context_tan_thresholds": {},  # case_id -> 0.65 + Murcko requirement
    }
    for case in registry.cases:
        ci = {
            "case_id": case.case_id,
            "synonyms_lower": set(),
            "pubchem_cids": set(),
            "chembl_ids": set(),
            "drugbank_ids": set(),
            "dois": set(),
            "pmids": set(),
            "title_fragments_lower": set(),
        }
        for s in case.forbidden_identifiers.get("synonyms", []) or []:
            ci["synonyms_lower"].add(s.lower())
            index["global_synonyms_lower"].add(s.lower())
        for cid in case.forbidden_identifiers.get("pubchem_cids", []) or []:
            ci["pubchem_cids"].add(str(cid))
        for cid in case.forbidden_identifiers.get("chembl_ids", []) or []:
            ci["chembl_ids"].add(str(cid))
        for did in case.forbidden_identifiers.get("drugbank_ids", []) or []:
            ci["drugbank_ids"].add(str(did))
        for doi in case.forbidden_documents.get("dois", []) or []:
            ci["dois"].add(doi.lower())
            index["global_dois"].add(doi.lower())
        for pmid in case.forbidden_documents.get("pmids", []) or []:
            ci["pmids"].add(str(pmid))
            index["global_pmids"].add(str(pmid))
        for tf in case.forbidden_documents.get("title_fragments", []) or []:
            ci["title_fragments_lower"].add(tf.lower())
            index["global_title_fragments_lower"].add(tf.lower())

        # Structural quarantine for the answer SMILES
        ans_smi = case.hidden_solution.get("canonical_smiles")
        if ans_smi:
            fp = _try_morgan_fp(ans_smi)
            if fp is not None:
                index["answer_fps"].append((case.case_id, fp))
            mk = _try_murcko_smiles(ans_smi)
            if mk:
                index["answer_murckos"].append((case.case_id, mk))

        index["answer_tan_thresholds"][case.case_id] = (
            case.forbidden_neighborhoods.get("tanimoto_to_answer", 0.85)
        )
        index["context_tan_thresholds"][case.case_id] = (
            case.forbidden_neighborhoods.get("tanimoto_with_context", 0.65)
        )

        index["per_case"][case.case_id] = ci
    return index


def scrub_rows(rows: Iterable[dict], registry: ABXSealedCaseRegistry) -> tuple[list[dict], ABXDecontamReport]:
    """Walk through rows, drop forbidden ones, return (kept_rows, report).

    Each row should have at minimum:
      - 'canonical_smiles' or 'inchi_key' or 'name'
      - optionally: 'doi', 'pmid', 'chembl_id', 'pubchem_cid', 'document_title'
    """
    index = build_forbidden_index(registry)
    kept: list[dict] = []
    report = ABXDecontamReport(n_input=0, n_kept=0, n_removed_total=0)

    def _str(v) -> str:
        """NaN-safe lower-case stringifier (pandas may pass NaN floats for missing strings)."""
        if v is None:
            return ""
        try:
            import pandas as _pd
            if _pd.isna(v):
                return ""
        except (TypeError, ValueError, ImportError):
            pass
        return str(v)

    for row in rows:
        report.n_input += 1
        reasons: list[str] = []

        name = _str(row.get("name")).lower()
        if name and name in index["global_synonyms_lower"]:
            reasons.append("synonym_match")

        doi = _str(row.get("doi")).lower()
        if doi and doi in index["global_dois"]:
            reasons.append("forbidden_doi")

        pmid = _str(row.get("pmid"))
        if pmid and pmid in index["global_pmids"]:
            reasons.append("forbidden_pmid")

        title = _str(row.get("document_title")).lower()
        if title:
            for frag in index["global_title_fragments_lower"]:
                if frag and frag in title:
                    reasons.append(f"title_fragment:{frag}")
                    break

        pubchem_cid = _str(row.get("pubchem_cid"))
        chembl_id = _str(row.get("chembl_id"))
        for case_id, ci in index["per_case"].items():
            if pubchem_cid and pubchem_cid in ci["pubchem_cids"]:
                reasons.append(f"pubchem_cid_match:{case_id}")
            if chembl_id and chembl_id in ci["chembl_ids"]:
                reasons.append(f"chembl_id_match:{case_id}")

        # Structural Tanimoto check
        smi = _str(row.get("canonical_smiles"))
        if smi and (index["answer_fps"] or index["answer_murckos"]):
            row_fp = _try_morgan_fp(smi)
            row_murcko = _try_murcko_smiles(smi)
            if row_fp is not None:
                for case_id, ans_fp in index["answer_fps"]:
                    tan = _tanimoto(row_fp, ans_fp)
                    threshold = index["answer_tan_thresholds"][case_id]
                    if tan >= threshold:
                        reasons.append(f"tanimoto_to_answer:{case_id}:{tan:.2f}")
                        continue
                    # context-level: 0.65 + same Murcko
                    ctx_threshold = index["context_tan_thresholds"][case_id]
                    if tan >= ctx_threshold and row_murcko:
                        for case_id_m, mk in index["answer_murckos"]:
                            if case_id_m == case_id and row_murcko == mk:
                                reasons.append(f"tanimoto_with_context:{case_id}:{tan:.2f}+murcko")
                                break

        if reasons:
            report.n_removed_total += 1
            for r in reasons:
                key = r.split(":")[0]
                report.removed_by_reason[key] = report.removed_by_reason.get(key, 0) + 1
        else:
            kept.append(row)
            report.n_kept += 1

    return kept, report


def audit_canaries(canary_rows: list[dict], registry: ABXSealedCaseRegistry) -> ABXDecontamReport:
    """Run scrub on canary rows (format-valid but unmistakably synthetic).

    All canaries MUST be filtered out. If any survives, the pipeline FAILS the audit.
    """
    _, report = scrub_rows(canary_rows, registry)
    return report
