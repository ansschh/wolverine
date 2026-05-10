"""Pass-0 decontamination: scrub raw rows against the sealed-case registry.

Three layers of removal:
  1. Identifier match  (exact SMILES, InChIKey, ChEMBL/PubChem ID, CAS, synonym)
  2. Document quarantine  (DOI, PMID, ChEMBL doc ID, patent)
  3. Neighborhood removal  (Tanimoto >= threshold, optionally with same-Murcko + same-target context)

Returns a `QuarantineReport` with per-reason counts so the audit pack can
explain every removal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from rasyn.schemas.registry import SealedCaseRegistry
from rasyn.utils.canonicalize import standardize_pair
from rasyn.utils.similarity import morgan_bits, murcko_match, tanimoto


@dataclass
class QuarantineReport:
    """Counts removed by reason; flat dict serialises cleanly to audit JSON."""

    total_input: int = 0
    total_kept: int = 0
    removed_by_smiles: int = 0
    removed_by_inchi_key: int = 0
    removed_by_synonym: int = 0
    removed_by_chembl_id: int = 0
    removed_by_pubchem_cid: int = 0
    removed_by_cas: int = 0
    removed_by_doi: int = 0
    removed_by_pmid: int = 0
    removed_by_chembl_doc_id: int = 0
    removed_by_assay_id: int = 0
    removed_by_neighbor_to_answer: int = 0
    removed_by_neighbor_with_context: int = 0
    canary_survivors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return d


@dataclass
class _ForbiddenIndex:
    """Pre-computed lookup tables across all sealed cases."""

    canonical_smiles: set[str]
    inchi_keys: set[str]
    synonyms_lower: set[str]
    chembl_ids: set[str]
    pubchem_cids: set[str]
    cas_numbers: set[str]
    dois: set[str]
    pmids: set[str]
    chembl_doc_ids: set[str]
    chembl_assay_ids: set[str]
    pubchem_aids: set[str]
    answer_morgan_fps: list  # list of (case_id, ExplicitBitVect, target_id, murcko_smiles)
    quarantine_thresholds: dict[str, tuple[float, float, bool, bool]]  # case_id -> (t_ans, t_ctx, same_murcko, same_target)


def build_forbidden_index(reg: SealedCaseRegistry) -> _ForbiddenIndex:
    """Pre-compute lookup tables and answer fingerprints from the registry."""
    canonical_smiles: set[str] = set()
    inchi_keys: set[str] = set()
    synonyms_lower: set[str] = set()
    chembl_ids: set[str] = set()
    pubchem_cids: set[str] = set()
    cas_numbers: set[str] = set()
    dois: set[str] = set()
    pmids: set[str] = set()
    chembl_doc_ids: set[str] = set()
    chembl_assay_ids: set[str] = set()
    pubchem_aids: set[str] = set()
    answer_fps: list = []
    thresholds: dict[str, tuple[float, float, bool, bool]] = {}

    for case in reg.cases:
        # parent + answer canonical SMILES + InChIKeys (if populated)
        for ref in (case.parent, case.answer):
            if ref.canonical_smiles:
                canonical_smiles.add(ref.canonical_smiles)
            if ref.inchi_key:
                inchi_keys.add(ref.inchi_key)
            if ref.chembl_id:
                chembl_ids.add(ref.chembl_id)
            if ref.pubchem_cid:
                pubchem_cids.add(ref.pubchem_cid)
            if ref.cas_number:
                cas_numbers.add(ref.cas_number)
            if ref.name:
                synonyms_lower.add(ref.name.lower())

        canonical_smiles.update(case.forbidden_identifiers.smiles_variants)
        inchi_keys.update(case.forbidden_identifiers.inchi_keys)
        chembl_ids.update(case.forbidden_identifiers.chembl_ids)
        pubchem_cids.update(case.forbidden_identifiers.pubchem_cids)
        cas_numbers.update(case.forbidden_identifiers.cas_numbers)
        synonyms_lower.update(s.lower() for s in case.forbidden_identifiers.synonyms)
        synonyms_lower.update(s.lower() for s in case.forbidden_identifiers.iupac_names)

        dois.update(case.forbidden_documents.dois)
        pmids.update(case.forbidden_documents.pmids)
        chembl_doc_ids.update(case.forbidden_documents.chembl_doc_ids)
        chembl_assay_ids.update(case.forbidden_assays.chembl_assay_ids)
        pubchem_aids.update(case.forbidden_assays.pubchem_aids)

        thresholds[case.case_id] = (
            case.quarantine.tanimoto_to_answer,
            case.quarantine.tanimoto_with_context,
            case.quarantine.require_same_murcko_for_context,
            case.quarantine.require_same_target_for_context,
        )

        if case.answer.canonical_smiles:
            fp = morgan_bits(case.answer.canonical_smiles)
            if fp is not None:
                from rasyn.utils.similarity import murcko_scaffold_smiles

                ms = murcko_scaffold_smiles(case.answer.canonical_smiles)
                target_id = "TODO_target_for_" + case.case_id  # populator should fill
                answer_fps.append((case.case_id, fp, target_id, ms))

    return _ForbiddenIndex(
        canonical_smiles=canonical_smiles,
        inchi_keys=inchi_keys,
        synonyms_lower=synonyms_lower,
        chembl_ids=chembl_ids,
        pubchem_cids=pubchem_cids,
        cas_numbers=cas_numbers,
        dois=dois,
        pmids=pmids,
        chembl_doc_ids=chembl_doc_ids,
        chembl_assay_ids=chembl_assay_ids,
        pubchem_aids=pubchem_aids,
        answer_morgan_fps=answer_fps,
        quarantine_thresholds=thresholds,
    )


def scrub_rows(
    rows: Iterable[dict],
    reg: SealedCaseRegistry,
    *,
    canonicalize: bool = True,
) -> tuple[list[dict], QuarantineReport]:
    """Apply Pass-0 quarantine to a stream of rows.

    Each row is a dict with keys (any subset; missing keys are ignored):
      - smiles, inchi_key, chembl_id, pubchem_cid, cas, synonyms (list)
      - doi, pmid, chembl_doc_id, chembl_assay_id, pubchem_aid
      - target_chembl_id, target_pref_name (used for context-aware neighbor)

    Returns (kept_rows, report).
    """
    idx = build_forbidden_index(reg)
    kept: list[dict] = []
    report = QuarantineReport()

    for row in rows:
        report.total_input += 1
        smiles = row.get("smiles")
        ik = row.get("inchi_key")
        if canonicalize and smiles and not ik:
            cs, computed_ik = standardize_pair(smiles)
            if cs:
                row = {**row, "smiles": cs}
                ik = computed_ik
                row["inchi_key"] = ik

        if ik and ik in idx.inchi_keys:
            report.removed_by_inchi_key += 1
            continue
        if smiles and smiles in idx.canonical_smiles:
            report.removed_by_smiles += 1
            continue
        if (cid := row.get("pubchem_cid")) and str(cid) in idx.pubchem_cids:
            report.removed_by_pubchem_cid += 1
            continue
        if (chid := row.get("chembl_id")) and chid in idx.chembl_ids:
            report.removed_by_chembl_id += 1
            continue
        if (cas := row.get("cas")) and cas in idx.cas_numbers:
            report.removed_by_cas += 1
            continue

        syns = row.get("synonyms") or []
        if any(s.lower() in idx.synonyms_lower for s in syns):
            report.removed_by_synonym += 1
            continue

        if (doi := row.get("doi")) and doi in idx.dois:
            report.removed_by_doi += 1
            continue
        if (pmid := row.get("pmid")) and str(pmid) in idx.pmids:
            report.removed_by_pmid += 1
            continue
        if (dcid := row.get("chembl_doc_id")) and dcid in idx.chembl_doc_ids:
            report.removed_by_chembl_doc_id += 1
            continue
        if (aid := row.get("chembl_assay_id")) and aid in idx.chembl_assay_ids:
            report.removed_by_assay_id += 1
            continue

        if smiles and idx.answer_morgan_fps:
            removed = False
            row_fp = morgan_bits(smiles)
            if row_fp is not None:
                for case_id, ans_fp, _target_id, ans_murcko in idx.answer_morgan_fps:
                    sim = tanimoto(row_fp, ans_fp)
                    t_ans, t_ctx, same_murcko_req, same_target_req = idx.quarantine_thresholds[case_id]
                    if sim >= t_ans:
                        report.removed_by_neighbor_to_answer += 1
                        removed = True
                        break
                    if sim >= t_ctx:
                        ok_murcko = (not same_murcko_req) or (
                            ans_murcko is not None and murcko_match(smiles, row.get("smiles", smiles))
                        )
                        # `same_target` requires a target field on the row; if missing, conservatively keep.
                        if ok_murcko:
                            report.removed_by_neighbor_with_context += 1
                            removed = True
                            break
            if removed:
                continue

        kept.append(row)
        report.total_kept += 1

    return kept, report
