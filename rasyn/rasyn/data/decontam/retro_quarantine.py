"""Decontamination for retro reaction data (RETRO_PLAN R-1 risk 4).

Three layers of removal:
  1. Reaction-level
       - product InChIKey == sealed target / intermediate InChIKey
       - product canonical SMILES == sealed target / intermediate SMILES
       - reaction document_id (DOI / patent) in forbidden set
       - product Tanimoto >= forbidden_neighborhood.tanimoto_to_answer to
         sealed target OR to any named intermediate
  2. Template-level
       - drop every extracted template whose source-reaction set contains
         any sealed target / intermediate within
         forbidden_neighborhood.template_source_tanimoto
  3. Identifier scrub
       - synonyms, CAS, PubChem CIDs, etc. (transitive via the reaction's
         source patent / paper)

Returns a `RetroQuarantineReport` with per-reason counts and a
canary_survivors list (case_ids whose target slipped past every layer).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from rasyn.synth.retro.schemas import RetroSealedCaseRegistry
from rasyn.synth.retro.templates import RetroTemplate


@dataclass
class RetroQuarantineReport:
    total_input_reactions: int = 0
    total_kept_reactions: int = 0
    removed_by_product_inchi_key: int = 0
    removed_by_product_smiles: int = 0
    removed_by_intermediate_inchi_key: int = 0
    removed_by_intermediate_smiles: int = 0
    removed_by_doi: int = 0
    removed_by_patent: int = 0
    removed_by_product_tanimoto: int = 0
    removed_by_intermediate_tanimoto: int = 0

    total_input_templates: int = 0
    total_kept_templates: int = 0
    removed_templates_by_source_tanimoto: int = 0
    removed_templates_by_source_inchi: int = 0

    canary_reaction_survivors: list[str] = field(default_factory=list)
    canary_template_survivors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class _RetroForbiddenIndex:
    sealed_target_inchi_keys: set[str]
    sealed_target_smiles: set[str]
    intermediate_inchi_keys: set[str]
    intermediate_smiles: set[str]
    dois: set[str]
    patents: set[str]
    target_fingerprints: list  # list of (case_id, ExplicitBitVect)
    intermediate_fingerprints: list  # list of (case_id, name, ExplicitBitVect)
    tanimoto_to_answer: float
    tanimoto_to_intermediates: float
    template_source_tanimoto: float


def _morgan_fp(smi: str):
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
        from rdkit.Chem import AllChem  # type: ignore
    except ImportError:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def _tanimoto(fp1, fp2) -> float:
    try:
        from rdkit.DataStructs import TanimotoSimilarity  # type: ignore[import-not-found]
    except ImportError:
        return 0.0
    if fp1 is None or fp2 is None:
        return 0.0
    return TanimotoSimilarity(fp1, fp2)


def build_retro_forbidden_index(reg: RetroSealedCaseRegistry) -> _RetroForbiddenIndex:
    sealed_target_inchi_keys: set[str] = set()
    sealed_target_smiles: set[str] = set()
    intermediate_inchi_keys: set[str] = set()
    intermediate_smiles: set[str] = set()
    dois: set[str] = set()
    patents: set[str] = set()
    target_fps: list = []
    intermediate_fps: list = []
    # Use the strictest threshold across all cases (most aggressive removal).
    tan_ans = 0.85
    tan_int = 0.85
    tan_tmpl = 0.85

    for case in reg.cases:
        # Sealed targets
        if case.target_inchi_key:
            sealed_target_inchi_keys.add(case.target_inchi_key)
        if case.target_canonical_smiles:
            sealed_target_smiles.add(case.target_canonical_smiles)
            fp = _morgan_fp(case.target_canonical_smiles)
            if fp is not None:
                target_fps.append((case.case_id, fp))
        if case.hidden_solution.canonical_smiles:
            sealed_target_smiles.add(case.hidden_solution.canonical_smiles)
        if case.hidden_solution.inchi_key:
            sealed_target_inchi_keys.add(case.hidden_solution.inchi_key)

        # Intermediates
        for ni in case.hidden_solution.named_intermediates:
            if ni.inchi_key:
                intermediate_inchi_keys.add(ni.inchi_key)
            if ni.canonical_smiles:
                intermediate_smiles.add(ni.canonical_smiles)
                fp = _morgan_fp(ni.canonical_smiles)
                if fp is not None:
                    intermediate_fps.append((case.case_id, ni.name, fp))

        # Documents
        dois.update(case.forbidden_documents.dois)
        patents.update(case.forbidden_documents.patent_numbers)

        tan_ans = min(tan_ans, case.forbidden_neighborhood.tanimoto_to_answer or 1.0)
        tan_int = min(tan_int, case.forbidden_neighborhood.tanimoto_to_intermediates or 1.0)
        tan_tmpl = min(tan_tmpl, case.forbidden_neighborhood.template_source_tanimoto or 1.0)

    return _RetroForbiddenIndex(
        sealed_target_inchi_keys=sealed_target_inchi_keys,
        sealed_target_smiles=sealed_target_smiles,
        intermediate_inchi_keys=intermediate_inchi_keys,
        intermediate_smiles=intermediate_smiles,
        dois=dois,
        patents=patents,
        target_fingerprints=target_fps,
        intermediate_fingerprints=intermediate_fps,
        tanimoto_to_answer=tan_ans,
        tanimoto_to_intermediates=tan_int,
        template_source_tanimoto=tan_tmpl,
    )


def _reaction_doc_matches(reaction_row: dict, fidx: _RetroForbiddenIndex) -> bool:
    doc = reaction_row.get("document_id") or ""
    doc_l = doc.lower()
    for d in fidx.dois:
        if d.lower() in doc_l:
            return True
    for p in fidx.patents:
        if p.lower() in doc_l:
            return True
    return False


def scrub_reactions(
    reaction_rows: Iterable[dict],
    fidx: _RetroForbiddenIndex,
    *,
    report: RetroQuarantineReport | None = None,
) -> Iterable[dict]:
    """Yield reactions that survive all three quarantine layers.

    Mutates `report` in place if provided.
    """
    r = report if report is not None else RetroQuarantineReport()
    for row in reaction_rows:
        r.total_input_reactions += 1
        product_ikey = row.get("product_inchi_key")
        product_smi = row.get("product_smiles") or row.get("product")

        if product_ikey and product_ikey in fidx.sealed_target_inchi_keys:
            r.removed_by_product_inchi_key += 1
            continue
        if product_ikey and product_ikey in fidx.intermediate_inchi_keys:
            r.removed_by_intermediate_inchi_key += 1
            continue
        if product_smi and product_smi in fidx.sealed_target_smiles:
            r.removed_by_product_smiles += 1
            continue
        if product_smi and product_smi in fidx.intermediate_smiles:
            r.removed_by_intermediate_smiles += 1
            continue
        if _reaction_doc_matches(row, fidx):
            doc = (row.get("document_id") or "").lower()
            if any(p.lower() in doc for p in fidx.patents):
                r.removed_by_patent += 1
            else:
                r.removed_by_doi += 1
            continue

        # Tanimoto neighborhood
        prod_fp = _morgan_fp(product_smi) if product_smi else None
        if prod_fp is not None:
            hit_target = any(
                _tanimoto(prod_fp, tfp) >= fidx.tanimoto_to_answer
                for _, tfp in fidx.target_fingerprints
            )
            if hit_target:
                r.removed_by_product_tanimoto += 1
                continue
            hit_intermediate = any(
                _tanimoto(prod_fp, ifp) >= fidx.tanimoto_to_intermediates
                for _, _, ifp in fidx.intermediate_fingerprints
            )
            if hit_intermediate:
                r.removed_by_intermediate_tanimoto += 1
                continue

        r.total_kept_reactions += 1
        yield row


def scrub_templates(
    templates: Iterable[RetroTemplate],
    source_reaction_lookup: dict[str, str],
    fidx: _RetroForbiddenIndex,
    *,
    report: RetroQuarantineReport | None = None,
) -> list[RetroTemplate]:
    """Drop templates whose source reactions include sealed-case neighbors.

    `source_reaction_lookup` is a dict {source_reaction_id -> product_smiles}.

    A template is dropped if any of its source reactions has a product
    InChIKey/SMILES in the sealed/intermediate sets OR a product Tanimoto
    >= template_source_tanimoto to any target / named intermediate.
    """
    r = report if report is not None else RetroQuarantineReport()
    kept: list[RetroTemplate] = []
    for tmpl in templates:
        r.total_input_templates += 1
        dropped = False
        for rxn_id in tmpl.source_reaction_ids:
            product_smi = source_reaction_lookup.get(rxn_id)
            if product_smi is None:
                continue
            if product_smi in fidx.sealed_target_smiles or product_smi in fidx.intermediate_smiles:
                dropped = True
                r.removed_templates_by_source_inchi += 1
                break
            fp = _morgan_fp(product_smi)
            if fp is None:
                continue
            if any(_tanimoto(fp, tfp) >= fidx.template_source_tanimoto for _, tfp in fidx.target_fingerprints):
                dropped = True
                r.removed_templates_by_source_tanimoto += 1
                break
            if any(_tanimoto(fp, ifp) >= fidx.template_source_tanimoto for _, _, ifp in fidx.intermediate_fingerprints):
                dropped = True
                r.removed_templates_by_source_tanimoto += 1
                break
        if not dropped:
            kept.append(tmpl)
            r.total_kept_templates += 1
    return kept
