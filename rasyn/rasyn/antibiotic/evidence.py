"""Spec §7.2 candidate evidence packet builder.

Returns the dict-shaped evidence packet exactly as specified:
  candidate_id
  structure: { canonical_smiles, inchi_key, molecular_graph }
  computed_descriptors: { molecular_weight, clogp, tpsa, hbd, hba,
                          formal_charge, rotatable_bonds, aromatic_ring_count, fsp3 }
  chemical_alerts: { pains_alerts, brenk_alerts, reactive_alerts, aggregation_risk }
  similarity_evidence: { nearest_known_antibiotic_similarity,
                         nearest_training_active_similarity,
                         nearest_training_inactive_similarity,
                         scaffold_novelty_score }
  predicted_evidence:  { antibacterial_scores_by_organism, cytotoxicity_risk,
                         hemolysis_risk, artifact_risk,
                         synthesizability_score, uncertainty_score }
  proposer_metadata:   { proposer_sources, seed_fragment_id, generation_guidance }

Alerts are evaluated via RDKit SMARTS catalogs (PAINS, Brenk). Reactive
alerts use a curated SMARTS list of common reactive groups
(Michael acceptors, alkyl halides, acyl chlorides, etc.). Aggregation
risk uses ChEMBL's aggregator-rule heuristic (high MW + high logP).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import hashlib


# Curated reactive SMARTS list (well-known electrophiles + colloidal-aggregator triggers).
REACTIVE_SMARTS: list[tuple[str, str]] = [
    ("acyl_halide",       "[C](=O)[Cl,Br,I]"),
    ("alkyl_halide",      "[#6;A][Cl,Br,I]"),
    ("aldehyde",          "[CX3H1](=O)[#6]"),
    ("isocyanate",        "N=C=O"),
    ("isothiocyanate",    "N=C=S"),
    ("epoxide",           "C1OC1"),
    ("aziridine",         "C1CN1"),
    ("vinyl_sulfone",     "C=CS(=O)(=O)"),
    ("michael_acceptor",  "C=CC(=O)"),
    ("disulfide",         "SS"),
    ("nitro_aromatic",    "[$(c[N+](=O)[O-])]"),
]


def _smarts_hits(mol, smarts_list: list[tuple[str, str]]) -> list[str]:
    from rdkit import Chem
    out = []
    for name, sm in smarts_list:
        patt = Chem.MolFromSmarts(sm)
        if patt and mol.HasSubstructMatch(patt):
            out.append(name)
    return out


def _pains_alerts(mol) -> list[str]:
    """Use RDKit's built-in FilterCatalog for PAINS / Brenk where available."""
    try:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        cat = FilterCatalog(params)
        return [m.GetDescription() for m in cat.GetMatches(mol)]
    except Exception:
        return []


def _brenk_alerts(mol) -> list[str]:
    try:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        cat = FilterCatalog(params)
        return [m.GetDescription() for m in cat.GetMatches(mol)]
    except Exception:
        return []


def _aggregation_risk(mw: float, clogp: float) -> str:
    """Heuristic from Shoichet et al. — colloidal aggregators tend to have
    MW > 400 AND clogP > 4. We bucket as low/moderate/high."""
    if mw > 500 and clogp > 5.5:
        return "high"
    if mw > 400 and clogp > 4.0:
        return "moderate"
    return "low"


def build_evidence_packet(
    candidate_smiles: str,
    *,
    candidate_id: Optional[str] = None,
    organism: Optional[str] = None,
    antibacterial_scores_by_organism: Optional[dict[str, float]] = None,
    cytotox_risk: Optional[float] = None,
    hemolysis_risk: Optional[float] = None,
    artifact_risk: Optional[float] = None,
    synthesizability_score: Optional[float] = None,
    uncertainty_score: Optional[float] = None,
    nearest_known_antibiotic_similarity: Optional[float] = None,
    nearest_training_active_similarity: Optional[float] = None,
    nearest_training_inactive_similarity: Optional[float] = None,
    scaffold_novelty_score: Optional[float] = None,
    proposer_sources: Optional[list[str]] = None,
    seed_fragment_id: Optional[str] = None,
    generation_guidance: Optional[dict] = None,
) -> dict:
    """Build the full §7.2 evidence packet dict.

    No fields are hallucinated; missing values stay None (Pydantic ABXRankerOutput
    consumers ignore None gracefully).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Lipinski, inchi
    mol = Chem.MolFromSmiles(candidate_smiles)
    if mol is None:
        return {"candidate_id": candidate_id, "structure": {"canonical_smiles": candidate_smiles}}

    canonical = Chem.MolToSmiles(mol, canonical=True)
    ik = inchi.MolToInchiKey(mol)
    if candidate_id is None:
        candidate_id = "CAND_" + hashlib.md5(canonical.encode()).hexdigest()[:10]

    mw = Descriptors.MolWt(mol)
    clogp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rb = Lipinski.NumRotatableBonds(mol)
    arc = Lipinski.NumAromaticRings(mol)
    fsp3 = Lipinski.FractionCSP3(mol)
    fc = Chem.rdmolops.GetFormalCharge(mol)

    pains = _pains_alerts(mol)
    brenk = _brenk_alerts(mol)
    reactive = _smarts_hits(mol, REACTIVE_SMARTS)

    return {
        "candidate_id": candidate_id,
        "structure": {
            "canonical_smiles": canonical,
            "inchi_key": ik,
            "molecular_graph": "rdkit_mol_native",
        },
        "computed_descriptors": {
            "molecular_weight": float(mw),
            "clogp": float(clogp),
            "tpsa": float(tpsa),
            "hbd": int(hbd),
            "hba": int(hba),
            "formal_charge": int(fc),
            "rotatable_bonds": int(rb),
            "aromatic_ring_count": int(arc),
            "fsp3": float(fsp3),
        },
        "chemical_alerts": {
            "pains_alerts": pains,
            "brenk_alerts": brenk,
            "reactive_alerts": reactive,
            "aggregation_risk": _aggregation_risk(mw, clogp),
        },
        "similarity_evidence": {
            "nearest_known_antibiotic_similarity": nearest_known_antibiotic_similarity,
            "nearest_training_active_similarity": nearest_training_active_similarity,
            "nearest_training_inactive_similarity": nearest_training_inactive_similarity,
            "scaffold_novelty_score": scaffold_novelty_score,
        },
        "predicted_evidence": {
            "antibacterial_scores_by_organism": antibacterial_scores_by_organism or {},
            "cytotoxicity_risk": cytotox_risk,
            "hemolysis_risk": hemolysis_risk,
            "artifact_risk": artifact_risk,
            "synthesizability_score": synthesizability_score,
            "uncertainty_score": uncertainty_score,
        },
        "proposer_metadata": {
            "proposer_sources": proposer_sources or [],
            "seed_fragment_id": seed_fragment_id,
            "generation_guidance": generation_guidance or {},
        },
    }
