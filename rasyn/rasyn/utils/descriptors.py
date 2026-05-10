"""RDKit descriptor block computation for the evidence builder."""

from __future__ import annotations

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors

    _HAVE_RDKIT = True
except ImportError:  # pragma: no cover
    _HAVE_RDKIT = False

from rasyn.schemas.evidence import DescriptorBlock, DescriptorDeltas


def _ensure():
    if not _HAVE_RDKIT:
        raise RuntimeError("RDKit not installed. `pip install -e '.[chem]'`")


def descriptor_block_from_smiles(smiles: str) -> DescriptorBlock | None:
    """Compute the canonical descriptor block. Returns None on parse failure."""
    _ensure()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return DescriptorBlock(
        mw=float(Descriptors.MolWt(mol)),
        log_p=float(Crippen.MolLogP(mol)),
        tpsa=float(Descriptors.TPSA(mol)),
        hbd=int(Lipinski.NumHDonors(mol)),
        hba=int(Lipinski.NumHAcceptors(mol)),
        rotatable_bonds=int(Lipinski.NumRotatableBonds(mol)),
        aromatic_rings=int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        fsp3=float(rdMolDescriptors.CalcFractionCSP3(mol)),
        formal_charge=int(Chem.GetFormalCharge(mol)),
    )


def descriptor_deltas(parent: DescriptorBlock, candidate: DescriptorBlock) -> DescriptorDeltas:
    """candidate - parent for each descriptor."""
    return DescriptorDeltas(
        delta_mw=candidate.mw - parent.mw,
        delta_log_p=candidate.log_p - parent.log_p,
        delta_tpsa=candidate.tpsa - parent.tpsa,
        delta_hbd=candidate.hbd - parent.hbd,
        delta_hba=candidate.hba - parent.hba,
        delta_rotatable_bonds=candidate.rotatable_bonds - parent.rotatable_bonds,
        delta_aromatic_rings=candidate.aromatic_rings - parent.aromatic_rings,
        delta_fsp3=candidate.fsp3 - parent.fsp3,
        delta_formal_charge=candidate.formal_charge - parent.formal_charge,
        delta_log_d=(
            (candidate.log_d_estimate - parent.log_d_estimate)
            if (candidate.log_d_estimate is not None and parent.log_d_estimate is not None)
            else None
        ),
        delta_pka=(
            (candidate.pka_estimate - parent.pka_estimate)
            if (candidate.pka_estimate is not None and parent.pka_estimate is not None)
            else None
        ),
    )
