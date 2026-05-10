"""Morgan fingerprints, Tanimoto similarity, Murcko scaffolds."""

from __future__ import annotations

from functools import lru_cache

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import DataStructs

    _HAVE_RDKIT = True
except ImportError:  # pragma: no cover
    _HAVE_RDKIT = False
    Chem = None  # type: ignore[assignment]


def _ensure():
    if not _HAVE_RDKIT:
        raise RuntimeError("RDKit not installed. `pip install -e '.[chem]'`")


@lru_cache(maxsize=100_000)
def morgan_bits(smiles: str, radius: int = 2, n_bits: int = 1024):
    """Returns an RDKit ExplicitBitVect Morgan fingerprint, or None."""
    _ensure()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto(a, b) -> float:
    """Tanimoto similarity of two RDKit ExplicitBitVect fingerprints."""
    _ensure()
    return DataStructs.TanimotoSimilarity(a, b)


def tanimoto_smiles(smiles_a: str, smiles_b: str, radius: int = 2, n_bits: int = 1024) -> float | None:
    """Tanimoto on two SMILES; returns None if either fails to parse."""
    fa = morgan_bits(smiles_a, radius, n_bits)
    fb = morgan_bits(smiles_b, radius, n_bits)
    if fa is None or fb is None:
        return None
    return tanimoto(fa, fb)


@lru_cache(maxsize=100_000)
def murcko_scaffold_smiles(smiles: str) -> str | None:
    """Bemis-Murcko scaffold SMILES, or None."""
    _ensure()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaff = MurckoScaffold.GetScaffoldForMol(mol)
    if scaff is None or scaff.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaff, canonical=True)


def murcko_match(smiles_a: str, smiles_b: str) -> bool:
    """True iff two molecules share Murcko scaffold (canonical-SMILES equality)."""
    sa = murcko_scaffold_smiles(smiles_a)
    sb = murcko_scaffold_smiles(smiles_b)
    return sa is not None and sa == sb


def heavy_atom_difference(smiles_a: str, smiles_b: str) -> int | None:
    """Absolute difference in heavy atom counts; None if either parse fails."""
    _ensure()
    ma, mb = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if ma is None or mb is None:
        return None
    return abs(ma.GetNumHeavyAtoms() - mb.GetNumHeavyAtoms())
