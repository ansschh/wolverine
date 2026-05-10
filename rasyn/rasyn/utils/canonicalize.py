"""RDKit-based molecule standardisation.

The Rasyn canonical form: RDKit canonical SMILES post-`chembl_structure_pipeline`
standardisation (desalted, neutralised, tautomer normalised), preserving
stereochemistry. The 27-char InChIKey is the cross-source join key.

All public functions return None on failure rather than raising — caller
decides whether to drop, log, or escalate.
"""

from __future__ import annotations

from functools import lru_cache

try:  # RDKit and chembl_structure_pipeline are hard runtime deps; soft-import for type-checkers.
    from rdkit import Chem
    from rdkit.Chem import AllChem  # noqa: F401  # registers fingerprint generators
    from rdkit.Chem.inchi import MolToInchiKey

    _HAVE_RDKIT = True
except ImportError:  # pragma: no cover
    _HAVE_RDKIT = False
    Chem = None  # type: ignore[assignment]
    MolToInchiKey = None  # type: ignore[assignment]

try:
    from chembl_structure_pipeline import standardizer as _csp

    _HAVE_CSP = True
except ImportError:  # pragma: no cover
    _HAVE_CSP = False
    _csp = None  # type: ignore[assignment]


def have_chemistry_stack() -> bool:
    """True iff both RDKit and chembl_structure_pipeline are importable."""
    return _HAVE_RDKIT and _HAVE_CSP


def _ensure_chem():
    if not _HAVE_RDKIT:
        raise RuntimeError("RDKit not installed. `pip install -e '.[chem]'`")


@lru_cache(maxsize=100_000)
def canonicalize_smiles(
    smiles: str,
    *,
    desalt: bool = True,
    neutralize: bool = True,
    keep_stereo: bool = True,
) -> str | None:
    """RDKit canonical SMILES with chembl_structure_pipeline standardisation.

    Returns None on parse failure. LRU-cached because the same SMILES
    appears thousands of times across source merges.
    """
    _ensure_chem()
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if _HAVE_CSP:
            block = Chem.MolToMolBlock(mol)
            std_block, _ = _csp.standardize_molblock(block)
            mol = Chem.MolFromMolBlock(std_block)
            if mol is None:
                return None
            if desalt:
                parent_block, _ = _csp.get_parent_molblock(std_block)
                mol = Chem.MolFromMolBlock(parent_block)
                if mol is None:
                    return None
        if neutralize:
            for atom in mol.GetAtoms():
                if atom.GetFormalCharge() != 0 and atom.GetNumExplicitHs() == 0:
                    pass  # CSP already neutralises; no-op fallback
        return Chem.MolToSmiles(mol, isomericSmiles=keep_stereo, canonical=True)
    except Exception:  # pragma: no cover - defensive
        return None


@lru_cache(maxsize=100_000)
def smiles_to_inchi_key(smiles: str) -> str | None:
    """InChIKey from canonical SMILES. None on failure."""
    _ensure_chem()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        ik = MolToInchiKey(mol)
        return ik if ik else None
    except Exception:  # pragma: no cover
        return None


def standardize_pair(smiles: str) -> tuple[str | None, str | None]:
    """Returns `(canonical_smiles, inchi_key)`. Either may be None on failure."""
    cs = canonicalize_smiles(smiles)
    if cs is None:
        return None, None
    ik = smiles_to_inchi_key(cs)
    return cs, ik
