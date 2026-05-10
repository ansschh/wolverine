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


def _csp_standardize(mol):
    """Run chembl_structure_pipeline standardisation. Returns mol or None.

    chembl_structure_pipeline returns vary across versions:
      - 1.x: standardize_molblock(block) -> str
      - some versions: -> tuple(str, exclude_flag) or (str, list_warnings)
    We handle both, and fall back to None silently so the caller can keep
    going with plain RDKit canonicalisation.
    """
    if not _HAVE_CSP:
        return None
    try:
        block = Chem.MolToMolBlock(mol)
        std = _csp.standardize_molblock(block)
        std_block = std[0] if isinstance(std, tuple) else std
        std_mol = Chem.MolFromMolBlock(std_block)
        return std_mol if std_mol is not None else None
    except Exception:
        return None


def _csp_parent(mol):
    """Run chembl_structure_pipeline desalt/parent. Returns mol or None on failure."""
    if not _HAVE_CSP:
        return None
    try:
        block = Chem.MolToMolBlock(mol)
        parent = _csp.get_parent_molblock(block)
        parent_block = parent[0] if isinstance(parent, tuple) else parent
        parent_mol = Chem.MolFromMolBlock(parent_block)
        return parent_mol if parent_mol is not None else None
    except Exception:
        return None


@lru_cache(maxsize=100_000)
def canonicalize_smiles(
    smiles: str,
    *,
    desalt: bool = True,
    neutralize: bool = True,
    keep_stereo: bool = True,
) -> str | None:
    """RDKit canonical SMILES with chembl_structure_pipeline standardisation.

    If `chembl_structure_pipeline` is installed and works, runs full
    standardise + desalt. If CSP fails (API mismatch / unsupported molecule),
    falls back to plain RDKit canonical SMILES — never returns None for a
    valid molecule just because CSP errored.

    Returns None ONLY on RDKit parse failure (truly invalid SMILES).
    LRU-cached because the same SMILES appears thousands of times across
    source merges.
    """
    _ensure_chem()
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Best-effort CSP standardise (fall back to plain RDKit if it errors).
        std_mol = _csp_standardize(mol) if _HAVE_CSP else None
        if std_mol is not None:
            mol = std_mol
            if desalt:
                parent_mol = _csp_parent(mol)
                if parent_mol is not None:
                    mol = parent_mol
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
