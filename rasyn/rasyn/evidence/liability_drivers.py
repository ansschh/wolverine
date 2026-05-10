"""Rule-based liability driver detection (v1; auxiliary models extend later).

Each liability has a small set of structural drivers we can detect via
SMARTS or descriptor thresholds. These feed the `liability_drivers_in_parent`
field of the evidence packet.
"""

from __future__ import annotations

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors

    _HAVE_RDKIT = True
except ImportError:  # pragma: no cover
    _HAVE_RDKIT = False
    Chem = None  # type: ignore[assignment]


# SMARTS heuristics. NOT exhaustive; these are the obvious red flags
# medicinal chemists screen for first.
HERG_SMARTS = {
    "tertiary_amine": "[$([NX3;H0;!R](C)(C)C)]",
    "basic_piperidine": "[$([NX3;H1;R1](C)C)]C1CCCCC1",
    "lipophilic_aromatic_amine": "c[NH]c",
}

SOLUBILITY_SMARTS = {
    "phenyl_ring": "c1ccccc1",
    "naphthalene": "c1ccc2ccccc2c1",
}

METSTAB_SMARTS = {
    "benzylic_ch": "[$([CH2]c)]",
    "ortho_methoxy": "[$(c1c([OCH3])cccc1)]",
    "tert_butyl": "C(C)(C)C",
    "n_methyl": "[$([NX3;H0]([CH3])[#6])]",
}

# Descriptor-threshold drivers (per liability).
HERG_DESCRIPTOR_DRIVERS = (
    ("high_logP", lambda d: d["log_p"] > 3.5),
    ("low_TPSA", lambda d: d["tpsa"] < 75),
)
SOLUBILITY_DESCRIPTOR_DRIVERS = (
    ("high_logP", lambda d: d["log_p"] > 4),
    ("high_aromatic_ring_count", lambda d: d["aromatic_rings"] >= 3),
    ("low_fsp3", lambda d: d["fsp3"] < 0.2),
)
METSTAB_DESCRIPTOR_DRIVERS = (
    ("high_logP", lambda d: d["log_p"] > 3.5),
)


_DRIVER_TABLE = {
    "hERG": (HERG_SMARTS, HERG_DESCRIPTOR_DRIVERS),
    "solubility": (SOLUBILITY_SMARTS, SOLUBILITY_DESCRIPTOR_DRIVERS),
    "metabolic_stability": (METSTAB_SMARTS, METSTAB_DESCRIPTOR_DRIVERS),
}


def _smarts_hits(mol, smarts_map: dict[str, str]) -> list[str]:
    hits: list[str] = []
    for name, smarts in smarts_map.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        if mol.HasSubstructMatch(patt):
            hits.append(name)
    return hits


def _descriptor_hits(descriptors: dict, table) -> list[str]:
    return [name for name, fn in table if fn(descriptors)]


def detect_liability_drivers(smiles: str, liability_type: str, descriptors: dict | None = None) -> list[str]:
    """Return a list of human-readable driver tags (e.g. 'high_logP', 'tertiary_amine')."""
    if not _HAVE_RDKIT:
        raise RuntimeError("RDKit required.")
    table = _DRIVER_TABLE.get(liability_type)
    if not table:
        return []
    smarts_map, desc_table = table
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    out: list[str] = list(_smarts_hits(mol, smarts_map))
    if descriptors is None:
        descriptors = {
            "log_p": float(Crippen.MolLogP(mol)),
            "tpsa": float(Descriptors.TPSA(mol)),
            "aromatic_rings": int(mol.GetRingInfo().NumAromaticRings()),
            "fsp3": float(sum(1 for a in mol.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP3) / max(mol.GetNumHeavyAtoms(), 1)),
        }
    out.extend(_descriptor_hits(descriptors, desc_table))
    return out
