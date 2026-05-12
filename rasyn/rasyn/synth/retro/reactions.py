"""Reaction-level utilities: canonicalization, atom-mapping, classification.

Used by R-1 (curation), R-2 (proposer training), R-3 (validator training).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# ===== Canonicalization =====

def canonicalize_smiles(smi: str) -> str | None:
    """Return canonical SMILES (RDKit) or None if unparseable."""
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except ImportError:
        return smi  # best-effort fallback when RDKit not available
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True)


def inchi_key_from_smiles(smi: str) -> str | None:
    """Return 27-char standard InChIKey or None if unparseable."""
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
        from rdkit.Chem.inchi import MolToInchiKey
    except ImportError:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return MolToInchiKey(m)
    except Exception:
        return None


def canonicalize_reaction(reactants: Iterable[str], product: str) -> tuple[list[str], str] | None:
    """Canonicalize each reactant + product. Returns (reactants, product) or None on failure."""
    canon_reactants: list[str] = []
    for r in reactants:
        c = canonicalize_smiles(r)
        if c is None:
            return None
        canon_reactants.append(c)
    canon_product = canonicalize_smiles(product)
    if canon_product is None:
        return None
    return canon_reactants, canon_product


# ===== Atom mapping (RXNMapper) =====

@dataclass
class AtomMapper:
    """Wrapper around RXNMapper. Loads the model lazily.

    Reference: https://github.com/rxn4chemistry/rxnmapper

    Usage:
        mapper = AtomMapper()
        result = mapper.map_one("CCO.CC(=O)Cl>>CCOC(C)=O")
        # result["mapped_rxn"] is the atom-mapped reaction SMILES.
    """

    batch_size: int = 32
    device: str = "cpu"
    _model = None  # type: ignore[assignment]

    def _load(self):
        if self._model is None:
            try:
                from rxnmapper import RXNMapper  # type: ignore[import-not-found]
            except ImportError as e:  # noqa: BLE001 -- explicit
                raise RuntimeError("rxnmapper not installed; pip install rxnmapper") from e
            self._model = RXNMapper()
        return self._model

    def map_one(self, rxn_smiles: str) -> dict | None:
        mapper = self._load()
        try:
            outs = mapper.get_attention_guided_atom_maps([rxn_smiles])
        except Exception:
            return None
        if not outs:
            return None
        return outs[0]

    def map_batch(self, rxn_smiles_list: list[str]) -> list[dict | None]:
        mapper = self._load()
        results: list[dict | None] = []
        for i in range(0, len(rxn_smiles_list), self.batch_size):
            chunk = rxn_smiles_list[i:i + self.batch_size]
            try:
                outs = mapper.get_attention_guided_atom_maps(chunk)
            except Exception:
                results.extend([None] * len(chunk))
                continue
            results.extend(outs)
        return results


# ===== Reaction classification (RXNFP) =====

@dataclass
class ReactionClassifier:
    """Wrapper around RXNFP (Schwaller et al., NMI 2021).

    Reference: https://github.com/rxn4chemistry/rxnfp
    Produces 256-d reaction fingerprints + a class label from a fine-tuned
    classifier head over 998 Schneider classes.
    """

    fingerprint_model_name: str = "rxnfp"  # see rxnfp.transformer_fingerprints
    _generator = None  # type: ignore[assignment]

    def _load(self):
        if self._generator is None:
            try:
                from rxnfp.transformer_fingerprints import (  # type: ignore[import-not-found]
                    RXNBERTFingerprintGenerator,
                    get_default_model_and_tokenizer,
                )
            except ImportError as e:  # noqa: BLE001 -- explicit
                raise RuntimeError("rxnfp not installed; pip install rxnfp") from e
            model, tokenizer = get_default_model_and_tokenizer()
            self._generator = RXNBERTFingerprintGenerator(model, tokenizer)
        return self._generator

    def fingerprint(self, rxn_smiles: str) -> list[float]:
        gen = self._load()
        return gen.convert(rxn_smiles)

    def fingerprint_batch(self, rxn_smiles_list: list[str]) -> list[list[float]]:
        gen = self._load()
        return gen.convert_batch(rxn_smiles_list)


# ===== Coarse class bucketing (heuristic; RETRO_PLAN R-1 v1 fallback) =====

_AMIDE_KEYS = ("amide", "peptide_coupling", "C(=O)N")
_SUZUKI_KEYS = ("Suzuki", "suzuki")
_BUCHWALD_KEYS = ("Buchwald", "buchwald_hartwig", "C-N_coupling")
_REDUCTIVE_AMINE_KEYS = ("reductive_amination",)
_SN2_KEYS = ("SN2",)
_SNAR_KEYS = ("SNAr",)
_NEGISHI_KEYS = ("Negishi",)
_WITTIG_KEYS = ("Wittig",)
_CLICK_KEYS = ("click", "azide_alkyne", "CuAAC")
_PROT_KEYS = ("protection", "deprotection")


def bucketize_class_name(raw: str | None) -> str:
    """Map a free-text or Schneider class name to one of the 12 coarse buckets.

    Returns one of:
        amide_coupling, suzuki_coupling, buchwald_hartwig,
        reductive_amination, sn2, sn_ar, negishi, wittig, click,
        protection_deprotection, other_cross_coupling, unclassified
    """
    if not raw:
        return "unclassified"
    s = raw.lower()
    if any(k.lower() in s for k in _AMIDE_KEYS):
        return "amide_coupling"
    if any(k.lower() in s for k in _SUZUKI_KEYS):
        return "suzuki_coupling"
    if any(k.lower() in s for k in _BUCHWALD_KEYS):
        return "buchwald_hartwig"
    if any(k.lower() in s for k in _REDUCTIVE_AMINE_KEYS):
        return "reductive_amination"
    if any(k.lower() in s for k in _SN2_KEYS):
        return "sn2"
    if any(k.lower() in s for k in _SNAR_KEYS):
        return "sn_ar"
    if any(k.lower() in s for k in _NEGISHI_KEYS):
        return "negishi"
    if any(k.lower() in s for k in _WITTIG_KEYS):
        return "wittig"
    if any(k.lower() in s for k in _CLICK_KEYS):
        return "click"
    if any(k.lower() in s for k in _PROT_KEYS):
        return "protection_deprotection"
    if "coupling" in s or "cross" in s:
        return "other_cross_coupling"
    return "unclassified"
