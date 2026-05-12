"""Retrieval proposer (RETRO_PLAN R-2 Channel 4).

FAISS index of product Morgan FPs (2048-bit, radius 2) over the curated
reaction corpus. At inference: given a target product, retrieve top-K
nearest reactions (by Tanimoto, approximated via dot-product on packed
bits) and return their reactants as candidate precursor sets.

This is the cheapest channel — no neural model is trained. The index is
built once in R-2 (build_retro_retrieval_index.py) and loaded at inference.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rasyn.synth.retro.proposers.base import RetroProposer
from rasyn.synth.retro.reactions import canonicalize_smiles, inchi_key_from_smiles
from rasyn.synth.retro.schemas import ProposerChannel, ProposerOutput, ReactionClass


@dataclass
class RetrievalProposerConfig:
    index_path: Path | None = None  # FAISS index file (built in R-2)
    metadata_path: Path | None = None  # pickle: list of dicts {reactants_smiles, reaction_class, source_id}
    n_bits: int = 2048
    top_k: int = 100


def _morgan_fp_bits(smi: str, n_bits: int = 2048):
    """Compute Morgan FP as a uint8 array of length n_bits (1/0)."""
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
        from rdkit.Chem import AllChem  # type: ignore
    except ImportError:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    from rdkit.DataStructs import ConvertToNumpyArray  # type: ignore
    ConvertToNumpyArray(bv, arr)
    return arr


class RetrievalProposer(RetroProposer):
    """FAISS-backed retrieval-by-precedent proposer.

    If FAISS or the index file is unavailable, falls back to brute-force
    numpy Tanimoto over the in-memory metadata (slower, OK for smoke
    tests of < 10k reference reactions).
    """

    channel: ProposerChannel = "retrieval"

    def __init__(self, cfg: RetrievalProposerConfig):
        self.cfg = cfg
        self.metadata: list[dict] = []
        if cfg.metadata_path and cfg.metadata_path.exists():
            with open(cfg.metadata_path, "rb") as fh:
                self.metadata = pickle.load(fh)
        self._index = None
        self._fingerprints = None  # fallback matrix
        if cfg.index_path and cfg.index_path.exists():
            try:
                import faiss  # type: ignore[import-not-found]
                self._index = faiss.read_index(str(cfg.index_path))
            except ImportError:
                pass
        if self._index is None:
            self._fingerprints = self._load_fingerprints_fallback()

    def _load_fingerprints_fallback(self) -> np.ndarray | None:
        """Build a (N, n_bits) uint8 matrix from metadata for brute-force search."""
        if not self.metadata:
            return None
        fps = []
        for entry in self.metadata:
            prod = entry.get("product_smiles")
            if not prod:
                fps.append(np.zeros(self.cfg.n_bits, dtype=np.uint8))
                continue
            fp = _morgan_fp_bits(prod, self.cfg.n_bits)
            fps.append(fp if fp is not None else np.zeros(self.cfg.n_bits, dtype=np.uint8))
        return np.stack(fps) if fps else None

    def _search(self, target_smiles: str, top_k: int) -> list[tuple[int, float]]:
        query_fp = _morgan_fp_bits(target_smiles, self.cfg.n_bits)
        if query_fp is None or not self.metadata:
            return []
        if self._index is not None:
            # Cast bit-vector to float32 for FAISS IndexFlatIP.
            q = query_fp.astype(np.float32).reshape(1, -1)
            distances, indices = self._index.search(q, top_k)
            # Convert dot-product to Tanimoto-ish: dot(a,b) / (|a|+|b|-dot)
            qsum = float(query_fp.sum())
            results = []
            for d, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                ref = self._fingerprints[idx].astype(np.float32).sum() if self._fingerprints is not None else qsum
                denom = qsum + ref - float(d)
                tan = float(d) / denom if denom > 0 else 0.0
                results.append((int(idx), tan))
            return results
        # Fallback brute force
        if self._fingerprints is None:
            return []
        sims = []
        q = query_fp.astype(np.float32)
        for i, fp in enumerate(self._fingerprints):
            f = fp.astype(np.float32)
            inter = float((q * f).sum())
            denom = q.sum() + f.sum() - inter
            tan = inter / denom if denom > 0 else 0.0
            sims.append((i, tan))
        sims.sort(key=lambda x: -x[1])
        return sims[:top_k]

    def propose(
        self,
        target_smiles: str,
        target_inchi_key: str,
        *,
        top_k: int = 10,
        reaction_class_hint: ReactionClass | None = None,
        **kwargs: Any,
    ) -> ProposerOutput:
        hits = self._search(target_smiles, max(top_k, self.cfg.top_k))
        candidates: list[list[str]] = []
        candidate_smiles: list[list[str]] = []
        confidences: list[float] = []
        class_preds: list[ReactionClass] = []
        source_ids: list[str] = []

        for idx, tan in hits[:top_k * 4]:  # over-fetch for class-filter
            if len(candidates) >= top_k:
                break
            ref = self.metadata[idx]
            if reaction_class_hint and ref.get("reaction_class") != reaction_class_hint:
                continue
            reactants = ref.get("reactant_smiles") or []
            canon = []
            ikeys = []
            ok = True
            for r in reactants:
                cs = canonicalize_smiles(r)
                if cs is None:
                    ok = False
                    break
                canon.append(cs)
                ik = inchi_key_from_smiles(cs)
                if ik is None:
                    ok = False
                    break
                ikeys.append(ik)
            if not ok:
                continue
            candidates.append(ikeys)
            candidate_smiles.append(canon)
            confidences.append(float(tan))
            class_preds.append(ref.get("reaction_class") or "unclassified")
            source_ids.append(str(ref.get("source_id") or ref.get("reaction_id") or ""))

        return ProposerOutput(
            channel=self.channel,
            target_inchi_key=target_inchi_key,
            target_smiles=target_smiles,
            reaction_class_hint=reaction_class_hint,
            candidates=candidates,
            candidate_smiles=candidate_smiles,
            confidences=confidences,
            reaction_class_predictions=class_preds,
            channel_metadata={"source_reaction_ids": source_ids},
        )
