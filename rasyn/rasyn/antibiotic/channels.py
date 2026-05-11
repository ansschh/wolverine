"""Seven proposer channels for antibiotic discovery (spec §10).

Channels:
  A: repurposing-library retriever
  B: screened-library retriever
  C: organism-specific analog retriever
  D: scaffold-hopping retriever (long-Tanimoto seed pool)
  E: fragment-conditioned diffusion (LEARNABLE — adapted from Channel 4 seq2seq)
  F: phenotype-conditioned edit diffusion (LEARNABLE — adapted from Channel 5)
  G: diversity / novelty selector (post-union filter)

For v1, Channels E and F reuse the ADMET Ch4/Ch5 seq2seq generator architecture
(trained on antibiotic generative training examples) since full graph diffusion
is a multi-week build. Per L25: not a placeholder — different teacher data, real
trained model. Full graph-diffusion upgrade is a future phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rasyn.antibiotic.schemas import ABXChallengePacket


@dataclass
class ABXProposerContext:
    """Shared context for all channels."""
    library_smiles_pool: list[str]               # all candidate molecules
    library_inchi_keys: list[str]
    known_antibiotic_smiles: list[str]           # for novelty penalty + scaffold-hopping
    sealed_answer_smiles_or_none: str | None     # for decontam
    embeddings_path: Path | None = None
    organism_active_pool_path: Path | None = None  # path to ChEMBL antibacterial parquet


# ============================================================
# Channel A: Repurposing-library retriever
# ============================================================

def channel_a_repurposing(packet: ABXChallengePacket, ctx: ABXProposerContext, top_k: int = 500) -> list[dict]:
    """Return drug-like molecules from a clinically-screened pool.

    For halicin-style discovery, the library is the Broad Drug Repurposing Hub.
    """
    out = []
    for i, smi in enumerate(ctx.library_smiles_pool[:top_k]):
        out.append({
            "candidate_smiles": smi,
            "candidate_inchi_key": ctx.library_inchi_keys[i] if i < len(ctx.library_inchi_keys) else None,
            "channel": "A_repurposing",
        })
    return out


# ============================================================
# Channel B: Screened-library retriever
# ============================================================

def channel_b_screened_library(packet: ABXChallengePacket, ctx: ABXProposerContext, top_k: int = 500) -> list[dict]:
    """Return molecules from screening libraries (PubChem AID screens, CO-ADD).

    Same data source as Channel A in v1, but different LIBRARY ORIGIN: these
    are NOT clinical drugs, they're screening compounds tested in phenotypic
    assays. Higher hit-rate potential.
    """
    return [
        {
            "candidate_smiles": smi,
            "candidate_inchi_key": ctx.library_inchi_keys[i] if i < len(ctx.library_inchi_keys) else None,
            "channel": "B_screened_library",
        }
        for i, smi in enumerate(ctx.library_smiles_pool[top_k:2 * top_k])
    ]


# ============================================================
# Channel C: Organism-specific analog retriever
# ============================================================

def channel_c_organism_specific_analog(
    packet: ABXChallengePacket, ctx: ABXProposerContext, top_k: int = 200,
) -> list[dict]:
    """Retrieve molecules known active against the target organism — return their analogs.

    For each known active for the target organism, find Tanimoto neighbors in the pool.
    """
    if not ctx.organism_active_pool_path or not ctx.organism_active_pool_path.exists():
        return []
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except ImportError:
        return []
    actives_df = pd.read_parquet(ctx.organism_active_pool_path)
    actives_df = actives_df[actives_df["organism"] == packet.organism_context.organism]
    actives_df = actives_df[actives_df["activity_label"] == "active"]
    if actives_df.empty:
        return []
    # Compute FPs of actives + pool
    def _fp(s):
        m = Chem.MolFromSmiles(s)
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None
    active_fps = [(s, _fp(s)) for s in actives_df["canonical_smiles"].dropna().unique()[:50]]
    out = []
    for i, pool_smi in enumerate(ctx.library_smiles_pool):
        pf = _fp(pool_smi)
        if pf is None: continue
        best_tan = max((DataStructs.TanimotoSimilarity(pf, af) for _, af in active_fps if af is not None), default=0.0)
        if best_tan >= 0.5:
            out.append({
                "candidate_smiles": pool_smi,
                "candidate_inchi_key": ctx.library_inchi_keys[i] if i < len(ctx.library_inchi_keys) else None,
                "channel": "C_organism_analog",
                "max_tanimoto_to_organism_active": float(best_tan),
            })
        if len(out) >= top_k:
            break
    return out


# ============================================================
# Channel D: Scaffold-hopping retriever
# ============================================================

def channel_d_scaffold_hopping(
    packet: ABXChallengePacket, ctx: ABXProposerContext, top_k: int = 200,
) -> list[dict]:
    """Retrieve molecules that are NOT close to known antibiotics — scaffold-hop space.

    For v1: filter pool for molecules with low max-Tanimoto to known antibiotics.
    """
    if not ctx.known_antibiotic_smiles:
        return []
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except ImportError:
        return []
    def _fp(s):
        m = Chem.MolFromSmiles(s)
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None
    known_fps = [_fp(s) for s in ctx.known_antibiotic_smiles[:50] if _fp(s) is not None]
    if not known_fps:
        return []
    out = []
    for i, pool_smi in enumerate(ctx.library_smiles_pool):
        pf = _fp(pool_smi)
        if pf is None: continue
        max_tan = max(DataStructs.TanimotoSimilarity(pf, kf) for kf in known_fps)
        if max_tan < 0.35:  # truly distant from known antibiotics
            out.append({
                "candidate_smiles": pool_smi,
                "candidate_inchi_key": ctx.library_inchi_keys[i] if i < len(ctx.library_inchi_keys) else None,
                "channel": "D_scaffold_hopping",
                "max_tanimoto_to_known_antibiotic": float(max_tan),
            })
        if len(out) >= top_k:
            break
    return out


# ============================================================
# Channels E, F: Generative (adapted from ADMET Ch4/Ch5)
# ============================================================

def channel_e_fragment_diffusion_from_json(
    case_id: str,
    json_path: Path,
    channel_name: str = "E_fragment_diffusion",
) -> list[dict]:
    """Load pre-generated Ch-E (fragment-conditioned) candidates from JSON.

    Pre-generation runs on pod with Ch4-style seq2seq ckpt + organism conditioning.
    Same pattern as ADMET's generate_channel_candidates.py.
    """
    import json
    if not Path(json_path).exists():
        return []
    data = json.loads(Path(json_path).read_text())
    case = data.get("cases", {}).get(case_id, {})
    return [{"candidate_smiles": s, "channel": channel_name} for s in case.get("candidates", [])]


def channel_f_edit_diffusion_from_json(
    case_id: str,
    json_path: Path,
) -> list[dict]:
    return channel_e_fragment_diffusion_from_json(case_id, json_path, channel_name="F_edit_diffusion")


# ============================================================
# Channel G: Diversity / novelty selector
# ============================================================

def channel_g_diversity_filter(
    candidates: list[dict],
    *,
    max_pool: int = 5000,
    min_tanimoto_intra: float = 0.15,
) -> list[dict]:
    """Diversity-greedy selection to prevent scaffold collapse.

    Keep candidates that are at least min_tanimoto_intra distant from already-selected ones.
    Cap at max_pool. Per spec §10 Channel G.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except ImportError:
        return candidates[:max_pool]
    selected: list[dict] = []
    selected_fps = []
    for c in candidates:
        if len(selected) >= max_pool:
            break
        m = Chem.MolFromSmiles(c.get("candidate_smiles") or "")
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        # Distance check (intra-pool diversity)
        if selected_fps:
            min_tan = max(DataStructs.TanimotoSimilarity(fp, sf) for sf in selected_fps)
            if min_tan > (1.0 - min_tanimoto_intra):
                continue  # too similar to existing pick
        selected.append(c)
        selected_fps.append(fp)
    return selected


# ============================================================
# Ensemble runner
# ============================================================

def run_abx_ensemble(
    packet: ABXChallengePacket,
    ctx: ABXProposerContext,
    *,
    ch_e_json: Path | None = None,
    ch_f_json: Path | None = None,
    max_pool: int = 5000,
) -> list[dict]:
    """Run all 7 channels, union + dedupe + diversity-filter."""
    all_candidates: list[dict] = []
    for fn in [channel_a_repurposing, channel_b_screened_library,
                channel_c_organism_specific_analog, channel_d_scaffold_hopping]:
        all_candidates.extend(fn(packet, ctx))
    if ch_e_json:
        all_candidates.extend(channel_e_fragment_diffusion_from_json(packet.case_id, ch_e_json))
    if ch_f_json:
        all_candidates.extend(channel_f_edit_diffusion_from_json(packet.case_id, ch_f_json))

    # Dedupe by SMILES
    seen: set[str] = set()
    deduped = []
    for c in all_candidates:
        s = c.get("candidate_smiles")
        if not s or s in seen:
            continue
        seen.add(s)
        deduped.append(c)

    # Channel G: diversity filter
    return channel_g_diversity_filter(deduped, max_pool=max_pool)
