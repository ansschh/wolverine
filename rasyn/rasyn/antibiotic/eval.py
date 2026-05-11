"""ABX evaluation harness (spec §21).

Modes:
  closed_hard_ranking — fixed candidate library with hidden hit + hard decoys;
                        compute hit rank / percentile, top-k recovery, enrichment,
                        penalty rates (toxicity, organism mismatch).
  open_proposer      — proposer generates candidates; metrics: hit recovery,
                        active-family recovery, novelty, validity, toxicity rates.
  prospective         — out of scope for v1 (requires wet-lab).

Returns a dict of metric values per case.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClosedRankingMetrics:
    case_id: str
    organism: str
    library_size: int
    hidden_hit_rank: int | None
    hidden_hit_percentile: float | None  # rank / library_size * 100
    top_1: bool
    top_10: bool
    top_50: bool
    top_100: bool
    top_1_pct: bool
    enrichment_at_top_100: float | None
    n_cytotoxic_in_top_100: int
    n_organism_mismatched_in_top_100: int
    n_known_controls_in_top_100: int
    verdict: str  # "discovered" | "near_miss" | "missed"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__annotations__}


@dataclass
class OpenProposerMetrics:
    case_id: str
    n_generated: int
    n_valid: int
    n_unique: int
    exact_hit_generated: bool
    active_family_analog_count: int
    median_novelty_vs_known: float | None
    pct_cytotoxic: float | None
    pct_synthesizable: float | None
    top_k_distinct_chemotypes: int

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__annotations__}


def _verdict_from_rank(rank: int | None, library_size: int) -> str:
    if rank is None:
        return "missed"
    if rank <= 10:
        return "discovered"
    if rank <= max(100, library_size // 100):
        return "near_miss"
    return "missed"


def closed_hard_ranking(
    ranked_candidates: list[dict],
    *,
    case_id: str,
    organism: str,
    hidden_hit_smiles: str | None,
    library_size: int,
    cytotoxic_smiles: set[str] | None = None,
    organism_mismatched_smiles: set[str] | None = None,
    known_control_smiles: set[str] | None = None,
) -> ClosedRankingMetrics:
    """Compute closed-ranking metrics given the ranker's output list.

    Each ranked_candidates item is a dict with at least 'candidate_smiles' and
    'final_discovery_score' (or equivalent). List is ordered best-first.
    """
    cytotoxic_smiles = cytotoxic_smiles or set()
    organism_mismatched_smiles = organism_mismatched_smiles or set()
    known_control_smiles = known_control_smiles or set()

    hit_rank: int | None = None
    if hidden_hit_smiles:
        for i, c in enumerate(ranked_candidates, start=1):
            if c.get("candidate_smiles") == hidden_hit_smiles:
                hit_rank = i
                break

    top_100 = ranked_candidates[:100]
    n_cyto = sum(1 for c in top_100 if c.get("candidate_smiles") in cytotoxic_smiles)
    n_org_miss = sum(1 for c in top_100 if c.get("candidate_smiles") in organism_mismatched_smiles)
    n_known = sum(1 for c in top_100 if c.get("candidate_smiles") in known_control_smiles)

    # Enrichment: P(hit in top-100) / (hits / library_size)
    enrichment = None
    if hidden_hit_smiles and library_size > 0:
        in_top = 1 if hit_rank and hit_rank <= 100 else 0
        baseline = 1.0 / library_size
        enrichment = (in_top / 100.0) / baseline if baseline > 0 else None

    return ClosedRankingMetrics(
        case_id=case_id,
        organism=organism,
        library_size=library_size,
        hidden_hit_rank=hit_rank,
        hidden_hit_percentile=(hit_rank / library_size * 100.0) if hit_rank else None,
        top_1=bool(hit_rank and hit_rank <= 1),
        top_10=bool(hit_rank and hit_rank <= 10),
        top_50=bool(hit_rank and hit_rank <= 50),
        top_100=bool(hit_rank and hit_rank <= 100),
        top_1_pct=bool(hit_rank and hit_rank <= max(1, library_size // 100)),
        enrichment_at_top_100=enrichment,
        n_cytotoxic_in_top_100=n_cyto,
        n_organism_mismatched_in_top_100=n_org_miss,
        n_known_controls_in_top_100=n_known,
        verdict=_verdict_from_rank(hit_rank, library_size),
    )


def open_proposer(
    generated_candidates: list[dict],
    *,
    case_id: str,
    hidden_hit_smiles: str | None = None,
    active_family_smiles: list[str] | None = None,
    known_antibiotic_smiles: list[str] | None = None,
) -> OpenProposerMetrics:
    """Compute open-mode metrics."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
        rdkit_ok = True
    except ImportError:
        rdkit_ok = False
    n_generated = len(generated_candidates)
    valid_smiles: set[str] = set()
    if rdkit_ok:
        for c in generated_candidates:
            m = Chem.MolFromSmiles(c.get("candidate_smiles") or "")
            if m is not None:
                valid_smiles.add(Chem.MolToSmiles(m, canonical=True))

    exact_hit = False
    if hidden_hit_smiles and rdkit_ok:
        ans_mol = Chem.MolFromSmiles(hidden_hit_smiles)
        if ans_mol is not None:
            ans_canon = Chem.MolToSmiles(ans_mol, canonical=True)
            exact_hit = ans_canon in valid_smiles

    # Active-family analog count
    family_count = 0
    if active_family_smiles and rdkit_ok:
        family_fps = [
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
            for s in active_family_smiles if Chem.MolFromSmiles(s) is not None
        ]
        for smi in valid_smiles:
            fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=2048)
            if any(DataStructs.TanimotoSimilarity(fp, ff) >= 0.7 for ff in family_fps):
                family_count += 1

    # Median novelty vs known antibiotics
    novelty = None
    if known_antibiotic_smiles and rdkit_ok:
        known_fps = [
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
            for s in known_antibiotic_smiles if Chem.MolFromSmiles(s) is not None
        ]
        novelties = []
        for smi in list(valid_smiles)[:1000]:
            fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=2048)
            max_tan = max((DataStructs.TanimotoSimilarity(fp, kf) for kf in known_fps), default=0.0)
            novelties.append(1.0 - max_tan)
        if novelties:
            novelties.sort()
            novelty = novelties[len(novelties) // 2]

    # Distinct chemotypes via Murcko
    distinct_murcko: set[str] = set()
    if rdkit_ok:
        for smi in list(valid_smiles)[:1000]:
            m = Chem.MolFromSmiles(smi)
            if m is None: continue
            sf = GetScaffoldForMol(m)
            if sf: distinct_murcko.add(Chem.MolToSmiles(sf))

    return OpenProposerMetrics(
        case_id=case_id,
        n_generated=n_generated,
        n_valid=len(valid_smiles),
        n_unique=len(valid_smiles),
        exact_hit_generated=exact_hit,
        active_family_analog_count=family_count,
        median_novelty_vs_known=novelty,
        pct_cytotoxic=None,  # would need predicted-tox scores per candidate
        pct_synthesizable=None,  # SAScore could fill this
        top_k_distinct_chemotypes=len(distinct_murcko),
    )


def save_metrics(metrics_obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(metrics_obj, "to_dict"):
        path.write_text(json.dumps(metrics_obj.to_dict(), indent=2))
    elif isinstance(metrics_obj, dict):
        path.write_text(json.dumps(metrics_obj, indent=2))
    else:
        path.write_text(str(metrics_obj))
