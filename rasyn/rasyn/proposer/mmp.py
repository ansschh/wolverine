"""Channel 2: matched-molecular-pair (MMP) transformation proposer.

Applies learned (or seeded) transformation rules of the form
`fragment_A -> fragment_B` to the parent. v1 ships a small seed rule list
covering common medicinal-chemistry edits; the full rule-mining pipeline runs
in Phase A-4 (pass 6 analog graph).
"""

from __future__ import annotations

from rasyn.proposer.base import Proposer, ProposerContext
from rasyn.schemas.challenge import ADMETChallengePacket
from rasyn.schemas.proposer import (
    CandidateAnnotation,
    ProposerOutput,
    TransformationDescriptor,
)
from rasyn.utils.canonicalize import canonicalize_smiles, smiles_to_inchi_key

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    _HAVE_RDKIT = True
except ImportError:  # pragma: no cover
    _HAVE_RDKIT = False
    Chem = None  # type: ignore[assignment]


# Seed transformation rule list (rxn SMARTS + transformation_class tag).
# Covers high-value classes: bioisostere swaps, polarity shifts, basicity tuning,
# prodrug motifs, soft-spot blocking. Mined rules from ChEMBL extend this.
SEED_RULES: list[tuple[str, str]] = [
    ("[c:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1>>[c:1]1[cH:2][n:3][cH:4][cH:5][cH:6]1", "phenyl_to_pyridyl"),
    ("[CH3:1][N:2]>>[H][N:2]", "n_demethylation"),
    ("[OH:1]>>[F:1]", "hydroxyl_to_fluoro_bioisostere"),
    ("[CH2:1][C:2](=O)[O:3]>>[CH2:1][C:2](=O)[NH:3]", "ester_to_amide"),
    ("[c:1][CH3:2]>>[c:1][F:2]", "methyl_to_fluoro"),
    ("[NX3;H0:1]([CH3])([CH3])[#6:2]>>[NX3;H0:1]([CH3])[#6:2]", "tertiary_to_secondary_amine"),
    ("[c:1][C:2](=O)[OH]>>[c:1][C:2](=O)[O][CH2][CH](N)C(C)C", "valyl_ester_prodrug"),
]


class MMPTransformerProposer(Proposer):
    channel = "mmp_transformer"

    def __init__(self, *, rules: list[tuple[str, str]] | None = None, max_per_rule: int = 16):
        self.rules = rules or SEED_RULES
        self.max_per_rule = max_per_rule
        if _HAVE_RDKIT:
            self._compiled = [(AllChem.ReactionFromSmarts(r), tag) for r, tag in self.rules]
        else:
            self._compiled = []

    def _apply_rule(self, mol, rxn) -> list[str]:
        try:
            products = rxn.RunReactants((mol,))
        except Exception:
            return []
        out: list[str] = []
        for product_set in products[: self.max_per_rule]:
            for product in product_set:
                try:
                    Chem.SanitizeMol(product)
                    smi = Chem.MolToSmiles(product, isomericSmiles=True, canonical=True)
                    cs = canonicalize_smiles(smi)
                    if cs:
                        out.append(cs)
                except Exception:
                    continue
        return out

    def propose(self, packet: ADMETChallengePacket, ctx: ProposerContext) -> ProposerOutput:
        if not _HAVE_RDKIT:
            return ProposerOutput(
                case_id=packet.case_id,
                channel=self.channel,
                candidates=[],
                raw_count=0,
                invalid_count=0,
                deduplicated_count=0,
            )

        parent = Chem.MolFromSmiles(packet.parent_canonical_smiles)
        if parent is None:
            return ProposerOutput(
                case_id=packet.case_id,
                channel=self.channel,
                candidates=[],
                raw_count=0,
                invalid_count=1,
                deduplicated_count=0,
            )

        annotations: list[CandidateAnnotation] = []
        seen: set[str] = set()
        raw = 0
        invalid = 0
        for rxn, tag in self._compiled:
            if rxn is None:
                continue
            products = self._apply_rule(parent, rxn)
            raw += len(products)
            for smi in products:
                if smi == packet.parent_canonical_smiles:
                    continue
                ik = smiles_to_inchi_key(smi)
                if not ik:
                    invalid += 1
                    continue
                if ik in seen:
                    continue
                seen.add(ik)
                annotations.append(
                    CandidateAnnotation(
                        candidate_id=f"mmp-{packet.case_id}-{tag}-{ik[:8]}",
                        canonical_smiles=smi,
                        inchi_key=ik,
                        parent_inchi_key=packet.parent_inchi_key,
                        proposer_sources=[self.channel],
                        transformation=TransformationDescriptor(
                            transformation_class=tag,
                            summary=f"applied MMP rule '{tag}'",
                            transformation_distance=None,
                        ),
                        proposer_confidence=0.5,
                    )
                )

        return ProposerOutput(
            case_id=packet.case_id,
            channel=self.channel,
            candidates=annotations,
            raw_count=raw,
            invalid_count=invalid,
            deduplicated_count=len(annotations),
        )
