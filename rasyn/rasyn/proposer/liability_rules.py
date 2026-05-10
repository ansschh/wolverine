"""Channel 3: liability-specific rule-pack proposer.

For each known liability the system loads a curated rule pack (a list of
medicinal-chemistry edits known to address that liability). Sub-proposers:
  - hERG: reduce basicity, add polarity, reduce lipophilicity
  - solubility: heteroatom insertion, reduce aromatic count, raise fsp3
  - metabolic stability: block soft spots (fluoro shielding, methyl removal)
  - prodrug/oral exposure: ester/amino-acid-ester/phosphate motifs
"""

from __future__ import annotations

from rasyn.proposer.base import Proposer, ProposerContext
from rasyn.proposer.mmp import MMPTransformerProposer
from rasyn.schemas.challenge import ADMETChallengePacket
from rasyn.schemas.proposer import ProposerOutput

LIABILITY_RULE_PACKS: dict[str, list[tuple[str, str]]] = {
    "hERG": [
        ("[NX3;H0:1]([CH3])([CH3])[#6:2]>>[NX3;H1:1]([CH3])[#6:2]", "herg_basicity_reduction"),
        ("[c:1][CH2:2][CH2:3][N:4]>>[c:1][CH2:2][C:3](=O)[O][N:4]", "herg_polarity_increase"),
        ("[c:1][C:2](C)(C)C>>[c:1][C:2](=O)O", "herg_lipophilicity_reduction"),
    ],
    "solubility": [
        ("[c:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1>>[c:1]1[n:2][cH:3][cH:4][cH:5][cH:6]1", "sol_phenyl_to_pyridyl"),
        ("[c:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1>>[c:1]1[cH:2][n:3][cH:4][cH:5][cH:6]1", "sol_phenyl_to_meta_pyridyl"),
        ("[c:1][CH3:2]>>[c:1][CH2:2][OH]", "sol_methyl_to_hydroxymethyl"),
    ],
    "metabolic_stability": [
        ("[c:1][CH2:2][c:3]>>[c:1][CF2:2][c:3]", "metstab_benzylic_fluoro_shield"),
        ("[c:1][CH3:2]>>[c:1][F:2]", "metstab_methyl_to_fluoro"),
        ("[c:1][O:2][CH3:3]>>[c:1][O:2][CHF2:3]", "metstab_methoxy_to_difluoromethoxy"),
    ],
    "oral_exposure": [
        ("[c:1][C:2](=O)[OH]>>[c:1][C:2](=O)[O][CH2][CH](N)C(C)C", "prodrug_l_valyl_ester"),
        ("[OH:1]>>[O:1]C(=O)CC", "prodrug_propionate_ester"),
        ("[OH:1]>>[O:1]P(=O)(O)O", "prodrug_phosphate"),
    ],
    "permeability": [
        ("[OH:1]>>[OCH3:1]", "perm_o_methylation"),
        ("[NH2:1]>>[NHCOCH3:1]", "perm_n_acetylation"),
    ],
}


class LiabilityRulesProposer(Proposer):
    channel = "liability_rules"

    def __init__(self, *, max_per_rule: int = 16):
        self.max_per_rule = max_per_rule

    def propose(self, packet: ADMETChallengePacket, ctx: ProposerContext) -> ProposerOutput:
        liability = packet.liability_context.liability_type
        rules = LIABILITY_RULE_PACKS.get(liability, [])
        if not rules:
            return ProposerOutput(
                case_id=packet.case_id,
                channel=self.channel,
                candidates=[],
                raw_count=0,
                invalid_count=0,
                deduplicated_count=0,
            )
        delegate = MMPTransformerProposer(rules=rules, max_per_rule=self.max_per_rule)
        out = delegate.propose(packet, ctx)
        # Re-tag the channel so attribution is correct.
        retagged = [
            ann.model_copy(update={"proposer_sources": [self.channel]}) for ann in out.candidates
        ]
        return out.model_copy(update={"channel": self.channel, "candidates": retagged})
