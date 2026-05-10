"""Per-case card: a single-page Markdown summary used in the audit pack.

Inputs:
  - the sealed case from the registry
  - the LockedPrediction
  - the eval result (Mode A + Mode B)
  - the answer reveal info (post-evaluation)

Output: Markdown.
"""

from __future__ import annotations

from rasyn.eval.harness import CaseEvalResult
from rasyn.schemas.locked import LockedPrediction
from rasyn.schemas.registry import SealedCase

CARD_TEMPLATE = """# Case: {case_id}

**Liability:** {liability_type}
**Rescue mode:** {rescue_mode}
**Parent:** {parent_name} (`{parent_smiles}`)
**Answer:** {answer_name}

## Locked prediction

- Locked at: `{locked_at}`
- System version: `{system_version}`
- Challenge packet hash: `{challenge_hash}`
- Model checkpoint hash: `{model_hash}`
- Output hash: `{output_hash}`

### Top-5 (locked)
{top_5}

### Top-10 (locked)
{top_10}

### Top-20 (locked)
{top_20}

## Mode A (open proposer) results

- Pool size: {a_pool}
- Rank of answer: {a_rank}
- exact recall@5/10/20: {a_e5}/{a_e10}/{a_e20}
- functional recall@5/10/20: {a_f5}/{a_f10}/{a_f20}
- MRR: {a_mrr:.4f}
- Invalid rate: {a_invalid:.3f}
- Per-channel attribution: `{a_channels}`

## Mode B (closed hard-ranking) results

- Pool size: {b_pool}
- Rank of answer: {b_rank}
- exact recall@5/10/20: {b_e5}/{b_e10}/{b_e20}
- functional recall@5/10/20: {b_f5}/{b_f10}/{b_f20}
- MRR: {b_mrr:.4f}

## Decontamination

- Forbidden synonyms removed: {n_synonyms}
- Forbidden documents quarantined: {n_docs}
- Tanimoto-to-answer threshold: {t_ans}
- Tanimoto-with-context threshold: {t_ctx}

## Notes
{notes}
"""


def render_per_case_card(
    case: SealedCase,
    locked: LockedPrediction,
    mode_a: CaseEvalResult | None,
    mode_b: CaseEvalResult | None,
    notes: str = "",
) -> str:
    def _fmt_top(ids: list[str]) -> str:
        return "\n".join(f"{i+1}. `{cid}`" for i, cid in enumerate(ids)) if ids else "_(empty)_"

    def _r(res: CaseEvalResult | None, attr: str, default=""):
        return getattr(res, attr) if res else default

    return CARD_TEMPLATE.format(
        case_id=case.case_id,
        liability_type=case.liability_type,
        rescue_mode=case.rescue_mode,
        parent_name=case.parent.name or "(unknown)",
        parent_smiles=case.parent.canonical_smiles or "(not yet populated)",
        answer_name=case.answer.name or "(unknown)",
        locked_at=locked.locked_at_utc,
        system_version=locked.system_version,
        challenge_hash=locked.challenge_packet_hash[:16] + "...",
        model_hash=locked.model_checkpoint_hash[:16] + "...",
        output_hash=locked.output_hash[:16] + "...",
        top_5=_fmt_top(locked.top_k_locked.get("5", [])),
        top_10=_fmt_top(locked.top_k_locked.get("10", [])),
        top_20=_fmt_top(locked.top_k_locked.get("20", [])),
        a_pool=_r(mode_a, "pool_size", "n/a"),
        a_rank=_r(mode_a, "rank_of_answer", "n/a"),
        a_e5=_r(mode_a, "exact_recall_at_5", "n/a"),
        a_e10=_r(mode_a, "exact_recall_at_10", "n/a"),
        a_e20=_r(mode_a, "exact_recall_at_20", "n/a"),
        a_f5=_r(mode_a, "functional_recall_at_5", "n/a"),
        a_f10=_r(mode_a, "functional_recall_at_10", "n/a"),
        a_f20=_r(mode_a, "functional_recall_at_20", "n/a"),
        a_mrr=float(_r(mode_a, "mrr", 0.0) or 0.0),
        a_invalid=float(_r(mode_a, "invalid_rate", 0.0) or 0.0),
        a_channels=_r(mode_a, "per_channel_attribution", {}),
        b_pool=_r(mode_b, "pool_size", "n/a"),
        b_rank=_r(mode_b, "rank_of_answer", "n/a"),
        b_e5=_r(mode_b, "exact_recall_at_5", "n/a"),
        b_e10=_r(mode_b, "exact_recall_at_10", "n/a"),
        b_e20=_r(mode_b, "exact_recall_at_20", "n/a"),
        b_f5=_r(mode_b, "functional_recall_at_5", "n/a"),
        b_f10=_r(mode_b, "functional_recall_at_10", "n/a"),
        b_f20=_r(mode_b, "functional_recall_at_20", "n/a"),
        b_mrr=float(_r(mode_b, "mrr", 0.0) or 0.0),
        n_synonyms=len(case.forbidden_identifiers.synonyms),
        n_docs=len(case.forbidden_documents.dois)
        + len(case.forbidden_documents.pmids)
        + len(case.forbidden_documents.chembl_doc_ids),
        t_ans=case.quarantine.tanimoto_to_answer,
        t_ctx=case.quarantine.tanimoto_with_context,
        notes=notes or "_(none)_",
    )
