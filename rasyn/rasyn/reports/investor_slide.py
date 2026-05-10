"""1-slide investor summary (Markdown, ready to paste into a deck).

The ENTIRE story in 8 sentences plus a 3-row table. Designed to survive
"this is cherry-picked" interrogation by referencing the audit pack hashes.
"""

from __future__ import annotations

from rasyn.eval.harness import CaseEvalResult

INVESTOR_SLIDE_TEMPLATE = """# Rasyn — held-out ADMET-rescue benchmark

We trained a chemistry AI from scratch on a contamination-controlled corpus.
The 3 sealed ADMET cases below were locked BEFORE training began. Predictions
were locked BEFORE answers were revealed. Decontamination is auditable.

| Case | Parent → Answer | Liability | Discovered? | Rank | Mode |
|---|---|---|---|---|---|
{rows}

**Decontamination:** every sealed-case identifier, document, assay, and
neighbourhood (Tanimoto ≥ 0.85 to answer; ≥ 0.65 within same scaffold + same
target) was quarantined before any pair mining. Canaries inserted before
cleaning, all canaries verified absent after.

**Audit pack:** `{audit_pack_path}` contains every hash (sealed_case_registry,
dataset_manifest, training_manifest, locked predictions). Reproducible
end-to-end. See `../technical_appendix.md` for the 13-section detail.

**System version:** `{system_version}`. Sealed-case-registry hash: `{registry_hash_short}`.
"""


def render_investor_slide(
    *,
    case_results: list[tuple[str, str, str, CaseEvalResult]],  # (case_id, "P -> A", liability, eval)
    audit_pack_path: str,
    system_version: str,
    registry_hash_short: str,
) -> str:
    rows = []
    for case_id, p_to_a, liability, res in case_results:
        discovered = "✓" if res.exact_recall_at_10 else "—"
        rank = res.rank_of_answer if res.rank_of_answer is not None else "—"
        rows.append(f"| {case_id} | {p_to_a} | {liability} | {discovered} | {rank} | {res.mode} |")
    return INVESTOR_SLIDE_TEMPLATE.format(
        rows="\n".join(rows),
        audit_pack_path=audit_pack_path,
        system_version=system_version,
        registry_hash_short=registry_hash_short,
    )
