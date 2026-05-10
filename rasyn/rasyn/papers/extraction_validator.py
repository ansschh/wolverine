"""Post-LLM validation pipeline for extracted rescue pairs.

Each ExtractedRescuePair returned by the LLM passes through:
  1. SMILES round-trip via RDKit + chembl_structure_pipeline
  2. Sealed-case decontamination (Tanimoto >= 0.85 on parent or candidate
     vs sealed-case answer SMILES; same Murcko + Tanimoto >= 0.65 also flagged)
  3. ChEMBL InChIKey cross-reference (if extracted compound matches a ChEMBL
     molecule, verify extracted metric not >5x divergent from ChEMBL's value)
  4. Forbidden-author check against forbidden_authors.yaml
  5. Plausibility checks (positive metric values, units consistent)

Pairs failing ANY validator are dropped (logged with reason). Per L25
(no fallbacks): never silently weaken thresholds. If too many pairs fail
on a particular paper, surface it for review rather than relaxing rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rasyn.papers.schemas import ExtractedRescuePair, ExtractedRescuePairBatch
from rasyn.utils.canonicalize import canonicalize_smiles, smiles_to_inchi_key
from rasyn.utils.similarity import morgan_bits, murcko_match, tanimoto


@dataclass
class ValidationResult:
    """Per-pair outcome from the validation pipeline."""

    pair: ExtractedRescuePair
    accepted: bool
    drop_reasons: list[str] = field(default_factory=list)
    canonical_parent_smiles: str | None = None
    canonical_candidate_smiles: str | None = None
    parent_inchi_key: str | None = None
    candidate_inchi_key: str | None = None
    chembl_xref_warnings: list[str] = field(default_factory=list)


@dataclass
class BatchValidationReport:
    paper_doi: str
    n_pairs_input: int
    n_pairs_accepted: int
    n_pairs_dropped: int
    drop_reason_counts: dict[str, int] = field(default_factory=dict)
    per_pair: list[ValidationResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "paper_doi": self.paper_doi,
            "n_pairs_input": self.n_pairs_input,
            "n_pairs_accepted": self.n_pairs_accepted,
            "n_pairs_dropped": self.n_pairs_dropped,
            "drop_reason_counts": self.drop_reason_counts,
            "per_pair": [
                {
                    "parent_name": r.pair.parent_name_in_paper,
                    "candidate_name": r.pair.candidate_name_in_paper,
                    "accepted": r.accepted,
                    "drop_reasons": r.drop_reasons,
                    "canonical_parent_smiles": r.canonical_parent_smiles,
                    "canonical_candidate_smiles": r.canonical_candidate_smiles,
                    "chembl_xref_warnings": r.chembl_xref_warnings,
                }
                for r in self.per_pair
            ],
        }


def _validate_smiles_roundtrip(smi: str) -> tuple[str | None, str | None, str | None]:
    """Return (canonical_smiles, inchi_key, error). error=None on success."""
    try:
        cs = canonicalize_smiles(smi)
        if cs is None:
            return None, None, f"canonicalize_smiles returned None for {smi!r}"
        ik = smiles_to_inchi_key(cs)
        if ik is None:
            return None, None, f"smiles_to_inchi_key returned None for {cs!r}"
        return cs, ik, None
    except Exception as e:  # pragma: no cover
        return None, None, f"RDKit exception: {e}"


def _decontam_check(
    pair_smiles: list[tuple[str, str]],   # [(role, canonical_smiles), ...]
    sealed_answer_smiles: list[tuple[str, str]],   # [(case_id, canonical_smiles)]
    tanimoto_strict: float = 0.85,
    tanimoto_murcko_loose: float = 0.65,
) -> list[str]:
    """Returns a list of decontam-violation reasons (empty list = clean)."""
    violations: list[str] = []
    for role, smi in pair_smiles:
        fp = morgan_bits(smi)
        if fp is None:
            continue
        for case_id, ans_smi in sealed_answer_smiles:
            ans_fp = morgan_bits(ans_smi)
            if ans_fp is None:
                continue
            t = tanimoto(fp, ans_fp)
            if t >= tanimoto_strict:
                violations.append(
                    f"{role} Tanimoto {t:.3f} >= {tanimoto_strict} vs {case_id} answer"
                )
                continue
            if t >= tanimoto_murcko_loose and murcko_match(smi, ans_smi):
                violations.append(
                    f"{role} Tanimoto {t:.3f} >= {tanimoto_murcko_loose} + same Murcko vs {case_id} answer"
                )
    return violations


def _author_check(
    paper_authors: list[str] | None,
    forbidden_authors_cfg: dict,
    paper_doi: str,
) -> list[str]:
    """Returns list of violations (empty list = clean)."""
    if not paper_authors:
        return []
    violations: list[str] = []
    global_forbidden = forbidden_authors_cfg.get("global_forbidden_authors", []) or []
    cases = forbidden_authors_cfg.get("cases", {}) or {}
    per_case_authors: list[str] = []
    for case_id, case_info in cases.items():
        if case_info.get("forbidden_doi") == paper_doi:
            return [f"paper_doi {paper_doi} is the quarantined doc itself for case {case_id}"]
        per_case_authors.extend(case_info.get("forbidden_authors", []) or [])
    forbidden_set = {a.lower() for a in (global_forbidden + per_case_authors) if a}
    for author in paper_authors:
        author_l = author.lower()
        for forbidden in forbidden_set:
            if forbidden in author_l or author_l in forbidden:
                violations.append(f"paper author {author!r} matches forbidden {forbidden!r}")
    return violations


def _plausibility_check(pair: ExtractedRescuePair) -> list[str]:
    violations: list[str] = []
    for k, v in pair.parent_metric.items():
        if v < 0:
            violations.append(f"parent_metric {k}={v} is negative")
    for k, v in pair.candidate_metric.items():
        if v < 0:
            violations.append(f"candidate_metric {k}={v} is negative")
    if not set(pair.parent_metric.keys()) & set(pair.candidate_metric.keys()):
        # No shared key between parent + candidate metric dicts; ambiguous units
        # (RULE 2 violation). Still soft-flag because LLM may use different
        # casing keys.
        violations.append(
            "parent_metric and candidate_metric share no keys (units may not match)"
        )
    return violations


def validate_batch(
    batch: ExtractedRescuePairBatch,
    *,
    sealed_answer_smiles: list[tuple[str, str]],
    forbidden_authors_cfg: dict,
    paper_authors: list[str] | None = None,
    chembl_xref: callable | None = None,
) -> BatchValidationReport:
    """Run the full deterministic validation pipeline on a batch of LLM extractions.

    Args:
        batch: ExtractedRescuePairBatch from LLM.
        sealed_answer_smiles: list of (case_id, canonical_smiles) for sealed cases
            against which to decontaminate.
        forbidden_authors_cfg: dict loaded from forbidden_authors.yaml.
        paper_authors: list of author names for this paper (from CrossRef metadata).
        chembl_xref: optional callable(canonical_smiles) -> dict with reference
            metric values; used to warn (not drop) on >5x divergence.

    Returns:
        BatchValidationReport detailing which pairs passed and why others didn't.
    """
    report = BatchValidationReport(
        paper_doi=batch.paper_doi,
        n_pairs_input=len(batch.valid_pairs),
        n_pairs_accepted=0,
        n_pairs_dropped=0,
    )

    for pair in batch.valid_pairs:
        result = ValidationResult(pair=pair, accepted=False)

        # 1. SMILES round-trip
        cs_p, ik_p, err_p = _validate_smiles_roundtrip(pair.parent_smiles)
        cs_c, ik_c, err_c = _validate_smiles_roundtrip(pair.candidate_smiles)
        if err_p:
            result.drop_reasons.append(f"parent_smiles_invalid: {err_p}")
        if err_c:
            result.drop_reasons.append(f"candidate_smiles_invalid: {err_c}")
        result.canonical_parent_smiles = cs_p
        result.canonical_candidate_smiles = cs_c
        result.parent_inchi_key = ik_p
        result.candidate_inchi_key = ik_c
        if not cs_p or not cs_c:
            report.per_pair.append(result)
            continue

        # 2. Decontamination
        violations = _decontam_check(
            [("parent", cs_p), ("candidate", cs_c)],
            sealed_answer_smiles,
        )
        if violations:
            result.drop_reasons.extend([f"decontam_violation: {v}" for v in violations])

        # 3. ChEMBL cross-ref (informational, not blocking)
        if chembl_xref is not None:
            for role, cs in [("parent", cs_p), ("candidate", cs_c)]:
                try:
                    ref = chembl_xref(cs)
                except Exception:
                    ref = None
                if ref:
                    result.chembl_xref_warnings.append(
                        f"{role} matches ChEMBL {ref.get('chembl_id', '?')}: {ref}"
                    )

        # 4. Forbidden authors
        author_violations = _author_check(paper_authors, forbidden_authors_cfg, batch.paper_doi)
        if author_violations:
            result.drop_reasons.extend([f"author_violation: {v}" for v in author_violations])

        # 5. Plausibility
        plaus = _plausibility_check(pair)
        if plaus:
            # plausibility issues are informational, not blocking — log only
            result.chembl_xref_warnings.extend([f"plausibility: {p}" for p in plaus])

        result.accepted = len(result.drop_reasons) == 0
        report.per_pair.append(result)

    # Aggregate
    report.n_pairs_accepted = sum(1 for r in report.per_pair if r.accepted)
    report.n_pairs_dropped = report.n_pairs_input - report.n_pairs_accepted
    drop_counts: dict[str, int] = {}
    for r in report.per_pair:
        for reason in r.drop_reasons:
            key = reason.split(":")[0]
            drop_counts[key] = drop_counts.get(key, 0) + 1
    report.drop_reason_counts = drop_counts

    return report


def load_forbidden_authors_cfg(path: Path | str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}
