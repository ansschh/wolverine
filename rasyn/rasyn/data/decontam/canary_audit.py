"""Canary audit: verify zero canaries survive decontamination.

Run AFTER the full data pipeline. If any canary appears in the cleaned
output, the pipeline halts and fails the audit.
"""

from __future__ import annotations

from dataclasses import dataclass

from rasyn.schemas.registry import Canary


@dataclass
class CanaryAuditResult:
    total_canaries: int
    survivors: list[Canary]
    passed: bool

    def to_dict(self) -> dict:
        return {
            "total_canaries": self.total_canaries,
            "survivor_count": len(self.survivors),
            "survivor_ids": [c.canary_id for c in self.survivors],
            "passed": self.passed,
        }


def audit_against_rows(canaries: list[Canary], rows: list[dict]) -> CanaryAuditResult:
    """Check every canary against every row by layer-appropriate match.

    A canary "survives" if its payload appears in the corresponding field of
    any row (smiles == payload for layer='smiles', etc.). Title/text canaries
    are substring-matched (case-insensitive) against any text-bearing field.
    """
    survivors: list[Canary] = []
    text_fields_per_row = [
        " ".join(str(v) for v in row.values() if isinstance(v, str)).lower() for row in rows
    ]

    for canary in canaries:
        payload = canary.payload
        layer = canary.layer
        survived = False
        if layer == "smiles":
            survived = any(r.get("smiles") == payload for r in rows)
        elif layer == "inchi_key":
            survived = any(r.get("inchi_key") == payload for r in rows)
        elif layer == "synonym":
            for r in rows:
                if payload in (r.get("synonyms") or []):
                    survived = True
                    break
        elif layer == "doi":
            survived = any(r.get("doi") == payload for r in rows)
        elif layer == "pmid":
            survived = any(str(r.get("pmid")) == payload for r in rows)
        elif layer == "chembl_id":
            survived = any(r.get("chembl_id") == payload or r.get("chembl_doc_id") == payload for r in rows)
        elif layer == "patent":
            survived = any(payload in (r.get("patents") or []) for r in rows)
        elif layer == "title_text":
            payload_lc = payload.lower()
            survived = any(payload_lc in tx for tx in text_fields_per_row)
        if survived:
            survivors.append(canary)

    return CanaryAuditResult(total_canaries=len(canaries), survivors=survivors, passed=len(survivors) == 0)
