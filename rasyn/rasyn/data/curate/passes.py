"""12-pass curation pipeline (Phase A-4).

Order matches PLAN.md §5 / `rasyn_curating_the_dataset.md`. Each pass is
called explicitly so a failure in pass N can be retried without rerunning N-1.
Pass 4 (paper rows) is intentionally a no-op until the paper-extraction
methodology workstream completes (PLAN.md §16).

This module is a SKELETON — each pass writes/reads the right files and logs
what it did. The actual SQL/parquet transforms inside each pass land
incrementally as Track A progresses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rasyn.data.registry.loader import load_sealed_case_registry


@dataclass
class CuratePaths:
    raw_dir: Path = Path("rasyn/data/raw")
    clean_dir: Path = Path("rasyn/data/clean")
    audit_dir: Path = Path("rasyn/data/clean/audits")

    def ensure(self) -> "CuratePaths":
        for d in (self.raw_dir, self.clean_dir, self.audit_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class CurateLog:
    """Append-only log of what each pass did. Written to clean/curate_log.json."""

    passes: list[dict] = field(default_factory=list)

    def record(self, pass_id: str, status: str, **info) -> None:
        self.passes.append({"pass_id": pass_id, "status": status, **info})

    def write(self, path: Path) -> None:
        path.write_text(json.dumps({"passes": self.passes}, indent=2), encoding="utf-8")


def pass_0_seal_and_decontam(paths: CuratePaths, log: CurateLog) -> None:
    """Apply Pass-0 quarantine to all raw sources before any pair mining."""
    reg = load_sealed_case_registry()
    log.record("pass_0", "skeleton", n_cases=len(reg.cases), note="real scrub runs once raw data lands")


def pass_1_chembl_activity_contexts(paths: CuratePaths, log: CurateLog) -> None:
    log.record("pass_1", "skeleton", note="needs ChEMBL bulk SQLite (~20 GB)")


def pass_2_pubchem_admet_facts(paths: CuratePaths, log: CurateLog) -> None:
    log.record("pass_2", "skeleton", note="needs PubChem BioAssay subset")


def pass_3_tdc_molnet_auxiliary(paths: CuratePaths, log: CurateLog) -> None:
    log.record("pass_3", "skeleton", note="TDC + MoleculeNet via Python API")


def pass_4_paper_rows_skipped_v1(paths: CuratePaths, log: CurateLog) -> None:
    log.record("pass_4", "skipped", note="paper extraction deferred (PLAN.md §16)")


def pass_5_internal_skipped_v1(paths: CuratePaths, log: CurateLog) -> None:
    log.record("pass_5", "skipped", note="no internal data per scope lock")


def pass_6_analog_graph(paths: CuratePaths, log: CurateLog) -> None:
    log.record("pass_6", "skeleton", note="ECFP4 Tanimoto + Murcko + MMP + MCS")


def pass_7_pair_generation(paths: CuratePaths, log: CurateLog) -> None:
    log.record("pass_7", "skeleton", note="parent-candidate edges with activity context + liability")


def pass_8_retention_bucketing(paths: CuratePaths, log: CurateLog) -> None:
    log.record("pass_8", "skeleton", note="3x/10x/100x potency-fold buckets")


def pass_9_liability_improvement_labels(paths: CuratePaths, log: CurateLog) -> None:
    log.record("pass_9", "skeleton", note="endpoint-specific improvement categories")


def pass_10_hard_negatives(paths: CuratePaths, log: CurateLog) -> None:
    log.record("pass_10", "skeleton", note="5 types: improved-not-active, retained-not-fixed, wrong-liability, new-liability, heuristic-trap")


def pass_11_local_ranking_tasks(paths: CuratePaths, log: CurateLog) -> None:
    log.record("pass_11", "skeleton", note="(parent, liability, [candidates]) groupings")


def pass_12_quality_tiers_and_rationales(paths: CuratePaths, log: CurateLog) -> None:
    log.record("pass_12", "skeleton", note="gold/silver/bronze + structured rationale auto-gen")


def run_all_passes(paths: CuratePaths | None = None) -> Path:
    """Execute every pass in order. Writes the curate log and returns its path."""
    paths = (paths or CuratePaths()).ensure()
    log = CurateLog()
    sequence = [
        pass_0_seal_and_decontam,
        pass_1_chembl_activity_contexts,
        pass_2_pubchem_admet_facts,
        pass_3_tdc_molnet_auxiliary,
        pass_4_paper_rows_skipped_v1,
        pass_5_internal_skipped_v1,
        pass_6_analog_graph,
        pass_7_pair_generation,
        pass_8_retention_bucketing,
        pass_9_liability_improvement_labels,
        pass_10_hard_negatives,
        pass_11_local_ranking_tasks,
        pass_12_quality_tiers_and_rationales,
    ]
    for fn in sequence:
        fn(paths, log)
    log_path = paths.audit_dir / "curate_log.json"
    log.write(log_path)
    return log_path
