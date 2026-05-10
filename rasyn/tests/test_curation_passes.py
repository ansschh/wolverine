"""Curation pipeline orchestrator: skeleton-mode invariants."""

from __future__ import annotations

import json
from pathlib import Path

from rasyn.data.curate.passes import CuratePaths, run_all_passes


def test_run_all_passes_writes_log(tmp_path: Path):
    paths = CuratePaths(
        raw_dir=tmp_path / "raw",
        clean_dir=tmp_path / "clean",
        audit_dir=tmp_path / "clean/audits",
    )
    log_path = run_all_passes(paths)
    assert log_path.exists()
    data = json.loads(log_path.read_text())
    pass_ids = [p["pass_id"] for p in data["passes"]]
    assert pass_ids == [f"pass_{i}" for i in range(13)]


def test_pass_4_paper_rows_marked_skipped(tmp_path: Path):
    paths = CuratePaths(
        raw_dir=tmp_path / "raw",
        clean_dir=tmp_path / "clean",
        audit_dir=tmp_path / "clean/audits",
    )
    log_path = run_all_passes(paths)
    data = json.loads(log_path.read_text())
    pass_4 = next(p for p in data["passes"] if p["pass_id"] == "pass_4")
    assert pass_4["status"] == "skipped"
    assert "PLAN.md §16" in pass_4["note"]


def test_pass_5_internal_marked_skipped(tmp_path: Path):
    paths = CuratePaths(
        raw_dir=tmp_path / "raw",
        clean_dir=tmp_path / "clean",
        audit_dir=tmp_path / "clean/audits",
    )
    log_path = run_all_passes(paths)
    data = json.loads(log_path.read_text())
    pass_5 = next(p for p in data["passes"] if p["pass_id"] == "pass_5")
    assert pass_5["status"] == "skipped"
