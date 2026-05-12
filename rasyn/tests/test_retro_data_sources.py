"""Hermetic tests for retro data-source parsers (no network)."""

from __future__ import annotations

import csv
import gzip
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from rasyn.data.sources import buyables as buyables_src
from rasyn.data.sources import uspto as uspto_src


# ===== USPTO-50K parser =====

def _build_uspto_50k_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "uspto_50k.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["reactants>reagents>production", "class"])
        writer.writeheader()
        writer.writerow({
            "reactants>reagents>production": "CCO.CC(=O)Cl>>CCOC(C)=O",
            "class": "amide_coupling",
        })
        writer.writerow({
            "reactants>reagents>production": "c1ccccc1Br.c1ccccc1B(O)O>>c1ccccc1-c1ccccc1",
            "class": "suzuki_coupling",
        })
        zf.writestr("train.csv", buf.getvalue())
    return zip_path


def test_uspto_50k_parser_basic(tmp_path):
    zip_path = _build_uspto_50k_zip(tmp_path)
    rows = list(uspto_src.iter_uspto_50k(zip_path))
    assert len(rows) == 2
    assert rows[0]["source"] == "uspto_50k"
    assert rows[0]["product"] == "CCOC(C)=O"
    assert rows[0]["mapped_rxn_smiles"] == rows[0]["rxn_smiles"]


def test_uspto_50k_parser_split_label(tmp_path):
    zip_path = _build_uspto_50k_zip(tmp_path)
    rows = list(uspto_src.iter_uspto_50k(zip_path))
    assert rows[0]["split"] == "train"


def test_split_rxn_smiles_round_trip():
    reactants, reagents, product = uspto_src._split_rxn_smiles("CCO.CC(=O)Cl>>CCOC(C)=O")
    assert reactants == ["CCO", "CC(=O)Cl"]
    assert reagents == []
    assert product == "CCOC(C)=O"


def test_split_rxn_smiles_rejects_bad_format():
    with pytest.raises(ValueError):
        uspto_src._split_rxn_smiles("CCO+CCN")


# ===== USPTO-full parser =====

def _build_uspto_full_tarball(tmp_path: Path) -> Path:
    """Tar.gz containing one .jsonl with 2 reactions."""
    tar_path = tmp_path / "uspto_full.tar.gz"
    jsonl_bytes = io.BytesIO()
    for i, rxn in enumerate([
        {"rxn_smiles": "CCO.CC(=O)Cl>>CCOC(C)=O", "patent_id": "US1234", "year": 2010, "yield_pct": 88.0},
        {"rxn_smiles": "c1ccccc1Br.c1ccccc1B(O)O>>c1ccccc1-c1ccccc1", "patent_id": "US5678", "year": 2011},
    ]):
        jsonl_bytes.write((json.dumps(rxn) + "\n").encode("utf-8"))
    jsonl_bytes.seek(0)
    with tarfile.open(tar_path, "w:gz") as tf:
        info = tarfile.TarInfo("uspto_reactions.jsonl")
        info.size = len(jsonl_bytes.getvalue())
        tf.addfile(info, jsonl_bytes)
    return tar_path


def test_uspto_full_parser_basic(tmp_path):
    tar_path = _build_uspto_full_tarball(tmp_path)
    rows = list(uspto_src.iter_uspto_full(tar_path))
    assert len(rows) == 2
    assert rows[0]["source"] == "uspto_full"
    assert rows[0]["product"] == "CCOC(C)=O"
    assert rows[0]["patent_id"] == "US1234"
    assert rows[0]["yield_pct"] == 88.0


# ===== USPTO-LLM parser =====

def _build_uspto_llm_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "uspto_llm.zip"
    payload = {
        "patent_id": "US9999",
        "reactions": [
            {"rxn_smiles": "CCO.CC(=O)Cl>>CCOC(C)=O",
             "solvent": "DMF", "temperature": "rt", "catalyst": "none", "yield": 91.5},
            {"rxn_smiles": "c1ccccc1Br.c1ccccc1B(O)O>>c1ccccc1-c1ccccc1",
             "solvent": "dioxane", "temperature": "80C", "catalyst": "Pd(PPh3)4"},
        ],
    }
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("uspto_llm.jsonl", json.dumps(payload) + "\n")
    return zip_path


def test_uspto_llm_parser_extracts_conditions(tmp_path):
    zip_path = _build_uspto_llm_zip(tmp_path)
    rows = list(uspto_src.iter_uspto_llm(zip_path))
    assert len(rows) == 2
    assert rows[0]["source"] == "uspto_llm"
    assert rows[0]["solvent_raw"] == "DMF"
    assert rows[0]["yield_pct"] == 91.5
    assert rows[1]["catalyst_raw"] == "Pd(PPh3)4"


def test_uspto_llm_parser_patent_id_in_record_id(tmp_path):
    zip_path = _build_uspto_llm_zip(tmp_path)
    rows = list(uspto_src.iter_uspto_llm(zip_path))
    assert rows[0]["source_record_id"].startswith("US9999:")


# ===== Buyables: cost-tier classification =====

def test_classify_cost_tier_tiers():
    assert buyables_src._classify_cost_tier(5.0) == "tier1"
    assert buyables_src._classify_cost_tier(50.0) == "tier2"
    assert buyables_src._classify_cost_tier(500.0) == "tier3"
    assert buyables_src._classify_cost_tier(None) == "unknown"


def test_buyables_config_default_snapshot_date():
    cfg = buyables_src.BuyablesConfig()
    assert cfg.snapshot_date == "2026-05-12"


# ===== ORD parser: skip-without-package guard =====

def test_ord_parser_raises_without_ord_schema(tmp_path):
    """ord-schema is heavyweight; CI may not have it. Verify we raise cleanly."""
    fake_pb = tmp_path / "fake.pb.gz"
    fake_pb.write_bytes(gzip.compress(b"\x00\x00\x00\x00"))
    try:
        import ord_schema  # noqa: F401
        pytest.skip("ord_schema installed; skipping the 'no-package' path")
    except ImportError:
        from rasyn.data.sources import ord as ord_src
        with pytest.raises(RuntimeError, match="ord-schema not installed"):
            list(ord_src._stream_reactions_from_pb(fake_pb))
