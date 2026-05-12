"""USPTO reaction-corpus adapters (RETRO_PLAN R-1, MVP data pack item 1+2+3+9).

Supports three USPTO subsets:
  - USPTO-50K (figshare; ~50K atom-mapped reactions, 10 reaction classes).
  - USPTO-full (Lowe 2017; ~1.8M reactions from US patents 1976-2016).
  - USPTO-LLM (2025, Zenodo 14396156; 247K LLM-extracted reactions with
    conditions + step segmentation).

Strategy: download once into rasyn/data/raw/uspto/, parse on demand.

Note on atom mapping:
  - USPTO-50K ships pre-mapped.
  - USPTO-full and USPTO-LLM may or may not be mapped; we run RXNMapper
    in the curation orchestrator (R-1) to fill mapped_rxn_smiles.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.request import urlretrieve

DEFAULT_RAW_DIR = Path("rasyn/data/raw/uspto")

# Public download URLs (verified as of 2026-05-12). Subject to change.
USPTO_50K_URL = (
    "https://figshare.com/ndownloader/files/45032717"
)  # USPTO-50K raw, figshare 25459573.

USPTO_FULL_URL = (
    "https://figshare.com/ndownloader/files/8664379"
)  # USPTO-full (Lowe 2017), figshare 5104873; ~3 GB tar.gz of CSV-like files.

USPTO_LLM_URL = (
    "https://zenodo.org/records/14396156/files/USPTO-LLM.zip"
)  # USPTO-LLM (WWW 2025), Zenodo 14396156.


@dataclass
class USPTOConfig:
    raw_dir: Path = DEFAULT_RAW_DIR
    subset: str = "50k"  # "50k" | "full" | "llm"
    timeout_s: int = 600


# ---------- Download helpers ----------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_uspto(cfg: USPTOConfig) -> Path:
    """Download the requested USPTO subset into `cfg.raw_dir`.

    Idempotent: skips if already downloaded. Returns the path to the
    on-disk archive.
    """
    _ensure_dir(cfg.raw_dir)
    if cfg.subset == "50k":
        target = cfg.raw_dir / "uspto_50k.zip"
        url = USPTO_50K_URL
    elif cfg.subset == "full":
        target = cfg.raw_dir / "uspto_full.tar.gz"
        url = USPTO_FULL_URL
    elif cfg.subset == "llm":
        target = cfg.raw_dir / "uspto_llm.zip"
        url = USPTO_LLM_URL
    else:
        raise ValueError(f"unknown USPTO subset: {cfg.subset!r}")
    if not target.exists():
        urlretrieve(url, target)
    return target


# ---------- Parsers (subset-specific) ----------

def _split_rxn_smiles(rxn_smiles: str) -> tuple[list[str], list[str], str]:
    """Split a `reactants>reagents>products` SMILES into ([reactants], [reagents], product).

    Returns the *single* product SMILES (we discard multi-product reactions
    upstream of this function via filtering, but if there are multiple, we
    take the first; the orchestrator will fan out the rest).
    """
    parts = rxn_smiles.split(">")
    if len(parts) != 3:
        raise ValueError(f"reaction SMILES must have 3 '>'-separated parts: {rxn_smiles!r}")
    reactants_blk, reagents_blk, products_blk = parts
    reactants = [s for s in reactants_blk.split(".") if s]
    reagents = [s for s in reagents_blk.split(".") if s]
    products = [s for s in products_blk.split(".") if s]
    product = products[0] if products else ""
    return reactants, reagents, product


def iter_uspto_50k(archive_path: Path) -> Iterator[dict]:
    """Yield dicts with keys: rxn_smiles, mapped_rxn_smiles, reactants, reagents, product, source_record_id.

    USPTO-50K is bundled as a zip with CSV files (train/val/test). Returns
    one dict per reaction.
    """
    with zipfile.ZipFile(archive_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            with zf.open(name) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8", newline="")
                reader = csv.DictReader(text)
                for row_id, row in enumerate(reader):
                    rxn = row.get("reactants>reagents>production") or row.get("rxn_smiles") or row.get("reaction")
                    if not rxn:
                        continue
                    try:
                        reactants, reagents, product = _split_rxn_smiles(rxn)
                    except ValueError:
                        continue
                    yield {
                        "source": "uspto_50k",
                        "source_record_id": f"{Path(name).stem}:{row_id}",
                        "rxn_smiles": rxn,
                        "mapped_rxn_smiles": rxn,  # 50k ships mapped
                        "reactants": reactants,
                        "reagents": reagents,
                        "product": product,
                        "reaction_class_raw": row.get("class") or row.get("reaction_class"),
                        "split": Path(name).stem,
                    }


def iter_uspto_full(archive_path: Path) -> Iterator[dict]:
    """Yield reaction dicts from the Lowe USPTO-full tarball.

    The Lowe corpus ships as a tar.gz of (potentially many) JSONL or TSV
    files keyed by patent. The actual filenames vary between releases; this
    iterator handles both .jsonl(.gz) and .tsv layouts.
    """
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            fname = member.name
            if fname.endswith((".jsonl", ".jsonl.gz")):
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                if fname.endswith(".gz"):
                    fh = gzip.GzipFile(fileobj=fh)
                for line in io.TextIOWrapper(fh, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rxn = row.get("rxn_smiles") or row.get("reaction_smiles")
                    if not rxn:
                        continue
                    try:
                        reactants, reagents, product = _split_rxn_smiles(rxn)
                    except ValueError:
                        continue
                    yield {
                        "source": "uspto_full",
                        "source_record_id": row.get("source_id") or row.get("patent_id"),
                        "rxn_smiles": rxn,
                        "mapped_rxn_smiles": row.get("mapped_rxn_smiles"),
                        "reactants": reactants,
                        "reagents": reagents,
                        "product": product,
                        "year": row.get("year"),
                        "patent_id": row.get("patent_id"),
                        "yield_pct": row.get("yield_pct"),
                    }
            elif fname.endswith((".tsv", ".csv")):
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                text = io.TextIOWrapper(fh, encoding="utf-8")
                # Try sniff CSV vs TSV
                sample = text.read(4096)
                text.seek(0)
                delim = "\t" if "\t" in sample else ","
                reader = csv.DictReader(text, delimiter=delim)
                for row_id, row in enumerate(reader):
                    rxn = row.get("rxn_smiles") or row.get("reactionsmiles") or row.get("ReactionSmiles")
                    if not rxn:
                        continue
                    try:
                        reactants, reagents, product = _split_rxn_smiles(rxn)
                    except ValueError:
                        continue
                    yield {
                        "source": "uspto_full",
                        "source_record_id": f"{fname}:{row_id}",
                        "rxn_smiles": rxn,
                        "mapped_rxn_smiles": row.get("mapped_rxn_smiles"),
                        "reactants": reactants,
                        "reagents": reagents,
                        "product": product,
                        "year": row.get("Year") or row.get("year"),
                        "patent_id": row.get("PatentNumber") or row.get("patent_id"),
                    }


def iter_uspto_llm(archive_path: Path) -> Iterator[dict]:
    """Yield reaction dicts from the USPTO-LLM Zenodo zip (2025).

    USPTO-LLM ships as a zip of JSONL with LLM-extracted conditions per
    reaction step. Each top-level record may contain multiple `reactions`
    (one per step); we yield each as a separate dict.
    """
    with zipfile.ZipFile(archive_path) as zf:
        for name in zf.namelist():
            if not name.endswith((".jsonl", ".jsonl.gz", ".json")):
                continue
            with zf.open(name) as fh:
                stream = gzip.GzipFile(fileobj=fh) if name.endswith(".gz") else fh
                for line in io.TextIOWrapper(stream, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    patent_id = rec.get("patent_id") or rec.get("patent")
                    reactions = rec.get("reactions", [rec])
                    for r_idx, rxn_rec in enumerate(reactions):
                        rxn = rxn_rec.get("rxn_smiles") or rxn_rec.get("reaction_smiles")
                        if not rxn:
                            continue
                        try:
                            reactants, reagents, product = _split_rxn_smiles(rxn)
                        except ValueError:
                            continue
                        yield {
                            "source": "uspto_llm",
                            "source_record_id": f"{patent_id}:{r_idx}" if patent_id else f"{name}:{r_idx}",
                            "rxn_smiles": rxn,
                            "mapped_rxn_smiles": rxn_rec.get("mapped_rxn_smiles"),
                            "reactants": reactants,
                            "reagents": reagents,
                            "product": product,
                            "patent_id": patent_id,
                            "conditions_raw": rxn_rec.get("conditions"),
                            "temperature_raw": rxn_rec.get("temperature"),
                            "solvent_raw": rxn_rec.get("solvent"),
                            "catalyst_raw": rxn_rec.get("catalyst"),
                            "yield_pct": rxn_rec.get("yield"),
                        }


def stream_uspto(cfg: USPTOConfig) -> Iterator[dict]:
    """Top-level streamer: download (if needed) + parse the requested subset."""
    archive = download_uspto(cfg)
    if cfg.subset == "50k":
        yield from iter_uspto_50k(archive)
    elif cfg.subset == "full":
        yield from iter_uspto_full(archive)
    elif cfg.subset == "llm":
        yield from iter_uspto_llm(archive)
    else:
        raise ValueError(f"unknown USPTO subset: {cfg.subset!r}")
