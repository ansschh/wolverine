"""USPTO reaction-corpus adapters (RETRO_PLAN R-1, MVP data pack item 1+2+3+9).

Supports three USPTO subsets:
  - USPTO-50K (~50K atom-mapped reactions, 10 reaction classes).
      Primary source: HuggingFace `sagawa/USPTO-50K` (parquet).
      Legacy/zip fallback retained for backwards compatibility.
  - USPTO-full (Lowe 2017; ~1.8M reactions from US patents 1976-2016).
  - USPTO-LLM (2025, Zenodo 14396156; 247K LLM-extracted reactions with
    conditions + step segmentation).

Strategy: download once into rasyn/data/raw/uspto/, parse on demand.
Downloads go through `_download.download_validated` so 0-byte / HTML
responses don't get cached as fake archives. Each source has a list of
candidate URLs that are tried in order (source-rot resilience).
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

from ._download import DownloadError, download_validated

DEFAULT_RAW_DIR = Path("rasyn/data/raw/uspto")

# ---------- USPTO-50K URLs ----------
# Tried in order; first that yields a valid file wins.
#
# Primary: HF parquet (Sagawa archive of Schneider 50K). Single-file shard.
# Secondary: archived figshare ID 45032717 (was the original, now appears to
#   return 0 bytes — kept here as a last-resort attempt in case the article
#   gets restored).
USPTO_50K_PARQUET_URLS: list[str] = [
    "https://huggingface.co/datasets/sagawa/USPTO-50K/resolve/main/data/train-00000-of-00001.parquet",
]
USPTO_50K_ZIP_URLS: list[str] = [
    # Original figshare; left in but expected to fail until article is republished.
    "https://figshare.com/ndownloader/files/45032717",
]

# ---------- USPTO-full (Lowe 2017) ----------
USPTO_FULL_URLS: list[str] = [
    # Original figshare; large (~3 GB tar.gz)
    "https://figshare.com/ndownloader/files/8664379",
]

# ---------- USPTO-LLM (Zenodo 14396156) ----------
USPTO_LLM_URLS: list[str] = [
    "https://zenodo.org/records/14396156/files/USPTO-LLM.zip",
]


@dataclass
class USPTOConfig:
    raw_dir: Path = DEFAULT_RAW_DIR
    subset: str = "50k"  # "50k" | "full" | "llm"
    timeout_s: int = 600
    prefer_parquet: bool = True  # USPTO-50K only: try parquet mirror before zip


# ---------- Download ----------

def download_uspto(cfg: USPTOConfig) -> Path:
    """Download the requested USPTO subset into `cfg.raw_dir`.

    Idempotent + validated: a previously cached file is verified before
    being reused, and a stale/corrupt cached file is redownloaded.
    Returns the on-disk archive path.
    """
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    if cfg.subset == "50k":
        if cfg.prefer_parquet:
            target = cfg.raw_dir / "uspto_50k.parquet"
            try:
                return download_validated(
                    USPTO_50K_PARQUET_URLS,
                    target,
                    kind="parquet",
                    min_bytes=64 * 1024,  # parquet header alone is ~1 KiB; real file is MBs
                    timeout_s=cfg.timeout_s,
                )
            except DownloadError:
                # fall through to zip mirror
                pass
        target = cfg.raw_dir / "uspto_50k.zip"
        return download_validated(
            USPTO_50K_ZIP_URLS,
            target,
            kind="zip",
            min_bytes=256 * 1024,
            timeout_s=cfg.timeout_s,
        )
    if cfg.subset == "full":
        target = cfg.raw_dir / "uspto_full.tar.gz"
        return download_validated(
            USPTO_FULL_URLS,
            target,
            kind="tar.gz",
            min_bytes=64 * 1024 * 1024,  # USPTO-full is multi-GB; a real download is far larger
            timeout_s=cfg.timeout_s,
        )
    if cfg.subset == "llm":
        target = cfg.raw_dir / "uspto_llm.zip"
        return download_validated(
            USPTO_LLM_URLS,
            target,
            kind="zip",
            min_bytes=1024 * 1024,
            timeout_s=cfg.timeout_s,
        )
    raise ValueError(f"unknown USPTO subset: {cfg.subset!r}")


# ---------- Parsers (subset-specific) ----------

def _split_rxn_smiles(rxn_smiles: str) -> tuple[list[str], list[str], str]:
    """Split a `reactants>reagents>products` SMILES into ([reactants], [reagents], product)."""
    parts = rxn_smiles.split(">")
    if len(parts) != 3:
        raise ValueError(f"reaction SMILES must have 3 '>'-separated parts: {rxn_smiles!r}")
    reactants_blk, reagents_blk, products_blk = parts
    reactants = [s for s in reactants_blk.split(".") if s]
    reagents = [s for s in reagents_blk.split(".") if s]
    products = [s for s in products_blk.split(".") if s]
    product = products[0] if products else ""
    return reactants, reagents, product


def _iter_uspto_50k_zip(archive_path: Path) -> Iterator[dict]:
    """Iterate USPTO-50K records out of a zip-of-CSVs (legacy figshare format)."""
    with zipfile.ZipFile(archive_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            with zf.open(name) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8", newline="")
                reader = csv.DictReader(text)
                for row_id, row in enumerate(reader):
                    rxn = (
                        row.get("reactants>reagents>production")
                        or row.get("rxn_smiles")
                        or row.get("reaction")
                    )
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
                        "mapped_rxn_smiles": rxn,
                        "reactants": reactants,
                        "reagents": reagents,
                        "product": product,
                        "reaction_class_raw": row.get("class") or row.get("reaction_class"),
                        "split": Path(name).stem,
                    }


def _iter_uspto_50k_parquet(archive_path: Path) -> Iterator[dict]:
    """Iterate USPTO-50K records out of a parquet shard (HF mirror format).

    The HF `sagawa/USPTO-50K` schema uses `rxn_smiles` (or similar) per row.
    We probe a list of likely column names so the iterator is robust to
    minor schema drift across mirror revisions.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(archive_path)
    columns = set(table.column_names)
    # Find the reaction-SMILES column. Most mirrors call it 'rxn_smiles';
    # Sagawa's release uses 'reactants>reagents>production' verbatim.
    rxn_col = next(
        (
            c
            for c in (
                "rxn_smiles",
                "reaction_smiles",
                "reactants>reagents>production",
                "reactants>reagents>products",
                "reaction",
            )
            if c in columns
        ),
        None,
    )
    if rxn_col is None:
        raise RuntimeError(
            f"USPTO-50K parquet {archive_path} has no recognised reaction column; "
            f"saw columns {sorted(columns)}"
        )
    class_col = next((c for c in ("class", "reaction_class", "class_id") if c in columns), None)
    split_col = next((c for c in ("split", "set", "subset") if c in columns), None)
    id_col = next((c for c in ("id", "rxn_id", "reaction_id") if c in columns), None)

    rows = table.to_pylist()
    for row_idx, row in enumerate(rows):
        rxn = row.get(rxn_col)
        if not rxn:
            continue
        try:
            reactants, reagents, product = _split_rxn_smiles(rxn)
        except ValueError:
            continue
        split = row.get(split_col) if split_col else "all"
        rid = row.get(id_col) if id_col else f"row{row_idx}"
        yield {
            "source": "uspto_50k",
            "source_record_id": f"{split}:{rid}",
            "rxn_smiles": rxn,
            "mapped_rxn_smiles": rxn,
            "reactants": reactants,
            "reagents": reagents,
            "product": product,
            "reaction_class_raw": (str(row.get(class_col)) if class_col else None),
            "split": str(split),
        }


def iter_uspto_50k(archive_path: Path) -> Iterator[dict]:
    """Dispatch to the right iterator based on extension."""
    if archive_path.suffix == ".parquet":
        yield from _iter_uspto_50k_parquet(archive_path)
    else:
        yield from _iter_uspto_50k_zip(archive_path)


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
    """Yield reaction dicts from the USPTO-LLM Zenodo zip (2025)."""
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
