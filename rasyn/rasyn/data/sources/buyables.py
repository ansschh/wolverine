"""Buyables / building-block inventory adapter (RETRO_PLAN R-1 + L4).

Three free sources, frozen to a single snapshot date:
  - ZINC-22 in-stock tranches (cartblanche.docking.org, files.docking.org)
  - Enamine REAL Building Blocks free SMILES tier
  - eMolecules free monthly snapshot

Output: BuyabilityRecord parquet with InChIKey + canonical_smiles +
inventory_sources (list) + cost_tier + cost_per_g_usd + catalog_id +
snapshot_date.

Cost tiers (per RETRO_PLAN L4):
  - tier1: <= $10/g  (preferred for headline `tier1-only-route-found`)
  - tier2: $10-$100/g
  - tier3: > $100/g
  - unknown: no pricing data
"""
from __future__ import annotations

import csv
import gzip
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ._download import download_validated

DEFAULT_RAW_DIR = Path("rasyn/data/raw/buyables")

ZINC22_INSTOCK_URLS: list[str] = [
    "https://files.docking.org/zinc22/zinc-22-in-stock.smi.gz",
]
ENAMINE_REAL_BB_URLS: list[str] = [
    "https://enamine.net/files/REAL_BB_Database/Enamine_Building_Blocks_Stock.sdf",
]
EMOLECULES_FREE_URLS: list[str] = [
    "https://downloads.emolecules.com/free/2026-05-01/version.smi.gz",
]


@dataclass
class BuyablesConfig:
    raw_dir: Path = DEFAULT_RAW_DIR
    snapshot_date: str = "2026-05-12"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _classify_cost_tier(cost_per_g_usd: float | None) -> str:
    if cost_per_g_usd is None:
        return "unknown"
    if cost_per_g_usd <= 10.0:
        return "tier1"
    if cost_per_g_usd <= 100.0:
        return "tier2"
    return "tier3"


# ---------- ZINC-22 ----------

def download_zinc22(cfg: BuyablesConfig) -> Path:
    _ensure_dir(cfg.raw_dir)
    target = cfg.raw_dir / "zinc22_instock.smi.gz"
    return download_validated(
        ZINC22_INSTOCK_URLS,
        target,
        kind="gz",
        min_bytes=512 * 1024,
    )


def stream_zinc22(cfg: BuyablesConfig) -> Iterator[dict]:
    """Yield buyability dicts from ZINC-22 in-stock list.

    ZINC SMI files are whitespace-separated: SMILES then ZINC ID.
    Pricing is not in the bulk SMI; we leave cost_per_g_usd=None
    (cost_tier='unknown').
    """
    path = download_zinc22(cfg)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 1:
                continue
            smi = parts[0]
            zinc_id = parts[1] if len(parts) > 1 else None
            yield {
                "source": "ZINC22",
                "smiles_raw": smi,
                "catalog_id": zinc_id,
                "cost_per_g_usd": None,
                "snapshot_date": cfg.snapshot_date,
            }


# ---------- Enamine REAL Building Blocks ----------

def download_enamine_bb(cfg: BuyablesConfig) -> Path:
    _ensure_dir(cfg.raw_dir)
    target = cfg.raw_dir / "Enamine_BB_Stock.sdf"
    return download_validated(
        ENAMINE_REAL_BB_URLS,
        target,
        kind=None,  # SDF is text; we validate via min_bytes only
        min_bytes=1024 * 1024,
    )


def stream_enamine_bb(cfg: BuyablesConfig) -> Iterator[dict]:
    """Yield buyability dicts from Enamine BB SDF.

    Uses RDKit's SDMolSupplier to iterate records, extracting SMILES,
    catalog ID (Enamine 'idnumber' tag), and price-per-gram if present.
    """
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except ImportError as e:  # noqa: BLE001 -- explicit
        raise RuntimeError("rdkit required to parse Enamine SDF") from e

    path = download_enamine_bb(cfg)
    suppl = Chem.SDMolSupplier(str(path))
    for mol in suppl:
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol, canonical=True)
        catalog_id = mol.GetPropsAsDict().get("idnumber") or mol.GetPropsAsDict().get("ID")
        price_per_g = None
        for key in ("Price_1g_USD", "Price (USD/g)", "Price/g"):
            if mol.HasProp(key):
                try:
                    price_per_g = float(mol.GetProp(key))
                    break
                except ValueError:
                    pass
        yield {
            "source": "Enamine_REAL_BB",
            "smiles_raw": smi,
            "catalog_id": str(catalog_id) if catalog_id else None,
            "cost_per_g_usd": price_per_g,
            "snapshot_date": cfg.snapshot_date,
        }


# ---------- eMolecules ----------

def download_emolecules(cfg: BuyablesConfig) -> Path:
    _ensure_dir(cfg.raw_dir)
    target = cfg.raw_dir / "emolecules_free.smi.gz"
    return download_validated(
        EMOLECULES_FREE_URLS,
        target,
        kind="gz",
        min_bytes=512 * 1024,
    )


def stream_emolecules(cfg: BuyablesConfig) -> Iterator[dict]:
    """Yield buyability dicts from eMolecules free SMI gzip.

    The free tier is structures-only (no pricing). Use it as breadth, not
    cost-tier signal.
    """
    path = download_emolecules(cfg)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        # eMolecules SMI: SMILES, emol_id, parent_id, optional tier
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            smi = parts[0]
            catalog_id = parts[1] if len(parts) > 1 else None
            yield {
                "source": "eMolecules",
                "smiles_raw": smi,
                "catalog_id": catalog_id,
                "cost_per_g_usd": None,
                "snapshot_date": cfg.snapshot_date,
            }


def stream_all_buyables(cfg: BuyablesConfig) -> Iterator[dict]:
    """Stream all three sources sequentially.

    The orchestrator deduplicates by canonical InChIKey + unions the
    inventory_sources column.
    """
    yield from stream_zinc22(cfg)
    yield from stream_enamine_bb(cfg)
    yield from stream_emolecules(cfg)
