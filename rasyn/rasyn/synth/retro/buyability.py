"""Buyability index (RETRO_PLAN R-4 + L4).

Frozen InChIKey-keyed lookup over the union of ZINC22 in-stock,
Enamine REAL BB, eMolecules. Loaded once at planner startup from
`rasyn/data/clean/retro/buyables.parquet`.

Tier-1 fast path: if `tier1_only=True`, the planner only accepts leaves
whose `cost_tier == "tier1"` (<= $10/g). The headline
`tier1-only-route-found` rate is reported separately from `any-buyable`
to guard against inflated success per RETRO_PLAN Risk 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rasyn.synth.retro.schemas import BuyabilityRecord


@dataclass
class BuyabilityIndexConfig:
    parquet_path: Path | None = None
    tier1_only: bool = False


class BuyabilityIndex:
    """In-memory dict {inchi_key -> BuyabilityRecord}."""

    def __init__(self, cfg: BuyabilityIndexConfig):
        self.cfg = cfg
        self._records: dict[str, BuyabilityRecord] = {}
        if cfg.parquet_path and cfg.parquet_path.exists():
            self._load_parquet(cfg.parquet_path)

    def _load_parquet(self, path: Path) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return
        for row in pq.read_table(path).to_pylist():
            try:
                rec = BuyabilityRecord(
                    inchi_key=row["inchi_key"],
                    canonical_smiles=row["canonical_smiles"],
                    inventory_sources=list(row.get("inventory_sources") or []),
                    cost_tier=row.get("cost_tier", "unknown"),
                    cost_per_g_usd=row.get("cost_per_g_usd"),
                    catalog_id=row.get("catalog_id"),
                    snapshot_date=row.get("snapshot_date", "unknown"),
                )
            except Exception:
                continue
            self._records[rec.inchi_key] = rec

    def add_record(self, record: BuyabilityRecord) -> None:
        self._records[record.inchi_key] = record

    def is_buyable(self, inchi_key: str) -> bool:
        rec = self._records.get(inchi_key)
        if rec is None:
            return False
        if self.cfg.tier1_only:
            return rec.cost_tier == "tier1"
        return True

    def lookup(self, inchi_key: str) -> BuyabilityRecord | None:
        return self._records.get(inchi_key)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, inchi_key: str) -> bool:
        return inchi_key in self._records
