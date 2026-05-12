"""R-1 curation orchestrator for Rasyn-Retro.

Pipeline:
  1. Download + parse: USPTO subsets + ORD + buyables sources.
  2. Canonicalize SMILES + compute InChIKey for products / reactants.
  3. Atom-map unmapped reactions with RXNMapper (optional, GPU-friendly).
  4. Heuristically classify each reaction into the 12 coarse buckets.
  5. Deduplicate reactions by (mapped_rxn_smiles, product_inchi_key).
  6. Decontaminate against the sealed-case registry (3 layers).
  7. Extract templates with RDChiral; decontaminate templates.
  8. Emit:
       rasyn/data/clean/retro/reactions_bronze.parquet  (USPTO-full + ORD)
       rasyn/data/clean/retro/reactions_silver.parquet  (USPTO-50K + USPTO-LLM)
       rasyn/data/clean/retro/buyables.parquet
       rasyn/data/clean/retro/templates.pkl
       artifacts/retro_decontam_audit/audit.json

Smoke mode (--smoke) runs a tiny slice for local verification.

Run on a single 5x A100 pod (~4-8 GPU-h for RXNMapper at full scale,
1-2 days wall for end-to-end including downloads). Smoke run finishes
in <60s locally.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

# Lazy imports for pyarrow / pandas / rdkit so a smoke run can degrade gracefully
# when not all deps are available.

from rasyn.data.decontam.retro_quarantine import (
    RetroQuarantineReport,
    build_retro_forbidden_index,
    scrub_reactions,
    scrub_templates,
)
from rasyn.data.sources import buyables as buyables_src
from rasyn.data.sources import ord as ord_src
from rasyn.data.sources import uspto as uspto_src
from rasyn.synth.retro.reactions import (
    bucketize_class_name,
    canonicalize_reaction,
    inchi_key_from_smiles,
)
from rasyn.synth.retro.registry import load_retro_sealed_case_registry
from rasyn.synth.retro.templates import RetroTemplate, extract_templates_bulk

DEFAULT_OUT_DIR = Path("rasyn/data/clean/retro")
DEFAULT_AUDIT_DIR = Path("artifacts/retro_decontam_audit")

logger = logging.getLogger("retro_curation")


# ---------- Row builders ----------

def _build_reaction_row(parsed: dict, *, quality_tier: str) -> dict | None:
    """Canonicalize parsed reaction + compute InChIKeys.

    Returns a dict with the keys expected by the Reaction schema's
    fields. Returns None if the reaction is unparseable.
    """
    reactants = parsed.get("reactants") or []
    product = parsed.get("product") or ""
    if not reactants or not product:
        return None
    canon = canonicalize_reaction(reactants, product)
    if canon is None:
        return None
    canon_reactants, canon_product = canon

    reactant_inchi_keys = []
    for r in canon_reactants:
        ik = inchi_key_from_smiles(r)
        if ik is None:
            return None
        reactant_inchi_keys.append(ik)
    product_inchi_key = inchi_key_from_smiles(canon_product)
    if product_inchi_key is None:
        return None

    return {
        "source": parsed.get("source"),
        "source_record_id": parsed.get("source_record_id"),
        "reactant_smiles": canon_reactants,
        "reagent_smiles": parsed.get("reagents") or [],
        "solvent_smiles": parsed.get("solvents") or [],
        "catalyst_smiles": parsed.get("catalysts") or [],
        "product_smiles": canon_product,
        "reactant_inchi_keys": reactant_inchi_keys,
        "product_inchi_key": product_inchi_key,
        "mapped_rxn_smiles": parsed.get("mapped_rxn_smiles"),
        "reaction_class": bucketize_class_name(parsed.get("reaction_class_raw")),
        "yield_pct": parsed.get("yield_pct"),
        "document_id": parsed.get("document_id") or parsed.get("patent_id"),
        "quality_tier": quality_tier,
    }


def _iter_reactions_all(
    *,
    use_uspto_50k: bool,
    use_uspto_full: bool,
    use_uspto_llm: bool,
    use_ord: bool,
    limit_per_source: int | None,
) -> Iterator[dict]:
    if use_uspto_50k:
        cfg = uspto_src.USPTOConfig(subset="50k")
        for i, r in enumerate(uspto_src.stream_uspto(cfg)):
            if limit_per_source is not None and i >= limit_per_source:
                break
            yield r
    if use_uspto_full:
        cfg = uspto_src.USPTOConfig(subset="full")
        for i, r in enumerate(uspto_src.stream_uspto(cfg)):
            if limit_per_source is not None and i >= limit_per_source:
                break
            yield r
    if use_uspto_llm:
        cfg = uspto_src.USPTOConfig(subset="llm")
        for i, r in enumerate(uspto_src.stream_uspto(cfg)):
            if limit_per_source is not None and i >= limit_per_source:
                break
            yield r
    if use_ord:
        cfg = ord_src.ORDConfig()
        for i, r in enumerate(ord_src.stream_ord(cfg)):
            if limit_per_source is not None and i >= limit_per_source:
                break
            yield r


def _atom_map_inplace(rows: list[dict], *, device: str = "cpu") -> None:
    """Fill mapped_rxn_smiles for rows that lack one. Modifies in-place.

    Uses RXNMapper batched. Falls back silently when rxnmapper is missing
    (rows keep their existing mapped_rxn_smiles or None).
    """
    try:
        from rasyn.synth.retro.reactions import AtomMapper
    except ImportError:
        return
    mapper = AtomMapper(device=device)
    todo_idx = [i for i, row in enumerate(rows) if not row.get("mapped_rxn_smiles")]
    if not todo_idx:
        return
    rxn_smiles_list = [
        ".".join(rows[i]["reactant_smiles"]) + ">>" + rows[i]["product_smiles"]
        for i in todo_idx
    ]
    try:
        results = mapper.map_batch(rxn_smiles_list)
    except Exception as e:
        logger.warning("atom-mapping failed: %s", e)
        return
    for idx, result in zip(todo_idx, results):
        if result is None:
            continue
        rows[idx]["mapped_rxn_smiles"] = result.get("mapped_rxn")


def _dedup_by_mapped_rxn(rows: Iterable[dict]) -> list[dict]:
    """Deduplicate by (mapped_rxn_smiles, product_inchi_key)."""
    seen: set[tuple] = set()
    kept: list[dict] = []
    for row in rows:
        key = (row.get("mapped_rxn_smiles") or row.get("product_inchi_key"), row.get("product_inchi_key"))
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


# ---------- Buyables builder ----------

def _build_buyables_table(snapshot_date: str, smoke: bool) -> list[dict]:
    cfg = buyables_src.BuyablesConfig(snapshot_date=snapshot_date)
    grouped: dict[str, dict] = {}
    # ZINC22 + Enamine + eMolecules
    sources_iter = (
        buyables_src.stream_zinc22(cfg) if not smoke else iter([
            {"source": "ZINC22", "smiles_raw": "CCO", "catalog_id": "ZINC000000",
             "cost_per_g_usd": None, "snapshot_date": snapshot_date},
            {"source": "ZINC22", "smiles_raw": "CC(=O)O", "catalog_id": "ZINC000001",
             "cost_per_g_usd": None, "snapshot_date": snapshot_date},
        ])
    )
    for row in sources_iter:
        smi = row["smiles_raw"]
        from rasyn.synth.retro.reactions import canonicalize_smiles, inchi_key_from_smiles
        cs = canonicalize_smiles(smi)
        if cs is None:
            continue
        ik = inchi_key_from_smiles(cs)
        if ik is None:
            continue
        existing = grouped.get(ik)
        if existing is None:
            grouped[ik] = {
                "inchi_key": ik,
                "canonical_smiles": cs,
                "inventory_sources": [row["source"]],
                "cost_tier": buyables_src._classify_cost_tier(row["cost_per_g_usd"]),
                "cost_per_g_usd": row["cost_per_g_usd"],
                "catalog_id": row["catalog_id"],
                "snapshot_date": snapshot_date,
            }
        else:
            if row["source"] not in existing["inventory_sources"]:
                existing["inventory_sources"].append(row["source"])
            # Prefer the cheaper price
            if row["cost_per_g_usd"] is not None and (
                existing["cost_per_g_usd"] is None or row["cost_per_g_usd"] < existing["cost_per_g_usd"]
            ):
                existing["cost_per_g_usd"] = row["cost_per_g_usd"]
                existing["cost_tier"] = buyables_src._classify_cost_tier(row["cost_per_g_usd"])
                existing["catalog_id"] = row["catalog_id"]
    return list(grouped.values())


# ---------- Main ----------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    p.add_argument("--smoke", action="store_true",
                   help="Run a tiny smoke slice (USPTO-50K small subset, no full downloads).")
    p.add_argument("--limit-per-source", type=int, default=None,
                   help="Cap reactions per source (debug).")
    p.add_argument("--use-uspto-50k", action="store_true", default=True)
    p.add_argument("--use-uspto-full", action="store_true", default=False)
    p.add_argument("--use-uspto-llm", action="store_true", default=False)
    p.add_argument("--use-ord", action="store_true", default=False)
    p.add_argument("--skip-atom-mapping", action="store_true", default=False)
    p.add_argument("--rxnmapper-device", default="cpu")
    p.add_argument("--snapshot-date", default="2026-05-12")
    p.add_argument("--template-min-frequency", type=int, default=5)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    out_dir = args.out_dir
    audit_dir = args.audit_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.use_uspto_50k = True
        args.use_uspto_full = False
        args.use_uspto_llm = False
        args.use_ord = False
        if args.limit_per_source is None:
            args.limit_per_source = 200

    logger.info("loading sealed-case registry")
    reg = load_retro_sealed_case_registry()
    fidx = build_retro_forbidden_index(reg)
    report = RetroQuarantineReport()

    logger.info("streaming + canonicalizing reactions")
    bronze_rows: list[dict] = []
    silver_rows: list[dict] = []
    t0 = time.time()
    for parsed in _iter_reactions_all(
        use_uspto_50k=args.use_uspto_50k,
        use_uspto_full=args.use_uspto_full,
        use_uspto_llm=args.use_uspto_llm,
        use_ord=args.use_ord,
        limit_per_source=args.limit_per_source,
    ):
        src = parsed.get("source")
        tier = "silver" if src in ("uspto_50k", "uspto_llm") else "bronze"
        row = _build_reaction_row(parsed, quality_tier=tier)
        if row is None:
            continue
        (silver_rows if tier == "silver" else bronze_rows).append(row)
    logger.info(
        "parsed %d bronze + %d silver rows in %.1fs",
        len(bronze_rows), len(silver_rows), time.time() - t0,
    )

    if not args.skip_atom_mapping:
        logger.info("atom-mapping bronze + silver (RXNMapper, device=%s)", args.rxnmapper_device)
        _atom_map_inplace(bronze_rows, device=args.rxnmapper_device)
        _atom_map_inplace(silver_rows, device=args.rxnmapper_device)

    logger.info("dedup bronze + silver")
    bronze_rows = _dedup_by_mapped_rxn(bronze_rows)
    silver_rows = _dedup_by_mapped_rxn(silver_rows)
    logger.info("dedup: %d bronze + %d silver", len(bronze_rows), len(silver_rows))

    logger.info("decontaminating reactions")
    bronze_kept = list(scrub_reactions(bronze_rows, fidx, report=report))
    silver_kept = list(scrub_reactions(silver_rows, fidx, report=report))
    logger.info("decontam: kept %d bronze + %d silver", len(bronze_kept), len(silver_kept))

    logger.info("extracting templates from kept reactions")
    template_input = (
        (row.get("mapped_rxn_smiles") or "", row.get("source_record_id") or "")
        for row in bronze_kept + silver_kept
        if row.get("mapped_rxn_smiles")
    )
    templates = extract_templates_bulk(template_input, min_frequency=args.template_min_frequency)
    logger.info("extracted %d templates", len(templates))

    logger.info("decontaminating templates")
    source_lookup: dict[str, str] = {
        row.get("source_record_id") or "": row.get("product_smiles") or ""
        for row in bronze_kept + silver_kept
    }
    templates_kept = scrub_templates(templates, source_lookup, fidx, report=report)
    logger.info("decontam templates: kept %d of %d", len(templates_kept), len(templates))

    logger.info("building buyables")
    buyables_rows = _build_buyables_table(args.snapshot_date, args.smoke)
    logger.info("built %d buyables", len(buyables_rows))

    logger.info("writing parquet + pickle outputs")
    _write_outputs(out_dir, bronze_kept, silver_kept, buyables_rows, templates_kept)

    audit = {
        "snapshot_date": args.snapshot_date,
        "smoke_mode": args.smoke,
        "report": report.to_dict(),
        "out_dir": str(out_dir),
        "counts": {
            "bronze_rows": len(bronze_kept),
            "silver_rows": len(silver_kept),
            "templates": len(templates_kept),
            "buyables": len(buyables_rows),
        },
    }
    (audit_dir / "audit.json").write_text(json.dumps(audit, indent=2))
    logger.info("audit -> %s", audit_dir / "audit.json")
    return 0


def _write_outputs(
    out_dir: Path,
    bronze: list[dict],
    silver: list[dict],
    buyables: list[dict],
    templates: list[RetroTemplate],
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        # Fallback to pickle if pyarrow missing
        for name, rows in (("reactions_bronze", bronze), ("reactions_silver", silver), ("buyables", buyables)):
            with open(out_dir / f"{name}.pkl", "wb") as fh:
                pickle.dump(rows, fh)
    else:
        for name, rows in (("reactions_bronze", bronze), ("reactions_silver", silver), ("buyables", buyables)):
            if not rows:
                continue
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, out_dir / f"{name}.parquet")
    with open(out_dir / "templates.pkl", "wb") as fh:
        pickle.dump(templates, fh)


if __name__ == "__main__":
    raise SystemExit(main())
