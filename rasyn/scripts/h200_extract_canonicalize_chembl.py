"""Extract ChEMBL bulk SQLite + canonicalize molecules + decontaminate.

Run after `chembl_35_sqlite.tar.gz` finishes downloading. Streams molecules
from ChEMBL's molecule_dictionary, canonicalizes via RDKit, runs Pass-0
decontamination against the sealed-case registry, writes parquet.

Outputs (rasyn/data/clean/):
  - molecules_canonical.parquet    [chembl_id, canonical_smiles, inchi_key, max_phase]
  - chembl_extract_report.json     {n_input, n_canonicalized, n_after_decontam}
  - oxs_compound_search.json       any ChEMBL rows whose synonyms / pref_name match 'OXS00*'

Run:
  cd ~/wolverine/rasyn && python scripts/h200_extract_canonicalize_chembl.py
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sqlite3
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd

from rasyn.data.decontam.quarantine import scrub_rows
from rasyn.data.registry.loader import load_sealed_case_registry
from rasyn.utils.canonicalize import standardize_pair

CHEMBL_TAR = Path("rasyn/data/raw/chembl/chembl_35_sqlite.tar.gz")
EXTRACT_DIR = Path("rasyn/data/raw/chembl/extracted")
OUT_DIR = Path("rasyn/data/clean")
OUT_PARQUET = OUT_DIR / "molecules_canonical.parquet"
OUT_REPORT = OUT_DIR / "chembl_extract_report.json"
OUT_OXS = OUT_DIR / "oxs_compound_search.json"


def find_sqlite_path(extract_dir: Path) -> Path | None:
    for p in extract_dir.rglob("chembl_*_sqlite/*.db"):
        return p
    for p in extract_dir.rglob("*.db"):
        return p
    return None


def extract_tarball(tar_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    db = find_sqlite_path(extract_dir)
    if db is None:
        raise RuntimeError(f"No .db found under {extract_dir}")
    print(f"  -> {db} ({db.stat().st_size / 1e9:.1f} GB)")
    return db


def _canon_one(args: tuple[str, str, str, int | None]) -> dict | None:
    chembl_id, smi, ik, max_phase = args
    cs, computed_ik = standardize_pair(smi)
    if cs is None:
        return None
    return {
        "chembl_id": chembl_id,
        "canonical_smiles": cs,
        "inchi_key": computed_ik or ik,
        "max_phase": max_phase,
    }


def stream_molecules(db_path: Path, limit: int | None = None):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sql = """
        SELECT md.chembl_id, cs.canonical_smiles, cs.standard_inchi_key, md.max_phase
        FROM molecule_dictionary md
        JOIN compound_structures cs ON cs.molregno = md.molregno
        WHERE cs.canonical_smiles IS NOT NULL
    """
    if limit:
        sql += f" LIMIT {limit}"
    cur = conn.cursor()
    cur.execute(sql)
    while True:
        rows = cur.fetchmany(50_000)
        if not rows:
            break
        for r in rows:
            yield r
    conn.close()


def search_oxs_compounds(db_path: Path) -> list[dict]:
    """Find any ChEMBL row whose synonym or pref_name matches OXS00* — try to recover OXS008474."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    out: list[dict] = []
    sql = """
        SELECT md.chembl_id, md.pref_name, ms.synonyms, cs.canonical_smiles, cs.standard_inchi_key
        FROM molecule_dictionary md
        LEFT JOIN molecule_synonyms ms ON ms.molregno = md.molregno
        LEFT JOIN compound_structures cs ON cs.molregno = md.molregno
        WHERE (md.pref_name LIKE 'OXS00%' OR ms.synonyms LIKE 'OXS00%')
    """
    try:
        cur.execute(sql)
        for chembl_id, pref, syn, smi, ik in cur.fetchall():
            out.append({
                "chembl_id": chembl_id,
                "pref_name": pref,
                "synonym": syn,
                "canonical_smiles": smi,
                "inchi_key": ik,
            })
    except Exception as e:
        out.append({"error": str(e)})
    conn.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="Limit molecules processed (debug)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    p.add_argument("--no-decontam", action="store_true")
    p.add_argument("--skip-extract", action="store_true", help="Use existing extracted dir")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_extract:
        db = find_sqlite_path(EXTRACT_DIR)
        if db is None:
            print("FATAL: --skip-extract but no DB found", file=sys.stderr)
            return 2
    else:
        if not CHEMBL_TAR.exists():
            print(f"FATAL: {CHEMBL_TAR} not found. Run wget first.", file=sys.stderr)
            return 2
        db = extract_tarball(CHEMBL_TAR, EXTRACT_DIR)

    print(f"[{time.strftime('%H:%M:%S')}] OXS compound search...")
    oxs_hits = search_oxs_compounds(db)
    OUT_OXS.write_text(json.dumps(oxs_hits, indent=2))
    print(f"  Found {len(oxs_hits)} OXS-prefixed entries -> {OUT_OXS}")
    for hit in oxs_hits[:10]:
        print(f"    {hit}")

    print(f"[{time.strftime('%H:%M:%S')}] Canonicalising molecules ({args.workers} workers)...")
    rows = list(stream_molecules(db, limit=args.limit))
    print(f"  Loaded {len(rows):,} raw rows")

    t0 = time.time()
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            results = pool.imap_unordered(_canon_one, rows, chunksize=2000)
            kept = [r for r in results if r is not None]
    else:
        kept = [r for r in (_canon_one(row) for row in rows) if r is not None]
    print(f"  Canonicalized {len(kept):,} / {len(rows):,} in {time.time() - t0:.1f}s")

    # Pass-0 decontamination
    decontam_report: dict = {}
    if not args.no_decontam:
        reg = load_sealed_case_registry()
        scrub_rows_input = [
            {"smiles": r["canonical_smiles"], "inchi_key": r["inchi_key"], "chembl_id": r["chembl_id"]}
            for r in kept
        ]
        kept_idx = []
        for i, r in enumerate(scrub_rows_input):
            kept_idx.append((i, r))
        # Scrub all at once for efficiency
        kept_after, report = scrub_rows([r for _, r in kept_idx], reg, canonicalize=False)
        survived_chembl = {r["chembl_id"] for r in kept_after}
        kept_final = [r for r in kept if r["chembl_id"] in survived_chembl]
        decontam_report = report.to_dict()
    else:
        kept_final = kept

    df = pd.DataFrame(kept_final)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, compression="zstd", index=False)
    print(f"[{time.strftime('%H:%M:%S')}] Wrote {len(df):,} rows to {OUT_PARQUET} ({OUT_PARQUET.stat().st_size / 1e6:.1f} MB)")

    OUT_REPORT.write_text(json.dumps({
        "raw_rows": len(rows),
        "canonicalized": len(kept),
        "after_decontam": len(kept_final),
        "elapsed_seconds": time.time() - t0,
        "decontam_report": decontam_report,
        "oxs_search_hits": len(oxs_hits),
    }, indent=2))
    print(f"  Report: {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
