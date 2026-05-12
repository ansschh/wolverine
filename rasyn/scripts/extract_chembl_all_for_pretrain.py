"""Extract ALL ChEMBL 36 canonical SMILES + approved-drug subset.

Two outputs:
  1. rasyn/data/clean/chembl_all_smiles.parquet
     Every distinct canonical_smiles in chembl_36 compound_structures.
     This feeds the 200M SMILES-LM pretrain.

  2. rasyn/data/clean/chembl_approved_drugs.parquet
     Subset where molecule_dictionary.max_phase >= 4 (FDA approved).
     ~14K drugs — equivalent to DrugBank's openly-redistributable subset.
     This feeds the broadened Channel A library.

Usage:
    python scripts/extract_chembl_all_for_pretrain.py \\
        --db /root/chembl_local/chembl_36.db \\
        --out rasyn/data/clean
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-len", type=int, default=5)
    p.add_argument("--max-len", type=int, default=200)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    # ---- 1. All molecules ----
    _log("Extracting all canonical_smiles from compound_structures...")
    sql_all = """
        SELECT DISTINCT cs.canonical_smiles
        FROM compound_structures cs
        WHERE cs.canonical_smiles IS NOT NULL
    """
    df_all = pd.read_sql_query(sql_all, conn)
    n_raw = len(df_all)
    _log(f"  raw distinct rows: {n_raw:,}")

    # Length filter (drop fragments + huge polymers)
    sl = df_all["canonical_smiles"].astype(str).str.len()
    keep = (sl >= args.min_len) & (sl <= args.max_len)
    df_all = df_all[keep].reset_index(drop=True)
    _log(f"  after length filter [{args.min_len}, {args.max_len}]: {len(df_all):,}")

    df_all.to_parquet(args.out / "chembl_all_smiles.parquet", compression="zstd", index=False)
    _log(f"  -> {args.out / 'chembl_all_smiles.parquet'}")

    # ---- 2. Approved drugs subset ----
    _log("Extracting FDA-approved subset (max_phase >= 4)...")
    sql_drugs = """
        SELECT DISTINCT cs.canonical_smiles, md.chembl_id, md.pref_name,
                        md.max_phase
        FROM molecule_dictionary md
        JOIN compound_structures cs ON cs.molregno = md.molregno
        WHERE md.max_phase >= 4
          AND cs.canonical_smiles IS NOT NULL
    """
    df_drugs = pd.read_sql_query(sql_drugs, conn)
    _log(f"  approved drugs rows: {len(df_drugs):,}")
    df_drugs["source"] = "chembl_approved_drugs"
    df_drugs.to_parquet(args.out / "chembl_approved_drugs.parquet", compression="zstd", index=False)
    _log(f"  -> {args.out / 'chembl_approved_drugs.parquet'}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
