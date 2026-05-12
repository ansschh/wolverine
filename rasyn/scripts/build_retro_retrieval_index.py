"""Build the FAISS retrieval index for the retro retrieval proposer.

Loads the curated reactions parquet from R-1, computes 2048-bit Morgan
fingerprints of every product, packs them into a FAISS IndexFlatIP, and
pickles a parallel metadata list for the retrieval proposer.

Run on CPU (~10-30 min on the silver+bronze union, depending on size).
No GPU required.

Outputs:
    checkpoints/retro_retrieval_v1/index.faiss
    checkpoints/retro_retrieval_v1/metadata.pkl
    checkpoints/retro_retrieval_v1/fingerprints.npy (uint8, N x 2048)
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("build_retrieval_index")


def _load_reactions(parquets: list[Path]) -> list[dict]:
    import pyarrow.parquet as pq
    rows: list[dict] = []
    for p in parquets:
        if not p.exists():
            logger.warning("missing parquet: %s", p)
            continue
        tbl = pq.read_table(p)
        rows.extend(tbl.to_pylist())
    return rows


def _morgan_fp_bits(smi: str, n_bits: int = 2048) -> np.ndarray | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.DataStructs import ConvertToNumpyArray
    except ImportError:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    ConvertToNumpyArray(bv, arr)
    return arr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reactions", nargs="+", type=Path,
                   default=[Path("rasyn/data/clean/retro/reactions_bronze.parquet"),
                            Path("rasyn/data/clean/retro/reactions_silver.parquet")])
    p.add_argument("--out", type=Path, default=Path("checkpoints/retro_retrieval_v1"))
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--max-rows", type=int, default=None, help="cap rows for smoke")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    logger.info("loading parquets %s", [str(p) for p in args.reactions])
    rows = _load_reactions(args.reactions)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    logger.info("loaded %d reaction rows", len(rows))

    fps = np.zeros((len(rows), args.n_bits), dtype=np.uint8)
    metadata: list[dict] = []
    n_skipped = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        prod = row.get("product_smiles") or row.get("product")
        fp = _morgan_fp_bits(prod, args.n_bits) if prod else None
        if fp is None:
            n_skipped += 1
            continue
        fps[i] = fp
        metadata.append({
            "row_index": i,
            "product_smiles": prod,
            "product_inchi_key": row.get("product_inchi_key"),
            "reactant_smiles": row.get("reactant_smiles") or [],
            "reaction_class": row.get("reaction_class") or "unclassified",
            "source": row.get("source"),
            "source_id": row.get("source_record_id"),
            "reaction_id": row.get("source_record_id"),
        })
        if (i + 1) % 50000 == 0:
            logger.info("  fp %d/%d (%.0f rows/s)", i + 1, len(rows),
                        (i + 1) / max(time.time() - t0, 1e-9))
    logger.info("computed %d fingerprints (%d skipped)", len(metadata), n_skipped)

    # Filter to non-skipped only
    keep_idx = [m["row_index"] for m in metadata]
    fps_kept = fps[keep_idx]

    logger.info("building FAISS IndexFlatIP")
    try:
        import faiss
        index = faiss.IndexFlatIP(args.n_bits)
        index.add(fps_kept.astype(np.float32))
        faiss.write_index(index, str(args.out / "index.faiss"))
        logger.info("wrote %s", args.out / "index.faiss")
    except ImportError:
        logger.warning("faiss not installed; skipping index file (fallback brute-force in proposer)")

    np.save(args.out / "fingerprints.npy", fps_kept)
    with open(args.out / "metadata.pkl", "wb") as fh:
        pickle.dump(metadata, fh)
    logger.info("wrote %s + %s", args.out / "fingerprints.npy", args.out / "metadata.pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
