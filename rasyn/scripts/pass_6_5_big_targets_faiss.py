"""Pass 6.5: supplementary big-target analog edges via FAISS top-K NN.

Pass 6 in `build_rescue_pair_dataset.py` capped per-target molecule count
at [5, 500] to avoid combinatorial blowup. That excluded high-data targets
(hERG, CYPs, kinases) entirely. Pass 6.5 brings them back via per-molecule
top-K nearest neighbors using FAISS binary index over Morgan fingerprints.

Approach:
  1. Read assay_facts.parquet, find targets with >500 unique molecules.
  2. For each big target:
     a. Compute Morgan fingerprints (1024-bit) for all its molecules.
     b. Build FAISS IndexBinaryFlat (exact Hamming distance).
     c. For each molecule, query top-K (default 20) nearest neighbors.
     d. Convert Hamming -> Tanimoto, keep edges with Tanimoto >= 0.5.
  3. Murcko match + heavy_atom_diff for the new edges.
  4. APPEND to analog_edges.parquet.
  5. Re-run downstream Passes 7-13 on the augmented edge set.

Output: analog_edges.parquet (augmented), then re-runs of Pass 7-13 will
overwrite rescue_pair_candidates.parquet + candidate_sets.parquet with
big-target-included data.

Run:
    cd ~/wolverine/rasyn
    python scripts/pass_6_5_big_targets_faiss.py
    # then re-run downstream:
    python scripts/build_rescue_pair_dataset.py --passes 7,8,9,10,11,12,13
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from rasyn.utils.similarity import morgan_bits, murcko_match

DATA_DIR = Path("rasyn/data/clean")
ASSAY_FACTS = DATA_DIR / "assay_facts.parquet"
MOLECULES = DATA_DIR / "molecules_canonical.parquet"
ANALOG_EDGES = DATA_DIR / "analog_edges.parquet"


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _morgan_to_uint8(smi: str, n_bits: int = 1024) -> np.ndarray | None:
    """Pack 1024-bit Morgan FP into 128-byte uint8 array for FAISS binary index."""
    fp = morgan_bits(smi, n_bits=n_bits)
    if fp is None:
        return None
    arr = np.zeros((n_bits,), dtype=np.uint8)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    # Pack into bytes (FAISS binary index expects packed bytes)
    return np.packbits(arr).astype(np.uint8)


def _tanimoto_from_hamming(fp_a: np.ndarray, fp_b: np.ndarray, n_bits: int = 1024) -> float:
    """Tanimoto from popcount + Hamming (binary fingerprints).

    For 1024-bit binary FPs A, B:
       |A AND B| = (popcount(A) + popcount(B) - Hamming) / 2
       |A OR B|  = (popcount(A) + popcount(B) + Hamming) / 2
       Tanimoto  = |A AND B| / |A OR B|
                = (pa + pb - h) / (pa + pb + h)
    """
    # Unpack to bits to count popcounts
    a_bits = np.unpackbits(fp_a)
    b_bits = np.unpackbits(fp_b)
    pa = int(a_bits.sum())
    pb = int(b_bits.sum())
    hamming = int(np.bitwise_xor(a_bits, b_bits).sum())
    denom = pa + pb + hamming
    if denom == 0:
        return 0.0
    return (pa + pb - hamming) / denom


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-molecules", type=int, default=501,
                   help="Only process targets with >= this many molecules (those excluded by Pass 6).")
    p.add_argument("--top-k", type=int, default=20, help="Top-K nearest neighbors per molecule.")
    p.add_argument("--min-tanimoto", type=float, default=0.5)
    p.add_argument("--out-edges", type=Path, default=ANALOG_EDGES)
    args = p.parse_args()

    if not ASSAY_FACTS.exists():
        raise SystemExit(f"FATAL: {ASSAY_FACTS} not found. Run Pass 1 first.")
    if not MOLECULES.exists():
        raise SystemExit(f"FATAL: {MOLECULES} not found.")

    try:
        import faiss
    except ImportError:
        raise SystemExit("FATAL: faiss-cpu not installed. `pip install faiss-cpu`")

    _log(f"Loading assay_facts + molecules...")
    facts = pd.read_parquet(ASSAY_FACTS, columns=["molecule_chembl_id", "target_chembl_id"])
    facts = facts.dropna(subset=["molecule_chembl_id", "target_chembl_id"])
    mols_df = pd.read_parquet(MOLECULES)
    smiles_by_id = dict(zip(mols_df["chembl_id"], mols_df["canonical_smiles"]))

    target_to_molecules: dict[str, set[str]] = defaultdict(set)
    for mid, tid in zip(facts["molecule_chembl_id"], facts["target_chembl_id"]):
        target_to_molecules[tid].add(mid)

    big_targets = {t: ms for t, ms in target_to_molecules.items() if len(ms) >= args.min_molecules}
    _log(f"Found {len(big_targets):,} big targets (>= {args.min_molecules} molecules each)")

    # Sort by size for predictable progress
    big_targets_sorted = sorted(big_targets.items(), key=lambda kv: len(kv[1]), reverse=True)
    for tid, ms in big_targets_sorted[:10]:
        _log(f"  {tid}: {len(ms):,} molecules")

    new_edges: list[dict] = []
    t0 = time.time()
    for ti, (tid, mol_ids) in enumerate(big_targets_sorted):
        ids_with_smi = [mid for mid in mol_ids if mid in smiles_by_id]
        if len(ids_with_smi) < 2:
            continue

        _log(f"[{ti + 1}/{len(big_targets_sorted)}] target {tid} ({len(ids_with_smi):,} mols) ...")

        # Compute Morgan FPs as packed uint8 (128 bytes per FP for 1024 bits)
        fps_packed: list[np.ndarray] = []
        valid_ids: list[str] = []
        for mid in ids_with_smi:
            fp = _morgan_to_uint8(smiles_by_id[mid])
            if fp is None:
                continue
            fps_packed.append(fp)
            valid_ids.append(mid)
        if len(valid_ids) < 2:
            continue

        # FAISS binary index over packed FPs
        fps_arr = np.stack(fps_packed)  # (N, 128) uint8
        index = faiss.IndexBinaryFlat(1024)  # 1024-bit binary index
        index.add(fps_arr)

        # Top-K query (K+1 because each molecule's nearest neighbor is itself)
        k_query = min(args.top_k + 1, len(valid_ids))
        D, I = index.search(fps_arr, k_query)  # D = Hamming distance, I = neighbor indices

        # Convert each (i, j) pair to edge with Tanimoto
        target_edges_count = 0
        for i, (dists, neighbors) in enumerate(zip(D, I)):
            for d, j in zip(dists, neighbors):
                if j == -1 or j == i:  # skip self / invalid
                    continue
                if i >= j:  # de-dup symmetric pairs (a-b same as b-a)
                    continue
                # Compute exact Tanimoto from packed FPs
                tan = _tanimoto_from_hamming(fps_arr[i], fps_arr[j])
                if tan < args.min_tanimoto:
                    continue
                new_edges.append({
                    "parent_chembl_id": valid_ids[i],
                    "candidate_chembl_id": valid_ids[j],
                    "ecfp_tanimoto": float(tan),
                    "shared_target_chembl_id": tid,
                })
                target_edges_count += 1
        _log(f"    -> {target_edges_count:,} edges (cumulative: {len(new_edges):,})")

    if not new_edges:
        _log("No new edges from big targets; nothing to append.")
        return

    new_df = pd.DataFrame(new_edges)
    new_df = new_df.drop_duplicates(["parent_chembl_id", "candidate_chembl_id", "shared_target_chembl_id"])

    # Murcko match + heavy_atom_diff
    _log(f"Computing Murcko + heavy_atom_diff for {len(new_df):,} new edges...")
    from rdkit import Chem
    murcko_results = []
    heavy_diffs = []
    for _, r in new_df.iterrows():
        a = smiles_by_id.get(r["parent_chembl_id"])
        b = smiles_by_id.get(r["candidate_chembl_id"])
        if a is None or b is None:
            murcko_results.append(False)
            heavy_diffs.append(None)
            continue
        try:
            ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
            heavy_diffs.append(abs(ma.GetNumHeavyAtoms() - mb.GetNumHeavyAtoms()))
            murcko_results.append(murcko_match(a, b))
        except Exception:
            murcko_results.append(False)
            heavy_diffs.append(None)
    new_df["murcko_match"] = murcko_results
    new_df["heavy_atom_diff"] = heavy_diffs

    # Append to existing analog_edges.parquet (or create if absent)
    if args.out_edges.exists():
        existing = pd.read_parquet(args.out_edges)
        _log(f"Existing analog_edges.parquet: {len(existing):,} rows")
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(["parent_chembl_id", "candidate_chembl_id", "shared_target_chembl_id"])
    else:
        combined = new_df

    combined.to_parquet(args.out_edges, compression="zstd", index=False)
    _log(f"Pass 6.5 DONE: appended {len(new_df):,} new edges; total now {len(combined):,} -> {args.out_edges}")
    _log(f"Wall-clock: {(time.time() - t0) / 60:.1f} min")
    _log("Next: re-run downstream passes:")
    _log("  python scripts/build_rescue_pair_dataset.py --passes 7,8,9,10,11,12,13")


if __name__ == "__main__":
    main()
