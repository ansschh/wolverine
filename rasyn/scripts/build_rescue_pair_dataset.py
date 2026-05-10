"""Phase A-4 orchestrator: build the 4-table rescue-pair dataset.

Implements the 11 working passes from PLAN.md §5 / `rasyn_curating_the_dataset.md`.
Pass 4 (papers) deferred per L16. Pass 5 (internal) skipped per L2.

Outputs (rasyn/data/clean/):
    molecules_canonical.parquet  (already exists)
    assay_facts.parquet          (Passes 1+2+3)
    analog_edges.parquet         (Pass 6)
    rescue_pair_candidates.parquet (Passes 7-13)
    candidate_sets.parquet       (Pass 11)
    dataset_manifest.json        (frozen + hashed)
    decontam_audit_post.json     (canary audit on rescue pairs)

Run:
    cd ~/wolverine/rasyn
    python scripts/build_rescue_pair_dataset.py --all

Or run individual passes:
    python scripts/build_rescue_pair_dataset.py --passes 1,2,3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sqlite3
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rasyn.data.decontam.canary_audit import audit_against_rows
from rasyn.data.decontam.quarantine import build_forbidden_index, scrub_rows
from rasyn.data.registry.canary_generator import generate_canaries_for_registry
from rasyn.data.registry.loader import load_sealed_case_registry

# ---------- paths ----------

DATA_DIR = Path("rasyn/data")
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"

CHEMBL_TAR = RAW_DIR / "chembl/chembl_35_sqlite.tar.gz"
CHEMBL_EXTRACTED_DIR = RAW_DIR / "chembl/extracted"

MOLECULES_PARQUET = CLEAN_DIR / "molecules_canonical.parquet"
ASSAY_FACTS_PARQUET = CLEAN_DIR / "assay_facts.parquet"
ANALOG_EDGES_PARQUET = CLEAN_DIR / "analog_edges.parquet"
RESCUE_PAIRS_PARQUET = CLEAN_DIR / "rescue_pair_candidates.parquet"
CANDIDATE_SETS_PARQUET = CLEAN_DIR / "candidate_sets.parquet"
DATASET_MANIFEST = CLEAN_DIR / "dataset_manifest.json"
DECONTAM_POST = CLEAN_DIR / "decontam_audit_post.json"

# ---------- helpers ----------


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_chembl_extracted() -> Path:
    """Extract ChEMBL bulk SQLite if not already extracted. Returns path to .db file."""
    if not CHEMBL_TAR.exists():
        raise SystemExit(f"FATAL: {CHEMBL_TAR} not found. Run wget first.")
    CHEMBL_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    db_path = next(CHEMBL_EXTRACTED_DIR.rglob("*.db"), None)
    if db_path is not None:
        return db_path
    _log(f"Extracting {CHEMBL_TAR} ...")
    with tarfile.open(CHEMBL_TAR, "r:gz") as tar:
        tar.extractall(path=CHEMBL_EXTRACTED_DIR)
    db_path = next(CHEMBL_EXTRACTED_DIR.rglob("*.db"), None)
    if db_path is None:
        raise SystemExit("No .db file in extracted ChEMBL")
    _log(f"Extracted ChEMBL to {db_path}")
    return db_path


# liability_type lookup from ChEMBL standard_type values
CHEMBL_STDTYPE_TO_LIABILITY = {
    # hERG
    "IC50": None,  # generic; assigned per target
    "hERG IC50": "hERG",
    "Inhibition": None,
    # solubility
    "Solubility": "solubility",
    "logS": "solubility",
    "Aqueous solubility": "solubility",
    # metabolic stability
    "Cl": "metabolic_stability",
    "CL": "metabolic_stability",
    "Half life": "metabolic_stability",
    "Half-life": "metabolic_stability",
    "T1/2": "metabolic_stability",
    "Stability": "metabolic_stability",
    # oral exposure
    "Bioavailability": "oral_exposure",
    "F": "oral_exposure",
    "AUC": "oral_exposure",
    # permeability
    "Caco-2": "permeability",
    "PAMPA": "permeability",
    "Papp": "permeability",
}

# Targets known to be hERG (ChEMBL2407 = human hERG / KCNH2)
HERG_TARGET_IDS = {"CHEMBL240"}  # ChEMBL hERG (KCNH2)

# Standard types treated as target-binding potency (vs. ADMET liability).
# Per spec §5: pChEMBL is computed for IC50/EC50/Ki/Kd/Potency under specific
# validity conditions. Restrict activity index to these so we don't conflate
# binding potency with ADMET measurements that share the IC50 standard_type.
BINDING_STDTYPES = {"IC50", "Ki", "EC50", "Kd", "AC50", "Potency"}


def _liability_for_row(target_chembl_id: str | None, standard_type: str | None) -> str | None:
    """Map a ChEMBL activity row to one of the canonical liability_type values."""
    if standard_type is None:
        return None
    # hERG-specific by target
    if target_chembl_id in HERG_TARGET_IDS:
        return "hERG"
    return CHEMBL_STDTYPE_TO_LIABILITY.get(standard_type)


# ---------- Pass 1: ChEMBL activity contexts ----------


def pass_1_chembl_activities(*, batch_size: int = 50_000) -> Path:
    """Stream ChEMBL bulk SQLite activities -> partial assay_facts.parquet.

    Filters: standard_relation = '=', standard_value present, has canonical SMILES.
    Adds: liability_type (None if not in our taxonomy).
    """
    _log("Pass 1: ChEMBL activities -> assay_facts (chembl rows)")
    db_path = _ensure_chembl_extracted()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sql = """
        SELECT
            md.chembl_id          AS molecule_chembl_id,
            cs.canonical_smiles   AS canonical_smiles,
            cs.standard_inchi_key AS inchi_key,
            t.chembl_id           AS target_chembl_id,
            t.pref_name           AS target_pref_name,
            a.chembl_id           AS assay_chembl_id,
            a.assay_type          AS assay_type,
            d.chembl_id           AS document_chembl_id,
            d.doi                 AS doi,
            act.standard_type     AS standard_type,
            act.standard_relation AS standard_relation,
            act.standard_value    AS standard_value,
            act.standard_units    AS standard_units,
            act.pchembl_value     AS pchembl_value,
            act.data_validity_comment AS data_validity_comment
        FROM activities act
        JOIN assays a       ON a.assay_id = act.assay_id
        JOIN target_dictionary t ON t.tid = a.tid
        JOIN molecule_dictionary md ON md.molregno = act.molregno
        JOIN compound_structures cs ON cs.molregno = md.molregno
        JOIN docs d ON d.doc_id = act.doc_id
        WHERE act.standard_value IS NOT NULL
          AND cs.canonical_smiles IS NOT NULL
          AND act.standard_relation = '='
    """
    cur = conn.cursor()
    cur.execute(sql)

    # Stream-write to parquet using PyArrow row-group writer
    schema = pa.schema(
        [
            ("molecule_chembl_id", pa.string()),
            ("canonical_smiles", pa.string()),
            ("inchi_key", pa.string()),
            ("target_chembl_id", pa.string()),
            ("target_pref_name", pa.string()),
            ("assay_chembl_id", pa.string()),
            ("assay_type", pa.string()),
            ("document_chembl_id", pa.string()),
            ("doi", pa.string()),
            ("standard_type", pa.string()),
            ("standard_relation", pa.string()),
            ("standard_value", pa.float64()),
            ("standard_units", pa.string()),
            ("pchembl_value", pa.float64()),
            ("liability_type", pa.string()),
            ("source", pa.string()),
            ("data_validity_comment", pa.string()),
        ]
    )
    out = ASSAY_FACTS_PARQUET
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out, schema, compression="zstd")

    rows_total = 0
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        cols = [c[0] for c in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        df["liability_type"] = df.apply(
            lambda r: _liability_for_row(r["target_chembl_id"], r["standard_type"]), axis=1,
        )
        df["source"] = "chembl"
        df = df[
            [
                "molecule_chembl_id", "canonical_smiles", "inchi_key",
                "target_chembl_id", "target_pref_name",
                "assay_chembl_id", "assay_type",
                "document_chembl_id", "doi",
                "standard_type", "standard_relation", "standard_value", "standard_units",
                "pchembl_value", "liability_type", "source", "data_validity_comment",
            ]
        ]
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        writer.write_table(table)
        rows_total += len(df)
        if rows_total % 500_000 == 0:
            _log(f"  Pass 1: streamed {rows_total:,} rows")

    writer.close()
    conn.close()
    _log(f"Pass 1 DONE: {rows_total:,} ChEMBL activity rows -> {out}")
    return out


# ---------- Pass 2: PubChem ADMET/toxicity facts ----------


def pass_2_pubchem_facts() -> None:
    """Append PubChem AID-derived liability rows to assay_facts.parquet.

    Uses LIABILITY_AID_HINTS (hERG, solubility, metabolic_stability AIDs).
    Each AID is downloaded as CSV via PUG REST and joined into assay_facts.
    """
    _log("Pass 2: PubChem ADMET/toxicity facts -> assay_facts (pubchem rows)")
    from rasyn.data.sources.pubchem import LIABILITY_AID_HINTS

    new_rows: list[dict] = []
    for liability, aids in LIABILITY_AID_HINTS.items():
        for aid in aids:
            _log(f"  Pulling AID {aid} ({liability})")
            try:
                # PUG REST CSV endpoint
                from urllib.request import urlopen
                url = f"https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi?query=download&record_type=summary&actvty=any&aid={aid}&response_type=display"
                # Simpler: use the BioAssay summary endpoint
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/concise/CSV"
                with urlopen(url, timeout=60) as resp:
                    csv_text = resp.read().decode("utf-8", errors="replace")
                df = pd.read_csv(pd.io.common.StringIO(csv_text))
                # PubChem schemas vary by AID; minimal viable subset:
                # Expect at least: PUBCHEM_CID, Activity_Outcome, Activity_Score (optional)
                for _, row in df.iterrows():
                    cid = str(row.get("PUBCHEM_CID", "")) if "PUBCHEM_CID" in df.columns else None
                    outcome = row.get("PUBCHEM_ACTIVITY_OUTCOME") if "PUBCHEM_ACTIVITY_OUTCOME" in df.columns else None
                    if not cid or pd.isna(cid):
                        continue
                    new_rows.append({
                        "molecule_chembl_id": None,
                        "canonical_smiles": None,  # require lookup separately
                        "inchi_key": None,
                        "target_chembl_id": None,
                        "target_pref_name": None,
                        "assay_chembl_id": f"PUBCHEM_AID_{aid}",
                        "assay_type": "B",
                        "document_chembl_id": None,
                        "doi": None,
                        "standard_type": "Activity_Outcome",
                        "standard_relation": "=",
                        "standard_value": (1.0 if outcome == "Active" else 0.0 if outcome == "Inactive" else None),
                        "standard_units": "binary",
                        "pchembl_value": None,
                        "liability_type": liability,
                        "source": "pubchem",
                        "data_validity_comment": None,
                        "pubchem_cid": cid,
                    })
            except Exception as e:
                _log(f"  PubChem AID {aid} failed: {e}")
                continue
    if not new_rows:
        _log("Pass 2: no new PubChem rows; skipping append")
        return
    df = pd.DataFrame(new_rows).drop(columns=["pubchem_cid"], errors="ignore")
    existing = pd.read_parquet(ASSAY_FACTS_PARQUET) if ASSAY_FACTS_PARQUET.exists() else pd.DataFrame()
    combined = pd.concat([existing, df], ignore_index=True, sort=False)
    combined.to_parquet(ASSAY_FACTS_PARQUET, compression="zstd", index=False)
    _log(f"Pass 2 DONE: appended {len(df)} PubChem rows; total {len(combined):,}")


# ---------- Pass 3: TDC + MoleculeNet ----------


TDC_TO_LIABILITY = {
    "hERG": "hERG",
    "hERG_Karim": "hERG",
    "Solubility_AqSolDB": "solubility",
    "HydrationFreeEnergy_FreeSolv": "solubility",
    "Lipophilicity_AstraZeneca": "solubility",
    "Half_Life_Obach": "metabolic_stability",
    "Clearance_Hepatocyte_AZ": "metabolic_stability",
    "Clearance_Microsome_AZ": "metabolic_stability",
    "Bioavailability_Ma": "oral_exposure",
    "HIA_Hou": "oral_exposure",
    "Caco2_Wang": "permeability",
    "PAMPA_NCATS": "permeability",
}


def pass_3_tdc_molnet_facts() -> None:
    """Append TDC + MoleculeNet rows to assay_facts.parquet.

    TDC datasets are pulled (cached if previously downloaded). Each row is
    converted to a (molecule, liability_type, value) record.
    """
    _log("Pass 3: TDC + MolNet -> assay_facts (tdc/molnet rows)")
    new_rows: list[dict] = []

    try:
        from tdc.single_pred import ADME, Tox
    except ImportError:
        _log("  TDC not installed; skipping Pass 3")
        return

    for ds_name, liability in TDC_TO_LIABILITY.items():
        try:
            ds_cls = ADME if ds_name in {
                "Solubility_AqSolDB", "HydrationFreeEnergy_FreeSolv", "Lipophilicity_AstraZeneca",
                "Half_Life_Obach", "Clearance_Hepatocyte_AZ", "Clearance_Microsome_AZ",
                "Bioavailability_Ma", "HIA_Hou", "Caco2_Wang", "PAMPA_NCATS",
            } else Tox
            ds = ds_cls(name=ds_name)
            df = ds.get_data()
            _log(f"  {ds_name}: {len(df)} rows ({liability})")
        except Exception as e:
            _log(f"  {ds_name} failed: {e}")
            continue

        smi_col = "Drug" if "Drug" in df.columns else df.columns[1]
        y_col = "Y" if "Y" in df.columns else df.columns[-1]
        for _, r in df.iterrows():
            try:
                smi = str(r[smi_col]).strip()
                y = float(r[y_col])
            except Exception:
                continue
            new_rows.append({
                "molecule_chembl_id": None,
                "canonical_smiles": smi,
                "inchi_key": None,
                "target_chembl_id": None,
                "target_pref_name": None,
                "assay_chembl_id": f"TDC_{ds_name}",
                "assay_type": "B",
                "document_chembl_id": None,
                "doi": None,
                "standard_type": ds_name,
                "standard_relation": "=",
                "standard_value": y,
                "standard_units": None,
                "pchembl_value": None,
                "liability_type": liability,
                "source": f"tdc:{ds_name}",
                "data_validity_comment": None,
            })

    if not new_rows:
        _log("Pass 3: no new TDC rows; skipping")
        return
    df = pd.DataFrame(new_rows)
    existing = pd.read_parquet(ASSAY_FACTS_PARQUET) if ASSAY_FACTS_PARQUET.exists() else pd.DataFrame()
    combined = pd.concat([existing, df], ignore_index=True, sort=False)
    combined.to_parquet(ASSAY_FACTS_PARQUET, compression="zstd", index=False)
    _log(f"Pass 3 DONE: appended {len(df)} TDC rows; total {len(combined):,}")


# ---------- Pass 6: Analog graph ----------


def _morgan_array(smi: str, n_bits: int = 1024):
    """Return Morgan FP as numpy uint8 array (1024 bits packed)."""
    fp = morgan_bits(smi, n_bits=n_bits)
    if fp is None:
        return None
    arr = np.zeros((n_bits,), dtype=np.uint8)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def pass_6_analog_graph(*, min_tanimoto: float = 0.5, max_pairs_per_target: int = 5000) -> Path:
    """Within-target analog graph: ECFP4 Tanimoto + Murcko match + heavy-atom diff.

    Constrains pair generation to molecules that share a target_chembl_id
    in assay_facts.parquet (avoids combinatorial blowup over 2.47M molecules).
    """
    _log(f"Pass 6: analog graph (Tanimoto >= {min_tanimoto}, within-target)")
    from rasyn.utils.similarity import morgan_bits, murcko_match, tanimoto

    if not ASSAY_FACTS_PARQUET.exists():
        raise SystemExit("Pass 6 requires assay_facts.parquet (run Pass 1 first)")
    if not MOLECULES_PARQUET.exists():
        raise SystemExit("Pass 6 requires molecules_canonical.parquet")

    # Build molecule lookup
    mols_df = pd.read_parquet(MOLECULES_PARQUET)
    smiles_by_id = dict(zip(mols_df["chembl_id"], mols_df["canonical_smiles"]))
    _log(f"  Loaded {len(smiles_by_id):,} canonical molecules")

    # Build per-target molecule sets from assay_facts
    facts = pd.read_parquet(ASSAY_FACTS_PARQUET, columns=["molecule_chembl_id", "target_chembl_id"])
    facts = facts.dropna(subset=["molecule_chembl_id", "target_chembl_id"])
    target_to_molecules: dict[str, set[str]] = defaultdict(set)
    for mid, tid in zip(facts["molecule_chembl_id"], facts["target_chembl_id"]):
        target_to_molecules[tid].add(mid)
    _log(f"  {len(target_to_molecules):,} targets with at least one molecule")

    # Filter targets to those with 5..500 molecules (avoid too small or too big)
    targets_keep = {t: ms for t, ms in target_to_molecules.items() if 5 <= len(ms) <= 500}
    _log(f"  {len(targets_keep):,} targets in [5, 500] molecule range")

    edges_rows: list[dict] = []
    target_count = 0
    for tid, mol_ids in targets_keep.items():
        target_count += 1
        if target_count % 200 == 0:
            _log(f"    progress: {target_count}/{len(targets_keep)} targets, {len(edges_rows):,} edges")
        ids_with_smi = [mid for mid in mol_ids if mid in smiles_by_id]
        if len(ids_with_smi) < 2:
            continue
        # Compute fingerprints for this target's molecules
        fps = {}
        for mid in ids_with_smi:
            fp = morgan_bits(smiles_by_id[mid])
            if fp is not None:
                fps[mid] = fp
        ids = list(fps.keys())
        # Pairwise Tanimoto
        n_pairs_in_target = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sim = tanimoto(fps[ids[i]], fps[ids[j]])
                if sim < min_tanimoto:
                    continue
                a, b = ids[i], ids[j]
                edges_rows.append({
                    "parent_chembl_id": a,
                    "candidate_chembl_id": b,
                    "ecfp_tanimoto": float(sim),
                    "shared_target_chembl_id": tid,
                })
                n_pairs_in_target += 1
                if n_pairs_in_target >= max_pairs_per_target:
                    break
            if n_pairs_in_target >= max_pairs_per_target:
                break

    if not edges_rows:
        raise SystemExit("Pass 6: no edges produced (check assay_facts and Tanimoto threshold)")

    df = pd.DataFrame(edges_rows)
    # De-duplicate by (parent, candidate, target) — same pair across targets stays separate
    df = df.drop_duplicates(["parent_chembl_id", "candidate_chembl_id", "shared_target_chembl_id"])

    # Add Murcko match + heavy-atom diff
    _log(f"  Computing Murcko match + heavy_atom_diff for {len(df):,} edges...")
    murcko_results = []
    heavy_diffs = []
    for _, r in df.iterrows():
        a = smiles_by_id.get(r["parent_chembl_id"])
        b = smiles_by_id.get(r["candidate_chembl_id"])
        if a is None or b is None:
            murcko_results.append(False)
            heavy_diffs.append(None)
            continue
        try:
            from rdkit import Chem
            ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
            heavy_diffs.append(abs(ma.GetNumHeavyAtoms() - mb.GetNumHeavyAtoms()))
            murcko_results.append(murcko_match(a, b))
        except Exception:
            murcko_results.append(False)
            heavy_diffs.append(None)
    df["murcko_match"] = murcko_results
    df["heavy_atom_diff"] = heavy_diffs

    df.to_parquet(ANALOG_EDGES_PARQUET, compression="zstd", index=False)
    _log(f"Pass 6 DONE: {len(df):,} analog edges -> {ANALOG_EDGES_PARQUET}")
    return ANALOG_EDGES_PARQUET


# ---------- Pass 7: Pair generation (dual-lookup per spec §5-7 + Table 3) ----------


def pass_7_pair_generation() -> Path:
    """Generate parent-candidate rescue-pair candidates via dual-lookup.

    Per spec §5-7 + Table 3, each pair carries TWO independent fact references:
      - activity_evidence: median pchembl at shared binding target. Restricted
        to BINDING_STDTYPES with no liability_type tag.
      - liability_evidence: per-molecule median ADMET value keyed by
        (liability_type, standard_type) so direction is comparable. Pair rows
        are emitted for each (liability_type, standard_type) common between
        parent + candidate.

    Earlier single-row implementation conflated activity and liability into one
    assay_facts row, causing 99.84% of Pass 9 improvement labels to be unknown
    and 100% of Pass 10 hard-negs to be None.

    Pair row schema:
        pair_id, parent_chembl_id, candidate_chembl_id, parent_smiles,
        candidate_smiles, target_chembl_id, liability_type, liability_endpoint,
        parent_activity_pchembl, candidate_activity_pchembl,
        parent_liability_value, candidate_liability_value,
        parent_liability_n_measurements, candidate_liability_n_measurements,
        ecfp_tanimoto, murcko_match, heavy_atom_diff
    """
    _log("Pass 7: pair generation (dual-lookup)")
    if not ANALOG_EDGES_PARQUET.exists():
        raise SystemExit("Pass 7 requires analog_edges.parquet (run Pass 6 / 6.5)")
    edges = pd.read_parquet(ANALOG_EDGES_PARQUET)
    facts = pd.read_parquet(ASSAY_FACTS_PARQUET)
    mols_df = pd.read_parquet(MOLECULES_PARQUET)
    smiles_by_id = dict(zip(mols_df["chembl_id"], mols_df["canonical_smiles"]))
    _log(f"  Loaded {len(edges):,} edges, {len(facts):,} facts, {len(smiles_by_id):,} canonical mols")

    # ----- Index 1: activity (binding potency) per (mol, target) -----
    act_facts = facts[
        facts["pchembl_value"].notna()
        & facts["standard_type"].isin(BINDING_STDTYPES)
        & facts["liability_type"].isna()  # exclude ADMET-tagged rows
        & facts["molecule_chembl_id"].notna()
        & facts["target_chembl_id"].notna()
    ]
    act_idx = (
        act_facts.groupby(["molecule_chembl_id", "target_chembl_id"], as_index=False)
        .agg(activity_pchembl=("pchembl_value", "median"))
    )
    activity_lookup: dict[tuple[str, str], float] = {}
    for r in act_idx.itertuples(index=False):
        activity_lookup[(r.molecule_chembl_id, r.target_chembl_id)] = float(r.activity_pchembl)
    _log(f"  Activity index: {len(activity_lookup):,} (mol, target) potency entries")

    # ----- Index 2: liability per molecule, keyed by (liability_type, standard_type) -----
    liab_facts = facts[
        facts["liability_type"].notna()
        & facts["standard_value"].notna()
        & facts["molecule_chembl_id"].notna()
    ]
    liab_idx = (
        liab_facts.groupby(
            ["molecule_chembl_id", "liability_type", "standard_type"], as_index=False
        ).agg(
            liability_value=("standard_value", "median"),
            n_measurements=("standard_value", "count"),
        )
    )
    liability_lookup: dict[str, dict[tuple[str, str], tuple[float, int]]] = {}
    for r in liab_idx.itertuples(index=False):
        liability_lookup.setdefault(r.molecule_chembl_id, {})[
            (r.liability_type, r.standard_type)
        ] = (float(r.liability_value), int(r.n_measurements))
    _log(f"  Liability index: {len(liability_lookup):,} mols with at least one ADMET fact")

    # ----- Generate pair rows -----
    pair_rows: list[dict] = []
    pid_seen: set[str] = set()
    n_with_common = 0
    n_no_common = 0
    n_skipped_no_signal = 0

    for e in edges.itertuples(index=False):
        a, b, tid = e.parent_chembl_id, e.candidate_chembl_id, e.shared_target_chembl_id
        a_pchembl = activity_lookup.get((a, tid))
        b_pchembl = activity_lookup.get((b, tid))
        a_liabs = liability_lookup.get(a, {})
        b_liabs = liability_lookup.get(b, {})
        common_liabs = set(a_liabs.keys()) & set(b_liabs.keys())

        has_potency_pair = a_pchembl is not None and b_pchembl is not None

        if common_liabs:
            n_with_common += 1
        else:
            n_no_common += 1
            if not has_potency_pair:
                # Neither activity-retention nor liability-improvement signal
                # is computable for this edge — skip per L33 (don't pad).
                n_skipped_no_signal += 1
                continue

        smi_a = smiles_by_id.get(a)
        smi_b = smiles_by_id.get(b)
        ecfp = float(e.ecfp_tanimoto)
        murcko = bool(e.murcko_match)
        heavy_diff = int(e.heavy_atom_diff) if pd.notna(e.heavy_atom_diff) else None

        emit_keys: list[tuple[str, str] | None] = (
            list(common_liabs) if common_liabs else [None]
        )

        for liab_key in emit_keys:
            liab_type = liab_key[0] if liab_key else None
            liab_endpoint = liab_key[1] if liab_key else None
            for parent_id, cand_id, p_smi, c_smi in (
                (a, b, smi_a, smi_b),
                (b, a, smi_b, smi_a),
            ):
                pid = f"{parent_id}_{cand_id}_{tid}_{liab_type or 'NA'}_{liab_endpoint or 'NA'}"
                if pid in pid_seen:
                    continue
                pid_seen.add(pid)

                if liab_key:
                    p_liab = liability_lookup.get(parent_id, {}).get(liab_key, (None, 0))
                    c_liab = liability_lookup.get(cand_id, {}).get(liab_key, (None, 0))
                else:
                    p_liab = (None, 0)
                    c_liab = (None, 0)

                pair_rows.append({
                    "pair_id": pid,
                    "parent_chembl_id": parent_id,
                    "candidate_chembl_id": cand_id,
                    "parent_smiles": p_smi,
                    "candidate_smiles": c_smi,
                    "target_chembl_id": tid,
                    "liability_type": liab_type,
                    "liability_endpoint": liab_endpoint,
                    "parent_activity_pchembl": activity_lookup.get((parent_id, tid)),
                    "candidate_activity_pchembl": activity_lookup.get((cand_id, tid)),
                    "parent_liability_value": p_liab[0],
                    "candidate_liability_value": c_liab[0],
                    "parent_liability_n_measurements": p_liab[1],
                    "candidate_liability_n_measurements": c_liab[1],
                    "ecfp_tanimoto": ecfp,
                    "murcko_match": murcko,
                    "heavy_atom_diff": heavy_diff,
                })

    df = pd.DataFrame(pair_rows)
    df.to_parquet(RESCUE_PAIRS_PARQUET, compression="zstd", index=False)
    _log(
        f"Pass 7 DONE: {len(df):,} pair rows -> {RESCUE_PAIRS_PARQUET} | "
        f"edges_w_common_liab={n_with_common:,} edges_no_common={n_no_common:,} "
        f"edges_skipped_no_signal={n_skipped_no_signal:,}"
    )
    return RESCUE_PAIRS_PARQUET


# ---------- Pass 8: Activity-retention bucketing ----------


def _retention_bucket(parent_pchembl: float | None, cand_pchembl: float | None) -> str:
    """Map activity delta to retention bucket per spec (3x/10x/100x folds)."""
    if parent_pchembl is None or cand_pchembl is None or pd.isna(parent_pchembl) or pd.isna(cand_pchembl):
        return "unknown"
    delta = parent_pchembl - cand_pchembl  # positive = candidate worse
    fold = 10 ** delta if delta >= 0 else 10 ** (-delta)
    # When delta>0 candidate is X-fold weaker than parent
    if delta <= 0.477:  # 3x
        return "strong"
    if delta <= 1.0:    # 10x
        return "acceptable"
    if delta <= 2.0:    # 100x
        return "weak"
    return "failed"


def pass_8_retention_buckets() -> None:
    _log("Pass 8: activity-retention bucketing")
    df = pd.read_parquet(RESCUE_PAIRS_PARQUET)
    df["activity_retention_bucket"] = [
        _retention_bucket(p, c)
        for p, c in zip(df["parent_activity_pchembl"], df["candidate_activity_pchembl"])
    ]
    df.to_parquet(RESCUE_PAIRS_PARQUET, compression="zstd", index=False)
    _log(f"Pass 8 DONE: bucket counts: {df['activity_retention_bucket'].value_counts().to_dict()}")


# ---------- Pass 9: Liability-improvement labeling ----------


def _improvement_category(parent_v: float | None, cand_v: float | None, liability: str | None) -> str:
    """Per-liability improvement category. None values -> 'unknown'."""
    if parent_v is None or cand_v is None or pd.isna(parent_v) or pd.isna(cand_v):
        return "unknown"
    if liability == "solubility":
        ratio = cand_v / parent_v if parent_v > 0 else 1
        if ratio >= 10: return "large"
        if ratio >= 5: return "moderate"
        if ratio >= 2: return "minor"
        if ratio >= 0.8: return "none"
        return "worse"
    if liability == "hERG":
        # Higher IC50 = better (less inhibition)
        ratio = cand_v / parent_v if parent_v > 0 else 1
        if ratio >= 10: return "large"
        if ratio >= 3: return "moderate"
        if ratio >= 1.5: return "minor"
        if ratio >= 0.7: return "none"
        return "worse"
    if liability in ("metabolic_stability", "permeability"):
        ratio = cand_v / parent_v if parent_v > 0 else 1
        if ratio >= 3: return "large"
        if ratio >= 1.5: return "moderate"
        if ratio >= 1.1: return "minor"
        if ratio >= 0.85: return "none"
        return "worse"
    if liability == "oral_exposure":
        ratio = cand_v / parent_v if parent_v > 0 else 1
        if ratio >= 3: return "large"
        if ratio >= 1.5: return "moderate"
        if ratio >= 1.1: return "minor"
        return "worse" if ratio < 0.8 else "none"
    return "unknown"


def pass_9_liability_labels() -> None:
    _log("Pass 9: liability-improvement category labeling")
    df = pd.read_parquet(RESCUE_PAIRS_PARQUET)
    # Use *_liability_value (per-mol median at common (liability_type, standard_type)),
    # NOT *_activity_value. Earlier code used activity_value which was the SAME row as
    # the potency lookup — conflated activity and liability.
    df["liability_improvement_category"] = [
        _improvement_category(p, c, lib)
        for p, c, lib in zip(
            df["parent_liability_value"],
            df["candidate_liability_value"],
            df["liability_type"],
        )
    ]
    df.to_parquet(RESCUE_PAIRS_PARQUET, compression="zstd", index=False)
    _log(f"Pass 9 DONE: improvement counts: {df['liability_improvement_category'].value_counts().to_dict()}")


# ---------- Pass 10: Hard-negative construction ----------


def pass_10_hard_negatives() -> None:
    """Label hard-negative type per spec §15. 5 types."""
    _log("Pass 10: hard-negative construction")
    df = pd.read_parquet(RESCUE_PAIRS_PARQUET)

    def _hn_type(row) -> str | None:
        ret = row["activity_retention_bucket"]
        imp = row["liability_improvement_category"]
        # Type 1: improved liability but lost activity
        if imp in ("large", "moderate") and ret == "failed":
            return "improved_but_activity_lost"
        # Type 2: retained activity but liability not fixed
        if ret in ("strong", "acceptable") and imp in ("none", "worse"):
            return "retained_but_liability_unfixed"
        # Type 4: new liability introduced (stub: detected via descriptors later)
        # Type 3 (wrong liability) and Type 5 (heuristic trap): require external context; flag at training time.
        return None

    df["hard_negative_type"] = df.apply(_hn_type, axis=1)
    df.to_parquet(RESCUE_PAIRS_PARQUET, compression="zstd", index=False)
    _log(f"Pass 10 DONE: hard-negative type counts: {df['hard_negative_type'].value_counts(dropna=False).to_dict()}")


# ---------- Pass 11: Local ranking-task assembly ----------


def pass_11_ranking_tasks() -> Path:
    """Group candidates per (parent, liability) into ranking tasks."""
    _log("Pass 11: local ranking-task assembly")
    df = pd.read_parquet(RESCUE_PAIRS_PARQUET)
    grouped = df.groupby(["parent_chembl_id", "liability_type", "target_chembl_id"], dropna=False)
    rows: list[dict] = []
    for (parent_id, liab, tid), grp in grouped:
        if pd.isna(liab):
            continue  # skip groups with no liability label
        rows.append({
            "ranking_task_id": f"task_{parent_id}_{liab}_{tid}",
            "parent_chembl_id": parent_id,
            "liability_type": liab,
            "target_chembl_id": tid,
            "candidate_ids": grp["candidate_chembl_id"].tolist(),
            "rescue_labels": [
                f"{ret}|{imp}" for ret, imp in zip(grp["activity_retention_bucket"], grp["liability_improvement_category"])
            ],
            "hard_negative_types": grp["hard_negative_type"].tolist(),
            "n_candidates": len(grp),
        })
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(CANDIDATE_SETS_PARQUET, compression="zstd", index=False)
    _log(f"Pass 11 DONE: {len(out_df):,} ranking tasks -> {CANDIDATE_SETS_PARQUET}")
    return CANDIDATE_SETS_PARQUET


# ---------- Pass 12: Quality-tier assignment ----------


def pass_12_quality_tiers() -> None:
    """Mark gold/silver/bronze/auxiliary per spec §10.

    Silver requires (per spec §10): measured activity AND measured liability,
    plausible analog relationship, reliable endpoint direction.
    Translated to columns:
      - has_act: parent + candidate activity_pchembl present
      - has_liab: parent + candidate liability_value present
      - retention bucket non-unknown
      - improvement category non-unknown
      - Murcko match True
      - heavy_atom_diff <= 5 (medicinal-chemistry analog plausibility)

    Bronze: pair has both activity and liability values but fails one or more
    of the silver structural / signal criteria.
    Auxiliary: pair lacks one or both of activity / liability (used only for
    auxiliary predictor training, not the core rescue objective).
    Gold: paper-curated (deferred per L16 — populated by P-1 to P-5 workstream).
    """
    _log("Pass 12: quality-tier assignment")
    df = pd.read_parquet(RESCUE_PAIRS_PARQUET)

    has_act = df["parent_activity_pchembl"].notna() & df["candidate_activity_pchembl"].notna()
    has_liab = df["parent_liability_value"].notna() & df["candidate_liability_value"].notna()
    ret_ok = df["activity_retention_bucket"].notna() & (df["activity_retention_bucket"] != "unknown")
    imp_ok = df["liability_improvement_category"].notna() & (df["liability_improvement_category"] != "unknown")
    murcko_ok = df["murcko_match"].fillna(False).astype(bool)
    heavy_ok = df["heavy_atom_diff"].notna() & (df["heavy_atom_diff"] <= 5)

    silver_mask = has_act & has_liab & ret_ok & imp_ok & murcko_ok & heavy_ok
    bronze_mask = has_act & has_liab & ~silver_mask

    tiers = pd.Series("auxiliary", index=df.index)
    tiers[bronze_mask] = "bronze"
    tiers[silver_mask] = "silver"
    df["quality_tier"] = tiers

    df.to_parquet(RESCUE_PAIRS_PARQUET, compression="zstd", index=False)
    _log(f"Pass 12 DONE: tier counts: {df['quality_tier'].value_counts().to_dict()}")


# ---------- Pass 13: Structured rationale ----------


def pass_13_rationales() -> None:
    """Auto-generate structured rationale fields per spec §4.7."""
    _log("Pass 13: structured rationale auto-generation")
    df = pd.read_parquet(RESCUE_PAIRS_PARQUET)

    from rasyn.evidence.liability_drivers import detect_liability_drivers

    parent_drivers = []
    candidate_changes = []
    expected_directions = []
    failure_risks = []

    for _, r in df.iterrows():
        liab = r.get("liability_type")
        psmi = r.get("parent_smiles")
        csmi = r.get("candidate_smiles")
        if liab is None or psmi is None or csmi is None:
            parent_drivers.append([])
            candidate_changes.append([])
            expected_directions.append({})
            failure_risks.append([])
            continue
        try:
            pd_drivers = detect_liability_drivers(psmi, liab)
            cd_drivers = detect_liability_drivers(csmi, liab)
            parent_drivers.append(pd_drivers)
            candidate_changes.append(sorted(set(pd_drivers) - set(cd_drivers)))
            expected_directions.append({liab: "decrease" if r["liability_improvement_category"] in ("large", "moderate", "minor") else "uncertain"})
            failure_risks.append(sorted(set(cd_drivers) - set(pd_drivers)))
        except Exception:
            parent_drivers.append([])
            candidate_changes.append([])
            expected_directions.append({})
            failure_risks.append([])

    df["liability_drivers_in_parent"] = parent_drivers
    df["modified_features"] = candidate_changes
    df["expected_delta_direction"] = expected_directions
    df["failure_mode_risks"] = failure_risks
    df.to_parquet(RESCUE_PAIRS_PARQUET, compression="zstd", index=False)
    _log(f"Pass 13 DONE: rationale columns added")


# ---------- Final: manifest + canary audit ----------


def finalize() -> None:
    """Build dataset_manifest.json + run canary audit on rescue_pair_candidates."""
    _log("Finalize: dataset manifest + canary audit")
    files = {
        "molecules_canonical": MOLECULES_PARQUET,
        "assay_facts": ASSAY_FACTS_PARQUET,
        "analog_edges": ANALOG_EDGES_PARQUET,
        "rescue_pair_candidates": RESCUE_PAIRS_PARQUET,
        "candidate_sets": CANDIDATE_SETS_PARQUET,
    }
    manifest = {
        "version": "0.1.0",
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {},
    }
    for name, path in files.items():
        if path.exists():
            manifest["files"][name] = {
                "path": str(path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
                "rows": int(pd.read_parquet(path).shape[0]),
            }

    # Canary audit on rescue_pair_candidates
    if RESCUE_PAIRS_PARQUET.exists():
        reg = load_sealed_case_registry()
        canaries = generate_canaries_for_registry(reg, per_layer=4)
        df = pd.read_parquet(RESCUE_PAIRS_PARQUET)
        rows_for_audit = df.to_dict(orient="records")
        result = audit_against_rows(canaries, rows_for_audit)
        manifest["canary_audit"] = result.to_dict()
        with open(DECONTAM_POST, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        _log(f"  Canary audit: {len(result.survivors)} survivors out of {result.total_canaries}")
        if not result.passed:
            _log("  WARNING: canary audit FAILED")

    with open(DATASET_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    _log(f"Finalize DONE: manifest -> {DATASET_MANIFEST}")


# ---------- main orchestrator ----------


PASS_FUNCTIONS = {
    1: pass_1_chembl_activities,
    2: pass_2_pubchem_facts,
    3: pass_3_tdc_molnet_facts,
    6: pass_6_analog_graph,
    7: pass_7_pair_generation,
    8: pass_8_retention_buckets,
    9: pass_9_liability_labels,
    10: pass_10_hard_negatives,
    11: pass_11_ranking_tasks,
    12: pass_12_quality_tiers,
    13: pass_13_rationales,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--passes", type=str, default=None,
                   help="Comma-separated pass numbers to run (e.g. '1,2,3'). Default: all enabled.")
    p.add_argument("--all", action="store_true", help="Run all enabled passes (1,2,3,6,7,8,9,10,11,12,13) + finalize")
    p.add_argument("--skip-finalize", action="store_true")
    args = p.parse_args()

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    if args.passes:
        passes = [int(x) for x in args.passes.split(",")]
    elif args.all:
        passes = sorted(PASS_FUNCTIONS.keys())
    else:
        raise SystemExit("Specify --passes <list> or --all")

    t0 = time.time()
    for n in passes:
        if n not in PASS_FUNCTIONS:
            raise SystemExit(f"Pass {n} not implemented (4=papers deferred per L16; 5=internal skipped per L2)")
        _log(f">>> Pass {n} starting")
        PASS_FUNCTIONS[n]()
        _log(f"<<< Pass {n} done in {time.time() - t0:.1f}s")

    if not args.skip_finalize:
        finalize()
    _log(f"All done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
