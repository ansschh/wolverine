"""Data-source adapters for antibiotic discovery.

Per spec §9, six sources:
  9.1 CO-ADD antimicrobial screens
  9.2 PubChem BioAssay
  9.3 ChEMBL bioactivity (filtered for antibacterial)
  9.4 Drug Repurposing Hub
  9.5 Literature (deferred, decontaminated separately)
  9.6 Internal (not applicable; public-only per L2)
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


# ----------------------------------------------------------------
# Organism + target taxonomies
# ----------------------------------------------------------------

# ChEMBL target IDs for the v1 antibacterial organisms.
# Source: ChEMBL document layer / target_dictionary entries.
ORGANISM_TO_CHEMBL_TARGETS: dict[str, list[str]] = {
    "E.coli": ["CHEMBL354", "CHEMBL353", "CHEMBL352"],            # E. coli
    "S.aureus": ["CHEMBL613714", "CHEMBL613715"],                 # S. aureus / MRSA
    "MRSA": ["CHEMBL613718"],                                      # MRSA
    "K.pneumoniae": ["CHEMBL613722"],                              # K. pneumoniae
    "A.baumannii": ["CHEMBL613752", "CHEMBL613753"],              # A. baumannii
    "P.aeruginosa": ["CHEMBL613721", "CHEMBL613734"],             # P. aeruginosa
    "N.gonorrhoeae": ["CHEMBL613720"],                             # N. gonorrhoeae
    "M.tuberculosis": ["CHEMBL613785", "CHEMBL613786"],           # MTB
}

# PubChem AIDs known to be antibacterial / antimicrobial screening assays.
# Source: NIH MLP + community curation (representative; not exhaustive).
ANTIBACTERIAL_PUBCHEM_AIDS: list[int] = [
    686970,   # E. coli growth inhibition counterscreen
    932,      # Antibacterial active broad screen
    1842,     # S. aureus phenotypic
    1832,     # M. tuberculosis high-throughput
    540303,   # Counter-screen mammalian cytotoxicity HEK293
    624414,   # Hemolysis red blood cell
    1224978,  # PAINS aggregator panel
]


def _log(msg: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------
# ChEMBL antibacterial extraction
# ----------------------------------------------------------------

def extract_chembl_antibacterial(
    chembl_db_path: str | Path,
    out_parquet: Path,
    organisms: list[str] | None = None,
) -> Path:
    """Stream ChEMBL bulk SQLite for antibacterial assay rows.

    Filters:
      - target_chembl_id IN known antibacterial targets (per ORGANISM_TO_CHEMBL_TARGETS)
      - standard_relation = '='
      - standard_value present
      - has canonical SMILES + InChIKey

    Output columns: molecule_chembl_id, canonical_smiles, inchi_key, target_chembl_id,
                    target_pref_name, organism_tag, assay_chembl_id, assay_type,
                    document_chembl_id, doi, standard_type, standard_value, standard_units,
                    activity_label.
    """
    if organisms is None:
        organisms = list(ORGANISM_TO_CHEMBL_TARGETS.keys())
    target_to_org = {
        tid: org for org in organisms for tid in ORGANISM_TO_CHEMBL_TARGETS.get(org, [])
    }
    if not target_to_org:
        _log("No antibacterial targets in scope; skipping ChEMBL extraction.")
        return out_parquet

    conn = sqlite3.connect(f"file:{chembl_db_path}?mode=ro", uri=True)
    placeholders = ",".join("?" * len(target_to_org))
    sql = f"""
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
            act.pchembl_value     AS pchembl_value
        FROM activities act
        JOIN assays a       ON a.assay_id = act.assay_id
        JOIN target_dictionary t ON t.tid = a.tid
        JOIN molecule_dictionary md ON md.molregno = act.molregno
        JOIN compound_structures cs ON cs.molregno = md.molregno
        JOIN docs d ON d.doc_id = act.doc_id
        WHERE t.chembl_id IN ({placeholders})
          AND act.standard_value IS NOT NULL
          AND cs.canonical_smiles IS NOT NULL
          AND act.standard_relation = '='
    """
    _log(f"Streaming ChEMBL antibacterial rows for {len(organisms)} organisms...")
    rows = []
    cur = conn.cursor()
    cur.execute(sql, list(target_to_org.keys()))
    cols = [c[0] for c in cur.description]
    while True:
        chunk = cur.fetchmany(20_000)
        if not chunk:
            break
        df = pd.DataFrame(chunk, columns=cols)
        df["organism_tag"] = df["target_chembl_id"].map(target_to_org)
        # Activity label: MIC <=4 ug/mL OR pchembl_value >=5 -> active
        def _label(r):
            unit = (r.get("standard_units") or "").lower()
            val = r.get("standard_value")
            if val is None:
                return "unknown"
            if "mic" in (r.get("standard_type") or "").lower() or "min inhib" in (r.get("standard_type") or "").lower():
                if unit in ("ug/ml", "ug.ml-1", "ug ml-1") and val <= 4.0:
                    return "active"
                if unit in ("um", "umol/l") and val <= 8.0:
                    return "active"
                return "inactive"
            pchembl = r.get("pchembl_value")
            if pchembl is not None and pchembl >= 5.0:
                return "active"
            return "weak"
        df["activity_label"] = df.apply(_label, axis=1)
        rows.append(df)
    conn.close()
    if not rows:
        _log("No ChEMBL antibacterial rows extracted.")
        return out_parquet
    full = pd.concat(rows, ignore_index=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_parquet, compression="zstd", index=False)
    _log(f"ChEMBL antibacterial: {len(full):,} rows -> {out_parquet}")
    return out_parquet


# ----------------------------------------------------------------
# PubChem BioAssay extraction (antibacterial AIDs)
# ----------------------------------------------------------------

def extract_pubchem_antibacterial(out_parquet: Path) -> Path:
    """Fetch the antibacterial-relevant PubChem AIDs and return active/inactive rows."""
    rows = []
    for aid in ANTIBACTERIAL_PUBCHEM_AIDS:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/concise/CSV"
        _log(f"PubChem AID {aid}...")
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                csv_text = resp.read().decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(csv_text))
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            _log(f"  PubChem AID {aid} failed: {e}")
            continue
        cid_col = next((c for c in df.columns if "PUBCHEM_CID" in c), None)
        outcome_col = next((c for c in df.columns if "ACTIVITY_OUTCOME" in c), None)
        if not cid_col or not outcome_col:
            continue
        for _, r in df.iterrows():
            cid = str(r[cid_col]) if pd.notna(r[cid_col]) else None
            outcome = r[outcome_col] if pd.notna(r[outcome_col]) else None
            if not cid or not outcome:
                continue
            label = "active" if outcome == "Active" else ("inactive" if outcome == "Inactive" else "unknown")
            rows.append({
                "pubchem_cid": cid,
                "pubchem_aid": aid,
                "activity_outcome": outcome,
                "activity_label": label,
                "source": "pubchem_bioassay",
            })
    if not rows:
        return out_parquet
    full = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_parquet, compression="zstd", index=False)
    _log(f"PubChem antibacterial: {len(full):,} rows -> {out_parquet}")
    return out_parquet


# ----------------------------------------------------------------
# Drug Repurposing Hub
# ----------------------------------------------------------------

DRUG_REPURPOSING_HUB_URL = (
    "https://repo-hub.broadinstitute.org/repurposing/data/repurposing_drugs_20231220.txt"
)


def fetch_drug_repurposing_hub(out_parquet: Path) -> Path:
    """Download the Drug Repurposing Hub annotated drug list."""
    try:
        with urllib.request.urlopen(DRUG_REPURPOSING_HUB_URL, timeout=120) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        _log(f"Drug Repurposing Hub fetch failed: {e}")
        return out_parquet
    df = pd.read_csv(io.StringIO(text), sep="\t")
    df["source"] = "drug_repurposing_hub"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, compression="zstd", index=False)
    _log(f"Drug Repurposing Hub: {len(df):,} drugs -> {out_parquet}")
    return out_parquet


# ----------------------------------------------------------------
# CO-ADD
# ----------------------------------------------------------------

def load_coadd_csv(coadd_csv_path: str | Path, out_parquet: Path) -> Path:
    """Convert a CO-ADD CSV (downloaded by user) to standardized parquet.

    CO-ADD downloads are gated behind email registration at co-add.org.
    Expects a CSV with cols including: 'Cmpd ID', 'SMILES', 'Organism', 'Inhibition %', etc.
    """
    p = Path(coadd_csv_path)
    if not p.exists():
        _log(f"CO-ADD CSV {coadd_csv_path} not found; skipping.")
        return out_parquet
    df = pd.read_csv(p, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for k_lower, k_real in cols.items():
        if "smiles" in k_lower:
            rename[k_real] = "canonical_smiles"
        elif "organism" in k_lower:
            rename[k_real] = "organism"
        elif "inhibition" in k_lower or "growth inhibition" in k_lower:
            rename[k_real] = "growth_inhibition_pct"
        elif "compound" in k_lower and "id" in k_lower:
            rename[k_real] = "coadd_id"
    df = df.rename(columns=rename)
    if "growth_inhibition_pct" in df.columns:
        df["activity_label"] = df["growth_inhibition_pct"].fillna(0).apply(
            lambda v: "active" if v >= 80 else ("weak" if v >= 50 else "inactive")
        )
    else:
        df["activity_label"] = "unknown"
    df["source"] = "co_add"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, compression="zstd", index=False)
    _log(f"CO-ADD: {len(df):,} rows -> {out_parquet}")
    return out_parquet
