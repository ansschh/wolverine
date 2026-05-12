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
        # Activity label: MIC <=4 ug/mL OR pchembl_value >=5 -> active.
        # NaN-safe: pandas returns numpy NaN floats for missing strings.
        def _safe_str(v) -> str:
            if v is None:
                return ""
            try:
                if pd.isna(v):
                    return ""
            except (TypeError, ValueError):
                pass
            return str(v).lower()

        def _safe_float(v):
            if v is None:
                return None
            try:
                if pd.isna(v):
                    return None
                return float(v)
            except (TypeError, ValueError):
                return None

        def _label(r):
            unit = _safe_str(r.get("standard_units"))
            stype = _safe_str(r.get("standard_type"))
            val = _safe_float(r.get("standard_value"))
            if val is None:
                return "unknown"
            if "mic" in stype or "min inhib" in stype:
                if unit in ("ug/ml", "ug.ml-1", "ug ml-1") and val <= 4.0:
                    return "active"
                if unit in ("um", "umol/l") and val <= 8.0:
                    return "active"
                return "inactive"
            pchembl = _safe_float(r.get("pchembl_value"))
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


def fetch_drug_repurposing_hub(out_parquet: Path, *, verify_ssl: bool = True) -> Path:
    """Download the Drug Repurposing Hub annotated drug list.

    The Broad Hub frequently presents an SSL chain older certifi bundles can't
    verify. Try in three escalating modes:
      1. system-default TLS
      2. certifi-resolved CA bundle (forced)
      3. unverified context with a printed warning (NOT silent — last resort)
    """
    import ssl

    def _try(ctx, label: str):
        try:
            with urllib.request.urlopen(DRUG_REPURPOSING_HUB_URL, timeout=120, context=ctx) as resp:
                _log(f"  fetched DRH via {label}")
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            _log(f"  DRH fetch via {label} failed: {e}")
            return None

    text = _try(ssl.create_default_context(), "system-default")
    if text is None:
        try:
            import certifi
            text = _try(ssl.create_default_context(cafile=certifi.where()), "certifi")
        except ImportError:
            pass
    if text is None and verify_ssl is False:
        _log("  WARNING: falling back to unverified SSL context (verify_ssl=False)")
        text = _try(ssl._create_unverified_context(), "unverified")
    if text is None:
        _log("Drug Repurposing Hub fetch failed all available SSL attempts.")
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

# Map CO-ADD scientific names → the closed Organism literal in schemas.py
COADD_ORGANISM_MAP: dict[str, str] = {
    "Escherichia coli":             "E.coli",
    "Staphylococcus aureus":        "S.aureus",      # MRSA strains get re-labelled below
    "Klebsiella pneumoniae":        "K.pneumoniae",
    "Acinetobacter baumannii":      "A.baumannii",
    "Pseudomonas aeruginosa":       "P.aeruginosa",
    "Mycobacterium tuberculosis":   "MTB",
    "Mycobacterium tuberculosis H37Rv": "MTB",
    "Clostridium difficile":        "C.difficile",
    "Clostridioides difficile":     "C.difficile",
    "Helicobacter pylori":          "H.pylori",
    "Neisseria gonorrhoeae":        "N.gonorrhoeae",
    # Mammalian / fungal — these become counter-screens, not antibacterial facts.
    "Homo sapiens":                 "human_cell",
    "Candida albicans":             "fungal",
    "Cryptococcus neoformans":      "fungal",
}


def _coadd_organism_to_schema(organism: str | None, strain: str | None) -> str:
    """Map CO-ADD ORGANISM + STRAIN into our schema's Organism literal.
    Re-routes 'S.aureus' to 'MRSA' when STRAIN contains MRSA/ATCC 43300/etc.
    """
    if organism is None:
        return "unknown"
    base = COADD_ORGANISM_MAP.get(str(organism).strip(), "unknown")
    if base == "S.aureus" and strain and ("MRSA" in str(strain) or "43300" in str(strain)):
        return "MRSA"
    return base


def load_coadd_inhibition(inhibition_csv_path: str | Path, out_parquet: Path) -> Path:
    """Load CO-ADD InhibitionData (single-concentration primary screen, ~803K rows).

    Columns: COADD_ID, COMPOUND_CODE, PROJECT_ID, LIBRARY_NAME, ASSAY_ID, ORGANISM,
             STRAIN, NASSAYS, INHIB_AVE, INHIB_STD, CONC, SMILES.
    Activity-label rule: INHIB_AVE >= 80 → active; >= 50 → weak; else inactive.
    """
    p = Path(inhibition_csv_path)
    if not p.exists():
        _log(f"CO-ADD InhibitionData not found at {p}; skipping.")
        return out_parquet
    df = pd.read_csv(p, low_memory=False)
    _log(f"  CO-ADD InhibitionData raw rows: {len(df):,}")

    # Required columns
    need = ["SMILES", "ORGANISM", "INHIB_AVE", "ASSAY_ID", "COADD_ID"]
    if not all(c in df.columns for c in need):
        _log(f"  unexpected columns: {list(df.columns)[:15]}")
        return out_parquet
    df = df.dropna(subset=["SMILES", "ORGANISM"])

    # Activity label
    def _label(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return "unknown"
        if v >= 80:  return "active"
        if v >= 50:  return "weak"
        return "inactive"
    df["activity_label"] = df["INHIB_AVE"].apply(_label)

    # Organism normalization → schema literal
    df["organism_tag"] = df.apply(
        lambda r: _coadd_organism_to_schema(r.get("ORGANISM"), r.get("STRAIN")), axis=1,
    )

    out = pd.DataFrame({
        "canonical_smiles":      df["SMILES"].astype(str),
        "coadd_id":              df["COADD_ID"],
        "compound_code":         df.get("COMPOUND_CODE"),
        "assay_chembl_id":       df["ASSAY_ID"],     # reuse column name for schema compat
        "assay_type":            "B",                # binding-style single-conc
        "organism_tag":          df["organism_tag"],
        "strain":                df.get("STRAIN"),
        "standard_type":         "Inhibition",
        "standard_value":        df["INHIB_AVE"],
        "standard_units":        "%",
        "standard_relation":     "=",
        "activity_label":        df["activity_label"],
        "source":                "co_add_inhibition",
    })
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, compression="zstd", index=False)
    _log(f"CO-ADD InhibitionData: {len(out):,} normalized rows -> {out_parquet}")
    return out_parquet


def load_coadd_dose_response(dr_csv_path: str | Path, out_parquet: Path) -> Path:
    """Load CO-ADD DoseResponseData (confirmed actives, ~42K rows).

    Columns: COADD_ID, COMPOUND_CODE, SMILES, PROJECT_ID, LIBRARY_NAME, ASSAY_ID,
             ORGANISM, STRAIN, NASSAYS, DRVAL_TYPE, DRVAL_MEDIAN, DRVAL_UNIT, DMAX_AVE.
    DRVAL_TYPE is one of MIC / IC50 / CC50 (CC50 = HEK293 cytotoxicity counter-screen).
    Label rule:
      MIC + numeric value <= 4 ug/mL OR <= 8 uM → active
      MIC + value with '>' prefix → inactive
      CC50 → counter-screen row (NOT antibacterial fact)
    """
    p = Path(dr_csv_path)
    if not p.exists():
        _log(f"CO-ADD DoseResponseData not found at {p}; skipping.")
        return out_parquet
    df = pd.read_csv(p, low_memory=False)
    _log(f"  CO-ADD DoseResponseData raw rows: {len(df):,}")

    need = ["SMILES", "ORGANISM", "DRVAL_MEDIAN", "DRVAL_UNIT", "DRVAL_TYPE", "ASSAY_ID", "COADD_ID"]
    if not all(c in df.columns for c in need):
        _log(f"  unexpected columns: {list(df.columns)[:15]}")
        return out_parquet
    df = df.dropna(subset=["SMILES", "ORGANISM"])

    def _parse_value(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None, None
        s = str(v).strip()
        rel = "="
        if s.startswith(">"):
            rel = ">"; s = s[1:].strip()
        elif s.startswith("<"):
            rel = "<"; s = s[1:].strip()
        try:
            return float(s), rel
        except ValueError:
            return None, None

    parsed = df["DRVAL_MEDIAN"].apply(_parse_value)
    df["dr_value"] = [p[0] for p in parsed]
    df["dr_rel"]   = [p[1] for p in parsed]

    def _label(r):
        kind = str(r.get("DRVAL_TYPE") or "").upper()
        v    = r.get("dr_value")
        rel  = r.get("dr_rel")
        unit = str(r.get("DRVAL_UNIT") or "").lower()
        if v is None:
            return "unknown"
        if kind == "CC50":
            # Mammalian cell cytotox (HEK293) — counter-screen, not antibacterial fact.
            return "cytotox_counter_screen"
        if kind == "HC10":
            # Hemolysis 10% concentration — counter-screen for RBC lysis.
            return "hemolysis_counter_screen"
        if kind == "IC50":
            return "active" if (rel == "=" and v <= 8.0) else "weak"
        if kind == "MIC":
            if rel == ">":
                return "inactive"
            if unit in ("ug/ml", "ug.ml-1") and v <= 4.0:
                return "active"
            if unit in ("um", "umol/l") and v <= 8.0:
                return "active"
            return "weak"
        return "unknown"

    df["activity_label"] = df.apply(_label, axis=1)
    df["organism_tag"]   = df.apply(
        lambda r: _coadd_organism_to_schema(r.get("ORGANISM"), r.get("STRAIN")), axis=1,
    )

    out = pd.DataFrame({
        "canonical_smiles":  df["SMILES"].astype(str),
        "coadd_id":          df["COADD_ID"],
        "compound_code":     df.get("COMPOUND_CODE"),
        "assay_chembl_id":   df["ASSAY_ID"],
        "assay_type":        "B",
        "organism_tag":      df["organism_tag"],
        "strain":            df.get("STRAIN"),
        "standard_type":     df["DRVAL_TYPE"],
        "standard_value":    df["dr_value"],
        "standard_units":    df["DRVAL_UNIT"],
        "standard_relation": df["dr_rel"],
        "activity_label":    df["activity_label"],
        "source":            "co_add_dose_response",
    })
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, compression="zstd", index=False)
    _log(f"CO-ADD DoseResponseData: {len(out):,} normalized rows -> {out_parquet}")
    return out_parquet


def load_coadd_csv(coadd_csv_path: str | Path, out_parquet: Path) -> Path:
    """Back-compat wrapper. Routes to inhibition / dose-response loader by filename."""
    p = Path(coadd_csv_path)
    if not p.exists():
        _log(f"CO-ADD CSV {p} not found; skipping.")
        return out_parquet
    name = p.name.lower()
    if "inhibition" in name:
        return load_coadd_inhibition(p, out_parquet)
    if "doseresponse" in name or "dose_response" in name:
        return load_coadd_dose_response(p, out_parquet)
    # Generic / unknown file — fall back to permissive parse
    df = pd.read_csv(p, low_memory=False)
    df["source"] = "co_add_generic"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, compression="zstd", index=False)
    _log(f"CO-ADD (generic): {len(df):,} rows -> {out_parquet}")
    return out_parquet
