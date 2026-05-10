"""ChEMBL bulk SQLite + targeted REST adapter.

Strategy: prefer the bulk SQLite dump (~20 GB) for full-corpus mining since
it gives target/document/assay/molecule joins locally. Use the REST API only
for targeted lookups (e.g., name->ChEMBL ID for the registry populator).

The bulk SQLite is downloaded once; subsequent runs read from disk.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.request import urlopen, urlretrieve

CHEMBL_LATEST = "35"  # bump when ChEMBL releases a new version
CHEMBL_BULK_URL_TEMPLATE = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
    "chembl_{ver}/chembl_{ver}_sqlite.tar.gz"
)
CHEMBL_REST_BASE = "https://www.ebi.ac.uk/chembl/api/data"


@dataclass
class ChEMBLConfig:
    version: str = CHEMBL_LATEST
    bulk_dir: Path = Path("rasyn/data/raw/chembl")
    sqlite_path: Path | None = None  # set by `download_bulk` after extraction


def download_bulk(cfg: ChEMBLConfig) -> Path:
    """Download the ChEMBL bulk SQLite tarball into `cfg.bulk_dir`.

    NOTE: ~20 GB download. Idempotent: skips if archive exists.
    Returns the path to the .tar.gz; caller untars it.
    """
    cfg.bulk_dir.mkdir(parents=True, exist_ok=True)
    url = CHEMBL_BULK_URL_TEMPLATE.format(ver=cfg.version)
    target = cfg.bulk_dir / Path(url).name
    if not target.exists():
        urlretrieve(url, target)
    return target


def open_bulk_db(sqlite_path: Path) -> sqlite3.Connection:
    """Read-only connection to the ChEMBL bulk SQLite."""
    return sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)


def stream_molecules(conn: sqlite3.Connection, *, batch_size: int = 50_000) -> Iterator[dict]:
    """Stream (chembl_id, smiles, inchi_key, max_phase) rows from molecule_dictionary + compound_structures."""
    sql = """
        SELECT md.chembl_id, cs.canonical_smiles, cs.standard_inchi_key, md.max_phase
        FROM molecule_dictionary md
        JOIN compound_structures cs ON cs.molregno = md.molregno
        WHERE cs.canonical_smiles IS NOT NULL
    """
    cur = conn.cursor()
    cur.execute(sql)
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        for chembl_id, smiles, ik, max_phase in rows:
            yield {"chembl_id": chembl_id, "smiles": smiles, "inchi_key": ik, "max_phase": max_phase}


def stream_activities(conn: sqlite3.Connection, *, batch_size: int = 50_000) -> Iterator[dict]:
    """Stream activity rows joining molecule, target, assay, document."""
    sql = """
        SELECT
            md.chembl_id AS molecule_chembl_id,
            t.chembl_id AS target_chembl_id,
            t.pref_name AS target_pref_name,
            a.chembl_id AS assay_chembl_id,
            a.assay_type,
            d.chembl_id AS document_chembl_id,
            d.doi,
            act.standard_type,
            act.standard_relation,
            act.standard_value,
            act.standard_units,
            act.pchembl_value
        FROM activities act
        JOIN assays a ON a.assay_id = act.assay_id
        JOIN target_dictionary t ON t.tid = a.tid
        JOIN molecule_dictionary md ON md.molregno = act.molregno
        JOIN docs d ON d.doc_id = act.doc_id
        WHERE act.standard_value IS NOT NULL
    """
    cur = conn.cursor()
    cur.execute(sql)
    cols = [c[0] for c in cur.description]
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            yield dict(zip(cols, row, strict=True))


def lookup_molecule_by_name(name: str) -> dict | None:
    """REST: look up a single molecule by name. Returns first hit or None."""
    import json
    from urllib.parse import quote

    url = f"{CHEMBL_REST_BASE}/molecule/search?q={quote(name)}&limit=1&format=json"
    try:
        with urlopen(url, timeout=30) as resp:
            data = json.load(resp)
    except Exception:
        return None
    mols = data.get("molecules", [])
    if not mols:
        return None
    m = mols[0]
    cs = (m.get("molecule_structures") or {}).get("canonical_smiles")
    ik = (m.get("molecule_structures") or {}).get("standard_inchi_key")
    return {
        "chembl_id": m.get("molecule_chembl_id"),
        "name": name,
        "canonical_smiles": cs,
        "inchi_key": ik,
        "max_phase": m.get("max_phase"),
        "synonyms": [s["molecule_synonym"] for s in (m.get("molecule_synonyms") or [])],
    }
