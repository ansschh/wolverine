"""P-1b: PMC Open-Access bulk filter -- med-chem OA full-text corpus.

Downloads PMC's `oa_file_list.csv` (~100MB), filters to medicinal-chemistry
journals + post-2010 + papers with adequate content, then fetches full-text
XML via PMC efetch API.

Output:
  rasyn/data/clean/pmc_oa_index.parquet   (DOI, PMCID, journal, year, license)
  rasyn/data/clean/pmc_oa_text/{pmcid}.txt   (one text file per fetched paper)

Per L33 (quality > quantity): journal allow-list is curated, post-2010 only.

Run:
    python scripts/pmc_oa_filter.py \\
        --output-index rasyn/data/clean/pmc_oa_index.parquet \\
        --text-dir     rasyn/data/clean/pmc_oa_text \\
        --concurrency 8 \\
        --max-papers 5000

Resumable: skips PMCIDs already in text-dir.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PMC_OA_FILE_LIST_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Curated medicinal-chemistry journal allow-list. We can broaden later;
# starting tight per L33 (quality > quantity).
MED_CHEM_JOURNAL_PATTERNS = [
    # ACS family (some OA articles)
    r"\bJ Med Chem\b",
    r"\bJournal of Medicinal Chemistry\b",
    r"\bACS Med Chem Lett\b",
    r"\bACS Medicinal Chemistry Letters\b",
    # RSC family
    r"\bRSC Med Chem\b",
    r"\bMedChemComm\b",
    # Elsevier family
    r"\bEur J Med Chem\b",
    r"\bEuropean Journal of Medicinal Chemistry\b",
    r"\bBioorg Med Chem\b",
    r"\bBioorganic & Medicinal Chemistry\b",
    r"\bBioorg Med Chem Lett\b",
    r"\bBioorganic & Medicinal Chemistry Letters\b",
    # BMC / open
    r"\bBMC.*Pharmacology\b",
    r"\bMolecules\b",
    r"\bPharmaceutics\b",
    r"\bDrug Des Devel Ther\b",
    # General medicinal chemistry / DMPK-relevant
    r"\bDrug Metab Dispos\b",
    r"\bJ Pharmacol Exp Ther\b",
    r"\bPharmacology Research & Perspectives\b",
    # Patents-as-papers (less common)
    r"\bExpert Opin.*Drug Discov\b",
]
JOURNAL_RE = re.compile("|".join(MED_CHEM_JOURNAL_PATTERNS), re.IGNORECASE)


def _log(msg: str) -> None:
    print(msg, flush=True)


def download_pmc_oa_file_list(out_path: Path) -> Path:
    if out_path.exists():
        _log(f"PMC oa_file_list.csv already at {out_path} ({out_path.stat().st_size:,} bytes)")
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Downloading {PMC_OA_FILE_LIST_URL}...")
    urllib.request.urlretrieve(PMC_OA_FILE_LIST_URL, out_path)
    _log(f"  -> {out_path} ({out_path.stat().st_size:,} bytes)")
    return out_path


def parse_oa_file_list(csv_path: Path, *, min_year: int = 2010) -> pd.DataFrame:
    """Parse PMC oa_file_list.csv. Columns differ across PMC versions.

    Typical columns:
      File, Article Citation, Accession ID, Last Updated, PMID, License
    The 'Article Citation' field looks like:
      "J Med Chem. 2018 Jul 13;61(13):5727-5740"
    """
    _log(f"Parsing {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Try common column-name variants
    citation_col = next((c for c in df.columns if "citation" in c.lower()), None)
    pmcid_col = next((c for c in df.columns if c.lower() in ("accession id", "pmcid", "accession")), None)
    pmid_col = next((c for c in df.columns if c.lower() == "pmid"), None)
    license_col = next((c for c in df.columns if "license" in c.lower()), None)
    file_col = next((c for c in df.columns if c.lower() == "file"), None)

    if not citation_col or not pmcid_col:
        _log(f"  WARN: unexpected columns: {list(df.columns)}")
        return pd.DataFrame()

    # Parse year + journal from citation
    def _journal(cit: str) -> str:
        if not isinstance(cit, str):
            return ""
        return cit.split(".")[0].strip()

    def _year(cit: str) -> int | None:
        if not isinstance(cit, str):
            return None
        m = re.search(r"\b(19|20)\d{2}\b", cit)
        return int(m.group(0)) if m else None

    df["journal"] = df[citation_col].map(_journal)
    df["year"] = df[citation_col].map(_year)
    df["pmcid"] = df[pmcid_col]
    df["pmid"] = df[pmid_col] if pmid_col else None
    df["license"] = df[license_col] if license_col else None
    df["ftp_path"] = df[file_col] if file_col else None

    n_total = len(df)
    df = df[df["year"].notna() & (df["year"].astype(int) >= min_year)]
    _log(f"  filter year >= {min_year}: {n_total:,} -> {len(df):,}")

    df = df[df["journal"].astype(str).map(lambda j: bool(JOURNAL_RE.search(j or "")))]
    _log(f"  filter med-chem journals: -> {len(df):,}")

    return df[["pmcid", "pmid", "journal", "year", "license", "ftp_path"]].reset_index(drop=True)


def fetch_pmc_xml_via_efetch(pmcid: str, *, timeout: int = 90) -> str | None:
    cleaned = pmcid.replace("PMC", "")
    url = f"{EFETCH_URL}?db=pmc&id={cleaned}&rettype=xml"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def xml_to_text(xml_str: str) -> str:
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_str)
        parts: list[str] = []
        for el in root.iter():
            if el.text and el.text.strip():
                parts.append(el.text.strip())
            if el.tail and el.tail.strip():
                parts.append(el.tail.strip())
        text = "\n".join(parts)
        return re.sub(r"\n{3,}", "\n\n", text)
    except ET.ParseError:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", xml_str))


def fetch_one(record: dict, text_dir: Path) -> dict:
    pmcid = record["pmcid"]
    out = text_dir / f"{pmcid}.txt"
    if out.exists() and out.stat().st_size > 1000:
        return {**record, "status": "cached", "n_chars": out.stat().st_size}

    xml = fetch_pmc_xml_via_efetch(pmcid)
    if not xml or len(xml) < 1024:
        return {**record, "status": "fetch_failed", "n_chars": 0}

    text = xml_to_text(xml)
    if len(text) < 500:
        return {**record, "status": "text_too_small", "n_chars": len(text)}

    out.write_text(text, encoding="utf-8")
    time.sleep(0.4)  # PMC API rate-limit
    return {**record, "status": "ok", "n_chars": len(text)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--oa-list-csv", type=Path,
                   default=Path("rasyn/data/raw/pmc_oa_file_list.csv"))
    p.add_argument("--output-index", type=Path,
                   default=Path("rasyn/data/clean/pmc_oa_index.parquet"))
    p.add_argument("--text-dir", type=Path,
                   default=Path("rasyn/data/clean/pmc_oa_text"))
    p.add_argument("--min-year", type=int, default=2010)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-papers", type=int, default=None,
                   help="Cap the number of papers to fetch (debug / pilot).")
    p.add_argument("--skip-fetch", action="store_true",
                   help="Just build index, don't fetch full-text.")
    args = p.parse_args()

    download_pmc_oa_file_list(args.oa_list_csv)
    df = parse_oa_file_list(args.oa_list_csv, min_year=args.min_year)
    if df.empty:
        _log("No papers matched filters.")
        return 0

    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_index, compression="zstd", index=False)
    _log(f"Index: {len(df):,} med-chem OA candidates -> {args.output_index}")

    if args.skip_fetch:
        return 0

    args.text_dir.mkdir(parents=True, exist_ok=True)
    if args.max_papers:
        df = df.head(args.max_papers)
        _log(f"Limited fetch to first {len(df):,} papers (--max-papers)")

    _log(f"Fetching {len(df):,} full-text XMLs (concurrency={args.concurrency})...")
    t0 = time.time()
    n_done = 0
    n_ok = 0
    n_failed = 0
    statuses: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(fetch_one, row, args.text_dir) for row in df.to_dict("records")]
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            statuses.append(r)
            n_done += 1
            if r["status"] == "ok":
                n_ok += 1
            elif r["status"] != "cached":
                n_failed += 1
            if n_done % 100 == 0:
                elapsed = time.time() - t0
                _log(f"  {n_done}/{len(df)} | ok={n_ok} fail={n_failed} | {n_done/max(elapsed,1):.1f}/s")

    pd.DataFrame(statuses).to_parquet(
        args.output_index.with_suffix(".fetch_status.parquet"),
        compression="zstd", index=False,
    )
    _log(f"\nFetched ok: {n_ok:,} | failed: {n_failed:,} | total: {n_done:,}")
    _log(f"Text dir: {args.text_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
