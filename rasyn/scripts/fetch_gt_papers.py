"""Fetch full text of GT papers (PMC open-access first, user-provided PDF fallback).

For each ground-truth paper DOI:
  1. Try NCBI ID Converter to map DOI -> PMCID. If success, fetch fulltext XML
     via PMC efetch. Strip XML tags, save as text.
  2. If not in PMC, look for a user-provided PDF at
     `rasyn/papers/gt_papers_pdfs/{doi_safe}.pdf` (and optional `.si.pdf`).
     Parse via pymupdf, save concatenated text.
  3. If neither available, log and report — user must drop the PDF.

Output: rasyn/papers/gt_papers_text/{doi_safe}.txt

Run:
  python scripts/fetch_gt_papers.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


GT_PATH = Path("rasyn/papers/ground_truth_set.yaml")  # use the original (DOIs are there)
PDFS_DIR = Path("rasyn/papers/gt_papers_pdfs")
TEXT_DIR = Path("rasyn/papers/gt_papers_text")
FETCH_REPORT = Path("rasyn/papers/gt_papers_fetch_report.json")


def _log(msg: str) -> None:
    print(msg, flush=True)


def doi_to_safe(doi: str) -> str:
    """Filename-safe form of a DOI: replace non-alphanumeric characters with '_'."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", doi)


def doi_to_pmcid(doi: str) -> str | None:
    """Map a DOI to a PMCID via NCBI ID Converter API."""
    url = (
        f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        f"?ids={urllib.parse.quote(doi)}&format=json"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for record in data.get("records", []):
            if "pmcid" in record:
                return record["pmcid"]
        return None
    except Exception as e:
        _log(f"  DOI->PMCID lookup failed: {e}")
        return None


def fetch_pmc_xml(pmcid: str) -> str | None:
    """Fetch full-text XML for a PMCID via NCBI efetch."""
    pmcid_clean = pmcid.replace("PMC", "")
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid_clean}&rettype=xml"
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        _log(f"  PMC efetch failed: {e}")
        return None


def xml_to_text(xml_str: str) -> str:
    """Extract human-readable text from PMC fulltext XML.

    Keeps body text + table cells. Drops tags but preserves textual content.
    """
    try:
        root = ET.fromstring(xml_str)
        # Use itertext() to walk all text content depth-first
        parts: list[str] = []
        for el in root.iter():
            txt = el.text
            if txt and txt.strip():
                parts.append(txt.strip())
            tail = el.tail
            if tail and tail.strip():
                parts.append(tail.strip())
        text = "\n".join(parts)
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
    except ET.ParseError as e:
        _log(f"  XML parse failed; falling back to regex strip: {e}")
        text = re.sub(r"<[^>]+>", " ", xml_str)
        text = re.sub(r"\s+", " ", text)
        return text


def parse_pdf(pdf_path: Path) -> str | None:
    """Extract text from a PDF using pymupdf."""
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError:
            _log("  pymupdf not installed (`pip install pymupdf`); cannot parse PDFs.")
            return None

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        _log(f"  PDF open failed: {e}")
        return None

    parts: list[str] = []
    for page in doc:
        try:
            parts.append(page.get_text())
        except Exception as e:
            _log(f"  PDF page text extract failed (page skipped): {e}")
    doc.close()
    return "\n\n".join(parts)


def fetch_one(doi: str, gt_id: str) -> dict:
    """Fetch one paper. Returns a record dict for the report."""
    safe = doi_to_safe(doi)
    text_path = TEXT_DIR / f"{safe}.txt"

    record: dict = {
        "gt_id": gt_id,
        "doi": doi,
        "safe": safe,
        "text_path": str(text_path),
        "source": None,
        "n_chars": 0,
        "needs_manual_pdf": False,
        "pdf_paths_expected": [
            str(PDFS_DIR / f"{safe}.pdf"),
            str(PDFS_DIR / f"{safe}.si.pdf"),
        ],
    }

    if text_path.exists():
        record["source"] = "cached"
        record["n_chars"] = text_path.stat().st_size
        _log(f"[{gt_id}] CACHED ({record['n_chars']} bytes)")
        return record

    # 1) Try PMC
    _log(f"[{gt_id}] {doi}: trying PMC...")
    pmcid = doi_to_pmcid(doi)
    if pmcid:
        _log(f"  -> {pmcid}; fetching fulltext...")
        xml = fetch_pmc_xml(pmcid)
        if xml and len(xml) > 1024:
            text = xml_to_text(xml)
            if len(text) > 500:
                text_path.write_text(text, encoding="utf-8")
                record["source"] = "pmc"
                record["pmcid"] = pmcid
                record["n_chars"] = len(text)
                _log(f"  -> PMC text saved: {len(text):,} chars -> {text_path.name}")
                time.sleep(1)
                return record

    # 2) Fall back to user-provided PDF
    main_pdf = PDFS_DIR / f"{safe}.pdf"
    si_pdf = PDFS_DIR / f"{safe}.si.pdf"
    parts: list[str] = []
    if main_pdf.exists():
        _log(f"  PMC unavailable; parsing {main_pdf.name}...")
        t = parse_pdf(main_pdf)
        if t:
            parts.append("=== MAIN PAPER ===\n" + t)
    if si_pdf.exists():
        _log(f"  Also found SI: {si_pdf.name}; parsing...")
        t = parse_pdf(si_pdf)
        if t:
            parts.append("\n\n=== SUPPLEMENTARY INFORMATION ===\n" + t)

    if parts:
        text = "\n".join(parts)
        text_path.write_text(text, encoding="utf-8")
        record["source"] = "pdf"
        record["n_chars"] = len(text)
        _log(f"  -> PDF text saved: {len(text):,} chars -> {text_path.name}")
        time.sleep(1)
        return record

    # 3) Manual PDF needed
    record["source"] = "missing"
    record["needs_manual_pdf"] = True
    _log(f"  MISSING: drop PDF at {main_pdf} (and optional SI at {si_pdf})")
    time.sleep(1)
    return record


def main() -> int:
    if not GT_PATH.exists():
        _log(f"FATAL: {GT_PATH} not found")
        return 1
    gt = yaml.safe_load(GT_PATH.read_text())

    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for pair in gt["pairs"]:
        gt_id = pair["id"]
        # In the YAML, DOI is currently `paper_doi_TODO` (verification-pending).
        doi = pair.get("paper_doi_TODO") or pair.get("paper_doi")
        if not doi:
            _log(f"[{gt_id}] NO DOI - skipping")
            records.append({"gt_id": gt_id, "doi": None, "source": "no_doi"})
            continue
        rec = fetch_one(doi, gt_id)
        records.append(rec)

    summary = {
        "total": len(records),
        "by_source": {},
        "missing_pdfs": [],
        "records": records,
    }
    for r in records:
        src = r.get("source") or "unknown"
        summary["by_source"][src] = summary["by_source"].get(src, 0) + 1
        if r.get("needs_manual_pdf"):
            summary["missing_pdfs"].append(r["pdf_paths_expected"][0])

    FETCH_REPORT.write_text(json.dumps(summary, indent=2))

    _log("")
    _log("=" * 60)
    for src, n in summary["by_source"].items():
        _log(f"  {src}: {n}")
    _log("=" * 60)
    if summary["missing_pdfs"]:
        _log("\nDrop the following PDFs to retry:")
        for p in summary["missing_pdfs"]:
            _log(f"  {p}")
    _log(f"\nReport: {FETCH_REPORT}")
    _log(f"Texts:  {TEXT_DIR}/")

    return 0


if __name__ == "__main__":
    import urllib.parse  # used in doi_to_pmcid
    sys.exit(main())
