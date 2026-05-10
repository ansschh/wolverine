"""P-1a: Unpaywall sweep -- find OA URLs for ChEMBL DOIs.

For every unique DOI in assay_facts.parquet, query the Unpaywall API
(free, email-registered, ~100K req/day limit). Records best OA URL,
license, and journal-level OA status per DOI.

Output: chembl_doi_oa_index.parquet
  Cols: doi, is_oa, oa_url, oa_license, oa_version (publishedVersion /
        acceptedVersion / submittedVersion), journal_is_oa, n_oa_locations,
        error

Then `scripts/download_oa_papers.py` (separate) downloads + parses the URLs.

Run:
    python scripts/unpaywall_sweep.py \\
        --assay-facts rasyn/data/clean/assay_facts.parquet \\
        --output rasyn/data/clean/chembl_doi_oa_index.parquet \\
        --email anshtiwari9899@gmail.com \\
        --concurrency 16

Resumable: skips DOIs already in the output parquet.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

UNPAYWALL_BASE = "https://api.unpaywall.org/v2/"


def _log(msg: str) -> None:
    print(msg, flush=True)


def query_unpaywall(doi: str, email: str, timeout: int = 30) -> dict:
    """Returns a flat dict suitable for the output parquet."""
    url = UNPAYWALL_BASE + urllib.parse.quote(doi.strip())
    url += f"?email={urllib.parse.quote(email)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"doi": doi, "is_oa": False, "error": "not_found_in_unpaywall"}
        return {"doi": doi, "error": f"http_{e.code}"}
    except urllib.error.URLError as e:
        return {"doi": doi, "error": f"url_{type(e).__name__}: {e.reason}"}
    except Exception as e:
        return {"doi": doi, "error": f"exc: {type(e).__name__}: {e}"}

    is_oa = bool(data.get("is_oa", False))
    locations = data.get("oa_locations") or []
    best = data.get("best_oa_location") or (locations[0] if locations else None)
    return {
        "doi": doi,
        "is_oa": is_oa,
        "oa_url": (best or {}).get("url"),
        "oa_url_for_pdf": (best or {}).get("url_for_pdf"),
        "oa_license": (best or {}).get("license"),
        "oa_version": (best or {}).get("version"),
        "host_type": (best or {}).get("host_type"),
        "journal_is_oa": bool(data.get("journal_is_oa", False)),
        "journal_is_in_doaj": bool(data.get("journal_is_in_doaj", False)),
        "n_oa_locations": len(locations),
        "title": (data.get("title") or "")[:500],
        "year": data.get("year"),
        "journal_name": data.get("journal_name"),
        "publisher": data.get("publisher"),
        "error": None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--assay-facts", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("rasyn/data/clean/chembl_doi_oa_index.parquet"))
    p.add_argument("--email", required=True,
                   help="Unpaywall requires email registration (free).")
    p.add_argument("--concurrency", type=int, default=16,
                   help="Parallel API requests. Unpaywall is fine with 16-32.")
    p.add_argument("--max-dois", type=int, default=None,
                   help="Process at most N DOIs (debug / pilot).")
    p.add_argument("--checkpoint-every", type=int, default=2000)
    args = p.parse_args()

    if not args.assay_facts.exists():
        _log(f"FATAL: {args.assay_facts} not found")
        return 1

    _log(f"Loading DOIs from {args.assay_facts}...")
    facts = pd.read_parquet(args.assay_facts, columns=["doi"])
    dois = facts["doi"].dropna().astype(str).str.strip()
    dois = dois[dois.str.len() > 0]
    unique_dois = dois.drop_duplicates().tolist()
    _log(f"Found {len(unique_dois):,} unique DOIs in assay_facts")

    if args.max_dois:
        unique_dois = unique_dois[: args.max_dois]
        _log(f"Limited to first {len(unique_dois):,} DOIs (--max-dois)")

    # Resume support
    done_dois: set[str] = set()
    if args.output.exists():
        prev = pd.read_parquet(args.output)
        done_dois = set(prev["doi"].astype(str).tolist())
        _log(f"Resuming: {len(done_dois):,} DOIs already in {args.output}")

    todo = [d for d in unique_dois if d not in done_dois]
    if not todo:
        _log("All DOIs already swept.")
        return 0
    _log(f"Querying Unpaywall for {len(todo):,} new DOIs (concurrency={args.concurrency})")

    results: list[dict] = []
    t0 = time.time()
    n_oa = 0
    n_err = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(query_unpaywall, doi, args.email): doi for doi in todo}
        n_done = 0
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)
            n_done += 1
            if res.get("is_oa"):
                n_oa += 1
            if res.get("error"):
                n_err += 1
            if n_done % 200 == 0:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1)
                _log(f"  {n_done:,}/{len(todo):,} | OA found: {n_oa} | err: {n_err} | {rate:.1f} req/s")
            if n_done % args.checkpoint_every == 0:
                prev = pd.read_parquet(args.output) if args.output.exists() else pd.DataFrame()
                combined = pd.concat([prev, pd.DataFrame(results)], ignore_index=True, sort=False)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                combined.to_parquet(args.output, compression="zstd", index=False)
                results.clear()

    if results:
        prev = pd.read_parquet(args.output) if args.output.exists() else pd.DataFrame()
        combined = pd.concat([prev, pd.DataFrame(results)], ignore_index=True, sort=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(args.output, compression="zstd", index=False)

    elapsed = time.time() - t0
    full = pd.read_parquet(args.output)
    _log("")
    _log("=" * 60)
    _log(f"Total processed: {len(todo):,} | wall-clock: {elapsed/60:.1f} min")
    _log(f"OA found in this run: {n_oa:,} | errors: {n_err:,}")
    _log(f"Index now has {len(full):,} DOI rows; "
         f"{int(full['is_oa'].fillna(False).sum()):,} OA total ({100 * full['is_oa'].fillna(False).mean():.1f}%)")
    _log(f"Output: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
