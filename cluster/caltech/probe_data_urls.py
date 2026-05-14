"""Probe candidate URLs for retro raw data sources.

Why a Python script vs a bash one-liner: long URLs paste-mangle in SSH
sessions. This file ships as a single source of truth and produces tabular
output (http_status, size_bytes, content_type, url) we can act on.

Usage on the Caltech login node:

    cd /resnick/scratch/atiwari2/rasyn-retro
    git pull
    python cluster/caltech/probe_data_urls.py
"""
from __future__ import annotations

import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

USER_AGENT = "rasyn-retro-probe/1.0 (Mozilla/5.0)"
TIMEOUT_S = 25

CANDIDATES: dict[str, list[str]] = {
    "uspto_50k": [
        "https://huggingface.co/datasets/sagawa/USPTO-50K/resolve/main/data/train-00000-of-00001.parquet",
        "https://huggingface.co/datasets/sagawa/USPTO-50K/resolve/main/train.csv",
        "https://huggingface.co/datasets/sagawa/USPTO-50K/resolve/main/raw_train.csv",
        "https://huggingface.co/datasets/sagawa/USPTO-50K/resolve/main/USPTO_50K.csv",
        "https://huggingface.co/datasets/sagawa/USPTO-50K/resolve/main/USPTO50K.csv",
        "https://huggingface.co/datasets/sagawa/USPTO-50K/resolve/main/README.md",  # sanity
        "https://raw.githubusercontent.com/Hanjun-Dai/GLN/master/data/schneider50k/raw_train.csv",
        "https://raw.githubusercontent.com/Hanjun-Dai/GLN/master/data/schneider50k/raw_val.csv",
        "https://raw.githubusercontent.com/Hanjun-Dai/GLN/master/data/schneider50k/raw_test.csv",
        "https://raw.githubusercontent.com/pschwllr/MolecularTransformer/master/data/uspto_50k/train.txt",
        "https://raw.githubusercontent.com/uta-smile/RetroXpert/master/data/USPTO50K/raw_train.csv",
    ],
    "uspto_full": [
        "https://figshare.com/ndownloader/files/8664379",
        "https://zenodo.org/record/14796879",
        "https://huggingface.co/datasets/AspirinCode/USPTO-FULL/resolve/main/USPTO_FULL.csv",
    ],
    "buyables_zinc": [
        "https://files.docking.org/zinc22/zinc-22-in-stock.smi.gz",  # current (dead)
        "https://zinc20.docking.org/substances.smi?subset=in-stock&format=zinc_id+smiles+inchikey",
        "https://zinc.docking.org/substances/subsets/in-stock/",
        "https://files.docking.org/2D/AA/AAAA.smi.gz",  # ZINC20 tranche probe
    ],
    "buyables_enamine": [
        "https://enamine.net/files/REAL_BB_Database/Enamine_Building_Blocks_Stock.sdf",
        "https://enamine.net/building-blocks/real-database",  # landing page sanity
    ],
    "buyables_emolecules": [
        "https://downloads.emolecules.com/free/2026-05-01/version.smi.gz",  # known working
    ],
}


def probe(url: str) -> tuple[int, int, str]:
    req = Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    try:
        with urlopen(req, timeout=TIMEOUT_S) as r:
            status = r.status
            size = int(r.headers.get("Content-Length", 0) or 0)
            ctype = r.headers.get("Content-Type", "") or ""
            return status, size, ctype
    except HTTPError as e:
        return e.code, 0, ""
    except (URLError, TimeoutError, ConnectionError, OSError) as e:
        return -1, 0, str(e)[:60]


def fmt_size(n: int) -> str:
    if n <= 0:
        return "0"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TiB"


def main() -> int:
    for source, urls in CANDIDATES.items():
        print(f"\n=== {source} ===")
        for url in urls:
            status, size, ctype = probe(url)
            tag = "OK " if status == 200 and size > 1024 else "   "
            print(f"  {tag} {status:>4}  {fmt_size(size):>10}  {ctype[:30]:<30}  {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
