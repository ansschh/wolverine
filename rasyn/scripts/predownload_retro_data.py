"""Pre-download all retro raw data on a machine with clean internet.

Why this exists:
  Caltech HPC compute nodes have outbound network restrictions and silently
  fail on figshare/Zenodo/Enamine. Running urlretrieve from a sbatch
  produced 0-byte files that broke the curation pipeline downstream.

  This script does every download once on the login node (or a dev box),
  validates each archive, and prints a manifest with SHA256s and sizes.
  The R-1 sbatch then runs in offline mode: `predownload_only` + sbatch
  guard ensures we never get to atom-mapping with bogus data.

Usage:
  python -m scripts.predownload_retro_data \
      --raw-root rasyn/data/raw \
      --sources uspto_50k,uspto_full,zinc22,enamine,emolecules \
      [--skip-existing]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rasyn.data.sources import _download as _dl
from rasyn.data.sources import buyables as buyables_src
from rasyn.data.sources import uspto as uspto_src

ALL_SOURCES = {
    "uspto_50k",
    "uspto_full",
    "uspto_llm",
    "zinc22",
    "enamine",
    "emolecules",
}

logger = logging.getLogger("predownload_retro")


def _download_source(source: str, raw_root: Path) -> Path:
    if source == "uspto_50k":
        cfg = uspto_src.USPTOConfig(raw_dir=raw_root / "uspto", subset="50k", prefer_parquet=True)
        return uspto_src.download_uspto(cfg)
    if source == "uspto_full":
        cfg = uspto_src.USPTOConfig(raw_dir=raw_root / "uspto", subset="full")
        return uspto_src.download_uspto(cfg)
    if source == "uspto_llm":
        cfg = uspto_src.USPTOConfig(raw_dir=raw_root / "uspto", subset="llm")
        return uspto_src.download_uspto(cfg)
    if source == "zinc22":
        cfg = buyables_src.BuyablesConfig(raw_dir=raw_root / "buyables")
        return buyables_src.download_zinc22(cfg)
    if source == "enamine":
        cfg = buyables_src.BuyablesConfig(raw_dir=raw_root / "buyables")
        return buyables_src.download_enamine_bb(cfg)
    if source == "emolecules":
        cfg = buyables_src.BuyablesConfig(raw_dir=raw_root / "buyables")
        return buyables_src.download_emolecules(cfg)
    raise ValueError(f"unknown source {source!r}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path("rasyn/data/raw"),
        help="Root directory under which uspto/, buyables/, ord/ live.",
    )
    p.add_argument(
        "--sources",
        default="uspto_50k,uspto_full,zinc22,enamine,emolecules",
        help="Comma-separated list of sources to download.",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/retro_predownload_manifest.json"),
        help="Where to write the manifest JSON.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in ALL_SOURCES]
    if unknown:
        logger.error("unknown source(s): %s (known: %s)", unknown, sorted(ALL_SOURCES))
        return 2

    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    failures: list[tuple[str, str]] = []
    for source in sources:
        logger.info("downloading %s ...", source)
        try:
            path = _download_source(source, args.raw_root)
        except _dl.DownloadError as e:
            logger.error("[%s] FAILED: %s", source, e)
            failures.append((source, str(e)))
            continue
        size = path.stat().st_size
        sha = _dl.sha256_of(path)
        manifest[source] = {
            "path": str(path),
            "size_bytes": size,
            "size_human": _human(size),
            "sha256": sha,
        }
        logger.info("[%s] ok: %s (%s, sha256=%s)", source, path, _human(size), sha[:12])

    args.manifest.write_text(json.dumps(manifest, indent=2))
    logger.info("manifest -> %s", args.manifest)

    if failures:
        logger.error("=== FAILURES (%d) ===", len(failures))
        for src, err in failures:
            logger.error("  %s: %s", src, err)
        logger.error(
            "Refusing to claim success. Investigate the listed URLs and "
            "either restore them or add a mirror in rasyn/data/sources/."
        )
        return 1

    logger.info("=== all %d source(s) downloaded and validated ===", len(sources))
    return 0


def _human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PiB"


if __name__ == "__main__":
    sys.exit(main())
