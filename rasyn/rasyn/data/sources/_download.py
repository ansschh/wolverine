"""Validated, retrying HTTP download helper for raw data sources.

The previous `urllib.request.urlretrieve` calls had two failure modes that
silently corrupted the raw-data cache:

  1. Dead URLs that respond with 0-byte 404 bodies. `urlretrieve` wrote the
     empty body to disk as the target archive; the next idempotent run
     checked `target.exists()`, returned True, and the curation pipeline
     crashed downstream with `zipfile.BadZipFile`.

  2. HTML redirect / login-wall pages saved with a `.zip` extension. Same
     downstream failure pattern.

This helper fixes both by writing to a `.partial` sibling, validating after
download (size + content-shape callback), and only renaming on success.

It also uses `requests` with a real User-Agent and follows redirects
explicitly so figshare / Zenodo / GitHub all behave.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import time
import zipfile
import tarfile
from pathlib import Path
from typing import Callable, Iterable

DEFAULT_USER_AGENT = "rasyn-retro/1.0 (+https://github.com/ansschh/wolverine)"
DEFAULT_TIMEOUT_S = 600
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_S = 4


class DownloadError(RuntimeError):
    """Raised when a download cannot be validated after all retries."""


def _stream_to_file(url: str, dest: Path, *, timeout_s: int, user_agent: str) -> None:
    """Stream a single URL to `dest` (no validation, no retries)."""
    import requests  # local import so unit tests can monkeypatch

    headers = {"User-Agent": user_agent, "Accept": "*/*"}
    with requests.get(url, stream=True, allow_redirects=True, timeout=timeout_s, headers=headers) as r:
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB
                if chunk:
                    fh.write(chunk)


def _validate_zip(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise DownloadError(f"{path}: not a valid zip archive")


def _validate_tar_gz(path: Path) -> None:
    try:
        with tarfile.open(path, "r:gz") as tf:
            tf.next()  # touch first member; full open confirms gzip + tar header
    except (tarfile.TarError, OSError) as e:
        raise DownloadError(f"{path}: not a valid tar.gz archive ({e})") from e


def _validate_gzip(path: Path) -> None:
    import gzip
    try:
        with gzip.open(path, "rb") as fh:
            fh.read(1)
    except (OSError, EOFError) as e:
        raise DownloadError(f"{path}: not a valid gzip file ({e})") from e


def _validate_parquet(path: Path) -> None:
    try:
        import pyarrow.parquet as pq
        pq.read_metadata(path)
    except Exception as e:  # noqa: BLE001 -- want any parquet failure to fail validation
        raise DownloadError(f"{path}: not a valid parquet file ({e})") from e


def _validate_min_size(path: Path, min_bytes: int) -> None:
    sz = path.stat().st_size
    if sz < min_bytes:
        raise DownloadError(f"{path}: size {sz} bytes < required minimum {min_bytes}")


VALIDATORS: dict[str, Callable[[Path], None]] = {
    "zip": _validate_zip,
    "tar.gz": _validate_tar_gz,
    "gz": _validate_gzip,
    "parquet": _validate_parquet,
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_validated(
    url: str | Iterable[str],
    dest: Path,
    *,
    kind: str | None = None,
    min_bytes: int = 1024,
    retries: int = DEFAULT_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    user_agent: str = DEFAULT_USER_AGENT,
    extra_validator: Callable[[Path], None] | None = None,
    overwrite_existing_invalid: bool = True,
) -> Path:
    """Download `url` to `dest`, validate, then atomically rename into place.

    If `url` is iterable, each URL is tried in order. The first one that
    yields a valid file wins. This is the mechanism for source-rot
    resilience (figshare ID went stale → fall through to the HF mirror).

    Idempotent: if `dest` already exists and validates, return it without
    re-downloading. If it exists but fails validation and
    `overwrite_existing_invalid=True`, the stale file is removed and we
    re-download.

    Args:
        url: single URL or list of candidate URLs (tried in order).
        dest: target on-disk path. Download writes to dest.with_suffix(... + '.partial')
            and atomically renames after validation.
        kind: one of 'zip' | 'tar.gz' | 'gz' | 'parquet' | None. Selects the
            built-in content validator. None = skip content validation
            (still does size minimum).
        min_bytes: minimum acceptable size in bytes (default 1 KiB — catches
            0-byte responses and tiny error pages).
        retries: per-URL retry count.
        backoff_s: initial backoff (doubled each retry).
        timeout_s: per-request timeout.
        user_agent: HTTP User-Agent header (some hosts deny default Python UA).
        extra_validator: optional callable that raises on invalid content.
        overwrite_existing_invalid: if `dest` exists but fails validation,
            remove and re-download.

    Returns:
        Path to the validated file.

    Raises:
        DownloadError: all URLs exhausted without producing a valid file.
    """
    urls: list[str] = [url] if isinstance(url, str) else list(url)
    if not urls:
        raise ValueError("at least one URL required")

    # Idempotent: existing file passes validation -> done.
    if dest.exists():
        try:
            _run_validators(dest, kind=kind, min_bytes=min_bytes, extra=extra_validator)
            return dest
        except DownloadError:
            if not overwrite_existing_invalid:
                raise
            dest.unlink()

    partial = dest.with_suffix(dest.suffix + ".partial")
    last_err: Exception | None = None

    for url_idx, candidate in enumerate(urls):
        for attempt in range(retries):
            try:
                if partial.exists():
                    partial.unlink()
                _stream_to_file(candidate, partial, timeout_s=timeout_s, user_agent=user_agent)
                _run_validators(partial, kind=kind, min_bytes=min_bytes, extra=extra_validator)
                # Validation passed: atomic rename.
                os.replace(partial, dest)
                return dest
            except Exception as e:  # noqa: BLE001 -- want broad catch for retry
                last_err = e
                if partial.exists():
                    partial.unlink()
                if attempt < retries - 1:
                    time.sleep(backoff_s * (2 ** attempt))

    raise DownloadError(
        f"failed to download {dest.name} from any of {len(urls)} candidate URL(s); "
        f"last error: {last_err}"
    )


def _run_validators(
    path: Path,
    *,
    kind: str | None,
    min_bytes: int,
    extra: Callable[[Path], None] | None,
) -> None:
    _validate_min_size(path, min_bytes)
    if kind is not None:
        validator = VALIDATORS.get(kind)
        if validator is None:
            raise ValueError(f"unknown kind {kind!r}; expected one of {sorted(VALIDATORS)}")
        validator(path)
    if extra is not None:
        extra(path)
