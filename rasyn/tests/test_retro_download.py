"""Unit tests for rasyn.data.sources._download.

Covers the failure modes that caused the R-1 sbatch crash:
  - 0-byte responses must not be cached as fake archives.
  - HTML responses with .zip extension must be rejected.
  - Validation runs on existing cached file (idempotent re-use).
  - Fallback URL list is tried in order.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from rasyn.data.sources import _download as _dl


def _make_valid_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("hello.csv", "rxn_smiles\nCC>>CCO\n")


# ----- _validate_min_size -----

def test_validate_min_size_rejects_zero_byte(tmp_path: Path) -> None:
    p = tmp_path / "empty.zip"
    p.write_bytes(b"")
    with pytest.raises(_dl.DownloadError, match="size"):
        _dl._validate_min_size(p, min_bytes=1)


def test_validate_min_size_accepts_above_threshold(tmp_path: Path) -> None:
    p = tmp_path / "big.zip"
    p.write_bytes(b"x" * 2048)
    _dl._validate_min_size(p, min_bytes=1024)  # no raise


# ----- _validate_zip -----

def test_validate_zip_rejects_html(tmp_path: Path) -> None:
    p = tmp_path / "fake.zip"
    p.write_bytes(b"<!DOCTYPE html>\n<html><body>Not found</body></html>")
    with pytest.raises(_dl.DownloadError, match="not a valid zip"):
        _dl._validate_zip(p)


def test_validate_zip_accepts_real_zip(tmp_path: Path) -> None:
    p = tmp_path / "real.zip"
    _make_valid_zip(p)
    _dl._validate_zip(p)  # no raise


# ----- _run_validators -----

def test_run_validators_min_size_first(tmp_path: Path) -> None:
    p = tmp_path / "f.zip"
    p.write_bytes(b"")
    with pytest.raises(_dl.DownloadError, match="size"):
        _dl._run_validators(p, kind="zip", min_bytes=10, extra=None)


def test_run_validators_kind_validator_runs(tmp_path: Path) -> None:
    p = tmp_path / "f.zip"
    p.write_bytes(b"not a real zip but bigger than 1 byte" * 50)
    with pytest.raises(_dl.DownloadError, match="not a valid zip"):
        _dl._run_validators(p, kind="zip", min_bytes=1, extra=None)


# ----- download_validated -----

def test_download_validated_idempotent_existing_valid(tmp_path: Path, monkeypatch) -> None:
    """Cached valid file is reused without network calls."""
    target = tmp_path / "real.zip"
    _make_valid_zip(target)

    # If _stream_to_file is called, we know the cache wasn't honoured.
    def boom(url, dest, **kw):
        raise AssertionError("network should not have been touched")

    monkeypatch.setattr(_dl, "_stream_to_file", boom)
    result = _dl.download_validated(
        ["http://nowhere.invalid/x.zip"],
        target,
        kind="zip",
        min_bytes=1,
    )
    assert result == target


def test_download_validated_replaces_invalid_cached_file(tmp_path: Path, monkeypatch) -> None:
    """Existing zero-byte cache is wiped and re-downloaded."""
    target = tmp_path / "x.zip"
    target.write_bytes(b"")  # the original bug: 0-byte cached file

    def fake_stream(url, dest, **kw):
        _make_valid_zip(dest)

    monkeypatch.setattr(_dl, "_stream_to_file", fake_stream)
    result = _dl.download_validated(
        ["http://example/x.zip"],
        target,
        kind="zip",
        min_bytes=10,
    )
    assert result.exists()
    assert zipfile.is_zipfile(result)


def test_download_validated_falls_through_to_second_url(tmp_path: Path, monkeypatch) -> None:
    """First URL returns 0 bytes → second URL produces real zip → success."""
    target = tmp_path / "x.zip"
    calls: list[str] = []

    def fake_stream(url, dest, **kw):
        calls.append(url)
        if "dead" in url:
            dest.write_bytes(b"")  # mimic figshare's 0-byte response
        else:
            _make_valid_zip(dest)

    monkeypatch.setattr(_dl, "_stream_to_file", fake_stream)
    # First URL "dead.example/x.zip" → 0 bytes; should be rejected, retried,
    # then fall through to "live.example/x.zip".
    result = _dl.download_validated(
        ["http://dead.example/x.zip", "http://live.example/x.zip"],
        target,
        kind="zip",
        min_bytes=10,
        retries=1,
        backoff_s=0,
    )
    assert result.exists()
    assert zipfile.is_zipfile(result)
    assert "dead.example" in calls[0]
    assert any("live.example" in c for c in calls)


def test_download_validated_raises_when_all_urls_fail(tmp_path: Path, monkeypatch) -> None:
    """All URLs return 0 bytes → DownloadError after all retries."""
    target = tmp_path / "x.zip"

    def fake_stream(url, dest, **kw):
        dest.write_bytes(b"")

    monkeypatch.setattr(_dl, "_stream_to_file", fake_stream)
    with pytest.raises(_dl.DownloadError, match="failed to download"):
        _dl.download_validated(
            ["http://a.example/x.zip", "http://b.example/x.zip"],
            target,
            kind="zip",
            min_bytes=10,
            retries=1,
            backoff_s=0,
        )
    # No partial file left over.
    assert not target.with_suffix(".zip.partial").exists()
    assert not target.exists()


def test_download_validated_atomic_rename(tmp_path: Path, monkeypatch) -> None:
    """A failed validation must NOT leave the target path populated."""
    target = tmp_path / "x.zip"

    state = {"count": 0}

    def fake_stream(url, dest, **kw):
        state["count"] += 1
        if state["count"] == 1:
            # First attempt: produce garbage. Validator should reject and unlink.
            dest.write_bytes(b"xxx")
        else:
            _make_valid_zip(dest)

    monkeypatch.setattr(_dl, "_stream_to_file", fake_stream)
    # 2 retries: first writes garbage, second writes real zip.
    result = _dl.download_validated(
        ["http://x.example/x.zip"],
        target,
        kind="zip",
        min_bytes=1,
        retries=2,
        backoff_s=0,
    )
    assert result.exists()
    assert zipfile.is_zipfile(result)


def test_sha256_stable(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello rasyn retro")
    s = _dl.sha256_of(p)
    assert len(s) == 64
    # deterministic
    assert s == _dl.sha256_of(p)
