from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from neuromorphic.fly import DatasetFile, DatasetIntegrityError, fetch, verify_file
from neuromorphic.fly.datasets import file_digest, load_manifest

_PAYLOAD = b"pre_root_id,post_root_id,syn_count\n1,2,5\n"


def _spec(payload: bytes = _PAYLOAD, **overrides: Any) -> DatasetFile:
    fields: dict[str, Any] = {
        "dataset": "toy",
        "version": "1",
        "file_name": "edges.csv",
        "url": "https://example.org/edges.csv",
        "size_bytes": len(payload),
        "checksum_algorithm": "sha256",
        "checksum": hashlib.sha256(payload).hexdigest(),
        "license": "CC-BY-4.0",
    }
    fields.update(overrides)
    return DatasetFile(**fields)


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> list[str]:
    requested: list[str] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        requested.append(request.full_url)
        return _Response(payload)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return requested


def test_manifest_rejects_unsafe_entries() -> None:
    with pytest.raises(ValueError, match="https"):
        _spec(url="http://example.org/edges.csv")
    with pytest.raises(ValueError, match="bare file name"):
        _spec(file_name="../edges.csv")
    with pytest.raises(ValueError, match="md5"):
        _spec(checksum_algorithm="md5")


def test_fetch_verifies_and_reuses_local_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = _serve(monkeypatch, _PAYLOAD)
    spec = _spec()
    path = fetch(spec, tmp_path)
    assert path == tmp_path / "toy" / "1" / "edges.csv"
    assert path.read_bytes() == _PAYLOAD
    assert fetch(spec, tmp_path) == path
    assert requested == ["https://example.org/edges.csv"]


def test_fetch_discards_corrupt_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corrupt = _PAYLOAD.replace(b"5", b"6")
    _serve(monkeypatch, corrupt)
    with pytest.raises(DatasetIntegrityError, match="sha256 mismatch"):
        fetch(_spec(), tmp_path)
    assert list((tmp_path / "toy" / "1").iterdir()) == []


def test_fetch_stops_on_oversized_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, _PAYLOAD + b"extra")
    with pytest.raises(DatasetIntegrityError, match="exceeds"):
        fetch(_spec(), tmp_path)


def test_verify_file_reports_size_and_md5_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "edges.csv"
    path.write_bytes(_PAYLOAD)
    md5 = _spec(checksum_algorithm="md5", checksum=hashlib.md5(_PAYLOAD).hexdigest())
    verify_file(path, md5)
    with pytest.raises(DatasetIntegrityError, match="bytes"):
        verify_file(path, _spec(size_bytes=len(_PAYLOAD) + 1))
    with pytest.raises(DatasetIntegrityError, match="missing"):
        verify_file(tmp_path / "absent.csv", md5)


def test_git_blob_sha1_matches_git_object_id(tmp_path: Path) -> None:
    path = tmp_path / "edges.csv"
    path.write_bytes(_PAYLOAD)
    blob = hashlib.sha1(b"blob %d\0" % len(_PAYLOAD) + _PAYLOAD).hexdigest()
    verify_file(path, _spec(checksum_algorithm="git-blob-sha1", checksum=blob))
    # Known object ID of an empty git blob.
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    assert file_digest(empty, "git-blob-sha1") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_shipped_manifest_parses_and_sets_are_nested() -> None:
    manifest = Path(__file__).resolve().parents[2] / "configs" / "fly" / "datasets.yaml"
    sets = {name: load_manifest(manifest, name) for name in ("T0-min", "T0-paper", "T0-full")}
    names = {key: {spec.file_name for spec in specs} for key, specs in sets.items()}
    assert names["T0-min"] == {"Connectivity_783.parquet", "Completeness_783.csv"}
    assert names["T0-min"] < names["T0-paper"] < names["T0-full"]
    assert all(spec.license for specs in sets.values() for spec in specs)
    with pytest.raises(ValueError, match="no files"):
        load_manifest(manifest, "T9")
