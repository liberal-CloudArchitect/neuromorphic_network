"""Checksummed, explicitly approved retrieval of public fly connectome files.

Nothing in this module downloads implicitly.  Callers pass a manifest entry whose
URL, size and checksum were reviewed in ``docs/fly/dataset_inventory.md``; the
file is streamed to a ``.part`` sibling, verified, and only then atomically
renamed into place.  Local data lives under the git-ignored ``data/fly`` tree.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ChecksumAlgorithm = Literal["md5", "sha256", "git-blob-sha1"]

_CHUNK_BYTES = 1 << 20
_ALLOWED_SCHEMES = ("https://",)
_DIGEST_LENGTHS = {"md5": 32, "sha256": 64, "git-blob-sha1": 40}


@dataclass(frozen=True, slots=True)
class DatasetFile:
    """One reviewed file of a public dataset release."""

    dataset: str
    version: str
    file_name: str
    url: str
    size_bytes: int
    checksum_algorithm: ChecksumAlgorithm
    checksum: str
    license: str

    def __post_init__(self) -> None:
        if not all((self.dataset, self.version, self.file_name, self.license)):
            raise ValueError("dataset, version, file_name and license must be non-empty")
        if Path(self.file_name).name != self.file_name or self.file_name in {".", ".."}:
            raise ValueError("file_name must be a bare file name")
        if not self.url.startswith(_ALLOWED_SCHEMES):
            raise ValueError("dataset URLs must use https")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        expected_length = _DIGEST_LENGTHS.get(self.checksum_algorithm)
        if expected_length is None:
            raise ValueError(f"unsupported checksum algorithm: {self.checksum_algorithm}")
        digest = self.checksum.lower()
        if len(digest) != expected_length or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"checksum is not a valid {self.checksum_algorithm} hex digest")

    def local_path(self, root: Path) -> Path:
        return root / self.dataset / self.version / self.file_name


class DatasetIntegrityError(RuntimeError):
    """Raised when a local or downloaded file does not match its manifest."""


def file_digest(path: Path, algorithm: ChecksumAlgorithm) -> str:
    """Hex digest of ``path``.

    ``git-blob-sha1`` is the object ID GitHub publishes for repository files:
    SHA-1 over ``b"blob <size>\\0"`` followed by the file content.
    """

    if algorithm == "git-blob-sha1":
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(f"blob {path.stat().st_size}\0".encode())
    else:
        digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, spec: DatasetFile) -> None:
    """Raise unless ``path`` has the manifest size and checksum."""

    if not path.is_file():
        raise DatasetIntegrityError(f"missing dataset file: {path}")
    size = path.stat().st_size
    if size != spec.size_bytes:
        raise DatasetIntegrityError(
            f"{spec.file_name}: expected {spec.size_bytes} bytes, found {size}"
        )
    actual = file_digest(path, spec.checksum_algorithm)
    if actual != spec.checksum.lower():
        raise DatasetIntegrityError(
            f"{spec.file_name}: {spec.checksum_algorithm} mismatch ({actual})"
        )


def fetch(spec: DatasetFile, root: Path, *, timeout_seconds: float = 60.0) -> Path:
    """Download ``spec`` into ``root`` unless a verified copy already exists."""

    target = spec.local_path(root)
    if target.exists():
        verify_file(target, spec)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(spec.url, headers={"User-Agent": "neuromorphic-fly/0"})
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout_seconds) as response,
            partial.open("wb") as handle,
        ):
            written = 0
            while chunk := response.read(_CHUNK_BYTES):
                written += len(chunk)
                if written > spec.size_bytes:
                    raise DatasetIntegrityError(f"{spec.file_name}: response exceeds manifest size")
                handle.write(chunk)
        verify_file(partial, spec)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, target)
    return target


def load_manifest(path: Path, download_set: str) -> list[DatasetFile]:
    """Return the reviewed files of ``download_set`` from a manifest YAML."""

    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "fly-datasets-v1":
        raise ValueError("unsupported dataset manifest schema")
    selected = [
        DatasetFile(**{key: value for key, value in entry.items() if key != "sets"})
        for entry in document["files"]
        if download_set in entry["sets"]
    ]
    if not selected:
        raise ValueError(f"manifest has no files in set {download_set!r}")
    return selected
