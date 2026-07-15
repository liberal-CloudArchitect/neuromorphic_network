"""Run manifest creation and artifact checksums."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

RUN_MANIFEST_SCHEMA_VERSION = "run-manifest-v1"


def _git_output(*arguments: str) -> str:
    result = subprocess.run(["git", *arguments], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def current_git_state() -> tuple[str, bool]:
    """Return current commit and whether tracked files are dirty."""
    return _git_output("rev-parse", "HEAD"), bool(_git_output("status", "--porcelain"))


def file_sha256(path: Path) -> str:
    """Return a SHA-256 digest for an artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    run_id: str,
    seed: int,
    device: torch.device,
    task: dict[str, Any],
    model: dict[str, Any],
    optimizer: dict[str, Any],
    data: dict[str, Any],
    budget: dict[str, Any],
    parameters: int,
    estimated_macs: int,
    mac_coverage: float,
    unsupported_parameters: tuple[str, ...],
) -> dict[str, Any]:
    """Build a complete JSON-compatible run manifest."""
    commit, dirty = current_git_state()
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "failure": None,
        "git": {"commit": commit, "dirty": dirty},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "argv": sys.argv,
        },
        "device": str(device),
        "seed": seed,
        "task": task,
        "model": model,
        "optimizer": optimizer,
        "data": data,
        "budget": budget,
        "cost": {
            "trainable_parameters": parameters,
            "estimated_macs_per_sequence": estimated_macs,
            "mac_profiler_coverage": mac_coverage,
            "unsupported_parameters": list(unsupported_parameters),
            "latency_ms": None,
            "wall_clock_seconds": None,
        },
        "artifacts": {},
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a manifest using canonical, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
