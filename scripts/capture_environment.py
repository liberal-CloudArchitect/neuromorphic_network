#!/usr/bin/env python3
"""Emit a sanitized, reproducible environment manifest as JSON."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections import Counter
from datetime import UTC, datetime
from typing import TypedDict, cast

import torch

import neuromorphic


class CondaPackage(TypedDict, total=False):
    channel: str


def _conda_channel_counts() -> dict[str, int]:
    """Count installed Conda package origins without exposing channel URLs."""
    completed = subprocess.run(
        ["conda", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    packages = cast(list[CondaPackage], json.loads(completed.stdout))
    counts = Counter(package.get("channel", "unknown") for package in packages)
    return dict(sorted(counts.items()))


def build_manifest() -> dict[str, object]:
    """Build a machine-readable manifest without user-specific absolute paths."""
    mps = getattr(torch.backends, "mps", None)
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "conda_channel_counts": _conda_channel_counts(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "package_version": neuromorphic.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "accelerators": {
            "cpu": True,
            "mps_built": bool(mps is not None and mps.is_built()),
            "mps_available": bool(mps is not None and mps.is_available()),
            "cuda_available": torch.cuda.is_available(),
        },
    }


def main() -> None:
    print(json.dumps(build_manifest(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
