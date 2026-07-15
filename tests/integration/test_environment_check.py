from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_environment.py"
REQUIRED_KEYS = {
    "python_version",
    "torch_version",
    "platform",
    "accelerator",
    "smoke_ok",
}


def _run_check(requirement: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--require", requirement],
        check=False,
        capture_output=True,
        cwd=ROOT,
        text=True,
    )


def test_cpu_forward_backward_smoke() -> None:
    result = _run_check("cpu")
    assert result.returncode == 0, result.stderr or result.stdout
    report = json.loads(result.stdout)
    assert report.keys() >= REQUIRED_KEYS
    assert report["python_version"].startswith("3.12.")
    assert report["torch_version"].startswith("2.12.1")
    assert report["accelerator"] == "cpu"
    assert report["smoke_ok"] is True


def test_mps_forward_backward_smoke_when_available() -> None:
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        pytest.skip("MPS is not available on this host")
    result = _run_check("mps")
    assert result.returncode == 0, result.stderr or result.stdout
    report = json.loads(result.stdout)
    assert report["accelerator"] == "mps"
    assert report["smoke_ok"] is True


def test_required_unavailable_mps_returns_failure() -> None:
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        pytest.skip("This assertion only applies to hosts without MPS")
    result = _run_check("mps")
    assert result.returncode != 0
    report = json.loads(result.stdout)
    assert report["smoke_ok"] is False
    assert report["error"]
