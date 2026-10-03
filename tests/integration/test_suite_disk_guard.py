"""The frozen 20 GiB free-disk guard still stops P3/P4 suites before any cell runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from neuromorphic.training import p4_suite
from neuromorphic.training.p3_config import load_p3_suite_config
from neuromorphic.training.p3_suite import execute_p3_suite
from neuromorphic.training.p4_config import P4SuiteConfig

_LOW_FREE_BYTES = 19 * 1024**3


def _low_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    def disk_usage(path: object) -> shutil._ntuple_diskusage:
        del path
        return shutil._ntuple_diskusage(100 * 1024**3, 81 * 1024**3, _LOW_FREE_BYTES)

    monkeypatch.setattr(shutil, "disk_usage", disk_usage)


def _never_build(*args: object, **kwargs: object) -> object:
    raise AssertionError("no cell may start below the free-disk floor")


def test_p3_suite_stops_below_free_disk_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_p3_suite_config(Path("configs/experiments/p3/ci.yaml")).model_copy(
        update={"output_root": tmp_path, "run_id": "disk-guard"}
    )
    monkeypatch.setattr("neuromorphic.training.p3_suite._build_model", _never_build)
    _low_disk(monkeypatch)

    summary = execute_p3_suite(config)

    assert summary["status"] == "resource_limit"


def test_p4_suite_stops_below_free_disk_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = P4SuiteConfig.model_validate(
        {
            "schema_version": "p4-suite-v1",
            "protocol_version": "p4-protocol-v2",
            "profile": "qualification",
            "qualification_only": True,
            "device": "cpu",
            "output_root": str(tmp_path),
            "control_root": str(tmp_path / "control"),
            "run_id": "disk-guard",
            "seeds": [7],
            "data": {"train": 64, "validation": 32, "analysis": 32, "test": 32, "ood": 32},
            "budget": {
                "batch_size": 8,
                "shared_steps_per_task": 4,
                "per_task_steps": 4,
                "continual_steps_per_stage": 2,
                "validation_interval": 2,
                "checkpoint_interval": 2,
                "patience": 10,
                "min_delta": 0.001,
                "wall_clock_hours": 2,
                "bootstrap_samples": 200,
            },
        }
    )
    monkeypatch.setattr(p4_suite, "_build_model", _never_build)
    _low_disk(monkeypatch)

    p4_suite.run_p4_suite(config)

    # Qualification evidence then marks the incomplete run as failed; the guard
    # itself is visible as the resource-limited first cell.
    registry = json.loads((tmp_path / "disk-guard" / "registry.json").read_text(encoding="utf-8"))
    assert registry["completed_cells"] == 0
    assert registry["resource_limited_cells"] == 1
    assert registry["cells"][0]["status"] == "RESOURCE_LIMIT"
