from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_CONTROL_PATH = Path(__file__).resolve().parents[2] / "scripts/p5_control.py"
_SPEC = importlib.util.spec_from_file_location("p5_control_under_test", _CONTROL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load scripts/p5_control.py")
control = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(control)


def test_p5_profiles_are_device_scoped() -> None:
    cuda = control._load_profile(control.PROFILE_CONFIGS["pilot-cuda"])
    mps = control._load_profile(control.PROFILE_CONFIGS["pilot-mps"])

    assert cuda.device == "cuda"
    assert str(cuda.qualification_report).replace("\\", "/").startswith("artifacts/p5-cuda/")
    assert mps.device == "mps"
    assert str(mps.qualification_report).replace("\\", "/").startswith("artifacts/p5-mps/")


def test_p5_status_distinguishes_running_from_completed_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_path = tmp_path / "artifacts/p5/control/current.json"
    directory = tmp_path / "artifacts/runs/pilot"
    directory.mkdir(parents=True)
    current_path.parent.mkdir(parents=True)
    current_path.write_text(
        json.dumps(
            {
                "run_id": "pilot",
                "profile": "pilot-cuda",
                "pid": 123,
                "artifact_dir": "artifacts/runs/pilot",
                "runtime_config": "runtime.yaml",
            }
        ),
        encoding="utf-8",
    )
    (directory / "registry.json").write_text(
        json.dumps(
            {
                "status": "running",
                "cells": [
                    {"candidate_id": "preset-0", "status": "COMPLETED"},
                    {"candidate_id": "preset-1", "status": "RUNNING"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "CURRENT", current_path)
    monkeypatch.setattr(control, "_alive", lambda pid: True)
    monkeypatch.setattr(control, "_process_matches_launch", lambda value: True)

    result = control.status()

    assert result["suite_status"] == "running"
    assert result["completed_cells"] == 1
    assert result["current_candidate"] == "preset-1"
    assert result["terminal"] is False


def test_p5_status_marks_passed_pilot_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_path = tmp_path / "artifacts/p5/control/current.json"
    directory = tmp_path / "artifacts/runs/pilot"
    directory.mkdir(parents=True)
    current_path.parent.mkdir(parents=True)
    current_path.write_text(
        json.dumps(
            {
                "run_id": "pilot",
                "profile": "pilot-cuda",
                "pid": 123,
                "artifact_dir": "artifacts/runs/pilot",
                "runtime_config": "runtime.yaml",
            }
        ),
        encoding="utf-8",
    )
    (directory / "registry.json").write_text(
        json.dumps({"status": "pilot_passed", "cells": []}), encoding="utf-8"
    )
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "CURRENT", current_path)
    monkeypatch.setattr(control, "_alive", lambda pid: False)
    monkeypatch.setattr(control, "_process_matches_launch", lambda value: False)

    result = control.status()

    assert result["terminal"] is True
    assert result["resume_allowed"] is False
