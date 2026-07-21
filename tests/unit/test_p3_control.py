"""Safety and accounting tests for the long-running P3 controller."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

_CONTROL_PATH = Path(__file__).resolve().parents[2] / "scripts/p3_control.py"
_SPEC = importlib.util.spec_from_file_location("p3_control_under_test", _CONTROL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load scripts/p3_control.py for testing")
control = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(control)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_alive_treats_permission_denial_as_an_existing_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_signal(pid: int, signal_number: int) -> None:
        raise PermissionError

    monkeypatch.setattr(control.os, "kill", deny_signal)
    assert control._alive(1234)


def test_under_root_rejects_artifact_path_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="escapes"):
        control._under_root("../outside", label="artifact")


def test_status_uses_cumulative_heartbeat_after_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    current_path = tmp_path / "artifacts/p3/control/current.json"
    monkeypatch.setattr(control, "CURRENT", current_path)
    runtime = tmp_path / "artifacts/p3/control/run/qualification.runtime.yaml"
    raw = yaml.safe_load(Path("configs/experiments/p3/ci.yaml").read_text(encoding="utf-8"))
    runtime.parent.mkdir(parents=True)
    runtime.write_text(yaml.safe_dump(raw), encoding="utf-8")
    artifact_directory = tmp_path / "artifacts/runs/run"
    _write_json(
        artifact_directory / "registry.json",
        {"status": "stopped", "cells": [], "wall_clock_seconds": 105.0},
    )
    _write_json(
        artifact_directory / "heartbeat.json",
        {"suite_elapsed_seconds": 120.0, "updated_at": datetime.now(UTC).isoformat()},
    )
    _write_json(
        current_path,
        {
            "run_id": "run",
            "pid": 987_654,
            "started_at": (datetime.now(UTC) - timedelta(seconds=10)).isoformat(),
            "prior_wall_clock_seconds": 100.0,
            "runtime_config": str(runtime.relative_to(tmp_path)),
            "artifact_dir": str(artifact_directory.relative_to(tmp_path)),
        },
    )
    monkeypatch.setattr(control, "_alive", lambda pid: False)
    monkeypatch.setattr(control, "_process_matches_launch", lambda current: False)

    result = control.status()

    assert result["elapsed_seconds"] == pytest.approx(120.0)
    assert result["remaining_wall_clock_seconds"] == pytest.approx(7_080.0)


def test_status_stops_elapsed_clock_after_completed_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    current_path = tmp_path / "artifacts/p3/control/current.json"
    monkeypatch.setattr(control, "CURRENT", current_path)
    runtime = tmp_path / "artifacts/p3/control/run/full.runtime.yaml"
    raw = yaml.safe_load(Path("configs/experiments/p3/ci.yaml").read_text(encoding="utf-8"))
    runtime.parent.mkdir(parents=True)
    runtime.write_text(yaml.safe_dump(raw), encoding="utf-8")
    artifact_directory = tmp_path / "artifacts/runs/run"
    _write_json(
        artifact_directory / "registry.json",
        {"status": "completed", "cells": [], "wall_clock_seconds": 25.0},
    )
    _write_json(
        artifact_directory / "heartbeat.json",
        {"suite_elapsed_seconds": 20.0, "updated_at": datetime.now(UTC).isoformat()},
    )
    _write_json(
        current_path,
        {
            "run_id": "run",
            "pid": 987_654,
            "started_at": (datetime.now(UTC) - timedelta(days=2)).isoformat(),
            "prior_wall_clock_seconds": 0.0,
            "runtime_config": str(runtime.relative_to(tmp_path)),
            "artifact_dir": str(artifact_directory.relative_to(tmp_path)),
        },
    )
    monkeypatch.setattr(control, "_alive", lambda pid: False)
    monkeypatch.setattr(control, "_process_matches_launch", lambda current: False)

    result = control.status()

    assert result["elapsed_seconds"] == pytest.approx(25.0)


def test_resume_rejects_a_completed_matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text("profile: qualification\n", encoding="utf-8")
    artifact_directory = tmp_path / "artifacts/runs/run"
    _write_json(artifact_directory / "registry.json", {"status": "qualification_passed"})
    current = {
        "run_id": "run",
        "pid": 987_654,
        "git_commit": "abc",
        "runtime_config": "runtime.yaml",
        "runtime_config_sha256": control._sha256(runtime),
        "artifact_dir": "artifacts/runs/run",
    }
    monkeypatch.setattr(control, "_preflight", lambda for_resume: ("abc", {}))
    monkeypatch.setattr(control, "_current", lambda: current)
    monkeypatch.setattr(control, "_alive", lambda pid: False)
    monkeypatch.setattr(
        control,
        "_evidence_lock_hashes",
        lambda: {
            "qualification_lock_sha256": None,
            "ci_lock_sha256": None,
            "pilot_lock_sha256": None,
        },
    )

    with pytest.raises(RuntimeError, match="already complete"):
        control.resume()


def test_force_stop_refuses_a_reused_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    current = {"run_id": "run", "pid": 1234, "process_group": 1234}
    monkeypatch.setattr(control, "_current", lambda: current)
    monkeypatch.setattr(control, "_alive", lambda pid: True)
    monkeypatch.setattr(control, "_process_matches_launch", lambda launch: False)

    with pytest.raises(RuntimeError, match="does not match"):
        control.stop(force=True)


def test_freeze_qualification_requires_mps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    report = tmp_path / "run/qualification-report.json"
    _write_json(
        report,
        {
            "status": "PASSED",
            "qualification_only": True,
            "git_commit": "abc",
            "git_dirty": False,
            "device": "cpu",
        },
    )
    monkeypatch.setattr(
        control,
        "_git",
        lambda *arguments: "" if arguments[:2] == ("status", "--porcelain") else "abc",
    )

    with pytest.raises(RuntimeError, match="MPS"):
        control.freeze_qualification(report.parent)


def test_controller_reports_preflight_errors_as_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_start() -> dict[str, object]:
        raise RuntimeError("clean worktree required")

    monkeypatch.setattr(control, "start", fail_start)

    assert control.main(["start"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error == {
        "command": "start",
        "error": "clean worktree required",
        "error_type": "RuntimeError",
        "exit_code": 2,
    }
