"""Safety and accounting tests for the long-running P4 controller."""

from __future__ import annotations

import importlib.util
import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

_CONTROL_PATH = Path(__file__).resolve().parents[2] / "scripts/p4_control.py"
_SPEC = importlib.util.spec_from_file_location("p4_control_under_test", _CONTROL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load scripts/p4_control.py for testing")
control = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(control)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_status_uses_cumulative_heartbeat_after_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    current_path = tmp_path / "artifacts/p4/control/current.json"
    monkeypatch.setattr(control, "CURRENT", current_path)
    runtime = tmp_path / "artifacts/p4/control/run/qualification.runtime.yaml"
    raw = yaml.safe_load(
        Path("configs/experiments/p4/qualification.yaml").read_text(encoding="utf-8")
    )
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


def test_start_requires_mechanism_lock_for_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "CONTROL", tmp_path / "artifacts/p4/control")
    monkeypatch.setattr(control, "CURRENT", tmp_path / "artifacts/p4/control/current.json")
    monkeypatch.setattr(
        control, "QUALIFICATION_LOCK", tmp_path / "artifacts/p4/qualification-lock.json"
    )
    monkeypatch.setattr(control, "PILOT_LOCK", tmp_path / "artifacts/p4/pilot-lock.json")
    monkeypatch.setattr(control, "MECHANISM_LOCK", tmp_path / "artifacts/p4/mechanism-lock.json")
    qualification_report = tmp_path / "artifacts/runs/qual/qualification-report.json"
    pilot_selection = tmp_path / "artifacts/runs/pilot/pilot-selection.json"
    _write_json(qualification_report, {"status": "PASSED"})
    _write_json(pilot_selection, {"status": "PASSED"})
    _write_json(
        control.QUALIFICATION_LOCK,
        {
            "status": "PASSED",
            "expected_git_commit": "abc123",
            "qualification_report": str(qualification_report.relative_to(tmp_path)),
            "qualification_report_sha256": control._sha256(qualification_report),
        },
    )
    _write_json(
        control.PILOT_LOCK,
        {
            "status": "PASSED",
            "git_commit": "abc123",
            "pilot_selection": str(pilot_selection.relative_to(tmp_path)),
            "pilot_selection_sha256": control._sha256(pilot_selection),
            "selected_preset": "preset-3",
            "optimizer": {
                "learning_rate": 0.0003,
                "weight_decay": 0.01,
                "temporal_loss_weight": 0.1,
                "gradient_clip_norm": 1.0,
            },
        },
    )
    monkeypatch.setattr(control, "_background_preflight", lambda *, config, for_resume: "abc123")

    with pytest.raises(RuntimeError, match="mechanism lock is missing"):
        control.start("full")


def test_mechanism_start_requires_qualification_and_pilot_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "CONTROL", tmp_path / "artifacts/p4/control")
    monkeypatch.setattr(control, "CURRENT", tmp_path / "artifacts/p4/control/current.json")
    monkeypatch.setattr(
        control, "QUALIFICATION_LOCK", tmp_path / "artifacts/p4/qualification-lock.json"
    )
    monkeypatch.setattr(control, "PILOT_LOCK", tmp_path / "artifacts/p4/pilot-lock.json")
    monkeypatch.setattr(control, "MECHANISM_LOCK", tmp_path / "artifacts/p4/mechanism-lock.json")
    monkeypatch.setattr(control, "_background_preflight", lambda *, config, for_resume: "abc123")

    with pytest.raises(RuntimeError, match="qualification lock is missing"):
        control.start("mechanism")


def test_resume_rejects_changed_evidence_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "CURRENT", tmp_path / "artifacts/p4/control/current.json")
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(
        Path("configs/experiments/p4/mechanism.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    current = {
        "run_id": "run",
        "pid": 987_654,
        "git_commit": "abc",
        "runtime_config": "runtime.yaml",
        "runtime_config_sha256": control._sha256(runtime),
        "artifact_dir": "artifacts/runs/run",
        "qualification_lock_sha256": "unchanged",
        "pilot_lock_sha256": "pilot",
        "mechanism_lock_sha256": "mechanism",
    }
    monkeypatch.setattr(control, "_current", lambda: current)
    monkeypatch.setattr(control, "_background_preflight", lambda *, config, for_resume: "abc")

    monkeypatch.setattr(
        control,
        "_evidence_lock_hashes",
        lambda: {
            "qualification_lock_sha256": "different",
            "pilot_lock_sha256": "pilot",
            "mechanism_lock_sha256": "mechanism",
        },
    )

    with pytest.raises(RuntimeError, match="qualification_lock_sha256"):
        control.resume()


def test_foreground_mode_is_only_allowed_for_qualification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control, "_git", lambda *arguments: "abc123")

    with pytest.raises(RuntimeError, match="only allowed for qualification"):
        control.start("pilot", foreground=True)


def test_controller_reports_preflight_errors_as_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_start(profile: str, *, foreground: bool = False) -> dict[str, object]:
        raise RuntimeError("mechanism lock is missing")

    monkeypatch.setattr(control, "start", fail_start)

    assert control.main(["start", "full"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error == {
        "command": "start",
        "error": "mechanism lock is missing",
        "error_type": "RuntimeError",
        "exit_code": 2,
    }


def test_nonqualification_preflight_requires_successful_ci_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "CURRENT", tmp_path / "current.json")
    monkeypatch.setattr(control, "CI_LOCK", tmp_path / "ci-lock.json")

    def fake_git(*arguments: str) -> str:
        if arguments == ("status", "--porcelain"):
            return ""
        if arguments == ("rev-parse", "HEAD"):
            return "abc123"
        if arguments == ("rev-parse", "origin/main"):
            return "abc123"
        raise AssertionError(arguments)

    monkeypatch.setattr(control, "_git", fake_git)
    config = control._profile_config("pilot")
    with pytest.raises(RuntimeError, match="successful GitHub Actions record"):
        control._background_preflight(config=config, for_resume=False)


def test_record_ci_freezes_latest_success_for_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "CI_LOCK", tmp_path / "artifacts/p4/ci-lock.json")

    def fake_git(*arguments: str) -> str:
        values = {
            ("status", "--porcelain"): "",
            ("rev-parse", "HEAD"): "abc123",
            ("rev-parse", "origin/main"): "abc123",
            ("remote", "get-url", "origin"): "https://github.com/example/repo.git",
        }
        return values[arguments]

    payload = {
        "workflow_runs": [
            {
                "id": 41,
                "head_sha": "abc123",
                "conclusion": "success",
                "path": ".github/workflows/ci.yml",
                "html_url": "https://github.com/example/repo/actions/runs/41",
                "updated_at": "2026-07-21T00:00:00Z",
            }
        ]
    }
    monkeypatch.setattr(control, "_git", fake_git)
    monkeypatch.setattr(
        control.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps(payload).encode()),
    )

    result = control.record_ci()

    assert result["run_id"] == 41
    assert result["git_commit"] == "abc123"
    assert json.loads(control.CI_LOCK.read_text(encoding="utf-8"))["conclusion"] == "success"
