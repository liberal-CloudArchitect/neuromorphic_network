"""Safety and accounting tests for the long-running P4 controller."""

from __future__ import annotations

import importlib.util
import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

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


def test_terminal_resource_limit_reports_restart_not_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    current_path = tmp_path / "artifacts/p4/control/current.json"
    monkeypatch.setattr(control, "CURRENT", current_path)
    runtime = tmp_path / "runtime.yaml"
    raw = yaml.safe_load(
        Path("configs/experiments/p4/qualification.yaml").read_text(encoding="utf-8")
    )
    runtime.write_text(yaml.safe_dump(raw), encoding="utf-8")
    artifact_directory = tmp_path / "artifacts/runs/run"
    _write_json(
        artifact_directory / "registry.json",
        {
            "status": "resource_limit",
            "wall_clock_seconds": 7_200.0,
            "ended_at": "2026-08-22T00:00:00+00:00",
            "cells": [
                {"cell_id": "done", "status": "COMPLETED"},
                {"cell_id": "partial", "status": "RESOURCE_LIMIT"},
                {"cell_id": "pending", "status": "PENDING"},
            ],
        },
    )
    _write_json(
        current_path,
        {
            "run_id": "run",
            "pid": 987_654,
            "started_at": "2026-08-22T00:00:00+00:00",
            "prior_wall_clock_seconds": 7_200.0,
            "runtime_config": "runtime.yaml",
            "artifact_dir": "artifacts/runs/run",
        },
    )
    monkeypatch.setattr(control, "_alive", lambda pid: False)
    monkeypatch.setattr(control, "_process_matches_launch", lambda current: False)

    result = control.status()

    assert result["terminal"] is True
    assert result["resume_allowed"] is False
    assert result["retryable_cells"] == 0
    assert result["restart_required_cells"] == 2
    assert result["incomplete_cells"] == ["partial", "pending"]


def test_status_marks_terminal_run_as_not_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    current_path = tmp_path / "artifacts/p4/control/current.json"
    monkeypatch.setattr(control, "CURRENT", current_path)
    runtime = tmp_path / "artifacts/p4/control/run/pilot.runtime.yaml"
    raw = yaml.safe_load(Path("configs/experiments/p4/pilot.yaml").read_text(encoding="utf-8"))
    runtime.parent.mkdir(parents=True)
    runtime.write_text(yaml.safe_dump(raw), encoding="utf-8")
    artifact_directory = tmp_path / "artifacts/runs/run"
    ended_at = datetime.now(UTC).isoformat()
    _write_json(
        artifact_directory / "registry.json",
        {
            "status": "pilot_passed",
            "cells": [],
            "wall_clock_seconds": 75.0,
            "ended_at": ended_at,
        },
    )
    _write_json(
        artifact_directory / "heartbeat.json",
        {
            "suite_elapsed_seconds": 70.0,
            "updated_at": (datetime.now(UTC) - timedelta(minutes=10)).isoformat(),
        },
    )
    _write_json(
        current_path,
        {
            "run_id": "run",
            "pid": 987_654,
            "started_at": (datetime.now(UTC) - timedelta(seconds=10)).isoformat(),
            "prior_wall_clock_seconds": 50.0,
            "runtime_config": str(runtime.relative_to(tmp_path)),
            "artifact_dir": str(artifact_directory.relative_to(tmp_path)),
        },
    )
    monkeypatch.setattr(control, "_alive", lambda pid: False)
    monkeypatch.setattr(control, "_process_matches_launch", lambda current: False)

    result = control.status()

    assert result["terminal"] is True
    assert result["resume_allowed"] is False
    assert result["terminal_reason"] == "pilot_passed"
    assert result["ended_at"] == ended_at
    assert result["heartbeat_stale"] is True


def test_status_marks_stopped_dead_run_as_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    current_path = tmp_path / "artifacts/p4/control/current.json"
    monkeypatch.setattr(control, "CURRENT", current_path)
    runtime = tmp_path / "artifacts/p4/control/run/mechanism.runtime.yaml"
    raw = yaml.safe_load(Path("configs/experiments/p4/mechanism.yaml").read_text(encoding="utf-8"))
    runtime.parent.mkdir(parents=True)
    runtime.write_text(yaml.safe_dump(raw), encoding="utf-8")
    artifact_directory = tmp_path / "artifacts/runs/run"
    _write_json(
        artifact_directory / "registry.json",
        {"status": "stopped", "cells": [], "wall_clock_seconds": 75.0},
    )
    _write_json(
        artifact_directory / "heartbeat.json",
        {
            "suite_elapsed_seconds": 75.0,
            "updated_at": (datetime.now(UTC) - timedelta(minutes=10)).isoformat(),
        },
    )
    _write_json(
        current_path,
        {
            "run_id": "run",
            "pid": 987_654,
            "started_at": (datetime.now(UTC) - timedelta(seconds=10)).isoformat(),
            "prior_wall_clock_seconds": 50.0,
            "runtime_config": str(runtime.relative_to(tmp_path)),
            "artifact_dir": str(artifact_directory.relative_to(tmp_path)),
        },
    )
    monkeypatch.setattr(control, "_alive", lambda pid: False)
    monkeypatch.setattr(control, "_process_matches_launch", lambda current: False)

    result = control.status()

    assert result["terminal"] is False
    assert result["resume_allowed"] is True
    assert result["terminal_reason"] == "stopped"
    assert result["heartbeat_stale"] is True


def test_launch_spec_uses_platform_detached_flags_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control.sys, "platform", "win32", raising=False)

    spec = control._launch_spec(Path("runtime.yaml"))

    assert spec["command"] == [
        control.sys.executable,
        "-m",
        "neuromorphic.training.run",
        "--config",
        "runtime.yaml",
    ]
    assert spec["creationflags"] != 0
    assert int(spec["creationflags"]) & 0x01000000


def test_alive_uses_get_process_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(control.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(control.subprocess, "run", fake_run)

    assert control._alive(12_484) is True
    assert calls[0][:3] == ["powershell", "-NoProfile", "-Command"]
    assert "12484" in calls[0][3]


def test_cuda_qualification_profile_uses_engineering_cuda_config() -> None:
    config = control._profile_config("cuda-qualification")

    assert config.profile == "qualification"
    assert config.qualification_only is True
    assert config.device == "cuda"
    assert str(config.control_root).replace("\\", "/").endswith("artifacts/p4-cuda/control")


def test_cuda_pilot_profile_uses_cuda_lock_namespace() -> None:
    config = control._profile_config("cuda-pilot")

    assert config.profile == "pilot"
    assert config.qualification_only is True
    assert config.device == "cuda"
    assert str(config.qualification_report).replace("\\", "/") == (
        "artifacts/p4-cuda/qualification-lock.json"
    )


def test_cuda_mechanism_profile_uses_cuda_lock_namespace() -> None:
    config = control._profile_config("cuda-mechanism")

    assert config.profile == "mechanism"
    assert config.qualification_only is False
    assert config.device == "cuda"
    assert str(config.qualification_report).replace("\\", "/") == (
        "artifacts/p4-cuda/qualification-lock.json"
    )
    assert str(config.pilot_lock).replace("\\", "/") == ("artifacts/p4-cuda/pilot-lock.json")


def test_cuda_evidence_hashes_do_not_reuse_mps_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mps = tmp_path / "artifacts/p4/qualification-lock.json"
    cuda = tmp_path / "artifacts/p4-cuda/qualification-lock.json"
    mps.parent.mkdir(parents=True)
    cuda.parent.mkdir(parents=True)
    mps.write_text("mps", encoding="utf-8")
    cuda.write_text("cuda", encoding="utf-8")
    monkeypatch.setattr(control, "QUALIFICATION_LOCK", mps)
    monkeypatch.setattr(control, "CUDA_QUALIFICATION_LOCK", cuda)
    monkeypatch.setattr(control, "CUDA_PILOT_LOCK", tmp_path / "cuda-pilot.json")

    hashes = control._evidence_lock_hashes("cuda-pilot")

    assert hashes["qualification_lock_sha256"] == control._sha256(cuda)
    assert hashes["qualification_lock_sha256"] != control._sha256(mps)


def test_cuda_pilot_runtime_requires_cuda_qualification_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mps_lock = tmp_path / "artifacts/p4/qualification-lock.json"
    cuda_lock = tmp_path / "artifacts/p4-cuda/qualification-lock.json"
    mps_lock.parent.mkdir(parents=True)
    _write_json(mps_lock, {"status": "PASSED", "git_commit": "abc123"})
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "CONTROL", tmp_path / "artifacts/p4/control")
    monkeypatch.setattr(control, "QUALIFICATION_LOCK", mps_lock)
    monkeypatch.setattr(control, "CUDA_QUALIFICATION_LOCK", cuda_lock)

    with pytest.raises(RuntimeError, match="CUDA qualification lock is missing"):
        control._prepare_runtime("cuda-pilot", head="abc123")

    cuda_lock.parent.mkdir(parents=True)
    _write_json(cuda_lock, {"status": "PASSED", "git_commit": "abc123"})
    runtime, run_id = control._prepare_runtime("cuda-pilot", head="abc123")

    assert run_id.startswith("p4-cuda-pilot-abc123-")
    assert control.load_p4_suite_config(runtime).device == "cuda"


def test_cuda_mechanism_runtime_consumes_cuda_pilot_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cuda_qualification = tmp_path / "artifacts/p4-cuda/qualification-lock.json"
    cuda_pilot = tmp_path / "artifacts/p4-cuda/pilot-lock.json"
    cuda_qualification.parent.mkdir(parents=True)
    _write_json(cuda_qualification, {"status": "PASSED", "git_commit": "abc123"})
    _write_json(
        cuda_pilot,
        {
            "status": "PASSED",
            "git_commit": "abc123",
            "selected_preset": "preset-1",
            "optimizer": {
                "learning_rate": 0.0001,
                "weight_decay": 0.01,
                "temporal_loss_weight": 0.1,
                "gradient_clip_norm": 1.0,
            },
        },
    )
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "CONTROL", tmp_path / "artifacts/p4/control")
    monkeypatch.setattr(control, "CUDA_QUALIFICATION_LOCK", cuda_qualification)
    monkeypatch.setattr(control, "CUDA_PILOT_LOCK", cuda_pilot)

    runtime, run_id = control._prepare_runtime("cuda-mechanism", head="abc123")
    config = control.load_p4_suite_config(runtime)

    assert run_id.startswith("p4-cuda-mechanism-abc123-")
    assert config.device == "cuda"
    assert config.selected_preset == "preset-1"
    assert config.optimizer.learning_rate == 0.0001


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
        lambda profile=None: {
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


def test_qualification_preflight_replaces_dead_launch_without_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "CURRENT", tmp_path / "artifacts/p4/control/current.json")
    _write_json(
        control.CURRENT,
        {
            "pid": 12_484,
            "artifact_dir": "artifacts/runs/stale-launch",
        },
    )

    def fake_git(*arguments: str) -> str:
        if arguments == ("status", "--porcelain"):
            return ""
        if arguments in {("rev-parse", "HEAD"), ("rev-parse", "origin/main")}:
            return "abc123"
        raise AssertionError(arguments)

    monkeypatch.setattr(control, "_git", fake_git)
    monkeypatch.setattr(control, "_alive", lambda pid: False)
    monkeypatch.setattr(
        control.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=101 * 1024**3),
    )
    config = control._profile_config("qualification").model_copy(update={"device": "cpu"})

    assert control._background_preflight(config=config, for_resume=False) == "abc123"


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
