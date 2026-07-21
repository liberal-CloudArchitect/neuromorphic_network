"""Manage a long-running local P3 MPS suite with auditable state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from neuromorphic.training.p3_config import load_p3_suite_config
from neuromorphic.training.p3_suite import verify_p3_run

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "artifacts/p3/control"
LOCK = ROOT / "artifacts/p3/qualification-lock.json"
CI_LOCK = ROOT / "artifacts/p3/ci-lock.json"
PILOT_LOCK = ROOT / "artifacts/p3/pilot-lock.json"
CURRENT = CONTROL / "current.json"
FULL_CONFIG = ROOT / "configs/experiments/p3/full.yaml"
PILOT_CONFIG = ROOT / "configs/experiments/p3/pilot.yaml"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True, check=True, capture_output=True
    ).stdout.strip()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_matches_launch(current: dict[str, Any]) -> bool:
    pid = int(current.get("pid", -1))
    runtime = current.get("runtime_config")
    if pid <= 0 or not isinstance(runtime, str) or not _alive(pid):
        return False
    inspected = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        text=True,
        check=False,
        capture_output=True,
    )
    command = inspected.stdout.strip()
    return (
        inspected.returncode == 0 and "neuromorphic.training.run" in command and runtime in command
    )


def _under_root(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} path is missing")
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} path escapes the repository") from error
    return path


def _under_directory(base: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} path is missing")
    path = (base / value).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} path escapes its artifact directory") from error
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current() -> dict[str, Any]:
    if not CURRENT.is_file():
        raise FileNotFoundError("no P3 background run has been registered")
    return _json(CURRENT)


def _evidence_lock_hashes() -> dict[str, str | None]:
    return {
        "qualification_lock_sha256": _sha256(LOCK) if LOCK.is_file() else None,
        "ci_lock_sha256": _sha256(CI_LOCK) if CI_LOCK.is_file() else None,
        "pilot_lock_sha256": _sha256(PILOT_LOCK) if PILOT_LOCK.is_file() else None,
    }


def _preflight(*, for_resume: bool) -> tuple[str, dict[str, Any]]:
    if _git("status", "--porcelain"):
        raise RuntimeError("P3 full run requires a clean worktree")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "origin/main") != head:
        raise RuntimeError("P3 full run requires HEAD == origin/main")
    lock = _json(LOCK)
    if lock.get("status") != "PASSED" or lock.get("expected_git_commit") != head:
        raise RuntimeError("qualification lock is missing, failed, or belongs to another commit")
    report = _under_root(lock.get("qualification_report"), label="qualification report")
    if not report.is_file() or _sha256(report) != lock.get("qualification_report_sha256"):
        raise RuntimeError("qualification report checksum does not match")
    report_value = _json(report)
    if (
        report_value.get("status") != "PASSED"
        or report_value.get("git_commit") != head
        or report_value.get("git_dirty") is not False
        or report_value.get("device") != "mps"
        or report_value.get("config_hash") != lock.get("qualification_config_hash")
        or report_value.get("matrix_hash") != lock.get("qualification_matrix_hash")
    ):
        raise RuntimeError("qualification report does not match its frozen lock")
    ci_lock = _json(CI_LOCK)
    if (
        ci_lock.get("git_commit") != head
        or ci_lock.get("conclusion") != "success"
        or not isinstance(ci_lock.get("run_id"), int)
        or not isinstance(ci_lock.get("workflow_url"), str)
    ):
        raise RuntimeError("a successful GitHub Actions record for HEAD is required")
    if sys.version_info[:2] != (3, 12) or torch.__version__.split("+")[0] != "2.12.1":
        raise RuntimeError("P3 full run requires Python 3.12 and PyTorch 2.12.1")
    if not torch.backends.mps.is_available():
        raise RuntimeError("P3 full run requires an available MPS backend")
    if sys.platform == "darwin":
        power = subprocess.run(
            ["pmset", "-g", "batt"], text=True, check=True, capture_output=True
        ).stdout
        if "AC Power" not in power:
            raise RuntimeError("connect the Mac to AC power before starting P3")
    if shutil.disk_usage(ROOT).free < 100 * 1024**3:
        raise RuntimeError("P3 start requires at least 100 GiB free")
    if CURRENT.exists():
        current = _json(CURRENT)
        pid = int(current.get("pid", -1))
        if pid > 0 and _alive(pid):
            raise RuntimeError(f"another P3 background process is active: {pid}")
        if not for_resume:
            registry = _under_root(current.get("artifact_dir"), label="artifact directory")
            registry = registry / "registry.json"
            completed_pilot = (
                registry.is_file()
                and _json(registry).get("status") == "pilot_passed"
                and PILOT_LOCK.is_file()
            )
            if not completed_pilot:
                raise RuntimeError(
                    "a prior P3 run exists; inspect it and use resume instead of start"
                )
    return head, lock


def _launch(runtime_config: Path, run_id: str, *, resumed: bool) -> dict[str, Any]:
    control = CONTROL / run_id
    control.mkdir(parents=True, exist_ok=True)
    log = control / "runner.log"
    command = [
        "/usr/bin/nohup",
        "/usr/bin/caffeinate",
        "-ims",
        sys.executable,
        "-m",
        "neuromorphic.training.run",
        "--config",
        str(runtime_config),
    ]
    handle = log.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            return_code = None
    finally:
        handle.close()
    if return_code is not None:
        tail = log.read_text(encoding="utf-8")[-2_000:] if log.is_file() else ""
        raise RuntimeError(
            f"P3 background process exited during launch with code {return_code}: {tail}"
        )
    profile = load_p3_suite_config(runtime_config).profile
    artifact_directory = ROOT / "artifacts/runs" / run_id
    registry = artifact_directory / "registry.json"
    prior_wall_clock = (
        float(_json(registry).get("wall_clock_seconds", 0.0)) if registry.is_file() else 0.0
    )
    existing = _json(CURRENT) if resumed and CURRENT.is_file() else {}
    launched_at = datetime.now(UTC).isoformat()
    launch = {
        "schema_version": "p3-launch-v1",
        "run_id": run_id,
        "pid": process.pid,
        "process_group": process.pid,
        "resumed": resumed,
        "profile": profile,
        "started_at": launched_at,
        "first_started_at": existing.get(
            "first_started_at", existing.get("started_at", launched_at)
        ),
        "prior_wall_clock_seconds": prior_wall_clock,
        "python": sys.executable,
        "git_commit": _git("rev-parse", "HEAD"),
        "runtime_config": str(runtime_config.relative_to(ROOT)),
        "runtime_config_sha256": _sha256(runtime_config),
        **_evidence_lock_hashes(),
        "log": str(log.relative_to(ROOT)),
        "command": command,
        "artifact_dir": f"artifacts/runs/{run_id}",
    }
    _write(control / "launch.json", launch)
    _write(CURRENT, launch)
    return launch


def start() -> dict[str, Any]:
    head, _ = _preflight(for_resume=False)
    pilot_lock: dict[str, Any] | None = None
    if PILOT_LOCK.is_file():
        pilot_lock = _json(PILOT_LOCK)
        if pilot_lock.get("git_commit") != head or pilot_lock.get("status") != "PASSED":
            raise RuntimeError("pilot lock is failed or belongs to another commit")
        selection_path = _under_root(pilot_lock.get("pilot_selection"), label="pilot selection")
        if not selection_path.is_file() or _sha256(selection_path) != pilot_lock.get(
            "pilot_selection_sha256"
        ):
            raise RuntimeError("pilot selection checksum does not match")
        selection = _json(selection_path)
        if (
            selection.get("status") != "PASSED"
            or selection.get("git_commit") != head
            or selection.get("config_hash") != pilot_lock.get("pilot_config_hash")
            or selection.get("matrix_hash") != pilot_lock.get("pilot_matrix_hash")
            or selection.get("selected_presets") != pilot_lock.get("selected_presets")
        ):
            raise RuntimeError("pilot selection does not match its frozen lock")
    source_config = PILOT_CONFIG if pilot_lock is None else FULL_CONFIG
    raw = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("P3 background config must be a YAML object")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    profile = "pilot" if pilot_lock is None else "full"
    run_id = f"p3-{profile}-{head[:8]}-{timestamp}"
    raw["run_id"] = run_id
    raw["expected_git_commit"] = head
    if pilot_lock is not None:
        raw["selected_presets"] = pilot_lock["selected_presets"]
    runtime = CONTROL / run_id / f"{profile}.runtime.yaml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    load_p3_suite_config(runtime)
    launch = _launch(runtime, run_id, resumed=False)
    launch["phase"] = profile
    launch["status_command"] = "./scripts/p3_full_run.sh status"
    launch["logs_command"] = "./scripts/p3_full_run.sh logs"
    return launch


def status() -> dict[str, Any]:
    current = _current()
    pid = int(current["pid"])
    result = dict(current)
    process_alive = _alive(pid)
    result["process_alive"] = process_alive
    result["process_matches_launch"] = _process_matches_launch(current)
    started = datetime.fromisoformat(str(current["started_at"]))
    attempt_elapsed = max((datetime.now(UTC) - started).total_seconds(), 0.0)
    prior_elapsed = float(current.get("prior_wall_clock_seconds", 0.0))
    elapsed = prior_elapsed + attempt_elapsed if process_alive else prior_elapsed
    result["elapsed_seconds"] = elapsed
    registry = _under_root(current.get("artifact_dir"), label="artifact directory")
    registry = registry / "registry.json"
    if registry.is_file():
        value = _json(registry)
        cells = value.get("cells", [])
        registry_elapsed = value.get("wall_clock_seconds")
        if isinstance(registry_elapsed, (int, float)):
            elapsed = (
                max(elapsed, float(registry_elapsed)) if process_alive else float(registry_elapsed)
            )
            result["elapsed_seconds"] = elapsed
        result.update(
            {
                "suite_status": value.get("status"),
                "completed_cells": sum(cell.get("status") == "COMPLETED" for cell in cells),
                "failed_cells": sum(cell.get("status") == "FAILED" for cell in cells),
                "resource_limited_cells": sum(
                    cell.get("status") == "RESOURCE_LIMIT" for cell in cells
                ),
                "remaining_cells": sum(
                    cell.get("status") in {"PENDING", "RUNNING"} for cell in cells
                ),
                "retryable_cells": sum(cell.get("status") != "COMPLETED" for cell in cells),
                "total_cells": len(cells),
            }
        )
        config_path = _under_root(current.get("runtime_config"), label="runtime config")
        if config_path.is_file():
            config = load_p3_suite_config(config_path)
            budget = config.budget.wall_clock_hours * 3600.0
            result["remaining_wall_clock_seconds"] = max(budget - elapsed, 0.0)
        running = next((cell for cell in cells if cell.get("status") == "RUNNING"), None)
        if isinstance(running, dict) and isinstance(running.get("artifact_dir"), str):
            result["current_cell"] = running.get("cell_id")
            artifact_directory = _under_root(
                current.get("artifact_dir"), label="artifact directory"
            )
            cell_directory = _under_directory(
                artifact_directory,
                running["artifact_dir"],
                label="cell artifact directory",
            )
            checkpoints = sorted(cell_directory.glob("*.pt"), key=lambda path: path.stat().st_mtime)
            if checkpoints:
                result["latest_checkpoint"] = str(checkpoints[-1].relative_to(ROOT))
    heartbeat = _under_root(current.get("artifact_dir"), label="artifact directory")
    heartbeat = heartbeat / "heartbeat.json"
    if heartbeat.is_file():
        heartbeat_value = _json(heartbeat)
        result["heartbeat"] = heartbeat_value
        heartbeat_elapsed = heartbeat_value.get("suite_elapsed_seconds")
        if isinstance(heartbeat_elapsed, (int, float)):
            elapsed = max(elapsed, float(heartbeat_elapsed))
            result["elapsed_seconds"] = elapsed
            if registry.is_file():
                config_path = _under_root(current.get("runtime_config"), label="runtime config")
                budget = load_p3_suite_config(config_path).budget.wall_clock_hours * 3600.0
                result["remaining_wall_clock_seconds"] = max(budget - elapsed, 0.0)
    result["free_bytes"] = shutil.disk_usage(ROOT).free
    return result


def resume() -> dict[str, Any]:
    head, _ = _preflight(for_resume=True)
    current = _current()
    if current.get("git_commit") != head:
        raise RuntimeError("resume commit does not match the original launch")
    for name, value in _evidence_lock_hashes().items():
        if current.get(name) != value:
            raise RuntimeError(f"resume evidence lock changed after launch: {name}")
    pid = int(current["pid"])
    if _alive(pid):
        raise RuntimeError("the P3 process is already active")
    run_id = str(current["run_id"])
    runtime = _under_root(current.get("runtime_config"), label="runtime config")
    if _sha256(runtime) != current.get("runtime_config_sha256"):
        raise RuntimeError("runtime config changed after launch")
    registry_path = _under_root(current.get("artifact_dir"), label="artifact directory")
    registry_path = registry_path / "registry.json"
    if registry_path.is_file() and _json(registry_path).get("status") in {
        "qualification_passed",
        "pilot_passed",
        "completed",
    }:
        raise RuntimeError("the P3 run is already complete and cannot be resumed")
    stop_file = _under_root(current.get("artifact_dir"), label="artifact directory") / "STOP"
    stop_file.unlink(missing_ok=True)
    return _launch(runtime, run_id, resumed=True)


def stop(force: bool) -> dict[str, object]:
    current = _current()
    pid = int(current["pid"])
    if force:
        if _alive(pid):
            if not _process_matches_launch(current):
                raise RuntimeError("refusing to signal a PID that does not match the P3 launch")
            os.killpg(int(current["process_group"]), signal.SIGTERM)
        return {"run_id": current["run_id"], "forced": True}
    if not _alive(pid):
        return {"run_id": current["run_id"], "already_stopped": True}
    stop_file = _under_root(current.get("artifact_dir"), label="artifact directory") / "STOP"
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.touch()
    return {"run_id": current["run_id"], "graceful_stop_requested": True}


def verify() -> dict[str, object]:
    current = _current()
    if _alive(int(current["pid"])):
        raise RuntimeError("wait for the active P3 process to stop before verification")
    return verify_p3_run(_under_root(current.get("artifact_dir"), label="artifact directory"))


def freeze_qualification(directory: Path) -> dict[str, object]:
    if _git("status", "--porcelain"):
        raise RuntimeError("qualification can only be frozen from a clean worktree")
    directory = directory.resolve()
    report = directory / "qualification-report.json"
    value = _json(report)
    if value.get("status") != "PASSED" or value.get("qualification_only") is not True:
        raise RuntimeError("only a passed qualification run can be frozen")
    head = _git("rev-parse", "HEAD")
    if value.get("git_commit") != head or value.get("git_dirty") is not False:
        raise RuntimeError("qualification report must come from this clean HEAD")
    if value.get("device") != "mps":
        raise RuntimeError("formal qualification lock requires the complete MPS matrix")
    verification = verify_p3_run(directory)
    if verification.get("status") != "qualification_passed" or verification.get("missing_cells"):
        raise RuntimeError("qualification artifacts are incomplete or failed verification")
    relative = report.relative_to(ROOT)
    lock: dict[str, object] = {
        "schema_version": "p3-qualification-lock-v1",
        "status": "PASSED",
        "expected_git_commit": head,
        "qualification_run_id": value["run_id"],
        "qualification_report": str(relative),
        "qualification_report_sha256": _sha256(report),
        "qualification_config_hash": value.get("config_hash"),
        "qualification_matrix_hash": value.get("matrix_hash"),
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    _write(LOCK, lock)
    return lock


def freeze_pilot(directory: Path) -> dict[str, object]:
    if _git("status", "--porcelain"):
        raise RuntimeError("pilot selection can only be frozen from a clean worktree")
    directory = directory.resolve()
    report = directory / "pilot-selection.json"
    value = _json(report)
    if value.get("status") != "PASSED":
        raise RuntimeError("only a passed pilot selection can be frozen")
    head = _git("rev-parse", "HEAD")
    if value.get("git_commit") != head or value.get("git_dirty") is not False:
        raise RuntimeError("pilot selection must come from this clean HEAD")
    verification = verify_p3_run(directory)
    if verification.get("status") != "pilot_passed" or verification.get("missing_cells"):
        raise RuntimeError("pilot artifacts are incomplete or failed verification")
    selected = value.get("selected_presets")
    if not isinstance(selected, dict) or set(selected) != {"modular", "gru", "transformer"}:
        raise RuntimeError("pilot selection does not cover all confirmatory models")
    relative = report.relative_to(ROOT)
    lock: dict[str, object] = {
        "schema_version": "p3-pilot-lock-v1",
        "status": "PASSED",
        "git_commit": head,
        "pilot_run_id": directory.name,
        "pilot_selection": str(relative),
        "pilot_selection_sha256": _sha256(report),
        "pilot_config_hash": value.get("config_hash"),
        "pilot_matrix_hash": value.get("matrix_hash"),
        "selected_presets": selected,
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    _write(PILOT_LOCK, lock)
    return lock


def record_ci() -> dict[str, object]:
    if _git("status", "--porcelain"):
        raise RuntimeError("CI can only be recorded for a clean worktree")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "origin/main") != head:
        raise RuntimeError("CI can only be recorded when HEAD == origin/main")
    remote = _git("remote", "get-url", "origin")
    if remote.startswith("git@github.com:"):
        slug = remote.removeprefix("git@github.com:").removesuffix(".git")
    elif "github.com/" in remote:
        slug = remote.split("github.com/", 1)[1].removesuffix(".git")
    else:
        raise RuntimeError("origin is not a GitHub repository")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{slug}/actions/runs?head_sha={head}&status=completed&per_page=100",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "neuromorphic-p3"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    runs = [
        run
        for run in payload.get("workflow_runs", [])
        if run.get("head_sha") == head
        and run.get("conclusion") == "success"
        and run.get("path") == ".github/workflows/ci.yml"
    ]
    if not runs:
        raise RuntimeError("no successful completed GitHub Actions run exists for HEAD")
    selected = max(runs, key=lambda run: str(run.get("updated_at", "")))
    lock = {
        "schema_version": "p3-ci-lock-v1",
        "git_commit": head,
        "conclusion": "success",
        "run_id": selected["id"],
        "workflow_url": selected["html_url"],
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    _write(CI_LOCK, lock)
    return lock


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("start")
    subparsers.add_parser("status")
    subparsers.add_parser("logs")
    subparsers.add_parser("resume")
    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("verify")
    freeze_parser = subparsers.add_parser("freeze-qualification")
    freeze_parser.add_argument("directory", type=Path)
    pilot_parser = subparsers.add_parser("freeze-pilot")
    pilot_parser.add_argument("directory", type=Path)
    subparsers.add_parser("record-ci")
    parsed = parser.parse_args(arguments)
    try:
        if parsed.command == "start":
            result = start()
        elif parsed.command == "status":
            result = status()
        elif parsed.command == "logs":
            current = _current()
            log = _under_root(current.get("log"), label="runner log")
            tail_arguments = ["tail", "-n", "200"]
            if _alive(int(current["pid"])):
                tail_arguments.append("-f")
            tail_arguments.append(str(log))
            return subprocess.run(tail_arguments, check=False).returncode
        elif parsed.command == "resume":
            result = resume()
        elif parsed.command == "stop":
            result = stop(parsed.force)
        elif parsed.command == "verify":
            result = verify()
        elif parsed.command == "record-ci":
            result = record_ci()
        elif parsed.command == "freeze-pilot":
            result = freeze_pilot(parsed.directory)
        else:
            result = freeze_qualification(parsed.directory)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(
            json.dumps(
                {
                    "command": parsed.command,
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "exit_code": 2,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
