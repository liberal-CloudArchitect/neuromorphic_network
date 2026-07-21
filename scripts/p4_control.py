"""Manage long-running local P4 suites with auditable background control."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml

from neuromorphic.training.p4_config import P4SuiteConfig, load_p4_suite_config

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "artifacts/p4/control"
QUALIFICATION_LOCK = ROOT / "artifacts/p4/qualification-lock.json"
PILOT_LOCK = ROOT / "artifacts/p4/pilot-lock.json"
MECHANISM_LOCK = ROOT / "artifacts/p4/mechanism-lock.json"
CI_LOCK = ROOT / "artifacts/p4/ci-lock.json"
CURRENT = CONTROL / "current.json"

PROFILE_CONFIGS: dict[str, Path] = {
    "qualification": ROOT / "configs/experiments/p4/qualification.yaml",
    "pilot": ROOT / "configs/experiments/p4/pilot.yaml",
    "mechanism": ROOT / "configs/experiments/p4/mechanism.yaml",
    "full": ROOT / "configs/experiments/p4/full.yaml",
}
PROFILE = Literal["qualification", "pilot", "mechanism", "full"]
TERMINAL_STATUSES = {
    "qualification_passed",
    "pilot_passed",
    "mechanism_passed",
    "mechanism_failed",
    "completed",
    "completed_with_failures",
    "failed",
    "resource_limit",
}


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
        raise FileNotFoundError("no P4 background run has been registered")
    return _json(CURRENT)


def _evidence_lock_hashes() -> dict[str, str | None]:
    return {
        "qualification_lock_sha256": _sha256(QUALIFICATION_LOCK)
        if QUALIFICATION_LOCK.is_file()
        else None,
        "pilot_lock_sha256": _sha256(PILOT_LOCK) if PILOT_LOCK.is_file() else None,
        "mechanism_lock_sha256": _sha256(MECHANISM_LOCK) if MECHANISM_LOCK.is_file() else None,
        "ci_lock_sha256": _sha256(CI_LOCK) if CI_LOCK.is_file() else None,
    }


def _validate_lock_artifact(lock: dict[str, Any], *, label: str) -> None:
    for artifact_key, hash_key in (
        ("qualification_report", "qualification_report_sha256"),
        ("pilot_selection", "pilot_selection_sha256"),
        ("mechanism_report", "mechanism_report_sha256"),
    ):
        artifact = lock.get(artifact_key)
        expected_hash = lock.get(hash_key)
        if artifact is None and expected_hash is None:
            continue
        path = _under_root(artifact, label=f"{label} evidence")
        if not path.is_file() or _sha256(path) != expected_hash:
            raise RuntimeError(f"{label} evidence checksum does not match")


def _require_lock(path: Path, *, label: str, head: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} lock is missing")
    lock = _json(path)
    if lock.get("status") != "PASSED":
        raise RuntimeError(f"{label} lock is failed or missing")
    commit = lock.get("git_commit", lock.get("expected_git_commit"))
    if commit != head:
        raise RuntimeError(f"{label} lock belongs to another commit")
    _validate_lock_artifact(lock, label=label)
    return lock


def _background_preflight(*, config: P4SuiteConfig, for_resume: bool) -> str:
    if _git("status", "--porcelain"):
        raise RuntimeError("P4 background runs require a clean worktree")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "origin/main") != head:
        raise RuntimeError("P4 background runs require HEAD == origin/main")
    if config.profile != "qualification":
        ci_lock = _json(CI_LOCK) if CI_LOCK.is_file() else {}
        if (
            ci_lock.get("git_commit") != head
            or ci_lock.get("conclusion") != "success"
            or not isinstance(ci_lock.get("run_id"), int)
            or not isinstance(ci_lock.get("workflow_url"), str)
        ):
            raise RuntimeError("a successful GitHub Actions record for HEAD is required")
    if sys.version_info[:2] != (3, 12) or torch.__version__.split("+")[0] != "2.12.1":
        raise RuntimeError("P4 background runs require Python 3.12 and PyTorch 2.12.1")
    if config.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("P4 background runs require an available MPS backend")
        if sys.platform == "darwin":
            power = subprocess.run(
                ["pmset", "-g", "batt"], text=True, check=True, capture_output=True
            ).stdout
            if "AC Power" not in power:
                raise RuntimeError("connect the Mac to AC power before starting P4")
    if shutil.disk_usage(ROOT).free < 100 * 1024**3:
        raise RuntimeError("P4 background runs require at least 100 GiB free")
    if CURRENT.exists():
        current = _json(CURRENT)
        pid = int(current.get("pid", -1))
        if pid > 0 and _alive(pid):
            raise RuntimeError(f"another P4 background process is active: {pid}")
        if not for_resume:
            registry = _under_root(current.get("artifact_dir"), label="artifact directory")
            registry = registry / "registry.json"
            if not registry.is_file() or _json(registry).get("status") not in TERMINAL_STATUSES:
                raise RuntimeError(
                    "a prior P4 run exists; inspect it and use resume instead of start"
                )
    return head


def _profile_config(profile: PROFILE) -> P4SuiteConfig:
    return load_p4_suite_config(PROFILE_CONFIGS[profile])


def _prepare_runtime(profile: PROFILE, *, head: str) -> tuple[Path, str]:
    raw = yaml.safe_load(PROFILE_CONFIGS[profile].read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("P4 background config must be a YAML object")
    if profile in {"pilot", "mechanism", "full"}:
        _require_lock(QUALIFICATION_LOCK, label="qualification", head=head)
    if profile in {"mechanism", "full"}:
        pilot_lock = _require_lock(PILOT_LOCK, label="pilot", head=head)
        selected_preset = pilot_lock.get("selected_preset")
        optimizer = pilot_lock.get("optimizer")
        if not isinstance(selected_preset, str) or not isinstance(optimizer, dict):
            raise RuntimeError("pilot lock is missing its frozen preset or optimizer")
        raw["selected_preset"] = selected_preset
        raw["optimizer"] = optimizer
    if profile == "full":
        _require_lock(MECHANISM_LOCK, label="mechanism", head=head)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"p4-{profile}-{head[:8]}-{timestamp}"
    raw["run_id"] = run_id
    raw["expected_git_commit"] = head
    runtime = CONTROL / run_id / f"{profile}.runtime.yaml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    load_p4_suite_config(runtime)
    return runtime, run_id


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
            f"P4 background process exited during launch with code {return_code}: {tail}"
        )
    profile = load_p4_suite_config(runtime_config).profile
    artifact_directory = ROOT / "artifacts/runs" / run_id
    registry = artifact_directory / "registry.json"
    prior_wall_clock = (
        float(_json(registry).get("wall_clock_seconds", 0.0)) if registry.is_file() else 0.0
    )
    existing = _json(CURRENT) if resumed and CURRENT.is_file() else {}
    launched_at = datetime.now(UTC).isoformat()
    launch = {
        "schema_version": "p4-launch-v1",
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


def _run_foreground(runtime_config: Path, *, profile: PROFILE, run_id: str) -> dict[str, Any]:
    if profile != "qualification":
        raise RuntimeError("foreground mode is only allowed for qualification")
    command = [sys.executable, "-m", "neuromorphic.training.run", "--config", str(runtime_config)]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return {
        "schema_version": "p4-foreground-v1",
        "run_id": run_id,
        "profile": profile,
        "foreground": True,
        "command": command,
        "runtime_config": str(runtime_config.relative_to(ROOT)),
        "runtime_config_sha256": _sha256(runtime_config),
        "exit_code": completed.returncode,
    }


def start(profile: PROFILE, *, foreground: bool = False) -> dict[str, Any]:
    if foreground:
        if profile != "qualification":
            raise RuntimeError("foreground mode is only allowed for qualification")
        head = _git("rev-parse", "HEAD")
        runtime, run_id = _prepare_runtime(profile, head=head)
        return _run_foreground(runtime, profile=profile, run_id=run_id)
    config = _profile_config(profile)
    head = _background_preflight(config=config, for_resume=False)
    runtime, run_id = _prepare_runtime(profile, head=head)
    launch = _launch(runtime, run_id, resumed=False)
    launch["phase"] = profile
    launch["status_command"] = "./scripts/p4_run.sh status"
    launch["logs_command"] = "./scripts/p4_run.sh logs"
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
        if isinstance(cells, list):
            result.update(
                {
                    "suite_status": value.get("status"),
                    "completed_cells": sum(
                        isinstance(cell, dict) and cell.get("status") == "COMPLETED"
                        for cell in cells
                    ),
                    "failed_cells": sum(
                        isinstance(cell, dict) and cell.get("status") == "FAILED" for cell in cells
                    ),
                    "resource_limited_cells": sum(
                        isinstance(cell, dict) and cell.get("status") == "RESOURCE_LIMIT"
                        for cell in cells
                    ),
                    "remaining_cells": sum(
                        isinstance(cell, dict) and cell.get("status") in {"PENDING", "RUNNING"}
                        for cell in cells
                    ),
                    "retryable_cells": sum(
                        isinstance(cell, dict) and cell.get("status") != "COMPLETED"
                        for cell in cells
                    ),
                    "total_cells": len(cells),
                }
            )
        config_path = _under_root(current.get("runtime_config"), label="runtime config")
        if config_path.is_file():
            config = load_p4_suite_config(config_path)
            budget = config.budget.wall_clock_hours * 3600.0
            result["remaining_wall_clock_seconds"] = max(budget - elapsed, 0.0)
        running = next(
            (cell for cell in cells if isinstance(cell, dict) and cell.get("status") == "RUNNING"),
            None,
        )
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
                budget = load_p4_suite_config(config_path).budget.wall_clock_hours * 3600.0
                result["remaining_wall_clock_seconds"] = max(budget - elapsed, 0.0)
    result["free_bytes"] = shutil.disk_usage(ROOT).free
    return result


def resume() -> dict[str, Any]:
    current = _current()
    runtime = _under_root(current.get("runtime_config"), label="runtime config")
    config = load_p4_suite_config(runtime)
    head = _background_preflight(config=config, for_resume=True)
    if current.get("git_commit") != head:
        raise RuntimeError("resume commit does not match the original launch")
    for name, value in _evidence_lock_hashes().items():
        if current.get(name) != value:
            raise RuntimeError(f"resume evidence lock changed after launch: {name}")
    pid = int(current["pid"])
    if _alive(pid):
        raise RuntimeError("the P4 process is already active")
    if _sha256(runtime) != current.get("runtime_config_sha256"):
        raise RuntimeError("runtime config changed after launch")
    registry_path = _under_root(current.get("artifact_dir"), label="artifact directory")
    registry_path = registry_path / "registry.json"
    if registry_path.is_file() and _json(registry_path).get("status") in TERMINAL_STATUSES:
        raise RuntimeError("the P4 run is already complete and cannot be resumed")
    stop_file = _under_root(current.get("artifact_dir"), label="artifact directory") / "STOP"
    stop_file.unlink(missing_ok=True)
    return _launch(runtime, str(current["run_id"]), resumed=True)


def stop(force: bool) -> dict[str, object]:
    current = _current()
    pid = int(current["pid"])
    if force:
        if _alive(pid):
            if not _process_matches_launch(current):
                raise RuntimeError("refusing to signal a PID that does not match the P4 launch")
            os.killpg(int(current["process_group"]), signal.SIGTERM)
        return {"run_id": current["run_id"], "forced": True}
    if not _alive(pid):
        return {"run_id": current["run_id"], "already_stopped": True}
    stop_file = _under_root(current.get("artifact_dir"), label="artifact directory") / "STOP"
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.touch()
    return {"run_id": current["run_id"], "graceful_stop_requested": True}


def _verify_p4_run(directory: Path) -> dict[str, object]:
    try:
        module = importlib.import_module("neuromorphic.training.p4_suite")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "neuromorphic.training.p4_suite is not available in this checkout"
        ) from error
    verifier = getattr(module, "verify_p4_run", None)
    if not callable(verifier):
        raise RuntimeError("neuromorphic.training.p4_suite.verify_p4_run is unavailable")
    return cast(dict[str, object], verifier(directory))


def verify() -> dict[str, object]:
    current = _current()
    if _alive(int(current["pid"])):
        raise RuntimeError("wait for the active P4 process to stop before verification")
    return _verify_p4_run(_under_root(current.get("artifact_dir"), label="artifact directory"))


def record_ci() -> dict[str, object]:
    """Freeze the latest successful CI run for the clean origin/main SHA."""

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
        f"https://api.github.com/repos/{slug}/actions/runs"
        f"?head_sha={head}&status=completed&per_page=100",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "neuromorphic-p4"},
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
        "schema_version": "p4-ci-lock-v1",
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
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("profile", choices=tuple(PROFILE_CONFIGS))
    start_parser.add_argument("--foreground", action="store_true")
    subparsers.add_parser("status")
    subparsers.add_parser("logs")
    subparsers.add_parser("resume")
    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("verify")
    subparsers.add_parser("record-ci")
    parsed = parser.parse_args(arguments)
    try:
        if parsed.command == "start":
            result = start(parsed.profile, foreground=parsed.foreground)
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
        elif parsed.command == "record-ci":
            result = record_ci()
        else:
            result = verify()
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
    if parsed.command == "start" and result.get("foreground") is True:
        return int(result["exit_code"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
