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
CURRENT = CONTROL / "current.json"
FULL_CONFIG = ROOT / "configs/experiments/p3/full.yaml"


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
    except (ProcessLookupError, PermissionError):
        return False
    return True


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


def _preflight() -> tuple[str, dict[str, Any]]:
    if _git("status", "--porcelain"):
        raise RuntimeError("P3 full run requires a clean worktree")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "origin/main") != head:
        raise RuntimeError("P3 full run requires HEAD == origin/main")
    lock = _json(LOCK)
    if lock.get("status") != "PASSED" or lock.get("expected_git_commit") != head:
        raise RuntimeError("qualification lock is missing, failed, or belongs to another commit")
    report = ROOT / str(lock.get("qualification_report"))
    if not report.is_file() or _sha256(report) != lock.get("qualification_report_sha256"):
        raise RuntimeError("qualification report checksum does not match")
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
    return head, lock


def _launch(runtime_config: Path, run_id: str, *, resumed: bool) -> dict[str, Any]:
    control = CONTROL / run_id
    control.mkdir(parents=True, exist_ok=True)
    log = control / "runner.log"
    command = [
        "/usr/bin/caffeinate",
        "-ims",
        sys.executable,
        "-m",
        "neuromorphic.training.run",
        "--config",
        str(runtime_config),
    ]
    handle = log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    launch = {
        "schema_version": "p3-launch-v1",
        "run_id": run_id,
        "pid": process.pid,
        "process_group": process.pid,
        "resumed": resumed,
        "started_at": datetime.now(UTC).isoformat(),
        "python": sys.executable,
        "git_commit": _git("rev-parse", "HEAD"),
        "runtime_config": str(runtime_config.relative_to(ROOT)),
        "runtime_config_sha256": _sha256(runtime_config),
        "log": str(log.relative_to(ROOT)),
        "command": command,
        "artifact_dir": f"artifacts/runs/{run_id}",
    }
    _write(control / "launch.json", launch)
    _write(CURRENT, launch)
    return launch


def start() -> dict[str, Any]:
    head, _ = _preflight()
    raw = yaml.safe_load(FULL_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("full P3 config must be a YAML object")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"p3-full-{head[:8]}-{timestamp}"
    raw["run_id"] = run_id
    raw["expected_git_commit"] = head
    runtime = CONTROL / run_id / "full.runtime.yaml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    load_p3_suite_config(runtime)
    launch = _launch(runtime, run_id, resumed=False)
    launch["status_command"] = "./scripts/p3_full_run.sh status"
    launch["logs_command"] = "./scripts/p3_full_run.sh logs"
    return launch


def status() -> dict[str, Any]:
    current = _current()
    pid = int(current["pid"])
    result = dict(current)
    result["process_alive"] = _alive(pid)
    registry = ROOT / str(current["artifact_dir"]) / "registry.json"
    if registry.is_file():
        value = _json(registry)
        cells = value.get("cells", [])
        result.update(
            {
                "suite_status": value.get("status"),
                "completed_cells": sum(cell.get("status") == "COMPLETED" for cell in cells),
                "failed_cells": sum(cell.get("status") == "FAILED" for cell in cells),
                "remaining_cells": sum(
                    cell.get("status") in {"PENDING", "RUNNING"} for cell in cells
                ),
                "total_cells": len(cells),
            }
        )
    heartbeat = ROOT / str(current["artifact_dir"]) / "heartbeat.json"
    if heartbeat.is_file():
        result["heartbeat"] = _json(heartbeat)
    result["free_bytes"] = shutil.disk_usage(ROOT).free
    return result


def resume() -> dict[str, Any]:
    head, _ = _preflight()
    current = _current()
    if current.get("git_commit") != head:
        raise RuntimeError("resume commit does not match the original launch")
    pid = int(current["pid"])
    if _alive(pid):
        raise RuntimeError("the P3 process is already active")
    run_id = str(current["run_id"])
    runtime = ROOT / str(current["runtime_config"])
    if _sha256(runtime) != current.get("runtime_config_sha256"):
        raise RuntimeError("runtime config changed after launch")
    stop_file = ROOT / str(current["artifact_dir"]) / "STOP"
    stop_file.unlink(missing_ok=True)
    return _launch(runtime, run_id, resumed=True)


def stop(force: bool) -> dict[str, object]:
    current = _current()
    pid = int(current["pid"])
    if force:
        if _alive(pid):
            os.killpg(int(current["process_group"]), signal.SIGTERM)
        return {"run_id": current["run_id"], "forced": True}
    stop_file = ROOT / str(current["artifact_dir"]) / "STOP"
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.touch()
    return {"run_id": current["run_id"], "graceful_stop_requested": True}


def verify() -> dict[str, object]:
    current = _current()
    return verify_p3_run(ROOT / str(current["artifact_dir"]))


def freeze_qualification(directory: Path) -> dict[str, object]:
    report = directory / "qualification-report.json"
    value = _json(report)
    if value.get("status") != "PASSED" or value.get("qualification_only") is not True:
        raise RuntimeError("only a passed qualification run can be frozen")
    head = _git("rev-parse", "HEAD")
    relative = report.relative_to(ROOT)
    lock = {
        "schema_version": "p3-qualification-lock-v1",
        "status": "PASSED",
        "expected_git_commit": head,
        "qualification_run_id": value["run_id"],
        "qualification_report": str(relative),
        "qualification_report_sha256": _sha256(report),
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    _write(LOCK, lock)
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
    parsed = parser.parse_args(arguments)
    if parsed.command == "start":
        result = start()
    elif parsed.command == "status":
        result = status()
    elif parsed.command == "logs":
        current = _current()
        return subprocess.run(
            ["tail", "-f", str(ROOT / str(current["log"]))], check=False
        ).returncode
    elif parsed.command == "resume":
        result = resume()
    elif parsed.command == "stop":
        result = stop(parsed.force)
    elif parsed.command == "verify":
        result = verify()
    else:
        result = freeze_qualification(parsed.directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
