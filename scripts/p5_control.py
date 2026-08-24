"""Cross-platform background control for P5 qualification and pilot runs."""

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

from neuromorphic.training.p5_config import (
    P5PilotConfig,
    P5QualificationConfig,
    load_p5_pilot_config,
    load_p5_qualification_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "artifacts/p5/control"
CURRENT = CONTROL / "current.json"
CI_LOCK = ROOT / "artifacts/p5/ci-lock.json"
PROFILE_CONFIGS = {
    "qualification-mps": ROOT / "configs/experiments/p5/qualification-mps.yaml",
    "qualification-cuda": ROOT / "configs/experiments/p5/qualification-cuda.yaml",
    "pilot-mps": ROOT / "configs/experiments/p5/pilot-mps.yaml",
    "pilot-cuda": ROOT / "configs/experiments/p5/pilot-cuda.yaml",
}
type Profile = Literal["qualification-mps", "qualification-cuda", "pilot-mps", "pilot-cuda"]
TERMINAL = {"qualification_passed", "pilot_passed", "pilot_failed", "failed"}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True, check=True, capture_output=True
    ).stdout.strip()


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        return (
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) "
                    "{ exit 0 } else { exit 1 }",
                ],
                check=False,
                capture_output=True,
            ).returncode
            == 0
        )
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
    if sys.platform == "win32":
        inspected = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f'(Get-CimInstance Win32_Process -Filter "ProcessId = {pid}").CommandLine',
            ],
            text=True,
            check=False,
            capture_output=True,
        )
    else:
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


def _under_root(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("P5 artifact path is missing")
    path = (ROOT / value).resolve()
    path.relative_to(ROOT.resolve())
    return path


def _load_profile(path: Path) -> P5QualificationConfig | P5PilotConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("P5 config must be a YAML object")
    if raw.get("schema_version") == "p5-qualification-v1":
        return load_p5_qualification_config(path)
    return load_p5_pilot_config(path)


def _lock_for(config: P5QualificationConfig | P5PilotConfig, name: str) -> Path:
    return ROOT / config.control_root.parent / f"{name}-lock.json"


def _require_lock(path: Path, *, head: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} lock is missing")
    value = _json(path)
    if value.get("status") != "PASSED" or value.get("git_commit") != head:
        raise RuntimeError(f"{label} lock is failed or belongs to another commit")
    evidence = value.get(f"{label}_report") or value.get(f"{label}_selection")
    checksum = value.get(f"{label}_report_sha256") or value.get(f"{label}_selection_sha256")
    if not isinstance(evidence, str) or not isinstance(checksum, str):
        raise RuntimeError(f"{label} lock evidence is incomplete")
    path_value = _under_root(evidence)
    if not path_value.is_file() or _sha256(path_value) != checksum:
        raise RuntimeError(f"{label} lock evidence checksum does not match")
    return value


def _evidence_hashes(config: P5QualificationConfig | P5PilotConfig) -> dict[str, str | None]:
    qualification = _lock_for(config, "qualification")
    return {
        "qualification_lock_sha256": (_sha256(qualification) if qualification.is_file() else None),
        "ci_lock_sha256": _sha256(CI_LOCK) if CI_LOCK.is_file() else None,
    }


def _preflight(config: P5QualificationConfig | P5PilotConfig, *, for_resume: bool = False) -> str:
    if _git("status", "--porcelain"):
        raise RuntimeError("P5 background runs require a clean worktree")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "origin/main") != head:
        raise RuntimeError("P5 background runs require HEAD == origin/main")
    if isinstance(config, P5PilotConfig):
        ci = _json(CI_LOCK) if CI_LOCK.is_file() else {}
        if ci.get("git_commit") != head or ci.get("conclusion") != "success":
            raise RuntimeError("P5 pilot requires a successful same-SHA CI lock")
        _require_lock(_lock_for(config, "qualification"), head=head, label="qualification")
    if sys.version_info[:2] != (3, 12) or torch.__version__.split("+")[0] != "2.12.1":
        raise RuntimeError("P5 requires Python 3.12 and PyTorch 2.12.1")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("P5 MPS profile requires an available MPS backend")
    if config.device == "mps" and sys.platform == "darwin":
        power = subprocess.run(
            ["pmset", "-g", "batt"], text=True, check=True, capture_output=True
        ).stdout
        if "AC Power" not in power:
            raise RuntimeError("connect the Mac to AC power before starting P5")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P5 CUDA profile requires an available CUDA backend")
    if shutil.disk_usage(ROOT).free < 100 * 1024**3:
        raise RuntimeError("P5 background runs require at least 100 GiB free")
    if CURRENT.is_file():
        current = _json(CURRENT)
        if _alive(int(current.get("pid", -1))):
            raise RuntimeError("another P5 background process is active")
        if not for_resume and "pilot" in str(current.get("profile")):
            directory = _under_root(current.get("artifact_dir"))
            registry = directory / "registry.json"
            if registry.is_file() and _json(registry).get("status") not in TERMINAL:
                raise RuntimeError("a resumable P5 pilot exists; use resume instead of start")
    return head


def _prepare(profile: Profile, *, head: str) -> tuple[Path, str]:
    raw = yaml.safe_load(PROFILE_CONFIGS[profile].read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("P5 profile must be a YAML object")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"p5-{profile}-{head[:8]}-{timestamp}"
    raw["run_id"] = run_id
    if raw.get("schema_version") == "p5-pilot-v1":
        raw["expected_git_commit"] = head
    runtime = CONTROL / run_id / f"{profile}.runtime.yaml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    _load_profile(runtime)
    return runtime, run_id


def _launch(runtime: Path, run_id: str, profile: Profile, *, resumed: bool) -> dict[str, Any]:
    command = [sys.executable, "-m", "neuromorphic.training.run", "--config", str(runtime)]
    if sys.platform == "win32":
        launch_command = command
        flags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
        )
        start_new_session = False
    else:
        launch_command = (
            ["/usr/bin/nohup", "/usr/bin/caffeinate", "-ims", *command]
            if sys.platform == "darwin"
            else ["nohup", *command]
        )
        flags = 0
        start_new_session = True
    log = runtime.parent / "runner.log"
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            launch_command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            start_new_session=start_new_session,
        )
        try:
            return_code = process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            return_code = None
    if return_code is not None:
        tail = log.read_text(encoding="utf-8")[-2000:] if log.is_file() else ""
        raise RuntimeError(
            f"P5 background process exited during launch with code {return_code}: {tail}"
        )
    launched = {
        "schema_version": "p5-launch-v1",
        "run_id": run_id,
        "profile": profile,
        "pid": process.pid,
        "process_group": process.pid,
        "resumed": resumed,
        "started_at": datetime.now(UTC).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "python": sys.executable,
        "runtime_config": str(runtime.relative_to(ROOT)),
        "runtime_config_sha256": _sha256(runtime),
        "artifact_dir": f"artifacts/runs/{run_id}",
        "log": str(log.relative_to(ROOT)),
        "command": launch_command,
        **_evidence_hashes(_load_profile(runtime)),
    }
    _write(runtime.parent / "launch.json", launched)
    _write(CURRENT, launched)
    return launched


def start(profile: Profile) -> dict[str, Any]:
    config = _load_profile(PROFILE_CONFIGS[profile])
    head = _preflight(config)
    runtime, run_id = _prepare(profile, head=head)
    return _launch(runtime, run_id, profile, resumed=False)


def status() -> dict[str, Any]:
    current = _json(CURRENT)
    result = dict(current)
    alive = _alive(int(current["pid"]))
    result["process_alive"] = alive
    result["process_matches_launch"] = _process_matches_launch(current)
    directory = _under_root(current["artifact_dir"])
    registry = directory / "registry.json"
    summary = directory / "summary.json"
    if registry.is_file():
        value = _json(registry)
        result["suite_status"] = value.get("status")
        cells = value.get("cells", [])
        if isinstance(cells, list):
            result["completed_cells"] = sum(
                isinstance(cell, dict) and cell.get("status") == "COMPLETED" for cell in cells
            )
            result["total_cells"] = len(cells)
            result["current_candidate"] = next(
                (
                    cell.get("candidate_id")
                    for cell in cells
                    if isinstance(cell, dict) and cell.get("status") == "RUNNING"
                ),
                None,
            )
    elif summary.is_file():
        result["suite_status"] = _json(summary).get("status")
    else:
        result["suite_status"] = "running" if alive else "failed"
    heartbeat = directory / "heartbeat.json"
    if heartbeat.is_file():
        result["heartbeat"] = _json(heartbeat)
    result["terminal"] = result["suite_status"] in TERMINAL
    result["resume_allowed"] = (
        not alive and result["suite_status"] == "stopped" and "pilot" in str(current["profile"])
    )
    result["free_bytes"] = shutil.disk_usage(ROOT).free
    return result


def resume() -> dict[str, Any]:
    current = _json(CURRENT)
    if _alive(int(current["pid"])):
        raise RuntimeError("P5 process is already active")
    if "pilot" not in str(current["profile"]):
        raise RuntimeError("only P5 pilot runs can be resumed")
    directory = _under_root(current["artifact_dir"])
    registry = _json(directory / "registry.json")
    if registry.get("status") != "stopped":
        raise RuntimeError("P5 pilot is not in a resumable state")
    original = _under_root(current["runtime_config"])
    raw = yaml.safe_load(original.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("P5 runtime config is invalid")
    raw["resume"] = str(directory.relative_to(ROOT))
    runtime = original.parent / "resume.runtime.yaml"
    runtime.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = _load_profile(runtime)
    _preflight(config, for_resume=True)
    for name, value in _evidence_hashes(config).items():
        if current.get(name) != value:
            raise RuntimeError(f"P5 resume evidence lock changed: {name}")
    return _launch(runtime, str(current["run_id"]), cast(Profile, current["profile"]), resumed=True)


def stop(force: bool) -> dict[str, object]:
    current = _json(CURRENT)
    pid = int(current["pid"])
    if force and _alive(pid):
        if not _process_matches_launch(current):
            raise RuntimeError("refusing to signal a PID that does not match the P5 launch")
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=True)
        else:
            os.killpg(int(current["process_group"]), signal.SIGTERM)
        return {"run_id": current["run_id"], "forced": True}
    if not _alive(pid):
        return {"run_id": current["run_id"], "already_stopped": True}
    directory = _under_root(current["artifact_dir"])
    (directory / "STOP").touch()
    return {"run_id": current["run_id"], "graceful_stop_requested": True}


def verify() -> dict[str, object]:
    current = _json(CURRENT)
    if _alive(int(current["pid"])):
        raise RuntimeError("wait for the P5 process to stop before verification")
    module = importlib.import_module("neuromorphic.training.p5_suite")
    verifier = module.verify_p5_run
    return cast(dict[str, object], verifier(_under_root(current["artifact_dir"])))


def record_ci() -> dict[str, object]:
    if _git("status", "--porcelain") or _git("rev-parse", "HEAD") != _git(
        "rev-parse", "origin/main"
    ):
        raise RuntimeError("P5 CI lock requires clean HEAD == origin/main")
    head = _git("rev-parse", "HEAD")
    remote = _git("remote", "get-url", "origin")
    if remote.startswith("git@github.com:"):
        slug = remote.removeprefix("git@github.com:").removesuffix(".git")
    elif "github.com/" in remote:
        slug = remote.split("github.com/", 1)[1].removesuffix(".git")
    else:
        raise RuntimeError("origin is not a GitHub repository")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{slug}/actions/runs?head_sha={head}&per_page=100",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "neuromorphic-p5"},
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
        raise RuntimeError("no successful CI run exists for HEAD")
    selected = max(runs, key=lambda run: str(run.get("updated_at", "")))
    lock = {
        "schema_version": "p5-ci-lock-v1",
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
            result = start(cast(Profile, parsed.profile))
        elif parsed.command == "status":
            result = status()
        elif parsed.command == "resume":
            result = resume()
        elif parsed.command == "stop":
            result = stop(parsed.force)
        elif parsed.command == "verify":
            result = verify()
        elif parsed.command == "record-ci":
            result = record_ci()
        else:
            current = _json(CURRENT)
            log = _under_root(current["log"])
            if sys.platform == "win32":
                escaped = str(log).replace("'", "''")
                command = [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Get-Content -LiteralPath '{escaped}' -Tail 200"
                    + (" -Wait" if _alive(int(current["pid"])) else ""),
                ]
            else:
                command = ["tail", "-n", "200"]
                if _alive(int(current["pid"])):
                    command.append("-f")
                command.append(str(log))
            return subprocess.run(command, check=False).returncode
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(json.dumps({"error": str(error), "exit_code": 2}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
