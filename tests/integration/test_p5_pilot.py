from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from neuromorphic.training.p5_config import P5PilotConfig
from neuromorphic.training.p5_suite import execute_p5_pilot, verify_p5_run


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p5_pilot_micro_writes_v6_checkpoints_and_selection(tmp_path: Path) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, check=True, capture_output=True
    ).stdout.strip()
    report = tmp_path / "qualification-report.json"
    report.write_text(json.dumps({"status": "PASSED"}), encoding="utf-8")
    lock = tmp_path / "qualification-lock.json"
    lock.write_text(
        json.dumps(
            {
                "status": "PASSED",
                "git_commit": commit,
                "qualification_report": str(report),
                "qualification_report_sha256": _sha256(report),
            }
        ),
        encoding="utf-8",
    )
    config = P5PilotConfig(
        device="mps",
        control_root=tmp_path / "control",
        qualification_report=lock,
        output_root=tmp_path,
        run_id="pilot-fixture",
    ).model_copy(
        update={
            "device": "cpu",
            "train_samples": 64,
            "validation_samples": 8,
            "batch_size": 8,
            "steps_per_preset": 2,
            "validation_interval": 1,
            "checkpoint_interval": 1,
        }
    )

    result = execute_p5_pilot(config)

    assert result["status"] in {"pilot_passed", "pilot_failed"}
    directory = tmp_path / "pilot-fixture"
    for preset in ("preset-0", "preset-1", "preset-2", "preset-3"):
        assert (directory / "cells" / preset / "checkpoint.pt").is_file()
        assert (directory / "cells" / preset / "summary.json").is_file()
    verified = verify_p5_run(directory)
    assert verified["cells"] == 4
    assert verified["missing_cells"] == []
