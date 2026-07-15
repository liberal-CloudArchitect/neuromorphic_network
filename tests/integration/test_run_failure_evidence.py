from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuromorphic.training.config import RunConfig
from neuromorphic.training.run import execute, main


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig.model_validate(
        {
            "seed": 7,
            "device": "cpu",
            "output_root": str(tmp_path),
            "run_id": "injected-failure",
            "task": {"task_id": "associative_recall.v1", "profile": "smoke"},
            "model": {"kind": "gru", "hidden_size": 8},
            "training": {"batch_size": 2, "max_steps": 1},
        }
    )


def test_failed_run_preserves_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_training(**_: object) -> dict[str, object]:
        raise FloatingPointError("injected non-finite loss")

    monkeypatch.setattr("neuromorphic.training.run.train_baseline", fail_training)
    with pytest.raises(FloatingPointError, match="injected"):
        execute(_config(tmp_path))
    manifest = json.loads(
        (tmp_path / "injected-failure" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["failure"] == {
        "type": "FloatingPointError",
        "message": "injected non-finite loss",
    }


def test_cli_configuration_error_has_distinct_exit_code(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("task: {}\n", encoding="utf-8")
    assert main(["--config", str(invalid)]) == 2
