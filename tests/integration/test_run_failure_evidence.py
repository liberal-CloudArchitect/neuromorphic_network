from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neuromorphic.tasks import create_task
from neuromorphic.training.baselines import BaselineOutput, GRUBaseline
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
    task = create_task("associative_recall.v1")

    class NonFiniteGRU(GRUBaseline):
        def forward(self, inputs: torch.Tensor, valid_mask: torch.Tensor) -> BaselineOutput:
            logits = self.output_head(self.input_projection(inputs))
            return BaselineOutput(logits=logits * torch.tensor(float("nan")))

    model = NonFiniteGRU(input_dim=task.input_dim, num_classes=task.num_classes, hidden_size=8)
    monkeypatch.setattr("neuromorphic.training.run.build_baseline", lambda *_: model)
    with pytest.raises(FloatingPointError, match="loss is not finite"):
        execute(_config(tmp_path))
    manifest = json.loads(
        (tmp_path / "injected-failure" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["failure"] == {
        "type": "FloatingPointError",
        "message": "loss is not finite",
    }


def test_cli_configuration_error_has_distinct_exit_code(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("task: {}\n", encoding="utf-8")
    assert main(["--config", str(invalid)]) == 2
