from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neuromorphic.training.config import RunConfig, load_run_config


def test_smoke_config_parses_and_freezes_resume_compatibility() -> None:
    path = Path("configs/experiments/p1/associative_recall_smoke.yaml")
    config = load_run_config(path)
    assert config.task.task_id == "associative_recall.v1"
    assert config.training.max_steps == 200
    assert "resume" not in config.checkpoint_compatible_dict()


def test_sensitivity_modes_require_explicit_matching_targets() -> None:
    with pytest.raises(ValidationError, match="step and token targets"):
        RunConfig.model_validate(
            {
                "matching_mode": "train_compute",
                "task": {"task_id": "associative_recall.v1"},
            }
        )
    config = RunConfig.model_validate(
        {
            "matching_mode": "inference_cost",
            "task": {"task_id": "associative_recall.v1"},
            "model": {"target_macs": 1000, "target_latency_ms": 1.0},
        }
    )
    assert config.model.cost_tolerance == 0.05


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"task": {"task_id": "associative_recall.v1"}, "unknown": True})
