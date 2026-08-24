from __future__ import annotations

from pathlib import Path

import pytest

from neuromorphic.training.p5_config import (
    P5_PILOT_PRESETS,
    P5PilotConfig,
    P5QualificationConfig,
    load_p5_pilot_config,
    load_p5_qualification_config,
)


def test_p5_qualification_config_is_frozen_and_hashable() -> None:
    config = load_p5_qualification_config(Path("configs/experiments/p5/qualification-cpu.yaml"))

    assert config.seed == 7
    assert config.steps_per_task == 4
    assert config.dual_route_fraction == 0.25
    assert len(config.config_hash()) == 64


def test_p5_qualification_rejects_post_result_threshold_changes() -> None:
    with pytest.raises(ValueError, match="dual-route fraction"):
        P5QualificationConfig(dual_route_fraction=0.2)


def test_p5_pilot_is_validation_only_and_device_scoped() -> None:
    config = load_p5_pilot_config(Path("configs/experiments/p5/pilot-cuda.yaml"))

    assert config.device == "cuda"
    assert config.train_samples == 8192
    assert config.validation_samples == 2048
    assert config.steps_per_preset == 1000
    assert len(P5_PILOT_PRESETS) == 4
    assert (
        config.config_hash() == config.model_copy(update={"telemetry_enabled": True}).config_hash()
    )


def test_p5_pilot_rejects_structural_budget_changes() -> None:
    with pytest.raises(ValueError, match="structural optimizer"):
        P5PilotConfig(
            device="cuda",
            control_root=Path("artifacts/p5-cuda/control"),
            qualification_report=Path("artifacts/p5-cuda/qualification-lock.json"),
            dual_route_fraction=0.2,
        )
