from __future__ import annotations

from pathlib import Path

import pytest

from neuromorphic.training.p5_config import (
    P5_PILOT_PRESETS,
    P5MechanismConfig,
    P5PilotConfig,
    P5QualificationConfig,
    load_p5_mechanism_config,
    load_p5_pilot_config,
    load_p5_qualification_config,
)


def test_p5_qualification_config_is_frozen_and_hashable() -> None:
    config = load_p5_qualification_config(Path("configs/experiments/p5/qualification-cpu.yaml"))

    assert config.seed == 7
    assert config.steps_per_task == 4
    assert config.dual_route_fraction == 0.25
    assert config.protocol_version == "p5-protocol-v2"
    assert len(config.config_hash()) == 64


def test_p5_qualification_rejects_post_result_threshold_changes() -> None:
    with pytest.raises(ValueError, match="dual-route fraction"):
        P5QualificationConfig(dual_route_fraction=0.2)


def test_p5_configs_reject_retired_protocol_v1() -> None:
    with pytest.raises(ValueError, match="p5-protocol-v2"):
        P5QualificationConfig(protocol_version="p5-protocol-v1")  # type: ignore[arg-type]


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


def test_p5_mechanism_matrix_is_eight_cells_per_seed() -> None:
    micro = load_p5_mechanism_config(Path("configs/experiments/p5/mechanism-ci.yaml"))
    formal = load_p5_mechanism_config(Path("configs/experiments/p5/mechanism-cuda.yaml"))

    assert len(micro.matrix()) == 8
    assert len(formal.matrix()) == 24
    assert sum(cell.retrained for cell in formal.matrix()) == 12
    assert {cell.variant for cell in micro.matrix()} == {
        "full",
        "predictor-off",
        "surprise-off",
        "no-dual-route",
        "acute-surprise-off",
        "shuffle-surprise",
        "dense-memory",
        "no-semantic-reservation",
    }


def test_p5_formal_mechanism_rejects_missing_qualification_lock() -> None:
    with pytest.raises(ValueError, match="requires qualification and pilot"):
        P5MechanismConfig(
            profile="mechanism",
            device="cuda",
            seeds=(17, 29, 43),
            train_samples=8192,
            validation_samples=2048,
            analysis_samples=512,
            test_samples=2048,
            batch_size=64,
            steps_per_task=5000,
            validation_interval=100,
            checkpoint_interval=100,
            patience=10,
            bootstrap_samples=10000,
            wall_clock_hours=48,
            selected_preset="preset-1",
            control_root=Path("artifacts/p5-cuda/control"),
            mechanism_qualification_report=Path(
                "artifacts/p5-cuda/mechanism-qualification-lock.json"
            ),
        )
