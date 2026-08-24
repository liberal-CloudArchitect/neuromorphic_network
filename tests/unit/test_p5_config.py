from __future__ import annotations

from pathlib import Path

import pytest

from neuromorphic.training.p5_config import (
    P5QualificationConfig,
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
