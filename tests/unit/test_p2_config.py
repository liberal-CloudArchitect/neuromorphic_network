from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neuromorphic.training.p2_config import P2SuiteConfig, load_p2_suite_config


def test_gate_and_ci_p2_configs_parse() -> None:
    gate = load_p2_suite_config(Path("configs/experiments/p2/gate.yaml"))
    ci = load_p2_suite_config(Path("configs/experiments/p2/ci.yaml"))

    assert gate.profile == "gate"
    assert gate.budget.pretrain_steps_per_stage == 100
    assert gate.budget.joint_steps_per_task == 200
    assert gate.telemetry.paired_training
    assert ci.profile == "ci"
    assert ci.model.feature_dim == 32


def test_gate_profile_rejects_weakened_budget_or_telemetry() -> None:
    with pytest.raises(ValidationError, match="100 updates"):
        P2SuiteConfig.model_validate(
            {
                "schema_version": "p2-suite-v1",
                "profile": "gate",
                "budget": {"pretrain_steps_per_stage": 99},
            }
        )
    with pytest.raises(ValidationError, match="paired telemetry"):
        P2SuiteConfig.model_validate(
            {
                "schema_version": "p2-suite-v1",
                "profile": "gate",
                "telemetry": {"paired_training": False},
            }
        )


def test_checkpoint_compatibility_ignores_observation_and_output_fields() -> None:
    base = P2SuiteConfig.model_validate(
        {"schema_version": "p2-suite-v1", "profile": "ci", "run_id": "one"}
    )
    changed = base.model_copy(
        update={
            "run_id": "two",
            "output_root": Path("elsewhere"),
            "telemetry": base.telemetry.model_copy(update={"paired_training": False}),
        }
    )

    assert base.checkpoint_compatible_dict() == changed.checkpoint_compatible_dict()


def test_model_rejects_non_frozen_working_capacity() -> None:
    with pytest.raises(ValidationError, match="working slot capacity"):
        P2SuiteConfig.model_validate(
            {
                "schema_version": "p2-suite-v1",
                "profile": "ci",
                "model": {"feature_dim": 32, "working_slots": 4, "working_slot_dim": 16},
            }
        )
