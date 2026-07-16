"""Validated configuration for the P2 modular-network qualification suite."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

P2_TASK_ORDER = (
    "associative_recall.v1",
    "delayed_rule_switch.v1",
    "small_graph.v1",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class P2ModelConfig(_StrictModel):
    """Frozen dimensions and routing policy for the minimal P2 network."""

    feature_dim: int = Field(default=128, ge=16)
    episodic_slots: int = Field(default=16, ge=12)
    working_slots: int = Field(default=4, ge=1)
    working_slot_dim: int = Field(default=32, ge=4)
    action_embedding_dim: int = Field(default=32, ge=4)
    task_embedding_dim: int = Field(default=16, ge=4)
    router_top_k: int = Field(default=2, ge=1)
    router_capacity_factor: float = Field(default=1.25, ge=1.0)

    @model_validator(mode="after")
    def validate_frozen_shape(self) -> P2ModelConfig:
        if self.router_top_k != 2:
            raise ValueError("P2 requires router_top_k=2")
        if self.working_slots * self.working_slot_dim != self.feature_dim:
            raise ValueError("working slot capacity must equal feature_dim")
        return self


class P2OptimizerConfig(_StrictModel):
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-2, ge=0.0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)


class P2BudgetConfig(_StrictModel):
    """Exact update budgets; the CI profile may only reduce these values."""

    pretrain_steps_per_stage: int = Field(default=100, ge=1)
    joint_steps_per_task: int = Field(default=200, ge=1)
    batch_size: int = Field(default=64, ge=1)
    train_size: int = Field(default=64, ge=1)
    validation_size: int = Field(default=32, ge=1)
    test_size: int = Field(default=32, ge=1)
    ood_size: int = Field(default=32, ge=1)
    validation_interval_per_task: int = Field(default=25, ge=1)
    checkpoint_interval_per_task: int = Field(default=50, ge=1)
    tbptt_steps: int = Field(default=32, ge=1)


class P2LossWeights(_StrictModel):
    primary: float = Field(default=1.0, ge=0.0)
    episodic_retrieval: float = Field(default=0.1, ge=0.0)
    episodic_separation: float = Field(default=0.01, ge=0.0)
    working_consistency: float = Field(default=0.05, ge=0.0)
    working_gate: float = Field(default=0.001, ge=0.0)
    predictive_next_state: float = Field(default=0.1, ge=0.0)
    router_load_balance: float = Field(default=0.01, ge=0.0)
    router_communication: float = Field(default=0.001, ge=0.0)

    def by_loss_name(self) -> dict[str, float]:
        return {
            "primary": self.primary,
            "episodic.retrieval": self.episodic_retrieval,
            "episodic.separation": self.episodic_separation,
            "working.state_consistency": self.working_consistency,
            "working.gate_regularization": self.working_gate,
            "predictive.next_state": self.predictive_next_state,
            "router.load_balance": self.router_load_balance,
            "router.communication_cost": self.router_communication,
        }


class P2TelemetryConfig(_StrictModel):
    paired_training: bool = True
    reducer_version: str = "p2-reducer-v1"
    baseline_version: str = "p2-modular-v1"


class P2SuiteConfig(_StrictModel):
    """Complete P2 pretraining and paired joint-smoke configuration."""

    schema_version: Literal["p2-suite-v1"] = "p2-suite-v1"
    profile: Literal["gate", "ci"] = "gate"
    seed: int = Field(default=7, ge=0)
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    output_root: Path = Path("artifacts/runs")
    run_id: str | None = None
    resume: Path | None = None
    task_order: tuple[str, str, str] = P2_TASK_ORDER
    model: P2ModelConfig = P2ModelConfig()
    optimizer: P2OptimizerConfig = P2OptimizerConfig()
    budget: P2BudgetConfig = P2BudgetConfig()
    losses: P2LossWeights = P2LossWeights()
    telemetry: P2TelemetryConfig = P2TelemetryConfig()

    @model_validator(mode="after")
    def validate_suite(self) -> P2SuiteConfig:
        if self.task_order != P2_TASK_ORDER:
            raise ValueError(f"task_order must be {P2_TASK_ORDER!r}")
        if self.profile == "gate":
            if self.seed != 7:
                raise ValueError("the P2 gate profile requires seed 7")
            if self.budget.pretrain_steps_per_stage != 100:
                raise ValueError("the P2 gate profile requires 100 updates per pretrain stage")
            if self.budget.joint_steps_per_task != 200:
                raise ValueError("the P2 gate profile requires 200 updates per task")
            if not self.telemetry.paired_training:
                raise ValueError("the P2 gate profile requires paired telemetry training")
        return self

    def checkpoint_compatible_dict(self) -> dict[str, object]:
        """Return semantic fields; observation and output locations are excluded."""

        return self.model_dump(
            mode="json",
            exclude={"run_id", "output_root", "resume", "telemetry"},
        )


def load_p2_suite_config(path: Path) -> P2SuiteConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("P2 suite configuration must be a YAML object")
    return P2SuiteConfig.model_validate(value)


__all__ = [
    "P2_TASK_ORDER",
    "P2BudgetConfig",
    "P2LossWeights",
    "P2ModelConfig",
    "P2OptimizerConfig",
    "P2SuiteConfig",
    "P2TelemetryConfig",
    "load_p2_suite_config",
]
