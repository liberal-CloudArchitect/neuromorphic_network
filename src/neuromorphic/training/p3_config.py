"""Strict P3 suite and experiment-cell configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

P3_TASK_ORDER = (
    "associative_recall.v1",
    "delayed_rule_switch.v1",
    "small_graph.v1",
)
P3_FORMAL_SEEDS = (17, 29, 43)
type P3ModelId = Literal["modular", "gru", "transformer"]
P3_MODELS: tuple[P3ModelId, ...] = ("modular", "gru", "transformer")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class P3DataConfig(_StrictModel):
    train: int = Field(ge=1)
    validation: int = Field(ge=1)
    analysis: int = Field(ge=1)
    test: int = Field(ge=1)
    ood: int = Field(ge=1)


class P3BudgetConfig(_StrictModel):
    batch_size: int = Field(default=64, ge=1)
    shared_steps_per_task: int = Field(default=5_000, ge=1)
    per_task_steps: int = Field(default=5_000, ge=1)
    causal_steps: int = Field(default=5_000, ge=1)
    continual_steps_per_stage: int = Field(default=1_500, ge=1)
    validation_interval: int = Field(default=100, ge=1)
    checkpoint_interval: int = Field(default=100, ge=1)
    patience: int = Field(default=10, ge=1)
    min_delta: float = Field(default=0.001, ge=0.0)
    wall_clock_hours: float = Field(default=72.0, gt=0.0)
    bootstrap_samples: int = Field(default=10_000, ge=1)


class P3OptimizerConfig(_StrictModel):
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-2, ge=0.0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)


class InterventionSpec(_StrictModel):
    schema_version: Literal["intervention-v1"] = "intervention-v1"
    target: Literal[
        "none",
        "episodic_memory.v1",
        "working_memory.v1",
        "predictive_adapter.v1",
        "sparse_router.v1",
        "action_selector.v1",
        "sensory_encoder.v1",
    ] = "none"
    operation: Literal[
        "none",
        "no_read_write",
        "reset_every_step",
        "loss_zero",
        "dense",
        "fixed",
        "random",
        "direct_head",
        "frozen_random",
        "shallow",
    ] = "none"
    phase: Literal["train", "evaluate", "both"] = "both"
    seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_target_operation(self) -> InterventionSpec:
        allowed = {
            "none": {"none"},
            "episodic_memory.v1": {"no_read_write"},
            "working_memory.v1": {"reset_every_step"},
            "predictive_adapter.v1": {"loss_zero"},
            "sparse_router.v1": {"dense", "fixed", "random"},
            "action_selector.v1": {"direct_head"},
            "sensory_encoder.v1": {"frozen_random", "shallow"},
        }
        if self.operation not in allowed[self.target]:
            raise ValueError("intervention target and operation are incompatible")
        return self


class P3ExperimentCell(_StrictModel):
    schema_version: Literal["p3-experiment-v1"] = "p3-experiment-v1"
    cell_id: str = Field(min_length=1)
    cell_type: Literal["pilot", "shared", "per_task", "causal", "control", "continual"]
    regime: Literal["shared", "per_task", "continual"]
    model_id: Literal["modular", "gru", "transformer"]
    variant_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    task_id: str | None = None
    max_steps: int = Field(ge=0)
    mandatory: bool = True
    intervention: InterventionSpec = InterventionSpec()


class P3SuiteConfig(_StrictModel):
    schema_version: Literal["p3-suite-v1"] = "p3-suite-v1"
    protocol_version: Literal["p3-protocol-v2"] = "p3-protocol-v2"
    profile: Literal["qualification", "full"]
    qualification_only: bool
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    output_root: Path = Path("artifacts/runs")
    control_root: Path = Path("artifacts/p3/control")
    run_id: str | None = None
    resume: Path | None = None
    seeds: tuple[int, ...]
    data: P3DataConfig
    budget: P3BudgetConfig
    optimizer: P3OptimizerConfig = P3OptimizerConfig()
    task_order: tuple[str, str, str] = P3_TASK_ORDER
    expected_git_commit: str | None = None
    qualification_report: Path | None = None

    @model_validator(mode="after")
    def validate_profile(self) -> P3SuiteConfig:
        if self.task_order != P3_TASK_ORDER:
            raise ValueError(f"task_order must be {P3_TASK_ORDER!r}")
        if self.profile == "qualification":
            if not self.qualification_only or self.seeds != (7,):
                raise ValueError("qualification requires qualification_only=true and seed 7")
            expected_data = (64, 32, 32, 32, 32)
            actual_data = (
                self.data.train,
                self.data.validation,
                self.data.analysis,
                self.data.test,
                self.data.ood,
            )
            if actual_data != expected_data or self.budget.batch_size > 16:
                raise ValueError("qualification data and batch budget are frozen")
            if self.budget.bootstrap_samples != 200:
                raise ValueError("qualification requires 200 bootstrap samples")
        else:
            if self.qualification_only or self.seeds != P3_FORMAL_SEEDS:
                raise ValueError("full P3 requires formal seeds and qualification_only=false")
            if self.budget.shared_steps_per_task != 5_000:
                raise ValueError("full shared budget requires 5000 updates per task")
            if self.budget.wall_clock_hours != 72.0:
                raise ValueError("full P3 wall-clock limit is frozen at 72 hours")
            if self.budget.bootstrap_samples != 10_000:
                raise ValueError("full P3 requires 10000 bootstrap samples")
            expected_data = (8_192, 2_048, 512, 2_048, 2_048)
            actual_data = (
                self.data.train,
                self.data.validation,
                self.data.analysis,
                self.data.test,
                self.data.ood,
            )
            if actual_data != expected_data or self.budget.batch_size != 64:
                raise ValueError("full P3 data and batch budgets are frozen")
        return self

    def compatible_dict(self) -> dict[str, object]:
        return self.model_dump(
            mode="json",
            exclude={"run_id", "resume", "output_root", "control_root"},
        )

    def config_hash(self) -> str:
        encoded = json.dumps(self.compatible_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def matrix(self) -> tuple[P3ExperimentCell, ...]:
        return build_p3_matrix(self)


def _cell(
    cell_type: Literal["pilot", "shared", "per_task", "causal", "control", "continual"],
    regime: Literal["shared", "per_task", "continual"],
    model: Literal["modular", "gru", "transformer"],
    variant: str,
    seed: int,
    steps: int,
    *,
    task: str | None = None,
    intervention: InterventionSpec | None = None,
) -> P3ExperimentCell:
    task_label = "all" if task is None else task.replace(".v1", "")
    cell_id = f"{cell_type}__{regime}__{model}__{variant}__s{seed}__{task_label}"
    return P3ExperimentCell(
        cell_id=cell_id,
        cell_type=cell_type,
        regime=regime,
        model_id=model,
        variant_id=variant,
        seed=seed,
        task_id=task,
        max_steps=steps,
        intervention=intervention or InterventionSpec(),
    )


def build_p3_matrix(config: P3SuiteConfig) -> tuple[P3ExperimentCell, ...]:
    cells: list[P3ExperimentCell] = []
    if config.profile == "qualification":
        for model in P3_MODELS:
            for preset in range(4):
                cells.append(_cell("pilot", "shared", model, f"preset-{preset}", 7, 2))
    for seed in config.seeds:
        shared_steps = (
            4 * len(P3_TASK_ORDER)
            if config.profile == "qualification"
            else config.budget.shared_steps_per_task * len(P3_TASK_ORDER)
        )
        for model in P3_MODELS:
            cells.append(_cell("shared", "shared", model, "full", seed, shared_steps))
            for task in P3_TASK_ORDER:
                steps = 4 if config.profile == "qualification" else config.budget.per_task_steps
                cells.append(_cell("per_task", "per_task", model, "full", seed, steps, task=task))
            continual_steps = (
                2 * len(P3_TASK_ORDER)
                if config.profile == "qualification"
                else config.budget.continual_steps_per_stage * len(P3_TASK_ORDER)
            )
            cells.append(_cell("continual", "continual", model, "latin", seed, continual_steps))
        causal_specs = (
            (
                "episodic-no-read-write",
                "associative_recall.v1",
                InterventionSpec(target="episodic_memory.v1", operation="no_read_write", seed=seed),
            ),
            (
                "working-reset",
                "delayed_rule_switch.v1",
                InterventionSpec(
                    target="working_memory.v1", operation="reset_every_step", seed=seed
                ),
            ),
            (
                "predictive-loss-zero",
                "small_graph.v1",
                InterventionSpec(target="predictive_adapter.v1", operation="loss_zero", seed=seed),
            ),
        )
        for variant, task, intervention in causal_specs:
            steps = 4 if config.profile == "qualification" else config.budget.causal_steps
            cells.append(
                _cell(
                    "causal",
                    "per_task",
                    "modular",
                    variant,
                    seed,
                    steps,
                    task=task,
                    intervention=intervention,
                )
            )
        controls = (
            ("router-dense", "sparse_router.v1", "dense"),
            ("router-fixed", "sparse_router.v1", "fixed"),
            ("router-random", "sparse_router.v1", "random"),
            ("direct-head", "action_selector.v1", "direct_head"),
            ("frozen-random-encoder", "sensory_encoder.v1", "frozen_random"),
            ("shallow-encoder", "sensory_encoder.v1", "shallow"),
        )
        for variant, target, operation in controls:
            cells.append(
                _cell(
                    "control",
                    "shared",
                    "modular",
                    variant,
                    seed,
                    0,
                    intervention=InterventionSpec(
                        target=target,  # type: ignore[arg-type]
                        operation=operation,  # type: ignore[arg-type]
                        phase="evaluate",
                        seed=seed,
                    ),
                )
            )
    ids = [cell.cell_id for cell in cells]
    if len(ids) != len(set(ids)):
        raise RuntimeError("P3 matrix generated duplicate cell IDs")
    return tuple(cells)


def load_p3_suite_config(path: Path) -> P3SuiteConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("P3 suite configuration must be a YAML object")
    return P3SuiteConfig.model_validate(value)


__all__ = [
    "InterventionSpec",
    "P3BudgetConfig",
    "P3DataConfig",
    "P3ExperimentCell",
    "P3SuiteConfig",
    "build_p3_matrix",
    "load_p3_suite_config",
]
