"""Strict configuration and pre-registered matrix for P4 experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

P4_TASK_ORDER = (
    "associative_recall.v1",
    "delayed_rule_switch.v1",
    "small_graph.v1",
)
P4_FORMAL_SEEDS = (17, 29, 43)
P4_MODELS = cast(
    tuple[Literal["modular-v2", "gru", "transformer"], ...],
    ("modular-v2", "gru", "transformer"),
)
P4_PILOT_PRESETS: dict[str, tuple[float, float, float]] = {
    "preset-0": (1.0e-4, 1.0e-2, 0.05),
    "preset-1": (1.0e-4, 1.0e-2, 0.10),
    "preset-2": (3.0e-4, 1.0e-2, 0.05),
    "preset-3": (3.0e-4, 1.0e-2, 0.10),
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class P4DataConfig(_StrictModel):
    train: int = Field(ge=1)
    validation: int = Field(ge=1)
    analysis: int = Field(ge=1)
    test: int = Field(ge=1)
    ood: int = Field(ge=1)


class P4BudgetConfig(_StrictModel):
    batch_size: int = Field(ge=1)
    shared_steps_per_task: int = Field(ge=1)
    per_task_steps: int = Field(ge=1)
    continual_steps_per_stage: int = Field(ge=1)
    validation_interval: int = Field(ge=1)
    checkpoint_interval: int = Field(ge=1)
    patience: int = Field(ge=1)
    min_delta: float = Field(ge=0.0)
    wall_clock_hours: float = Field(gt=0.0)
    bootstrap_samples: int = Field(ge=1)


class P4OptimizerConfig(_StrictModel):
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-2, ge=0.0)
    temporal_loss_weight: float = Field(default=0.05, gt=0.0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)


class P4InterventionSpec(_StrictModel):
    schema_version: Literal["intervention-v2"] = "intervention-v2"
    operation: Literal[
        "none",
        "predictor_off",
        "loss_zero",
        "feedback_zero",
        "acute_feedback_off",
        "shuffle_forecast",
        "dense_memory",
        "legacy_capacity",
        "episodic_off",
        "working_reset",
        "direct_head",
        "frozen_random_encoder",
        "shallow_encoder",
    ] = "none"
    phase: Literal["train", "evaluate", "both"] = "both"


class P4ExperimentCell(_StrictModel):
    schema_version: Literal["p4-experiment-v1"] = "p4-experiment-v1"
    cell_id: str = Field(min_length=1)
    cell_type: Literal["pilot", "shared", "per_task", "causal", "control", "continual"]
    regime: Literal["shared", "per_task", "continual"]
    model_id: Literal["modular-v2", "gru", "transformer"]
    variant_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    task_id: str | None = None
    max_steps: int = Field(ge=0)
    mandatory: bool = True
    intervention: P4InterventionSpec = P4InterventionSpec()


class P4SuiteConfig(_StrictModel):
    schema_version: Literal["p4-suite-v1"] = "p4-suite-v1"
    protocol_version: Literal["p4-protocol-v1"] = "p4-protocol-v1"
    profile: Literal["qualification", "pilot", "mechanism", "full"]
    qualification_only: bool
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    output_root: Path = Path("artifacts/runs")
    control_root: Path = Path("artifacts/p4/control")
    run_id: str | None = None
    resume: Path | None = None
    seeds: tuple[int, ...]
    data: P4DataConfig
    budget: P4BudgetConfig
    optimizer: P4OptimizerConfig = P4OptimizerConfig()
    task_order: tuple[str, str, str] = P4_TASK_ORDER
    selected_preset: Literal["preset-0", "preset-1", "preset-2", "preset-3"] | None = None
    expected_git_commit: str | None = None
    qualification_report: Path | None = None
    pilot_lock: Path | None = None
    mechanism_report: Path | None = None

    @model_validator(mode="after")
    def validate_profile(self) -> P4SuiteConfig:
        if self.task_order != P4_TASK_ORDER:
            raise ValueError(f"task_order must be {P4_TASK_ORDER!r}")
        sizes = (
            self.data.train,
            self.data.validation,
            self.data.analysis,
            self.data.test,
            self.data.ood,
        )
        if self.profile == "qualification":
            if not self.qualification_only or self.seeds != (7,):
                raise ValueError("P4 qualification requires seed 7 and qualification_only=true")
            if sizes != (64, 32, 32, 32, 32) or self.budget.batch_size > 16:
                raise ValueError("P4 qualification data budget is frozen")
            if self.budget.bootstrap_samples != 200:
                raise ValueError("P4 qualification requires 200 bootstrap samples")
        elif self.profile == "pilot":
            if not self.qualification_only or self.seeds != (7,):
                raise ValueError("P4 pilot requires seed 7 and qualification_only=true")
            if sizes != (8_192, 2_048, 1, 1, 1) or self.budget.batch_size != 64:
                raise ValueError("P4 pilot may only access frozen train/validation budgets")
            if self.selected_preset is not None:
                raise ValueError("P4 pilot cannot consume a prior preset")
            if self.qualification_report is None:
                raise ValueError("P4 pilot requires a passed qualification lock")
        else:
            if self.qualification_only or self.seeds != P4_FORMAL_SEEDS:
                raise ValueError("formal P4 requires seeds 17, 29, 43")
            if sizes != (8_192, 2_048, 512, 2_048, 2_048):
                raise ValueError("formal P4 data sizes are frozen")
            if self.budget.batch_size != 64 or self.selected_preset is None:
                raise ValueError("formal P4 requires batch 64 and a frozen pilot preset")
            if self.qualification_report is None or self.pilot_lock is None:
                raise ValueError("formal P4 requires qualification and pilot locks")
            expected_hours = 24.0 if self.profile == "mechanism" else 72.0
            if self.budget.wall_clock_hours != expected_hours:
                raise ValueError(
                    f"{self.profile} wall-clock budget must be {expected_hours:g} hours"
                )
            if self.budget.bootstrap_samples != 10_000:
                raise ValueError("formal P4 requires 10000 bootstrap samples")
            expected_optimizer = P4_PILOT_PRESETS[self.selected_preset]
            actual_optimizer = (
                self.optimizer.learning_rate,
                self.optimizer.weight_decay,
                self.optimizer.temporal_loss_weight,
            )
            if actual_optimizer != expected_optimizer:
                raise ValueError("formal P4 optimizer must match the selected pilot preset")
            if self.profile == "full" and self.mechanism_report is None:
                raise ValueError("full P4 requires a frozen GATE-4-MECH report")
        if self.optimizer.gradient_clip_norm != 1.0:
            raise ValueError("P4 gradient clipping is frozen at 1.0")
        return self

    def matrix(self) -> tuple[P4ExperimentCell, ...]:
        return build_p4_matrix(self)

    def compatible_dict(self) -> dict[str, object]:
        value = self.model_dump(
            mode="json", exclude={"run_id", "resume", "output_root", "control_root"}
        )
        value["experiment_matrix"] = [cell.model_dump(mode="json") for cell in self.matrix()]
        return value

    def config_hash(self) -> str:
        encoded = json.dumps(self.compatible_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def matrix_hash(self) -> str:
        encoded = json.dumps(
            [cell.model_dump(mode="json") for cell in self.matrix()],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def _cell(
    cell_type: Literal["pilot", "shared", "per_task", "causal", "control", "continual"],
    regime: Literal["shared", "per_task", "continual"],
    model_id: Literal["modular-v2", "gru", "transformer"],
    variant: str,
    seed: int,
    steps: int,
    *,
    task_id: str | None = None,
    operation: str = "none",
    phase: Literal["train", "evaluate", "both"] = "both",
) -> P4ExperimentCell:
    task_label = "all" if task_id is None else task_id.removesuffix(".v1")
    return P4ExperimentCell(
        cell_id=f"{cell_type}__{regime}__{model_id}__{variant}__s{seed}__{task_label}",
        cell_type=cell_type,
        regime=regime,
        model_id=model_id,
        variant_id=variant,
        seed=seed,
        task_id=task_id,
        max_steps=steps,
        intervention=P4InterventionSpec(operation=operation, phase=phase),  # type: ignore[arg-type]
    )


def _mechanism_cells(seed: int, steps: int) -> list[P4ExperimentCell]:
    result = [
        _cell("shared", "shared", "modular-v2", "full", seed, steps),
        _cell(
            "causal",
            "shared",
            "modular-v2",
            "predictor-off",
            seed,
            steps,
            operation="predictor_off",
        ),
        _cell("causal", "shared", "modular-v2", "loss-zero", seed, steps, operation="loss_zero"),
        _cell(
            "causal",
            "shared",
            "modular-v2",
            "feedback-zero",
            seed,
            steps,
            operation="feedback_zero",
        ),
    ]
    for variant, operation in (
        ("acute-feedback-off", "acute_feedback_off"),
        ("shuffle-forecast", "shuffle_forecast"),
        ("dense-memory", "dense_memory"),
        ("legacy-capacity", "legacy_capacity"),
    ):
        result.append(
            _cell(
                "control",
                "shared",
                "modular-v2",
                variant,
                seed,
                0,
                operation=operation,
                phase="evaluate",
            )
        )
    return result


def build_p4_matrix(config: P4SuiteConfig) -> tuple[P4ExperimentCell, ...]:
    if config.profile == "pilot":
        return tuple(
            _cell("pilot", "shared", "modular-v2", f"preset-{index}", 7, 1_000)
            for index in range(4)
        )
    mechanism_steps = 12 if config.profile == "qualification" else 15_000
    seeds = (7,) if config.profile == "qualification" else config.seeds
    cells = [cell for seed in seeds for cell in _mechanism_cells(seed, mechanism_steps)]
    if config.profile in {"qualification", "mechanism"}:
        return tuple(cells)
    for seed in config.seeds:
        for model in ("gru", "transformer"):
            cells.append(_cell("shared", "shared", model, "full", seed, 15_000))
        for model in P4_MODELS:
            for task_id in P4_TASK_ORDER:
                cells.append(
                    _cell("per_task", "per_task", model, "full", seed, 5_000, task_id=task_id)
                )
            cells.append(_cell("continual", "continual", model, "latin", seed, 4_500))
        cells.extend(
            (
                _cell(
                    "causal",
                    "per_task",
                    "modular-v2",
                    "episodic-off",
                    seed,
                    5_000,
                    task_id=P4_TASK_ORDER[0],
                    operation="episodic_off",
                ),
                _cell(
                    "causal",
                    "per_task",
                    "modular-v2",
                    "working-reset",
                    seed,
                    5_000,
                    task_id=P4_TASK_ORDER[1],
                    operation="working_reset",
                ),
            )
        )
        for variant, operation in (
            ("direct-head", "direct_head"),
            ("frozen-random-encoder", "frozen_random_encoder"),
            ("shallow-encoder", "shallow_encoder"),
        ):
            cells.append(
                _cell(
                    "control",
                    "shared",
                    "modular-v2",
                    variant,
                    seed,
                    0,
                    operation=operation,
                    phase="evaluate",
                )
            )
    if len(cells) != 81:
        raise RuntimeError(f"P4 full matrix must contain 81 cells, got {len(cells)}")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise RuntimeError("P4 matrix contains duplicate cell IDs")
    return tuple(cells)


def load_p4_suite_config(path: Path) -> P4SuiteConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("P4 suite config must be a YAML object")
    return P4SuiteConfig.model_validate(value)


__all__ = [
    "P4_FORMAL_SEEDS",
    "P4_PILOT_PRESETS",
    "P4_TASK_ORDER",
    "P4BudgetConfig",
    "P4DataConfig",
    "P4ExperimentCell",
    "P4InterventionSpec",
    "P4OptimizerConfig",
    "P4SuiteConfig",
    "build_p4_matrix",
    "load_p4_suite_config",
]
