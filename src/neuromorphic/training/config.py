"""Validated experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base configuration model that rejects misspelled fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskConfig(StrictModel):
    """Procedural task selection."""

    task_id: Literal["associative_recall.v1", "delayed_rule_switch.v1", "small_graph.v1"]
    profile: Literal["smoke", "qualification"] = "smoke"


class ModelConfig(StrictModel):
    """Supported monolithic baseline architecture."""

    kind: Literal["gru", "transformer"] = "gru"
    hidden_size: int = Field(default=128, ge=8)
    layers: int = Field(default=1, ge=1)
    heads: int = Field(default=4, ge=1)
    feedforward_size: int = Field(default=512, ge=16)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    target_parameters: int | None = Field(default=None, ge=1)
    parameter_tolerance: float = Field(default=0.05, gt=0.0, le=0.25)
    target_macs: int | None = Field(default=None, ge=1)
    target_latency_ms: float | None = Field(default=None, gt=0.0)
    cost_tolerance: float = Field(default=0.05, gt=0.0, le=0.25)


class OptimizerConfig(StrictModel):
    """Optimizer settings shared by all baselines."""

    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-2, ge=0.0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)


class TrainingConfig(StrictModel):
    """Training budget and early-stopping rules."""

    batch_size: int = Field(default=64, ge=1)
    max_steps: int = Field(default=5_000, ge=1)
    eval_interval: int = Field(default=100, ge=1)
    patience: int = Field(default=10, ge=1)
    min_delta: float = Field(default=0.001, ge=0.0)
    tbptt_steps: int = Field(default=32, ge=1)
    checkpoint_interval: int = Field(default=100, ge=1)
    auxiliary_loss_weight: float = Field(default=0.1, ge=0.0)
    target_optimizer_steps: int | None = Field(default=None, ge=1)
    target_training_tokens: int | None = Field(default=None, ge=1)
    compute_tolerance: float = Field(default=0.05, gt=0.0, le=0.25)
    require_loss_decrease: bool = False


class RunConfig(StrictModel):
    """Complete command-line run configuration."""

    schema_version: Literal["run-config-v1"] = "run-config-v1"
    matching_mode: Literal["unmatched", "parameter", "train_compute", "inference_cost"] = (
        "unmatched"
    )
    seed: int = Field(default=7, ge=0)
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    output_root: Path = Path("artifacts/runs")
    run_id: str | None = None
    resume: Path | None = None
    task: TaskConfig
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    training: TrainingConfig = TrainingConfig()

    @model_validator(mode="after")
    def validate_matching_targets(self) -> RunConfig:
        """Require the comparison targets implied by the selected sensitivity mode."""
        if self.matching_mode == "parameter" and self.model.target_parameters is None:
            raise ValueError("parameter matching requires model.target_parameters")
        if self.matching_mode == "train_compute" and (
            self.training.target_optimizer_steps is None
            or self.training.target_training_tokens is None
        ):
            raise ValueError("train_compute matching requires step and token targets")
        if self.matching_mode == "inference_cost" and (
            self.model.target_macs is None or self.model.target_latency_ms is None
        ):
            raise ValueError("inference_cost matching requires MAC and latency targets")
        return self

    def checkpoint_compatible_dict(self) -> dict[str, object]:
        """Return fields that must remain stable across resume."""
        return self.model_dump(mode="json", exclude={"resume", "run_id", "output_root"})


def load_run_config(path: Path) -> RunConfig:
    """Load and validate one YAML experiment configuration."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("run configuration must be a YAML object")
    return RunConfig.model_validate(value)


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic accelerator without hard-coding MPS."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device
