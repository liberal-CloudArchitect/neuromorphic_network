"""Strict configuration for the P5 mechanism qualification suite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class P5QualificationConfig(_StrictModel):
    schema_version: Literal["p5-qualification-v1"] = "p5-qualification-v1"
    protocol_version: Literal["p5-protocol-v1"] = "p5-protocol-v1"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    seed: Literal[7] = 7
    train_samples: Literal[64] = 64
    validation_samples: Literal[32] = 32
    batch_size: Literal[8] = 8
    steps_per_task: Literal[4] = 4
    learning_rate: float = Field(default=3.0e-4, gt=0.0)
    weight_decay: float = Field(default=1.0e-2, ge=0.0)
    temporal_loss_weight: float = Field(default=0.1, gt=0.0)
    semantic_loss_weight: float = Field(default=0.01, gt=0.0)
    dual_budget_weight: float = Field(default=0.001, ge=0.0)
    dual_route_fraction: float = Field(default=0.25, gt=0.0, le=0.5)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)
    output_root: Path = Path("artifacts/runs")
    run_id: str | None = None

    @model_validator(mode="after")
    def validate_frozen_mechanism(self) -> P5QualificationConfig:
        if self.temporal_loss_weight != 0.1:
            raise ValueError("P5 qualification temporal loss weight is frozen at 0.1")
        if self.semantic_loss_weight != 0.01:
            raise ValueError("P5 qualification semantic loss weight is frozen at 0.01")
        if self.dual_route_fraction != 0.25:
            raise ValueError("P5 qualification dual-route fraction is frozen at 0.25")
        if self.gradient_clip_norm != 1.0:
            raise ValueError("P5 qualification gradient clipping is frozen at 1.0")
        return self

    def compatible_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"run_id", "output_root"})

    def config_hash(self) -> str:
        encoded = json.dumps(self.compatible_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def load_p5_qualification_config(path: Path) -> P5QualificationConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("P5 qualification config must be a YAML object")
    return P5QualificationConfig.model_validate(value)


__all__ = ["P5QualificationConfig", "load_p5_qualification_config"]
