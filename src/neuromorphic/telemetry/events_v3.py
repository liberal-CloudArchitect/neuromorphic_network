"""Detached P5 telemetry for predictive surprise and semantic routing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Literal

from neuromorphic.core.registry import ALL_MODULE_IDS
from neuromorphic.telemetry.events_v2 import TelemetryV2Scalar, _validate_metadata

SCHEMA_VERSION_V3 = "telemetry-v3"
SCIENTIFIC_DISCLAIMER_V3 = (
    "Artificial computational abstraction telemetry; not biological neural activity, "
    "brain tissue, BOLD, or a clinical measurement."
)

type TelemetryV3Phase = Literal["train", "evaluate", "replay"]


def _finite_nonnegative(value: float | None, name: str) -> None:
    if value is not None and (not math.isfinite(value) or value < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TelemetryV3Event:
    """One post-loss, detached scalar event for a complete P5 update."""

    event_id: str
    run_id: str
    global_step: int
    task: str
    phase: TelemetryV3Phase
    module_id: str
    compute_gate: bool
    surprise_count: int
    surprise_sum: float
    semantic_required: int
    semantic_executed: int
    semantic_coverage: float
    dual_count: int
    dual_fraction: float
    forecast_coverage: float
    forecast_error: float
    persistence_error: float
    scorer_grad_norm: float | None
    active_calls: float
    dense_calls: float
    metadata: Mapping[str, TelemetryV2Scalar] = field(default_factory=dict)
    scientific_disclaimer: str = field(default=SCIENTIFIC_DISCLAIMER_V3, init=False)
    schema_version: str = field(default=SCHEMA_VERSION_V3, init=False)

    def __post_init__(self) -> None:
        if not self.event_id or not self.run_id or not self.task:
            raise ValueError("telemetry-v3 identifiers cannot be empty")
        if self.global_step < 0:
            raise ValueError("global_step must be non-negative")
        if self.module_id not in ALL_MODULE_IDS:
            raise ValueError(f"unregistered telemetry module: {self.module_id}")
        if self.phase not in {"train", "evaluate", "replay"}:
            raise ValueError(f"invalid telemetry phase: {self.phase}")
        for integer_value, name in (
            (self.surprise_count, "surprise_count"),
            (self.semantic_required, "semantic_required"),
            (self.semantic_executed, "semantic_executed"),
            (self.dual_count, "dual_count"),
        ):
            if integer_value < 0:
                raise ValueError(f"{name} must be non-negative")
        for coverage_value, name in (
            (self.semantic_coverage, "semantic_coverage"),
            (self.dual_fraction, "dual_fraction"),
            (self.forecast_coverage, "forecast_coverage"),
        ):
            if not math.isfinite(coverage_value) or not 0.0 <= coverage_value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for metric_value, name in (
            (self.surprise_sum, "surprise_sum"),
            (self.forecast_error, "forecast_error"),
            (self.persistence_error, "persistence_error"),
            (self.scorer_grad_norm, "scorer_grad_norm"),
            (self.active_calls, "active_calls"),
            (self.dense_calls, "dense_calls"),
        ):
            _finite_nonnegative(metric_value, name)
        if self.semantic_executed > self.semantic_required:
            raise ValueError("semantic_executed cannot exceed semantic_required")
        if self.active_calls > self.dense_calls:
            raise ValueError("active_calls cannot exceed dense_calls")
        _validate_metadata(self.metadata)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = [
    "SCHEMA_VERSION_V3",
    "SCIENTIFIC_DISCLAIMER_V3",
    "TelemetryV3Event",
    "TelemetryV3Phase",
]
