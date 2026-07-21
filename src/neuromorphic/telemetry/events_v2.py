"""Detached scalar telemetry for the P4 recurrent predictor and semantic router.

The records describe an artificial computational abstraction.  They are not
measurements of biological tissue and deliberately contain no visualization or
anatomical-atlas fields.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Literal

from torch import Tensor

from neuromorphic.core.registry import ALL_MODULE_IDS

SCHEMA_VERSION_V2 = "telemetry-v2"
SCIENTIFIC_DISCLAIMER_V2 = (
    "Artificial computational abstraction telemetry; not biological neural activity, "
    "brain tissue, BOLD, or a clinical measurement."
)

type TelemetryV2Phase = Literal["train", "evaluate", "generate", "replay"]
type TelemetryV2EventType = Literal["compute", "predict", "route", "summary"]
type TelemetryV2Scalar = str | int | float | bool | None

_PHASES = frozenset(("train", "evaluate", "generate", "replay"))
_EVENT_TYPES = frozenset(("compute", "predict", "route", "summary"))
_FORBIDDEN_FIELD_FRAGMENTS = ("atlas", "web", "viewer")


def _require_text(value: str, name: str) -> None:
    if not value or value.isspace():
        raise ValueError(f"{name} must be a non-empty string")


def _finite_nonnegative(value: float | None, name: str) -> None:
    if value is not None and (not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and non-negative when present")


def _validate_metadata(metadata: Mapping[str, TelemetryV2Scalar]) -> None:
    if len(metadata) > 32:
        raise ValueError("telemetry metadata may contain at most 32 entries")
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("telemetry metadata keys must be strings")
        _require_text(key, "metadata key")
        lowered = key.lower()
        if any(fragment in lowered for fragment in _FORBIDDEN_FIELD_FRAGMENTS):
            raise ValueError("telemetry-v2 forbids atlas, web, and viewer metadata fields")
        if isinstance(value, str):
            if len(value) > 256:
                raise ValueError("telemetry metadata strings may contain at most 256 characters")
        elif isinstance(value, bool) or value is None:
            continue
        elif isinstance(value, int | float):
            number = float(value)
            if not math.isfinite(number) or abs(number) > 1.0e12:
                raise ValueError("telemetry metadata numbers must be finite and within +/-1e12")
        else:
            raise TypeError("telemetry metadata values must be detached scalar JSON values")


def detached_scalar(value: Tensor | TelemetryV2Scalar) -> TelemetryV2Scalar:
    """Detach a scalar tensor, or validate and return an existing JSON scalar."""

    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("telemetry tensors must contain exactly one scalar")
        value = value.detach().item()
    _validate_metadata({"value": value})
    return value


@dataclass(frozen=True, slots=True)
class TelemetryV2Event:
    """One bounded aggregate event emitted after model state and losses exist."""

    event_id: str
    run_id: str
    global_step: int
    task: str
    event_type: TelemetryV2EventType
    phase: TelemetryV2Phase
    module_id: str
    compute_gate: bool
    reserved_count: int | None = None
    learned_count: int | None = None
    raw_count: int | None = None
    executed_count: int | None = None
    capacity: int | None = None
    drop_count: int | None = None
    reserved_coverage: float | None = None
    forecast_coverage: float | None = None
    forecast_error: float | None = None
    persistence_error: float | None = None
    active_mac: float | None = None
    dense_mac: float | None = None
    metadata: Mapping[str, TelemetryV2Scalar] = field(default_factory=dict)
    scientific_disclaimer: str = field(default=SCIENTIFIC_DISCLAIMER_V2, init=False)
    schema_version: str = field(default=SCHEMA_VERSION_V2, init=False)

    def __post_init__(self) -> None:
        for text_value, name in (
            (self.event_id, "event_id"),
            (self.run_id, "run_id"),
            (self.task, "task"),
        ):
            _require_text(text_value, name)
        if self.global_step < 0:
            raise ValueError("global_step must be non-negative")
        if self.module_id not in ALL_MODULE_IDS:
            raise ValueError(f"unregistered telemetry module: {self.module_id}")
        if self.phase not in _PHASES:
            raise ValueError(f"invalid telemetry phase: {self.phase}")
        if self.event_type not in _EVENT_TYPES:
            raise ValueError(f"invalid telemetry event type: {self.event_type}")
        for integer_value, name in (
            (self.reserved_count, "reserved_count"),
            (self.learned_count, "learned_count"),
            (self.raw_count, "raw_count"),
            (self.executed_count, "executed_count"),
            (self.capacity, "capacity"),
            (self.drop_count, "drop_count"),
        ):
            if integer_value is not None and integer_value < 0:
                raise ValueError(f"{name} must be non-negative when present")
        for coverage, name in (
            (self.reserved_coverage, "reserved_coverage"),
            (self.forecast_coverage, "forecast_coverage"),
        ):
            if coverage is not None and (not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0):
                raise ValueError(f"{name} must be between zero and one")
        for metric_value, name in (
            (self.forecast_error, "forecast_error"),
            (self.persistence_error, "persistence_error"),
            (self.active_mac, "active_mac"),
            (self.dense_mac, "dense_mac"),
        ):
            _finite_nonnegative(metric_value, name)
        _validate_metadata(self.metadata)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping matching telemetry-v2."""

        return asdict(self)


__all__ = [
    "SCHEMA_VERSION_V2",
    "SCIENTIFIC_DISCLAIMER_V2",
    "TelemetryV2Event",
    "TelemetryV2EventType",
    "TelemetryV2Phase",
    "TelemetryV2Scalar",
    "detached_scalar",
]
