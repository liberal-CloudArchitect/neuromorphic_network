"""Serializable records conforming to ``schemas/telemetry-v1.json``."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Literal

from neuromorphic.core.registry import MODULE_IDS

SCHEMA_VERSION = "telemetry-v1"
SCIENTIFIC_DISCLAIMER = (
    "Artificial model telemetry; not BOLD, biological neural activity, or a clinical measurement."
)

type TelemetryPhase = Literal["train", "evaluate", "generate", "replay"]
type EventTag = Literal["retrieve", "write", "select", "interrupt", "replay", "predict", "route"]
type TelemetryScalar = str | int | float | bool | None

_PHASES = frozenset(("train", "evaluate", "generate", "replay"))
_EVENT_TAGS = frozenset(("retrieve", "write", "select", "interrupt", "replay", "predict", "route"))


def _finite_nonnegative(value: float | None, name: str) -> None:
    if value is not None and (not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and non-negative when present")


@dataclass(frozen=True, slots=True)
class TelemetryEdge:
    """Directed artificial information-flow observation."""

    target_module_id: str
    flow: float
    routed: bool = False

    def __post_init__(self) -> None:
        if self.target_module_id not in MODULE_IDS:
            raise ValueError(f"unregistered telemetry edge target: {self.target_module_id}")
        _finite_nonnegative(self.flow, "flow")


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """One bounded, schema-versioned observation of artificial module activity."""

    event_id: str
    parent_event_id: str | None
    run_id: str
    episode_id: str | int
    step: int
    token_id: int | None
    monotonic_time_ns: int
    module_clock: int
    phase: TelemetryPhase
    module_id: str
    source: str
    reducer_version: str
    baseline_version: str
    compute_gate: bool
    activity_raw: float | None
    activity_z: float | None
    routing_mass: float | None
    state_change: float | None
    surprise: float | None
    confidence: float | None
    event_tags: tuple[EventTag, ...] = ()
    dropped_since_last: int = 0
    edges: tuple[TelemetryEdge, ...] = ()
    metadata: Mapping[str, TelemetryScalar] = field(default_factory=dict)
    grad_rms: float | None = None
    update_rms: float | None = None
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        for text, name in (
            (self.event_id, "event_id"),
            (self.run_id, "run_id"),
            (self.source, "source"),
            (self.reducer_version, "reducer_version"),
            (self.baseline_version, "baseline_version"),
        ):
            if not text or text.isspace():
                raise ValueError(f"{name} must be a non-empty string")
        if self.module_id not in MODULE_IDS:
            raise ValueError(f"unregistered telemetry module: {self.module_id}")
        if self.phase not in _PHASES:
            raise ValueError(f"invalid telemetry phase: {self.phase}")
        for integer, name in (
            (self.step, "step"),
            (self.monotonic_time_ns, "monotonic_time_ns"),
            (self.module_clock, "module_clock"),
            (self.dropped_since_last, "dropped_since_last"),
        ):
            if integer < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.token_id is not None and self.token_id < 0:
            raise ValueError("token_id must be non-negative when present")
        for metric, name in (
            (self.routing_mass, "routing_mass"),
            (self.state_change, "state_change"),
            (self.surprise, "surprise"),
            (self.grad_rms, "grad_rms"),
            (self.update_rms, "update_rms"),
        ):
            _finite_nonnegative(metric, name)
        for activity, name in (
            (self.activity_raw, "activity_raw"),
            (self.activity_z, "activity_z"),
        ):
            if activity is not None and not math.isfinite(activity):
                raise ValueError(f"{name} must be finite when present")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be between zero and one")
        if len(set(self.event_tags)) != len(self.event_tags):
            raise ValueError("event_tags must be unique")
        if unknown_tags := set(self.event_tags).difference(_EVENT_TAGS):
            raise ValueError(f"unknown event tags: {sorted(unknown_tags)}")
        if len(self.metadata) > 32:
            raise ValueError("telemetry metadata may contain at most 32 entries")
        for key, metadata_value in self.metadata.items():
            if not key or key.isspace():
                raise ValueError("telemetry metadata keys must be non-empty")
            if isinstance(metadata_value, str) and len(metadata_value) > 256:
                raise ValueError("telemetry metadata strings may contain at most 256 characters")
            if isinstance(metadata_value, int | float) and not isinstance(metadata_value, bool):
                if not math.isfinite(float(metadata_value)) or abs(float(metadata_value)) > 1.0e12:
                    raise ValueError("telemetry metadata numbers must be finite and within +/-1e12")
            elif not (isinstance(metadata_value, str | bool) or metadata_value is None):
                raise TypeError("telemetry metadata values must be scalar JSON values")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping matching telemetry-v1."""

        return asdict(self)


__all__ = [
    "SCHEMA_VERSION",
    "SCIENTIFIC_DISCLAIMER",
    "EventTag",
    "TelemetryEdge",
    "TelemetryEvent",
    "TelemetryPhase",
    "TelemetryScalar",
]
