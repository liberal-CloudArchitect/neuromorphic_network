"""Tensor contracts for modular brain-inspired computation.

The types describe an artificial computational system. They neither represent
biological tissue nor claim one-to-one equivalence with brain regions.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

from neuromorphic.core.registry import MODULE_IDS

type Phase = Literal["train", "evaluate", "generate", "replay"]
type MetadataScalar = str | int | float | bool | None

_PHASES = frozenset(("train", "evaluate", "generate", "replay"))
_INTEGER_DTYPES = frozenset((torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64))
_MAX_METADATA_PROPERTIES = 32
_MAX_METADATA_STRING_LENGTH = 256
_MAX_METADATA_ABS_NUMBER = 1.0e12
_TRUSTED_INTERNAL_EXECUTION: ContextVar[bool] = ContextVar(
    "trusted_internal_execution", default=False
)


@contextmanager
def trusted_internal_execution() -> Iterator[None]:
    """Skip repeated tensor-value scans inside an already validated batch graph."""

    token = _TRUSTED_INTERNAL_EXECUTION.set(True)
    try:
        yield
    finally:
        _TRUSTED_INTERNAL_EXECUTION.reset(token)


def internal_execution_is_trusted() -> bool:
    return _TRUSTED_INTERNAL_EXECUTION.get()


def _require_nonempty(value: str, name: str) -> None:
    if not value or value.isspace():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_metadata(metadata: Mapping[str, MetadataScalar]) -> None:
    if len(metadata) > _MAX_METADATA_PROPERTIES:
        raise ValueError(f"metadata may contain at most {_MAX_METADATA_PROPERTIES} entries")
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        _require_nonempty(key, "metadata key")
        if isinstance(value, str):
            if len(value) > _MAX_METADATA_STRING_LENGTH:
                raise ValueError("metadata strings may contain at most 256 characters")
        elif isinstance(value, bool) or value is None:
            continue
        elif isinstance(value, int | float):
            number = float(value)
            if not math.isfinite(number) or abs(number) > _MAX_METADATA_ABS_NUMBER:
                raise ValueError("metadata numbers must be finite and within +/-1e12")
        else:
            raise TypeError("metadata values must be scalar JSON values")


@dataclass(frozen=True, slots=True)
class BrainPacket:
    """Time-aligned representation exchanged by computation modules."""

    representation: Tensor
    valid_mask: Tensor
    modality: str
    step_index: Tensor
    source_module: str
    goal_context: Tensor | None = None
    metadata: Mapping[str, MetadataScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_brain_packet(self)


def validate_brain_packet(packet: BrainPacket) -> None:
    """Validate shape, dtype, device, and bounded metadata invariants."""

    representation = packet.representation
    if representation.ndim != 3:
        raise ValueError("representation must have shape [B, T, F]")
    batch_size, sequence_length, feature_size = representation.shape
    if min(batch_size, sequence_length, feature_size) <= 0:
        raise ValueError("representation dimensions must be positive")
    if not representation.is_floating_point():
        raise TypeError("representation must use a floating-point dtype")
    if not internal_execution_is_trusted() and not torch.isfinite(representation).all().item():
        raise ValueError("representation must contain only finite values")

    expected_time_shape = (batch_size, sequence_length)
    if packet.valid_mask.shape != expected_time_shape:
        raise ValueError("valid_mask must have shape [B, T]")
    if packet.valid_mask.dtype is not torch.bool:
        raise TypeError("valid_mask must use torch.bool")
    if packet.valid_mask.device != representation.device:
        raise ValueError("valid_mask and representation must share a device")

    if packet.step_index.shape != expected_time_shape:
        raise ValueError("step_index must have shape [B, T]")
    if packet.step_index.dtype not in _INTEGER_DTYPES:
        raise TypeError("step_index must use an integer dtype")
    if packet.step_index.device != representation.device:
        raise ValueError("step_index and representation must share a device")
    if not internal_execution_is_trusted() and torch.any(packet.step_index < 0).item():
        raise ValueError("step_index values must be non-negative")

    _require_nonempty(packet.modality, "modality")
    _require_nonempty(packet.source_module, "source_module")
    if packet.source_module not in MODULE_IDS:
        raise ValueError(f"source_module is not registered: {packet.source_module}")

    if packet.goal_context is not None:
        goal_context = packet.goal_context
        if goal_context.ndim != 3 or goal_context.shape[:2] != expected_time_shape:
            raise ValueError("goal_context must have shape [B, T, G]")
        if goal_context.shape[2] <= 0:
            raise ValueError("goal_context feature size must be positive")
        if not goal_context.is_floating_point():
            raise TypeError("goal_context must use a floating-point dtype")
        if goal_context.device != representation.device:
            raise ValueError("goal_context and representation must share a device")
        if not internal_execution_is_trusted() and not torch.isfinite(goal_context).all().item():
            raise ValueError("goal_context must contain only finite values")

    _validate_metadata(packet.metadata)


@dataclass(frozen=True, slots=True)
class ModuleContext:
    """Execution context supplied to a module for one sequence chunk."""

    task_id: str
    phase: Phase
    reset_mask: Tensor
    eligible_modules: tuple[str, ...]
    telemetry_enabled: bool = False

    def __post_init__(self) -> None:
        validate_module_context(self)


def validate_module_context(
    context: ModuleContext,
    *,
    batch_size: int | None = None,
    sequence_length: int | None = None,
    device: torch.device | None = None,
) -> None:
    """Validate reset semantics and the eligible module registry subset."""

    _require_nonempty(context.task_id, "task_id")
    if context.phase not in _PHASES:
        raise ValueError(f"phase must be one of {sorted(_PHASES)}")
    if context.reset_mask.ndim != 2:
        raise ValueError("reset_mask must have shape [B, T]")
    if context.reset_mask.dtype is not torch.bool:
        raise TypeError("reset_mask must use torch.bool")
    if batch_size is not None and context.reset_mask.shape[0] != batch_size:
        raise ValueError("reset_mask batch dimension must match BrainPacket")
    if sequence_length is not None and context.reset_mask.shape[1] != sequence_length:
        raise ValueError("reset_mask time dimension must match BrainPacket")
    if device is not None and context.reset_mask.device != device:
        raise ValueError("reset_mask and BrainPacket must share a device")
    if len(set(context.eligible_modules)) != len(context.eligible_modules):
        raise ValueError("eligible_modules must not contain duplicates")
    unknown = set(context.eligible_modules).difference(MODULE_IDS)
    if unknown:
        raise ValueError(f"eligible_modules contains unregistered identifiers: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class ModuleState:
    """Versioned tensor state owned by exactly one registered module."""

    module_id: str
    version: str
    tensors: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_module_state(self)


def validate_module_state(
    state: ModuleState,
    *,
    device: torch.device | None = None,
) -> None:
    """Validate ownership, version, tensor names, and optional device placement."""

    if state.module_id not in MODULE_IDS:
        raise ValueError(f"unregistered module state owner: {state.module_id}")
    _require_nonempty(state.version, "version")
    for name, tensor in state.tensors.items():
        _require_nonempty(name, "state tensor name")
        if not isinstance(tensor, Tensor):
            raise TypeError("ModuleState values must be torch.Tensor instances")
        if device is not None and tensor.device != device:
            raise ValueError("all state tensors must use the requested device")


@runtime_checkable
class TelemetryRecord(Protocol):
    """Minimal serialization boundary implemented by telemetry-v1 events."""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable event mapping."""
        ...


@dataclass(frozen=True, slots=True)
class ModuleOutput:
    """Validated output bundle returned by every computation module."""

    packet: BrainPacket
    state: ModuleState
    prediction_logits: Tensor | None = None
    action_logits: Tensor | None = None
    auxiliary_losses: Mapping[str, Tensor] = field(default_factory=dict)
    telemetry_events: tuple[TelemetryRecord, ...] = ()

    def __post_init__(self) -> None:
        validate_module_output(self)


def validate_module_output(output: ModuleOutput) -> None:
    """Validate output alignment, scalar losses, and telemetry event types."""

    if not internal_execution_is_trusted():
        validate_brain_packet(output.packet)
        validate_module_state(output.state, device=output.packet.representation.device)
    if output.packet.source_module != output.state.module_id:
        raise ValueError("output packet source_module must match state module owner")
    batch_size, sequence_length, _ = output.packet.representation.shape
    for name, logits in (
        ("prediction_logits", output.prediction_logits),
        ("action_logits", output.action_logits),
    ):
        if logits is None:
            continue
        if logits.ndim != 3 or logits.shape[:2] != (batch_size, sequence_length):
            raise ValueError(f"{name} must have shape [B, T, C]")
        if logits.shape[2] <= 0 or not logits.is_floating_point():
            raise TypeError(f"{name} must have a positive floating-point class dimension")
        if logits.device != output.packet.representation.device:
            raise ValueError(f"{name} and packet representation must share a device")
        if not internal_execution_is_trusted() and not torch.isfinite(logits).all().item():
            raise ValueError(f"{name} must contain only finite values")

    for name, loss in output.auxiliary_losses.items():
        _require_nonempty(name, "auxiliary loss name")
        if not isinstance(loss, Tensor):
            raise TypeError("auxiliary losses must be torch.Tensor instances")
        if loss.ndim != 0:
            raise ValueError("auxiliary losses must be scalar tensors")
        if loss.device != output.packet.representation.device:
            raise ValueError("auxiliary losses and packet representation must share a device")
        if not loss.is_floating_point():
            raise TypeError("auxiliary losses must use a floating-point dtype")
        if not internal_execution_is_trusted() and not torch.isfinite(loss).item():
            raise ValueError("auxiliary losses must be finite")

    if any(not isinstance(event, TelemetryRecord) for event in output.telemetry_events):
        raise TypeError("telemetry_events must implement the TelemetryRecord protocol")


@runtime_checkable
class BrainModule(Protocol):
    """Structural interface implemented by all P2 computation modules."""

    module_id: str

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ModuleState:
        """Create empty module-owned state on the explicit device."""
        ...

    def forward(
        self,
        packet: BrainPacket,
        state: ModuleState,
        context: ModuleContext,
    ) -> ModuleOutput:
        """Execute one time-aligned sequence chunk."""
        ...

    def reset_state(self, state: ModuleState, reset_mask: Tensor) -> ModuleState:
        """Clear transient state for episode boundaries selected by reset_mask."""
        ...


__all__ = [
    "BrainModule",
    "BrainPacket",
    "MetadataScalar",
    "ModuleContext",
    "ModuleOutput",
    "ModuleState",
    "Phase",
    "TelemetryRecord",
    "internal_execution_is_trusted",
    "trusted_internal_execution",
    "validate_brain_packet",
    "validate_module_context",
    "validate_module_output",
    "validate_module_state",
]
