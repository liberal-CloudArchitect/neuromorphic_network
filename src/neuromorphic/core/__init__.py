"""Frozen P0 contracts shared by brain-inspired computation modules."""

from neuromorphic.core.contracts import (
    BrainModule,
    BrainPacket,
    ModuleContext,
    ModuleOutput,
    ModuleState,
    Phase,
    TelemetryRecord,
)
from neuromorphic.core.registry import (
    MODULE_IDS,
    OPTIONAL_EXPERT_IDS,
    REQUIRED_PATH_IDS,
    is_registered_module,
)

__all__ = [
    "MODULE_IDS",
    "OPTIONAL_EXPERT_IDS",
    "REQUIRED_PATH_IDS",
    "BrainModule",
    "BrainPacket",
    "ModuleContext",
    "ModuleOutput",
    "ModuleState",
    "Phase",
    "TelemetryRecord",
    "is_registered_module",
]
