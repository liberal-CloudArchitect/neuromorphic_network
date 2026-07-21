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
from neuromorphic.core.module_registry import ModuleRegistry
from neuromorphic.core.network_state import NetworkState
from neuromorphic.core.registry import (
    ALL_MODULE_IDS,
    MODULE_IDS,
    OPTIONAL_EXPERT_IDS,
    P4_MODULE_IDS,
    P4_OPTIONAL_EXPERT_IDS,
    P4_REQUIRED_PATH_IDS,
    REQUIRED_PATH_IDS,
    is_registered_module,
)

__all__ = [
    "ALL_MODULE_IDS",
    "MODULE_IDS",
    "OPTIONAL_EXPERT_IDS",
    "P4_MODULE_IDS",
    "P4_OPTIONAL_EXPERT_IDS",
    "P4_REQUIRED_PATH_IDS",
    "REQUIRED_PATH_IDS",
    "BrainModule",
    "BrainPacket",
    "ModuleContext",
    "ModuleOutput",
    "ModuleRegistry",
    "ModuleState",
    "NetworkState",
    "Phase",
    "TelemetryRecord",
    "is_registered_module",
]
