"""Versioned module identifiers frozen at GATE-0."""

from __future__ import annotations

from typing import Final

SENSORY_ENCODER: Final = "sensory_encoder.v1"
EPISODIC_MEMORY: Final = "episodic_memory.v1"
WORKING_MEMORY: Final = "working_memory.v1"
PREDICTIVE_ADAPTER: Final = "predictive_adapter.v1"
ACTION_SELECTOR: Final = "action_selector.v1"
SPARSE_ROUTER: Final = "sparse_router.v1"
PREDICTIVE_ADAPTER_V2: Final = "predictive_adapter.v2"
SPARSE_ROUTER_V2: Final = "sparse_router.v2"

MODULE_IDS: Final[tuple[str, ...]] = (
    SENSORY_ENCODER,
    EPISODIC_MEMORY,
    WORKING_MEMORY,
    PREDICTIVE_ADAPTER,
    ACTION_SELECTOR,
    SPARSE_ROUTER,
)

P4_MODULE_IDS: Final[tuple[str, ...]] = (
    SENSORY_ENCODER,
    EPISODIC_MEMORY,
    WORKING_MEMORY,
    PREDICTIVE_ADAPTER_V2,
    ACTION_SELECTOR,
    SPARSE_ROUTER_V2,
)

ALL_MODULE_IDS: Final[tuple[str, ...]] = (*MODULE_IDS, PREDICTIVE_ADAPTER_V2, SPARSE_ROUTER_V2)

P4_OPTIONAL_EXPERT_IDS: Final[tuple[str, ...]] = (EPISODIC_MEMORY, WORKING_MEMORY)
P4_REQUIRED_PATH_IDS: Final[tuple[str, ...]] = (
    SENSORY_ENCODER,
    PREDICTIVE_ADAPTER_V2,
    SPARSE_ROUTER_V2,
    ACTION_SELECTOR,
)

OPTIONAL_EXPERT_IDS: Final[tuple[str, ...]] = (
    EPISODIC_MEMORY,
    WORKING_MEMORY,
    PREDICTIVE_ADAPTER,
)

REQUIRED_PATH_IDS: Final[tuple[str, ...]] = (
    SENSORY_ENCODER,
    SPARSE_ROUTER,
    ACTION_SELECTOR,
)


def is_registered_module(module_id: str) -> bool:
    """Return whether *module_id* belongs to a frozen legacy or P4 graph."""

    return module_id in ALL_MODULE_IDS


__all__ = [
    "ACTION_SELECTOR",
    "ALL_MODULE_IDS",
    "EPISODIC_MEMORY",
    "MODULE_IDS",
    "OPTIONAL_EXPERT_IDS",
    "P4_MODULE_IDS",
    "P4_OPTIONAL_EXPERT_IDS",
    "P4_REQUIRED_PATH_IDS",
    "PREDICTIVE_ADAPTER",
    "PREDICTIVE_ADAPTER_V2",
    "REQUIRED_PATH_IDS",
    "SENSORY_ENCODER",
    "SPARSE_ROUTER",
    "SPARSE_ROUTER_V2",
    "WORKING_MEMORY",
    "is_registered_module",
]
