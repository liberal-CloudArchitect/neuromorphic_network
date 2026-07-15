"""Versioned module identifiers frozen at GATE-0."""

from __future__ import annotations

from typing import Final

SENSORY_ENCODER: Final = "sensory_encoder.v1"
EPISODIC_MEMORY: Final = "episodic_memory.v1"
WORKING_MEMORY: Final = "working_memory.v1"
PREDICTIVE_ADAPTER: Final = "predictive_adapter.v1"
ACTION_SELECTOR: Final = "action_selector.v1"
SPARSE_ROUTER: Final = "sparse_router.v1"

MODULE_IDS: Final[tuple[str, ...]] = (
    SENSORY_ENCODER,
    EPISODIC_MEMORY,
    WORKING_MEMORY,
    PREDICTIVE_ADAPTER,
    ACTION_SELECTOR,
    SPARSE_ROUTER,
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
    """Return whether *module_id* is one of the frozen P0 identifiers."""

    return module_id in MODULE_IDS


__all__ = [
    "ACTION_SELECTOR",
    "EPISODIC_MEMORY",
    "MODULE_IDS",
    "OPTIONAL_EXPERT_IDS",
    "PREDICTIVE_ADAPTER",
    "REQUIRED_PATH_IDS",
    "SENSORY_ENCODER",
    "SPARSE_ROUTER",
    "WORKING_MEMORY",
    "is_registered_module",
]
