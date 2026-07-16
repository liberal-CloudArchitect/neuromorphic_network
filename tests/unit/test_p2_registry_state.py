"""Unit tests for the P2 module registry and functional network state."""

from __future__ import annotations

import pytest
import torch

from neuromorphic.core.module_registry import ModuleRegistry
from neuromorphic.core.network_state import NetworkState
from neuromorphic.core.registry import MODULE_IDS
from neuromorphic.modules.action_selector import ActionSelector
from neuromorphic.modules.episodic_memory import EpisodicMemory
from neuromorphic.modules.predictive_adapter import PredictiveAdapter
from neuromorphic.modules.sensory_encoder import SensoryEncoder
from neuromorphic.modules.sparse_router import SparseRouter
from neuromorphic.modules.working_memory import WorkingMemory


def _registry() -> ModuleRegistry:
    return ModuleRegistry(
        (
            SensoryEncoder(),
            EpisodicMemory(),
            WorkingMemory(),
            PredictiveAdapter(),
            ActionSelector(),
            SparseRouter(),
        )
    )


def test_registry_is_complete_and_uses_frozen_order() -> None:
    registry = _registry()
    registry.require_complete()
    assert registry.ids == MODULE_IDS
    assert len(tuple(registry.parameters())) > 0
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(SensoryEncoder())


def test_registry_rejects_unknown_and_incomplete_implementations() -> None:
    class Unknown(SensoryEncoder):
        module_id = "unknown.v1"

    with pytest.raises(ValueError, match="unregistered"):
        ModuleRegistry((Unknown(),))
    with pytest.raises(ValueError, match="incomplete"):
        ModuleRegistry((SensoryEncoder(),)).require_complete()


def test_network_state_advances_only_valid_items_and_detaches_at_32() -> None:
    state = NetworkState.initial(_registry(), 2, device=torch.device("cpu"), dtype=torch.float32)
    assert state.valid_step_counts.tolist() == [0, 0]
    for _ in range(31):
        state, detached = state.advance(torch.tensor([True, False]))
        assert not detached.any()
    state, detached = state.advance(torch.tensor([True, True]))
    assert state.valid_step_counts.tolist() == [32, 1]
    assert detached.tolist() == [True, False]


def test_network_state_replace_reset_and_validation() -> None:
    registry = _registry()
    state = NetworkState.initial(registry, 2, device=torch.device("cpu"), dtype=torch.float32)
    working = registry.get("working_memory.v1")
    replacement = working.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    assert state.replace(replacement).get("working_memory.v1") is replacement
    state, _ = state.advance(torch.tensor([True, True]))
    assert state.reset_counts(torch.tensor([True, False])).valid_step_counts.tolist() == [0, 1]
    with pytest.raises(ValueError, match=r"shape \[B\]"):
        state.advance(torch.ones(2, 1, dtype=torch.bool))
