"""P6 modular graph: the v3 graph over four tasks with ``working_memory.v2`` (CR-015).

The predictor's detached per-step surprise, which already modulates routing,
also enters the update gate of ``working_memory.v2``.  ``working_memory="v1"``
builds the ``wm-v1`` mechanism cell on the same four-task base (erratum E1).
``working_surprise`` swaps only the surprise that working memory sees for the
``shuffle-surprise`` and ``surprise-zero`` inference cells; routing keeps the
original signal.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal, cast

import torch
from torch import Tensor

from neuromorphic.core.contracts import (
    BrainModule,
    BrainPacket,
    ModuleContext,
    ModuleOutput,
    ModuleState,
)
from neuromorphic.core.module_registry import ModuleRegistry
from neuromorphic.core.registry import (
    EPISODIC_MEMORY,
    P6_MODULE_IDS,
    WORKING_MEMORY,
    WORKING_MEMORY_V2,
)
from neuromorphic.modules.episodic_memory import EpisodicMemory
from neuromorphic.modules.network import TaskBoundaryAdapters
from neuromorphic.modules.network_v3 import ModularBrainNetworkV3
from neuromorphic.modules.p6_modules import (
    P6ActionSelector,
    P6TaskBoundaryAdapters,
    PredictiveAdapterV3P6,
    SparseRouterV3P6,
)
from neuromorphic.modules.sensory_encoder import SensoryEncoder
from neuromorphic.modules.working_memory import WorkingMemory
from neuromorphic.modules.working_memory_v2 import WorkingMemoryV2, WriteSource
from neuromorphic.tasks.base import TaskBatch
from neuromorphic.tasks.control import TaskControl
from neuromorphic.tasks.p6_control import p6_task_control_from_batch

WorkingMemoryVersion = Literal["v2", "v1"]
WorkingSurpriseMode = Literal["full", "shuffle", "zero"]
WORKING_SURPRISE_MODES: tuple[WorkingSurpriseMode, ...] = ("full", "shuffle", "zero")


def _default_registry_v4(
    feature_dim: int,
    *,
    episodic_slots: int,
    working_slots: int,
    working_slot_dim: int,
    action_embedding_dim: int,
    task_embedding_dim: int,
    dual_route_fraction: float,
    working_memory: WorkingMemoryVersion,
    write_source: WriteSource,
    surprise_gating: bool,
) -> ModuleRegistry:
    working: WorkingMemory | WorkingMemoryV2
    if working_memory == "v2":
        working = WorkingMemoryV2(
            feature_dim,
            slots=working_slots,
            slot_dim=working_slot_dim,
            write_source=write_source,
            surprise_gating=surprise_gating,
        )
    else:
        working = WorkingMemory(feature_dim, slots=working_slots, slot_dim=working_slot_dim)
    return ModuleRegistry(
        (
            SensoryEncoder(feature_dim=feature_dim),
            EpisodicMemory(feature_dim=feature_dim, slots=episodic_slots),
            working,
            PredictiveAdapterV3P6(feature_dim=feature_dim, action_dim=action_embedding_dim),
            P6ActionSelector(feature_dim=feature_dim),
            SparseRouterV3P6(
                feature_dim,
                task_embedding_dim=task_embedding_dim,
                dual_route_fraction=dual_route_fraction,
                working_module_id=working.module_id,
            ),
        )
    )


class ModularBrainNetworkV4(ModularBrainNetworkV3):
    """Execute the four-task P6 graph with surprise-gated working memory."""

    def __init__(
        self,
        *,
        feature_dim: int = 128,
        episodic_slots: int = 16,
        working_slots: int = 4,
        working_slot_dim: int = 32,
        action_embedding_dim: int = 32,
        task_embedding_dim: int = 16,
        dual_route_fraction: float = 0.25,
        working_memory: WorkingMemoryVersion = "v2",
        write_source: WriteSource = "enriched",
        surprise_gating: bool = True,
        tbptt_interval: int = 32,
    ) -> None:
        if working_memory not in {"v2", "v1"}:
            raise ValueError(f"unknown working memory version: {working_memory}")
        if working_memory == "v1" and (write_source != "enriched" or not surprise_gating):
            raise ValueError("write_source and surprise_gating apply to working_memory.v2 only")
        working_id = WORKING_MEMORY_V2 if working_memory == "v2" else WORKING_MEMORY
        self.module_ids = tuple(
            working_id if module_id == WORKING_MEMORY_V2 else module_id
            for module_id in P6_MODULE_IDS
        )
        self.optional_expert_ids = (EPISODIC_MEMORY, working_id)
        self.working_module_id = working_id
        self.working_surprise_mode: WorkingSurpriseMode = "full"
        registry = _default_registry_v4(
            feature_dim,
            episodic_slots=episodic_slots,
            working_slots=working_slots,
            working_slot_dim=working_slot_dim,
            action_embedding_dim=action_embedding_dim,
            task_embedding_dim=task_embedding_dim,
            dual_route_fraction=dual_route_fraction,
            working_memory=working_memory,
            write_source=write_source,
            surprise_gating=surprise_gating,
        )
        super().__init__(
            feature_dim=feature_dim,
            episodic_slots=episodic_slots,
            working_slots=working_slots,
            working_slot_dim=working_slot_dim,
            action_embedding_dim=action_embedding_dim,
            task_embedding_dim=task_embedding_dim,
            dual_route_fraction=dual_route_fraction,
            registry=registry,
            tbptt_interval=tbptt_interval,
        )

    @contextmanager
    def working_surprise(self, mode: WorkingSurpriseMode) -> Iterator[None]:
        """Feed working memory the full, batch-shuffled or zeroed surprise inside the block."""

        if mode not in WORKING_SURPRISE_MODES:
            raise ValueError(f"unknown working surprise mode: {mode}")
        previous = self.working_surprise_mode
        self.working_surprise_mode = mode
        try:
            yield
        finally:
            self.working_surprise_mode = previous

    def _build_boundary_adapters(self, feature_dim: int) -> TaskBoundaryAdapters:
        return P6TaskBoundaryAdapters(feature_dim)

    def _task_control(self, batch: TaskBatch) -> TaskControl:
        return p6_task_control_from_batch(batch)

    def _run_expert(
        self,
        module_id: str,
        module: BrainModule,
        packet: BrainPacket,
        state: ModuleState,
        context: ModuleContext,
        *,
        indices: Tensor,
        surprise: Tensor,
    ) -> ModuleOutput:
        if module_id != WORKING_MEMORY_V2 or not cast(WorkingMemoryV2, module).surprise_gating:
            return super()._run_expert(
                module_id, module, packet, state, context, indices=indices, surprise=surprise
            )
        mode = self.working_surprise_mode
        if mode == "full":
            gate = surprise
        elif mode == "shuffle":
            gate = surprise.roll(1, dims=0)
        elif mode == "zero":
            gate = torch.zeros_like(surprise)
        else:
            raise ValueError(f"unknown working surprise mode: {mode}")
        working = cast(WorkingMemoryV2, module)
        return working.forward(packet, state, context, surprise=gate.index_select(0, indices))


__all__ = [
    "WORKING_SURPRISE_MODES",
    "ModularBrainNetworkV4",
    "WorkingMemoryVersion",
    "WorkingSurpriseMode",
]
