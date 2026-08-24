"""P5 modular graph with residual forecasting and stable semantic routing."""

from __future__ import annotations

from neuromorphic.core.module_registry import ModuleRegistry
from neuromorphic.core.registry import (
    P5_MODULE_IDS,
    PREDICTIVE_ADAPTER_V3,
    SPARSE_ROUTER_V3,
)
from neuromorphic.modules.action_selector import ActionSelector
from neuromorphic.modules.episodic_memory import EpisodicMemory
from neuromorphic.modules.network_v2 import ModularBrainNetworkV2
from neuromorphic.modules.predictive_adapter_v3 import PredictiveAdapterV3
from neuromorphic.modules.sensory_encoder import SensoryEncoder
from neuromorphic.modules.sparse_router_v3 import SparseRouterV3
from neuromorphic.modules.working_memory import WorkingMemory


def _default_registry_v3(
    feature_dim: int,
    *,
    episodic_slots: int,
    working_slots: int,
    working_slot_dim: int,
    action_embedding_dim: int,
    task_embedding_dim: int,
    dual_route_fraction: float,
) -> ModuleRegistry:
    return ModuleRegistry(
        (
            SensoryEncoder(feature_dim=feature_dim),
            EpisodicMemory(feature_dim=feature_dim, slots=episodic_slots),
            WorkingMemory(
                feature_dim=feature_dim,
                slots=working_slots,
                slot_dim=working_slot_dim,
            ),
            PredictiveAdapterV3(
                feature_dim=feature_dim,
                action_dim=action_embedding_dim,
            ),
            ActionSelector(feature_dim=feature_dim),
            SparseRouterV3(
                feature_dim=feature_dim,
                task_embedding_dim=task_embedding_dim,
                dual_route_fraction=dual_route_fraction,
            ),
        )
    )


class ModularBrainNetworkV3(ModularBrainNetworkV2):
    """Execute the v3 graph while retaining P4 batch/state contracts."""

    module_ids = P5_MODULE_IDS
    predictive_module_id = PREDICTIVE_ADAPTER_V3
    router_module_id = SPARSE_ROUTER_V3

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
        registry: ModuleRegistry | None = None,
        tbptt_interval: int = 32,
    ) -> None:
        if registry is None:
            registry = _default_registry_v3(
                feature_dim,
                episodic_slots=episodic_slots,
                working_slots=working_slots,
                working_slot_dim=working_slot_dim,
                action_embedding_dim=action_embedding_dim,
                task_embedding_dim=task_embedding_dim,
                dual_route_fraction=dual_route_fraction,
            )
        super().__init__(
            feature_dim=feature_dim,
            episodic_slots=episodic_slots,
            working_slots=working_slots,
            working_slot_dim=working_slot_dim,
            action_embedding_dim=action_embedding_dim,
            task_embedding_dim=task_embedding_dim,
            registry=registry,
            tbptt_interval=tbptt_interval,
        )


__all__ = ["ModularBrainNetworkV3"]
