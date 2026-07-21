"""Artificial computation modules used by the P2 modular network."""

from neuromorphic.modules.action_selector import ActionSelector
from neuromorphic.modules.episodic_memory import EpisodicMemory
from neuromorphic.modules.network import ModularBrainNetwork, ModularBrainOutput
from neuromorphic.modules.network_v2 import ModularBrainNetworkV2, ModularBrainOutputV2
from neuromorphic.modules.predictive_adapter import PredictiveAdapter
from neuromorphic.modules.predictive_adapter_v2 import PredictiveAdapterV2
from neuromorphic.modules.sensory_encoder import SensoryEncoder
from neuromorphic.modules.sparse_router import RoutingDecision, SparseRouter
from neuromorphic.modules.sparse_router_v2 import RoutingDecisionV2, SparseRouterV2
from neuromorphic.modules.working_memory import WorkingMemory

__all__ = [
    "ActionSelector",
    "EpisodicMemory",
    "ModularBrainNetwork",
    "ModularBrainNetworkV2",
    "ModularBrainOutput",
    "ModularBrainOutputV2",
    "PredictiveAdapter",
    "PredictiveAdapterV2",
    "RoutingDecision",
    "RoutingDecisionV2",
    "SensoryEncoder",
    "SparseRouter",
    "SparseRouterV2",
    "WorkingMemory",
]
