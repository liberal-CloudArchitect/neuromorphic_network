"""Artificial computation modules used by the P2 modular network."""

from neuromorphic.modules.action_selector import ActionSelector
from neuromorphic.modules.episodic_memory import EpisodicMemory
from neuromorphic.modules.network import ModularBrainNetwork, ModularBrainOutput
from neuromorphic.modules.predictive_adapter import PredictiveAdapter
from neuromorphic.modules.sensory_encoder import SensoryEncoder
from neuromorphic.modules.sparse_router import RoutingDecision, SparseRouter
from neuromorphic.modules.working_memory import WorkingMemory

__all__ = [
    "ActionSelector",
    "EpisodicMemory",
    "ModularBrainNetwork",
    "ModularBrainOutput",
    "PredictiveAdapter",
    "RoutingDecision",
    "SensoryEncoder",
    "SparseRouter",
    "WorkingMemory",
]
