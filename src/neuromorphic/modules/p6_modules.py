"""P6 four-task variants of the v3 predictor, v3 router, selector and adapters (CR-015, E1).

Each class keeps its parent's equations and module ID and only widens the task
vocabulary from three to four: uncued rule switch is identified by goal-context
channel 104.  The router treats uncued steps exactly like Delayed Rule Switch
steps (reserved working memory, dual-route eligibility, dual budget).
"""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor, nn

from neuromorphic.core.contracts import BrainPacket
from neuromorphic.core.registry import EPISODIC_MEMORY, WORKING_MEMORY, WORKING_MEMORY_V2
from neuromorphic.modules._utils import require_goal_context
from neuromorphic.modules.action_selector import ActionSelector
from neuromorphic.modules.network import TaskBoundaryAdapters
from neuromorphic.modules.predictive_adapter_v3 import PredictiveAdapterV3
from neuromorphic.modules.sparse_router_v3 import SparseRouterV3
from neuromorphic.tasks.p6_control import (
    P6_GOAL_CONTEXT_DIM,
    P6_TASK_CLASS_COUNTS,
    P6_TASK_IDS,
    P6_TASK_INPUT_DIMS,
    UNCUED_RULE_SWITCH,
    UNCUED_TASK_CHANNEL,
    p6_task_one_hot,
)

_EVENT = slice(0, 5)


class P6TaskBoundaryAdapters(TaskBoundaryAdapters):
    """Task input projections for the four P6 tasks."""

    task_ids = P6_TASK_IDS
    input_dims = P6_TASK_INPUT_DIMS


class P6ActionSelector(ActionSelector):
    """``action_selector.v1`` with an isolated two-class uncued-rule-switch head."""

    class_counts = P6_TASK_CLASS_COUNTS

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__(feature_dim)
        self.heads[UNCUED_RULE_SWITCH.removesuffix(".v1")] = nn.Linear(
            feature_dim, P6_TASK_CLASS_COUNTS[UNCUED_RULE_SWITCH]
        )


class PredictiveAdapterV3P6(PredictiveAdapterV3):
    """``predictive_adapter.v3`` whose goal features carry four task bits."""

    goal_feature_dim: ClassVar[int] = 9

    def _goal_features(self, goal: Tensor) -> Tensor:
        return torch.cat((goal[..., _EVENT], p6_task_one_hot(goal)), dim=-1)


class SparseRouterV3P6(SparseRouterV3):
    """``sparse_router.v3`` over four tasks; uncued steps follow the DRS rules."""

    task_count: ClassVar[int] = 4

    def __init__(
        self,
        feature_dim: int = 128,
        *,
        task_embedding_dim: int = 16,
        dual_route_fraction: float = 0.25,
        surprise_weight: float = 1.0,
        working_module_id: str = WORKING_MEMORY_V2,
    ) -> None:
        super().__init__(
            feature_dim,
            task_embedding_dim=task_embedding_dim,
            dual_route_fraction=dual_route_fraction,
            surprise_weight=surprise_weight,
        )
        if working_module_id not in {WORKING_MEMORY, WORKING_MEMORY_V2}:
            raise ValueError(f"unsupported working-memory expert: {working_module_id}")
        self.expert_ids = (EPISODIC_MEMORY, working_module_id)

    @staticmethod
    def _task_index(goal_context: Tensor) -> Tensor:
        return p6_task_one_hot(goal_context).argmax(dim=-1)

    @staticmethod
    def _task_masks(packet: BrainPacket) -> tuple[Tensor, Tensor]:
        goal = require_goal_context(packet, minimum_features=P6_GOAL_CONTEXT_DIM)
        rule_switch = goal[..., 70].gt(0.5) | goal[..., UNCUED_TASK_CHANNEL].gt(0.5)
        return goal[..., 69].gt(0.5), rule_switch


__all__ = [
    "P6ActionSelector",
    "P6TaskBoundaryAdapters",
    "PredictiveAdapterV3P6",
    "SparseRouterV3P6",
]
