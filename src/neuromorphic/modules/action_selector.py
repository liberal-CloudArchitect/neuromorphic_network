"""Task-isolated action heads on a shared conflict-integration core."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from neuromorphic.core.contracts import BrainPacket, ModuleContext, ModuleOutput, ModuleState
from neuromorphic.core.registry import ACTION_SELECTOR
from neuromorphic.modules._utils import packet_from, require_goal_context, validate_inputs

TASK_CLASS_COUNTS = {
    "associative_recall.v1": 32,
    "delayed_rule_switch.v1": 2,
    "small_graph.v1": 4,
}


class ActionSelector(nn.Module):
    """Select task actions without sharing output-head parameters."""

    module_id = ACTION_SELECTOR
    state_version = "selector-state-v1"

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.integration = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict(
            {
                "associative_recall": nn.Linear(feature_dim, 32),
                "delayed_rule_switch": nn.Linear(feature_dim, 2),
                "small_graph": nn.Linear(feature_dim, 4),
            }
        )

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> ModuleState:
        del batch_size, device, dtype
        return ModuleState(self.module_id, self.state_version)

    def forward(
        self, packet: BrainPacket, state: ModuleState, context: ModuleContext
    ) -> ModuleOutput:
        validate_inputs(
            packet, state, context, module_id=self.module_id, version=self.state_version
        )
        try:
            class_count = TASK_CLASS_COUNTS[context.task_id]
        except KeyError as error:
            raise ValueError(f"unsupported action task: {context.task_id}") from error
        head_name = context.task_id.removesuffix(".v1")
        integrated = self.integration(packet.representation)
        integrated = torch.where(packet.valid_mask.unsqueeze(-1), integrated, packet.representation)
        logits = self.heads[head_name](integrated)
        if logits.shape[-1] != class_count:
            raise RuntimeError("task head class count does not match the frozen specification")
        if context.task_id == "small_graph.v1":
            goal = require_goal_context(packet)
            action_mask = goal[..., 37:41].gt(0.5)
            logits = logits.masked_fill(~action_mask, -torch.finfo(logits.dtype).max)
        logits = torch.where(packet.valid_mask.unsqueeze(-1), logits, torch.zeros_like(logits))
        return ModuleOutput(
            packet_from(packet, integrated, self.module_id),
            state,
            action_logits=logits,
        )

    def reset_state(self, state: ModuleState, reset_mask: Tensor) -> ModuleState:
        if state.module_id != self.module_id or state.version != self.state_version:
            raise ValueError("invalid action selector state")
        if reset_mask.ndim != 1 or reset_mask.dtype is not torch.bool:
            raise ValueError("reset_mask must be a boolean [B] tensor")
        return state


__all__ = ["TASK_CLASS_COUNTS", "ActionSelector"]
