"""Action-conditioned next-state prediction auxiliary module."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from neuromorphic.core.contracts import (
    BrainPacket,
    ModuleContext,
    ModuleOutput,
    ModuleState,
)
from neuromorphic.core.registry import PREDICTIVE_ADAPTER
from neuromorphic.modules._utils import (
    packet_from,
    require_goal_context,
    reset_tensor_rows,
    validate_inputs,
)


class PredictiveAdapter(nn.Module):
    """Predict graph transitions from the model's actual selected action."""

    module_id = PREDICTIVE_ADAPTER
    state_version = "predictive-state-v1"

    def __init__(
        self, feature_dim: int = 128, action_count: int = 32, action_dim: int = 32
    ) -> None:
        super().__init__()
        if min(feature_dim, action_count, action_dim) <= 0:
            raise ValueError("predictive dimensions must be positive")
        self.feature_dim = feature_dim
        self.action_count = action_count
        self.action_embedding = nn.Embedding(action_count, action_dim)
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim + action_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 16),
        )

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> ModuleState:
        del dtype
        return ModuleState(
            self.module_id,
            self.state_version,
            {"pending_action": torch.full((batch_size,), -1, device=device, dtype=torch.long)},
        )

    @staticmethod
    def dynamic_targets(action_nodes: Tensor, selected_action: Tensor) -> Tensor:
        """Resolve transition targets from action slots, never an oracle next-state field."""

        if action_nodes.ndim != 3:
            raise ValueError("action_nodes must have shape [B, T, A]")
        if selected_action.shape != action_nodes.shape[:2]:
            raise ValueError("selected_action must have shape [B, T]")
        if action_nodes.dtype != torch.long or selected_action.dtype != torch.long:
            raise TypeError("action_nodes and selected_action must use torch.long")
        valid_action = selected_action.ge(0) & selected_action.lt(action_nodes.shape[-1])
        safe_action = selected_action.clamp(0, action_nodes.shape[-1] - 1)
        target = action_nodes.gather(-1, safe_action.unsqueeze(-1)).squeeze(-1)
        return torch.where(valid_action & target.ge(0), target, torch.full_like(target, -100))

    def forward_with_action(
        self,
        packet: BrainPacket,
        state: ModuleState,
        context: ModuleContext,
        selected_action: Tensor,
        action_nodes: Tensor | None = None,
    ) -> ModuleOutput:
        if selected_action.shape != packet.valid_mask.shape:
            raise ValueError("selected_action must have shape [B, T]")
        if selected_action.dtype != torch.long:
            raise TypeError("selected_action must use torch.long")
        validate_inputs(
            packet, state, context, module_id=self.module_id, version=self.state_version
        )
        safe_action = selected_action.clamp(0, self.action_count - 1)
        action_features = self.action_embedding(safe_action)
        prediction_logits = self.predictor(
            torch.cat((packet.representation, action_features), dim=-1)
        )
        active = packet.valid_mask & selected_action.ge(0) & selected_action.lt(self.action_count)
        prediction_logits = torch.where(
            active.unsqueeze(-1), prediction_logits, torch.zeros_like(prediction_logits)
        )
        pending = state.tensors["pending_action"]
        for step in range(packet.representation.shape[1]):
            pending = reset_tensor_rows(pending, context.reset_mask[:, step], -1)
            pending = torch.where(active[:, step], selected_action[:, step], pending)
        next_state = ModuleState(self.module_id, self.state_version, {"pending_action": pending})
        losses: dict[str, Tensor] = {}
        if action_nodes is not None:
            targets = self.dynamic_targets(action_nodes, selected_action)
            selected = targets.ne(-100) & packet.valid_mask
            if torch.any(selected).item():
                losses["predictive_next_state"] = nn.functional.cross_entropy(
                    prediction_logits[selected], targets[selected]
                )
            else:
                losses["predictive_next_state"] = prediction_logits.sum() * 0.0
        else:
            losses["predictive_next_state"] = prediction_logits.sum() * 0.0
        return ModuleOutput(
            packet_from(packet, packet.representation, self.module_id),
            next_state,
            prediction_logits=prediction_logits,
            auxiliary_losses=losses,
        )

    def forward(
        self, packet: BrainPacket, state: ModuleState, context: ModuleContext
    ) -> ModuleOutput:
        goal = require_goal_context(packet)
        action_copy = goal[..., 72:104]
        selected_action = action_copy.argmax(dim=-1)
        selected_action = torch.where(
            action_copy.sum(dim=-1).gt(0), selected_action, torch.full_like(selected_action, -1)
        )
        return self.forward_with_action(packet, state, context, selected_action)

    def reset_state(self, state: ModuleState, reset_mask: Tensor) -> ModuleState:
        if state.module_id != self.module_id or state.version != self.state_version:
            raise ValueError("invalid predictive adapter state")
        if reset_mask.ndim != 1 or reset_mask.dtype is not torch.bool:
            raise ValueError("reset_mask must be a boolean [B] tensor")
        return ModuleState(
            self.module_id,
            self.state_version,
            {"pending_action": reset_tensor_rows(state.tensors["pending_action"], reset_mask, -1)},
        )


__all__ = ["PredictiveAdapter"]
