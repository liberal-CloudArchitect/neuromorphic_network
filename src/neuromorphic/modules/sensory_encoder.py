"""Shared sensory representation encoder for artificial task inputs."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from neuromorphic.core.contracts import (
    BrainPacket,
    ModuleContext,
    ModuleOutput,
    ModuleState,
)
from neuromorphic.core.registry import SENSORY_ENCODER
from neuromorphic.modules._utils import packet_from, validate_inputs


class SensoryEncoder(nn.Module):
    """Normalize and refine task-adapter features with a residual MLP."""

    module_id = SENSORY_ENCODER
    state_version = "sensory-state-v1"

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = feature_dim
        self.norm = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> ModuleState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        del device, dtype
        return ModuleState(self.module_id, self.state_version)

    def forward_inputs(
        self,
        inputs: Tensor,
        valid_mask: Tensor,
        step_index: Tensor,
        goal_context: Tensor | None,
        context: ModuleContext,
        *,
        modality: str = "symbolic",
        state: ModuleState | None = None,
    ) -> ModuleOutput:
        """Create the first legal packet directly from adapter features."""

        if inputs.ndim != 3 or inputs.shape[-1] != self.feature_dim:
            raise ValueError(f"inputs must have shape [B, T, {self.feature_dim}]")
        batch_size = inputs.shape[0]
        if state is None:
            state = self.initial_state(batch_size, device=inputs.device, dtype=inputs.dtype)
        packet = BrainPacket(
            representation=inputs,
            valid_mask=valid_mask,
            modality=modality,
            step_index=step_index,
            source_module=self.module_id,
            goal_context=goal_context,
        )
        return self.forward(packet, state, context)

    def forward(
        self, packet: BrainPacket, state: ModuleState, context: ModuleContext
    ) -> ModuleOutput:
        validate_inputs(
            packet, state, context, module_id=self.module_id, version=self.state_version
        )
        if packet.representation.shape[-1] != self.feature_dim:
            raise ValueError(f"representation feature size must be {self.feature_dim}")
        encoded = packet.representation + self.mlp(self.norm(packet.representation))
        encoded = torch.where(packet.valid_mask.unsqueeze(-1), encoded, packet.representation)
        return ModuleOutput(packet_from(packet, encoded, self.module_id), state)

    def reset_state(self, state: ModuleState, reset_mask: Tensor) -> ModuleState:
        if state.module_id != self.module_id or state.version != self.state_version:
            raise ValueError("invalid sensory encoder state")
        if reset_mask.ndim != 1 or reset_mask.dtype is not torch.bool:
            raise ValueError("reset_mask must be a boolean [B] tensor")
        return state


__all__ = ["SensoryEncoder"]
