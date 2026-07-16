"""Gated fixed-capacity working-memory abstraction."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from neuromorphic.core.contracts import BrainPacket, ModuleContext, ModuleOutput, ModuleState
from neuromorphic.core.registry import WORKING_MEMORY
from neuromorphic.modules._utils import masked_mean, packet_from, reset_tensor_rows, validate_inputs


class WorkingMemory(nn.Module):
    """Four learned 32-dimensional slots updated by module-owned gates."""

    module_id = WORKING_MEMORY
    state_version = "working-state-v1"

    def __init__(self, feature_dim: int = 128, slots: int = 4, slot_dim: int = 32) -> None:
        super().__init__()
        if min(feature_dim, slots, slot_dim) <= 0:
            raise ValueError("working-memory dimensions must be positive")
        self.feature_dim = feature_dim
        self.slots = slots
        self.slot_dim = slot_dim
        self.candidate = nn.Linear(feature_dim, slot_dim)
        self.update_gate = nn.Linear(feature_dim, slots)
        self.read_query = nn.Linear(feature_dim, slot_dim)
        self.read_projection = nn.Linear(slot_dim, feature_dim)

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> ModuleState:
        tensors = {
            "slots": torch.zeros(batch_size, self.slots, self.slot_dim, device=device, dtype=dtype)
        }
        return ModuleState(self.module_id, self.state_version, tensors)

    def forward(
        self, packet: BrainPacket, state: ModuleState, context: ModuleContext
    ) -> ModuleOutput:
        validate_inputs(
            packet, state, context, module_id=self.module_id, version=self.state_version
        )
        if packet.representation.shape[-1] != self.feature_dim:
            raise ValueError(f"representation feature size must be {self.feature_dim}")
        slots = state.tensors["slots"]
        outputs: list[Tensor] = []
        changes: list[Tensor] = []
        regularizers: list[Tensor] = []
        for step in range(packet.representation.shape[1]):
            slots = reset_tensor_rows(slots, context.reset_mask[:, step])
            representation = packet.representation[:, step]
            valid = packet.valid_mask[:, step]
            attention_logits = (
                torch.einsum("bsd,bd->bs", slots, self.read_query(representation))
                / self.slot_dim**0.5
            )
            attention = torch.softmax(attention_logits, dim=-1)
            read = torch.einsum("bs,bsd->bd", attention, slots)
            enriched = representation + self.read_projection(read)

            gates = torch.sigmoid(self.update_gate(representation))
            gates = gates * valid.unsqueeze(-1)
            candidate = torch.tanh(self.candidate(representation)).unsqueeze(1)
            updated = slots + gates.unsqueeze(-1) * (candidate - slots)
            updated = torch.where(valid[:, None, None], updated, slots)
            changes.append((updated - slots).square().mean(dim=(1, 2)))
            regularizers.append((gates * (1.0 - gates)).mean(dim=-1))
            slots = updated
            outputs.append(torch.where(valid.unsqueeze(-1), enriched, representation))

        consistency = masked_mean(torch.stack(changes, dim=1), packet.valid_mask)
        gate_regularization = masked_mean(torch.stack(regularizers, dim=1), packet.valid_mask)
        next_state = ModuleState(self.module_id, self.state_version, {"slots": slots})
        return ModuleOutput(
            packet_from(packet, torch.stack(outputs, dim=1), self.module_id),
            next_state,
            auxiliary_losses={
                "working_consistency": consistency,
                "working_gate_regularization": gate_regularization,
            },
        )

    def reset_state(self, state: ModuleState, reset_mask: Tensor) -> ModuleState:
        if state.module_id != self.module_id or state.version != self.state_version:
            raise ValueError("invalid working memory state")
        if reset_mask.ndim != 1 or reset_mask.dtype is not torch.bool:
            raise ValueError("reset_mask must be a boolean [B] tensor")
        return ModuleState(
            self.module_id,
            self.state_version,
            {"slots": reset_tensor_rows(state.tensors["slots"], reset_mask)},
        )


__all__ = ["WorkingMemory"]
