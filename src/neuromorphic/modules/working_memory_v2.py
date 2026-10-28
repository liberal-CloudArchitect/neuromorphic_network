"""Read-enriched, surprise-gated working memory (CR-015, p6-protocol-v1).

Reads exactly as ``working_memory.v1``.  Two changes affect the write:

- the candidate is computed from the read-enriched representation
  ``e_t = r_t + P·read_t`` instead of ``r_t`` (``write_source="enriched"``);
- the update-gate logit adds ``s_t·w``, where ``s_t`` is the detached
  forecast surprise of ``predictive_adapter.v3`` and ``w`` holds one weight per
  slot (``surprise_gating=True``).

``write_source="current"`` and ``surprise_gating=False`` are the frozen
``enrich-off`` and ``gate-off`` mechanism cells; with both, the module computes
``working_memory.v1``.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from neuromorphic.core.contracts import BrainPacket, ModuleContext, ModuleOutput, ModuleState
from neuromorphic.core.registry import WORKING_MEMORY_V2
from neuromorphic.modules._utils import masked_mean, packet_from, reset_tensor_rows, validate_inputs

WriteSource = Literal["enriched", "current"]


class WorkingMemoryV2(nn.Module):
    """Four learned 32-dimensional slots with read-enriched, surprise-gated writes."""

    module_id = WORKING_MEMORY_V2
    state_version = "working-state-v2"

    def __init__(
        self,
        feature_dim: int = 128,
        slots: int = 4,
        slot_dim: int = 32,
        *,
        write_source: WriteSource = "enriched",
        surprise_gating: bool = True,
        surprise_init: float = 2.0,
    ) -> None:
        super().__init__()
        if min(feature_dim, slots, slot_dim) <= 0:
            raise ValueError("working-memory dimensions must be positive")
        if write_source not in {"enriched", "current"}:
            raise ValueError(f"unknown write source: {write_source}")
        self.feature_dim = feature_dim
        self.slots = slots
        self.slot_dim = slot_dim
        self.write_source = write_source
        self.surprise_gating = surprise_gating
        self.candidate = nn.Linear(feature_dim, slot_dim)
        self.update_gate = nn.Linear(feature_dim, slots)
        self.read_query = nn.Linear(feature_dim, slot_dim)
        self.read_projection = nn.Linear(slot_dim, feature_dim)
        self.surprise_gate = nn.Parameter(torch.full((slots,), surprise_init))

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> ModuleState:
        tensors = {
            "slots": torch.zeros(batch_size, self.slots, self.slot_dim, device=device, dtype=dtype)
        }
        return ModuleState(self.module_id, self.state_version, tensors)

    def forward(
        self,
        packet: BrainPacket,
        state: ModuleState,
        context: ModuleContext,
        *,
        surprise: Tensor | None = None,
    ) -> ModuleOutput:
        """Run ``[B, T]`` steps; ``surprise`` is ``[B, T, 1]`` and required when gating."""

        validate_inputs(
            packet, state, context, module_id=self.module_id, version=self.state_version
        )
        if packet.representation.shape[-1] != self.feature_dim:
            raise ValueError(f"representation feature size must be {self.feature_dim}")
        gate_surprise = self._gate_surprise(packet, context, surprise)
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

            logits = self.update_gate(representation)
            if gate_surprise is not None:
                logits = logits + gate_surprise[:, step, None] * self.surprise_gate[None, :]
            gates = torch.sigmoid(logits) * valid.unsqueeze(-1)
            source = enriched if self.write_source == "enriched" else representation
            candidate = torch.tanh(self.candidate(source)).unsqueeze(1)
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

    def _gate_surprise(
        self, packet: BrainPacket, context: ModuleContext, surprise: Tensor | None
    ) -> Tensor | None:
        """Detached ``[B, T]`` gate input, zero on padding and episode-reset steps."""

        if not self.surprise_gating:
            return None
        if surprise is None:
            raise ValueError("working_memory.v2 with surprise gating requires surprise")
        if surprise.shape != (*packet.valid_mask.shape, 1):
            raise ValueError("surprise must have shape [B, T, 1]")
        if surprise.device != packet.representation.device:
            raise ValueError("surprise and packet must share a device")
        if not surprise.is_floating_point():
            raise TypeError("surprise must use a floating-point dtype")
        keep = packet.valid_mask & ~context.reset_mask
        return torch.where(keep, surprise[..., 0].detach(), torch.zeros_like(surprise[..., 0]))

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


__all__ = ["WorkingMemoryV2", "WriteSource"]
