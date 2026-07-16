"""Bounded episode-local key/value memory with strict read-before-write."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from neuromorphic.core.contracts import BrainPacket, ModuleContext, ModuleOutput, ModuleState
from neuromorphic.core.registry import EPISODIC_MEMORY
from neuromorphic.modules._utils import (
    masked_mean,
    packet_from,
    require_goal_context,
    reset_tensor_rows,
    validate_inputs,
)


class EpisodicMemory(nn.Module):
    """Sixteen-slot content-addressed memory for associative recall events."""

    module_id = EPISODIC_MEMORY
    state_version = "episodic-state-v1"

    def __init__(self, feature_dim: int = 128, slots: int = 16, key_dim: int = 32) -> None:
        super().__init__()
        if min(feature_dim, slots, key_dim) <= 0:
            raise ValueError("memory dimensions must be positive")
        self.feature_dim = feature_dim
        self.slots = slots
        self.key_dim = key_dim
        self.read_projection = nn.Linear(feature_dim, feature_dim)
        self.value_projection = nn.Linear(feature_dim, feature_dim)

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> ModuleState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        tensors = {
            "keys": torch.zeros(batch_size, self.slots, self.key_dim, device=device, dtype=dtype),
            "values": torch.zeros(
                batch_size, self.slots, self.feature_dim, device=device, dtype=dtype
            ),
            "occupied": torch.zeros(batch_size, self.slots, device=device, dtype=torch.bool),
            "write_index": torch.zeros(batch_size, device=device, dtype=torch.long),
            "pending_key": torch.zeros(batch_size, self.key_dim, device=device, dtype=dtype),
            "pending_value": torch.zeros(batch_size, self.feature_dim, device=device, dtype=dtype),
            "pending_valid": torch.zeros(batch_size, device=device, dtype=torch.bool),
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
        goal = require_goal_context(packet)
        current = state
        outputs: list[Tensor] = []
        retrieval_terms: list[Tensor] = []
        separation_terms: list[Tensor] = []
        for step in range(packet.representation.shape[1]):
            current = self.reset_state(current, context.reset_mask[:, step])
            representation = packet.representation[:, step]
            valid = packet.valid_mask[:, step]
            event = goal[:, step, :5]
            key = goal[:, step, 5 : 5 + self.key_dim]
            associative_task = context.task_id == "associative_recall.v1"
            store = valid & event[:, 0].gt(0.5) & associative_task
            query = valid & event[:, 2].gt(0.5) & associative_task

            keys = current.tensors["keys"]
            values = current.tensors["values"]
            occupied = current.tensors["occupied"]
            similarity = torch.einsum("bsk,bk->bs", keys, key)
            similarity = similarity.masked_fill(~occupied, -1.0e4)
            weights = torch.softmax(similarity, dim=-1) * occupied.to(representation.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            read = torch.einsum("bs,bsf->bf", weights, values)
            has_match = occupied.any(dim=-1) & query
            enriched = representation + self.read_projection(read) * has_match.unsqueeze(-1)
            outputs.append(torch.where(valid.unsqueeze(-1), enriched, representation))

            retrieval_error = (read - representation.detach()).square().mean(dim=-1)
            retrieval_terms.append(retrieval_error * query.to(retrieval_error.dtype))
            occupied_similarity = similarity.masked_fill(~occupied, 0.0).clamp_min(0.0)
            separation_terms.append(
                occupied_similarity.square().sum(dim=-1)
                / occupied.sum(dim=-1).clamp_min(1).to(representation.dtype)
                * store.to(representation.dtype)
            )

            pending_key = torch.where(store.unsqueeze(-1), key, current.tensors["pending_key"])
            pending_value = torch.where(
                store.unsqueeze(-1),
                self.value_projection(representation),
                current.tensors["pending_value"],
            )
            pending_valid = current.tensors["pending_valid"] | store
            current = ModuleState(
                self.module_id,
                self.state_version,
                {
                    **current.tensors,
                    "pending_key": pending_key,
                    "pending_value": pending_value,
                    "pending_valid": pending_valid,
                },
            )

        output = torch.stack(outputs, dim=1)
        retrieval = masked_mean(torch.stack(retrieval_terms, dim=1), packet.valid_mask)
        separation = masked_mean(torch.stack(separation_terms, dim=1), packet.valid_mask)
        return ModuleOutput(
            packet_from(packet, output, self.module_id),
            current,
            auxiliary_losses={
                "episodic_retrieval": retrieval,
                "episodic_separation": separation,
            },
        )

    def commit_pending(self, state: ModuleState) -> ModuleState:
        """Commit staged stores only after downstream action formation."""

        if state.module_id != self.module_id or state.version != self.state_version:
            raise ValueError("invalid episodic memory state")
        pending = state.tensors["pending_valid"]
        indices = state.tensors["write_index"]
        slot_mask = functional.one_hot(indices, self.slots).to(torch.bool) & pending.unsqueeze(-1)
        keys = torch.where(
            slot_mask.unsqueeze(-1),
            state.tensors["pending_key"].unsqueeze(1),
            state.tensors["keys"],
        )
        values = torch.where(
            slot_mask.unsqueeze(-1),
            state.tensors["pending_value"].unsqueeze(1),
            state.tensors["values"],
        )
        occupied = state.tensors["occupied"] | slot_mask
        write_index = torch.where(pending, (indices + 1).remainder(self.slots), indices)
        tensors = {
            **state.tensors,
            "keys": keys,
            "values": values,
            "occupied": occupied,
            "write_index": write_index,
            "pending_key": torch.zeros_like(state.tensors["pending_key"]),
            "pending_value": torch.zeros_like(state.tensors["pending_value"]),
            "pending_valid": torch.zeros_like(pending),
        }
        return ModuleState(self.module_id, self.state_version, tensors)

    def reset_state(self, state: ModuleState, reset_mask: Tensor) -> ModuleState:
        if state.module_id != self.module_id or state.version != self.state_version:
            raise ValueError("invalid episodic memory state")
        if reset_mask.ndim != 1 or reset_mask.dtype is not torch.bool:
            raise ValueError("reset_mask must be a boolean [B] tensor")
        fills: dict[str, float | int | bool] = {"write_index": 0}
        tensors = {
            name: reset_tensor_rows(
                tensor,
                reset_mask,
                fills.get(name, False if tensor.dtype is torch.bool else 0),
            )
            for name, tensor in state.tensors.items()
        }
        return ModuleState(self.module_id, self.state_version, tensors)


__all__ = ["EpisodicMemory"]
