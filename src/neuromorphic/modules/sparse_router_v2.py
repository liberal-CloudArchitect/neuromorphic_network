"""Top-1 router with explicit reservations and zero-drop per-step capacities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn

from neuromorphic.core.contracts import BrainPacket, ModuleContext, ModuleOutput, ModuleState
from neuromorphic.core.registry import EPISODIC_MEMORY, SPARSE_ROUTER_V2, WORKING_MEMORY
from neuromorphic.modules._utils import packet_from, require_goal_context, validate_inputs

_EVENT = slice(0, 5)
_TASK = slice(69, 72)
_EXPERT_IDS = (EPISODIC_MEMORY, WORKING_MEMORY)


@dataclass(frozen=True, slots=True)
class RoutingDecisionV2:
    """Explicitly track reservation, learned choice, and executed routing masks."""

    scores: Tensor
    reserved_mask: Tensor
    learned_mask: Tensor
    raw_mask: Tensor
    executed_mask: Tensor
    capacity: Tensor
    capacity_drops: int = 0

    def __post_init__(self) -> None:
        if self.scores.ndim != 3 or self.scores.shape[-1] != len(_EXPERT_IDS):
            raise ValueError("scores must have shape [B, T, 2]")
        expected = self.scores.shape
        for name, mask in (
            ("reserved_mask", self.reserved_mask),
            ("learned_mask", self.learned_mask),
            ("raw_mask", self.raw_mask),
            ("executed_mask", self.executed_mask),
        ):
            if mask.shape != expected:
                raise ValueError(f"{name} must align with scores")
            if mask.dtype is not torch.bool:
                raise TypeError(f"{name} must use torch.bool")
        if self.capacity.shape != (self.scores.shape[1], len(_EXPERT_IDS)):
            raise ValueError("capacity must have shape [T, 2]")
        if self.capacity.dtype != torch.long:
            raise TypeError("capacity must use torch.long")
        if self.capacity_drops != 0:
            raise ValueError("RoutingDecisionV2 does not permit dropped tokens")


class SparseRouterV2(nn.Module):
    """Route top-1 over episodic and working memory with optional reservations."""

    module_id = SPARSE_ROUTER_V2
    state_version = "router-state-v2"

    def __init__(self, feature_dim: int = 128, *, task_embedding_dim: int = 16) -> None:
        super().__init__()
        if min(feature_dim, task_embedding_dim) <= 0:
            raise ValueError("router dimensions must be positive")
        self.feature_dim = feature_dim
        self.task_embedding = nn.Embedding(3, task_embedding_dim)
        self.scorer = nn.Linear(feature_dim + task_embedding_dim, len(_EXPERT_IDS))
        self.fusion = nn.Linear(feature_dim, feature_dim)

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> ModuleState:
        del batch_size, device, dtype
        return ModuleState(self.module_id, self.state_version)

    def _scores(self, packet: BrainPacket) -> Tensor:
        if packet.goal_context is not None and packet.goal_context.shape[-1] >= 72:
            task_one_hot = packet.goal_context[..., _TASK]
            task_index = task_one_hot.argmax(dim=-1)
        else:
            task_index = torch.zeros_like(packet.step_index)
        task_features = self.task_embedding(task_index)
        return cast(Tensor, self.scorer(torch.cat((packet.representation, task_features), dim=-1)))

    @staticmethod
    def _reservation_mask(packet: BrainPacket) -> Tensor:
        goal = require_goal_context(packet, minimum_features=72)
        is_associative_recall = goal[..., 69].gt(0.5)
        event = goal[..., _EVENT]
        reserve_episodic = is_associative_recall & (event[..., 0].gt(0.5) | event[..., 2].gt(0.5))
        reserved = torch.zeros(
            (*packet.valid_mask.shape, len(_EXPERT_IDS)),
            dtype=torch.bool,
            device=packet.representation.device,
        )
        reserved[..., 0] = reserve_episodic & packet.valid_mask
        return reserved

    def route(
        self,
        packet: BrainPacket,
        *,
        mode: Literal["learned", "dense", "no_reservation", "legacy_capacity"] = "learned",
    ) -> RoutingDecisionV2:
        if mode not in {"learned", "dense", "no_reservation", "legacy_capacity"}:
            raise ValueError(f"unsupported routing mode: {mode}")
        scores = self._scores(packet)
        valid = packet.valid_mask.unsqueeze(-1)
        reserved = self._reservation_mask(packet)
        learned = torch.zeros_like(reserved)

        if mode == "dense":
            raw = valid.expand_as(scores).clone()
            executed = raw.clone()
        else:
            order = torch.argsort(scores, dim=-1, descending=True, stable=True)
            learned.scatter_(-1, order[..., :1], True)
            learned &= valid
            if mode in {"no_reservation", "legacy_capacity"}:
                raw = learned
            else:
                has_reservation = reserved.any(dim=-1, keepdim=True)
                raw = reserved | (learned & ~has_reservation)
            executed = raw.clone()

        valid_per_step = packet.valid_mask.to(torch.long).sum(dim=0, keepdim=False)
        if mode == "legacy_capacity":
            # Frozen P3-style control: no reservation and a per-expert quota.
            # Overflow is rerouted to the other expert rather than dropped.
            capacity_per_step = torch.ceil(valid_per_step.to(scores.dtype) * 1.25 / 2).to(
                torch.long
            )
            capacity = capacity_per_step.unsqueeze(-1).expand(-1, len(_EXPERT_IDS)).clone()
            for step in range(scores.shape[1]):
                for expert in range(len(_EXPERT_IDS)):
                    candidates = raw[:, step, expert]
                    order = torch.argsort(scores[:, step, expert], descending=True, stable=True)
                    ranked = torch.zeros_like(candidates, dtype=torch.long)
                    ranked[order] = torch.arange(
                        scores.shape[0], device=scores.device, dtype=torch.long
                    )
                    overflow = candidates & ranked.ge(capacity_per_step[step])
                    executed[overflow, step, expert] = False
                    executed[overflow, step, 1 - expert] = True
        else:
            # Capacity is an allowance, not observed load. Each P4 expert may
            # receive every valid row, so reservations cannot be displaced.
            capacity = valid_per_step.unsqueeze(-1).expand(-1, len(_EXPERT_IDS)).clone()
        return RoutingDecisionV2(scores, reserved, learned, raw, executed, capacity, 0)

    def forward(
        self, packet: BrainPacket, state: ModuleState, context: ModuleContext
    ) -> ModuleOutput:
        validate_inputs(
            packet, state, context, module_id=self.module_id, version=self.state_version
        )
        decision = self.route(packet)
        valid = packet.valid_mask.unsqueeze(-1).to(packet.representation.dtype)
        unreserved = (~decision.reserved_mask.any(dim=-1, keepdim=True)).to(valid.dtype)
        balance_valid = valid * unreserved
        probabilities = torch.softmax(decision.scores, dim=-1)
        mean_probability = (probabilities * balance_valid).sum(dim=(0, 1)) / (
            balance_valid.sum().clamp_min(1.0)
        )
        load_balance = (mean_probability - 1.0 / len(_EXPERT_IDS)).square().mean()
        communication = (
            decision.executed_mask.to(probabilities.dtype) * probabilities * valid
        ).sum() / valid.sum().clamp_min(1.0)
        return ModuleOutput(
            packet_from(packet, packet.representation, self.module_id),
            state,
            auxiliary_losses={
                "router_load_balance": load_balance,
                "router_communication_cost": communication,
            },
        )

    def combine(
        self,
        encoder_packet: BrainPacket,
        expert_packets: Mapping[str, BrainPacket],
        decision: RoutingDecisionV2,
        state: ModuleState,
        context: ModuleContext,
    ) -> ModuleOutput:
        validate_inputs(
            encoder_packet,
            state,
            context,
            module_id=self.module_id,
            version=self.state_version,
        )
        if decision.scores.shape[:2] != encoder_packet.valid_mask.shape:
            raise ValueError("routing decision and encoder packet must align")
        unknown = set(expert_packets).difference(_EXPERT_IDS)
        if unknown:
            raise ValueError(f"combine received unknown experts: {sorted(unknown)}")
        masked_scores = decision.scores.masked_fill(~decision.executed_mask, -1.0e4)
        weights = torch.softmax(masked_scores, dim=-1) * decision.executed_mask.to(
            masked_scores.dtype
        )
        combined = encoder_packet.representation
        for index, module_id in enumerate(_EXPERT_IDS):
            expert = expert_packets.get(module_id)
            if expert is None:
                continue
            if expert.source_module != module_id:
                raise ValueError("expert packet source does not match mapping key")
            if expert.representation.shape != combined.shape:
                raise ValueError("expert packet representation must align with encoder")
            contribution = expert.representation - encoder_packet.representation
            combined = combined + contribution * weights[..., index].unsqueeze(-1)
        fused = encoder_packet.representation + self.fusion(combined)
        fused = torch.where(
            encoder_packet.valid_mask.unsqueeze(-1), fused, encoder_packet.representation
        )
        return ModuleOutput(packet_from(encoder_packet, fused, self.module_id), state)

    def reset_state(self, state: ModuleState, reset_mask: Tensor) -> ModuleState:
        if state.module_id != self.module_id or state.version != self.state_version:
            raise ValueError("invalid sparse router state")
        if reset_mask.ndim != 1 or reset_mask.dtype is not torch.bool:
            raise ValueError("reset_mask must be a boolean [B] tensor")
        return state


__all__ = ["RoutingDecisionV2", "SparseRouterV2"]
