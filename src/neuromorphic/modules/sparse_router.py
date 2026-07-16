"""Deterministic hard top-k router with capacity-aware expert reassignment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from neuromorphic.core.contracts import BrainPacket, ModuleContext, ModuleOutput, ModuleState
from neuromorphic.core.registry import OPTIONAL_EXPERT_IDS, SPARSE_ROUTER
from neuromorphic.modules._utils import packet_from, validate_inputs


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Both score-top-k and capacity-executed masks, kept explicitly distinct."""

    scores: Tensor
    raw_top2_mask: Tensor
    executed_mask: Tensor
    capacity: Tensor
    rerouted_mask: Tensor
    capacity_drops: int = 0

    def __post_init__(self) -> None:
        if self.scores.ndim != 3 or self.scores.shape[-1] != len(OPTIONAL_EXPERT_IDS):
            raise ValueError("scores must have shape [B, T, 3]")
        if self.raw_top2_mask.shape != self.scores.shape:
            raise ValueError("raw_top2_mask must align with scores")
        if self.executed_mask.shape != self.scores.shape:
            raise ValueError("executed_mask must align with scores")
        if self.rerouted_mask.shape != self.scores.shape:
            raise ValueError("rerouted_mask must align with scores")
        if any(
            mask.dtype is not torch.bool
            for mask in (self.raw_top2_mask, self.executed_mask, self.rerouted_mask)
        ):
            raise TypeError("routing masks must use torch.bool")
        if self.capacity.shape != (self.scores.shape[1],):
            raise ValueError("capacity must have shape [T]")
        if self.capacity.dtype != torch.long:
            raise TypeError("capacity must use torch.long")
        if self.capacity_drops < 0:
            raise ValueError("capacity_drops must be non-negative")


class SparseRouter(nn.Module):
    """Select exactly two of three optional experts for each valid token."""

    module_id = SPARSE_ROUTER
    state_version = "router-state-v1"

    def __init__(
        self,
        feature_dim: int = 128,
        *,
        task_embedding_dim: int = 16,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        if top_k != 2:
            raise ValueError("P2 sparse routing requires top_k=2")
        if capacity_factor < 1.0:
            raise ValueError("capacity_factor must be at least 1.0")
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.task_embedding = nn.Embedding(3, task_embedding_dim)
        self.scorer = nn.Linear(feature_dim + task_embedding_dim, len(OPTIONAL_EXPERT_IDS))
        self.fusion = nn.Linear(feature_dim, feature_dim)

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> ModuleState:
        del batch_size, device, dtype
        return ModuleState(self.module_id, self.state_version)

    def _scores(self, packet: BrainPacket) -> Tensor:
        if packet.goal_context is not None and packet.goal_context.shape[-1] >= 72:
            task_one_hot = packet.goal_context[..., 69:72]
            task_index = task_one_hot.argmax(dim=-1)
        else:
            task_index = torch.zeros_like(packet.step_index)
        task_features = self.task_embedding(task_index)
        return cast(Tensor, self.scorer(torch.cat((packet.representation, task_features), dim=-1)))

    def route(
        self,
        packet: BrainPacket,
        forced_experts: tuple[str, ...] | None = None,
    ) -> RoutingDecision:
        scores = self._scores(packet)
        raw = torch.zeros_like(scores, dtype=torch.bool)
        if forced_experts is None:
            order = torch.argsort(scores, dim=-1, descending=True, stable=True)
            raw.scatter_(-1, order[..., : self.top_k], True)
        else:
            if len(forced_experts) != self.top_k or len(set(forced_experts)) != self.top_k:
                raise ValueError("forced_experts must contain exactly two unique experts")
            unknown = set(forced_experts).difference(OPTIONAL_EXPERT_IDS)
            if unknown:
                raise ValueError(f"forced_experts contains unknown IDs: {sorted(unknown)}")
            forced_indices = torch.tensor(
                [OPTIONAL_EXPERT_IDS.index(module_id) for module_id in forced_experts],
                device=scores.device,
            )
            raw[..., forced_indices] = True
        raw &= packet.valid_mask.unsqueeze(-1)
        executed = raw.clone()
        capacities = torch.zeros(
            packet.representation.shape[1], device=scores.device, dtype=torch.long
        )
        expert_count = len(OPTIONAL_EXPERT_IDS)

        # Stable device-native greedy reassignment. Source and target experts are
        # visited in frozen registry order. Within a pair, score penalty is sorted
        # stably, so equal penalties retain batch-index order.
        for step in range(packet.representation.shape[1]):
            token_count = packet.valid_mask[:, step].sum()
            capacity = torch.ceil(
                token_count.to(scores.dtype) * self.capacity_factor * self.top_k / expert_count
            ).to(torch.long)
            capacities[step] = capacity
            step_executed = executed[:, step]
            for source in range(expert_count):
                for target in range(expert_count):
                    if source == target:
                        continue
                    loads = step_executed.to(torch.long).sum(dim=0)
                    moves = torch.minimum(
                        (loads[source] - capacity).clamp_min(0),
                        (capacity - loads[target]).clamp_min(0),
                    )
                    excluded = (~step_executed).to(torch.long).argmax(dim=-1)
                    candidates = step_executed[:, source] & excluded.eq(target)
                    penalty = scores[:, step, target] - scores[:, step, source]
                    ranked = torch.argsort(penalty.masked_fill(~candidates, torch.inf), stable=True)
                    ranks = torch.empty_like(ranked)
                    ranks.scatter_(0, ranked, torch.arange(ranked.numel(), device=scores.device))
                    chosen = candidates & ranks.lt(moves)
                    step_executed[:, source] &= ~chosen
                    step_executed[:, target] |= chosen
            executed[:, step] = step_executed
        rerouted = raw ^ executed
        return RoutingDecision(scores, raw, executed, capacities, rerouted, 0)

    def forward(
        self, packet: BrainPacket, state: ModuleState, context: ModuleContext
    ) -> ModuleOutput:
        validate_inputs(
            packet, state, context, module_id=self.module_id, version=self.state_version
        )
        decision = self.route(packet)
        probabilities = torch.softmax(decision.scores, dim=-1)
        valid = packet.valid_mask.unsqueeze(-1).to(probabilities.dtype)
        mean_probability = (probabilities * valid).sum(dim=(0, 1)) / valid.sum().clamp_min(1.0)
        load_balance = (mean_probability - 1.0 / len(OPTIONAL_EXPERT_IDS)).square().mean()
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
        decision: RoutingDecision,
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
        unknown = set(expert_packets).difference(OPTIONAL_EXPERT_IDS)
        if unknown:
            raise ValueError(f"combine received unknown experts: {sorted(unknown)}")
        weights = torch.softmax(
            decision.scores.masked_fill(~decision.executed_mask, -1.0e4), dim=-1
        )
        combined = encoder_packet.representation
        for index, module_id in enumerate(OPTIONAL_EXPERT_IDS):
            if module_id not in expert_packets:
                continue
            expert = expert_packets[module_id]
            if expert.source_module != module_id:
                raise ValueError("expert packet source does not match mapping key")
            if expert.representation.shape != combined.shape:
                raise ValueError("expert packet representation must align with encoder")
            contribution = expert.representation - encoder_packet.representation
            active_weight = weights[..., index] * decision.executed_mask[..., index]
            combined = combined + contribution * active_weight.unsqueeze(-1)
        combined = encoder_packet.representation + self.fusion(combined)
        combined = torch.where(
            encoder_packet.valid_mask.unsqueeze(-1), combined, encoder_packet.representation
        )
        return ModuleOutput(packet_from(encoder_packet, combined, self.module_id), state)

    def reset_state(self, state: ModuleState, reset_mask: Tensor) -> ModuleState:
        if state.module_id != self.module_id or state.version != self.state_version:
            raise ValueError("invalid sparse router state")
        if reset_mask.ndim != 1 or reset_mask.dtype is not torch.bool:
            raise ValueError("reset_mask must be a boolean [B] tensor")
        return state


__all__ = ["RoutingDecision", "SparseRouter"]
