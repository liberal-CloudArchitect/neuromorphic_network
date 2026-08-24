"""Semantic memory routing with bounded uncertainty-driven dual execution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neuromorphic.core.contracts import BrainPacket, ModuleContext, ModuleOutput, ModuleState
from neuromorphic.core.registry import EPISODIC_MEMORY, SPARSE_ROUTER_V3, WORKING_MEMORY
from neuromorphic.modules._utils import packet_from, require_goal_context, validate_inputs
from neuromorphic.modules.sparse_router_v2 import RoutingDecisionV2, SparseRouterV2

_EVENT = slice(0, 5)
_TASK = slice(69, 72)
_EXPERT_IDS = (EPISODIC_MEMORY, WORKING_MEMORY)


class SparseRouterV3(SparseRouterV2):
    """Reserve task-critical memory and dual-route only a bounded DRS subset.

    AR store/query tokens require episodic memory.  Every valid DRS token
    requires working memory, while at most ``dual_route_fraction`` of rows per
    step also execute episodic memory according to score ambiguity and detached
    predictive surprise.  Straight-through fusion lets the task loss train the
    scorer even when the forward pass executes a single expert.
    """

    module_id = SPARSE_ROUTER_V3
    state_version = "router-state-v3"

    def __init__(
        self,
        feature_dim: int = 128,
        *,
        task_embedding_dim: int = 16,
        dual_route_fraction: float = 0.25,
        surprise_weight: float = 1.0,
    ) -> None:
        super().__init__(feature_dim=feature_dim, task_embedding_dim=task_embedding_dim)
        if not 0.0 <= dual_route_fraction <= 0.5:
            raise ValueError("dual_route_fraction must be in [0, 0.5]")
        if surprise_weight < 0.0:
            raise ValueError("surprise_weight must be non-negative")
        self.dual_route_fraction = dual_route_fraction
        self.surprise_weight = surprise_weight
        self.scorer = nn.Linear(feature_dim + task_embedding_dim + 1, len(_EXPERT_IDS))

    def _scores(self, packet: BrainPacket, surprise: Tensor | None = None) -> Tensor:
        if packet.goal_context is not None and packet.goal_context.shape[-1] >= 72:
            task_index = packet.goal_context[..., _TASK].argmax(dim=-1)
        else:
            task_index = torch.zeros_like(packet.step_index)
        if surprise is None:
            surprise = torch.zeros(
                (*packet.valid_mask.shape, 1),
                dtype=packet.representation.dtype,
                device=packet.representation.device,
            )
        if surprise.shape != (*packet.valid_mask.shape, 1):
            raise ValueError("surprise must have shape [B, T, 1]")
        if surprise.device != packet.representation.device:
            raise ValueError("surprise and packet must share a device")
        task_features = self.task_embedding(task_index)
        return cast(
            Tensor,
            self.scorer(torch.cat((packet.representation, task_features, surprise), dim=-1)),
        )

    @staticmethod
    def _task_masks(packet: BrainPacket) -> tuple[Tensor, Tensor]:
        goal = require_goal_context(packet, minimum_features=72)
        return goal[..., 69].gt(0.5), goal[..., 70].gt(0.5)

    @classmethod
    def _reservation_mask(cls, packet: BrainPacket) -> Tensor:
        goal = require_goal_context(packet, minimum_features=72)
        is_ar, is_drs = cls._task_masks(packet)
        event = goal[..., _EVENT]
        reserve_episodic = is_ar & (event[..., 0].gt(0.5) | event[..., 2].gt(0.5))
        reserved = torch.zeros(
            (*packet.valid_mask.shape, len(_EXPERT_IDS)),
            dtype=torch.bool,
            device=packet.representation.device,
        )
        reserved[..., 0] = reserve_episodic & packet.valid_mask
        reserved[..., 1] = is_drs & packet.valid_mask
        return reserved

    def route(
        self,
        packet: BrainPacket,
        *,
        mode: Literal["learned", "dense", "no_reservation", "legacy_capacity"] = "learned",
        surprise: Tensor | None = None,
    ) -> RoutingDecisionV2:
        if mode != "learned":
            return super().route(packet, mode=mode)
        scores = self._scores(packet, surprise)
        valid = packet.valid_mask.unsqueeze(-1)
        reserved = self._reservation_mask(packet)
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        learned = torch.zeros_like(reserved)
        learned.scatter_(-1, order[..., :1], True)
        learned &= valid
        has_reservation = reserved.any(dim=-1, keepdim=True)
        raw = reserved | (learned & ~has_reservation)

        _, is_drs = self._task_masks(packet)
        if self.dual_route_fraction > 0.0:
            surprise_values = (
                torch.zeros_like(packet.valid_mask, dtype=scores.dtype)
                if surprise is None
                else surprise.squeeze(-1).to(scores.dtype)
            )
            ambiguity = -torch.abs(scores[..., 0] - scores[..., 1])
            priority = ambiguity + self.surprise_weight * surprise_values
            for step in range(scores.shape[1]):
                candidates = is_drs[:, step] & packet.valid_mask[:, step]
                count = int(candidates.sum().item())
                if count == 0:
                    continue
                quota = min(count, math.floor(count * self.dual_route_fraction))
                if quota == 0:
                    continue
                ranked = torch.argsort(priority[:, step], descending=True, stable=True)
                ranked = ranked[candidates.index_select(0, ranked)]
                chosen = ranked[:quota]
                learned[chosen, step, 0] = True
                raw[chosen, step, 0] = True

        valid_per_step = packet.valid_mask.to(torch.long).sum(dim=0)
        capacity = valid_per_step.unsqueeze(-1).expand(-1, len(_EXPERT_IDS)).clone()
        return RoutingDecisionV2(scores, reserved, learned, raw, raw.clone(), capacity, 0)

    def routing_losses(
        self,
        packet: BrainPacket,
        decision: RoutingDecisionV2,
    ) -> dict[str, Tensor]:
        valid = packet.valid_mask
        reserved = decision.reserved_mask.any(dim=-1) & valid
        zero = decision.scores.sum() * 0.0
        semantic_alignment = (
            F.cross_entropy(
                decision.scores[reserved],
                decision.reserved_mask[reserved].to(torch.long).argmax(dim=-1),
            )
            if torch.any(reserved).item()
            else zero
        )
        unreserved = valid & ~decision.reserved_mask.any(dim=-1)
        probabilities = torch.softmax(decision.scores, dim=-1)
        count = unreserved.to(probabilities.dtype).sum()
        mean_probability = (probabilities * unreserved.unsqueeze(-1).to(probabilities.dtype)).sum(
            dim=(0, 1)
        ) / count.clamp_min(1.0)
        load_balance = torch.where(
            count.gt(0),
            (mean_probability - 0.5).square().mean(),
            zero,
        )
        _, is_drs = self._task_masks(packet)
        drs_valid = is_drs & valid
        drs_count = drs_valid.to(probabilities.dtype).sum()
        episodic_probability = (
            probabilities[..., 0] * drs_valid.to(probabilities.dtype)
        ).sum() / drs_count.clamp_min(1.0)
        dual_budget = torch.where(
            drs_count.gt(0),
            torch.relu(episodic_probability - self.dual_route_fraction).square(),
            zero,
        )
        communication = decision.executed_mask.to(probabilities.dtype).sum() / (
            valid.to(probabilities.dtype).sum().clamp_min(1.0) * len(_EXPERT_IDS)
        )
        return {
            "router.semantic_alignment": semantic_alignment,
            "router.load_balance": load_balance,
            "router.dual_budget": dual_budget,
            "router.communication_cost": communication,
        }

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
        unknown = set(expert_packets).difference(_EXPERT_IDS)
        if unknown:
            raise ValueError(f"combine received unknown experts: {sorted(unknown)}")
        soft = torch.softmax(decision.scores, dim=-1)
        masked = decision.scores.masked_fill(~decision.executed_mask, -1.0e4)
        hard = torch.softmax(masked, dim=-1) * decision.executed_mask.to(masked.dtype)
        weights = hard.detach() + soft - soft.detach()
        weights = weights * encoder_packet.valid_mask.unsqueeze(-1)
        combined = encoder_packet.representation
        for index, module_id in enumerate(_EXPERT_IDS):
            expert = expert_packets.get(module_id)
            if expert is None:
                continue
            if expert.source_module != module_id:
                raise ValueError("expert packet source does not match mapping key")
            contribution = expert.representation - encoder_packet.representation
            combined = combined + contribution * weights[..., index].unsqueeze(-1)
        fused = encoder_packet.representation + self.fusion(combined)
        fused = torch.where(
            encoder_packet.valid_mask.unsqueeze(-1), fused, encoder_packet.representation
        )
        return ModuleOutput(packet_from(encoder_packet, fused, self.module_id), state)


__all__ = ["SparseRouterV3"]
