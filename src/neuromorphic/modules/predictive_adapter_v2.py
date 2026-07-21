"""Action-conditioned latent forecasting with bounded feedback carry-over."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neuromorphic.core.contracts import BrainPacket, ModuleContext, ModuleOutput, ModuleState
from neuromorphic.core.registry import PREDICTIVE_ADAPTER_V2
from neuromorphic.modules._utils import (
    packet_from,
    require_goal_context,
    reset_tensor_rows,
    validate_inputs,
)

_EVENT = slice(0, 5)
_TASK = slice(69, 72)
_ACTION_COPY = slice(72, 104)


@dataclass(frozen=True, slots=True)
class PredictiveConsumeResult:
    """Causal result of comparing the current latent with the prior forecast."""

    output: ModuleOutput
    transition_mask: Tensor
    forecast_error: Tensor
    persistence_error: Tensor
    feedback_delta: Tensor


class PredictiveAdapterV2(nn.Module):
    """Consume last-step forecasts safely and stage the next latent forecast."""

    module_id = PREDICTIVE_ADAPTER_V2
    state_version = "predictive-state-v2"

    def __init__(
        self, feature_dim: int = 128, action_count: int = 32, action_dim: int = 32
    ) -> None:
        super().__init__()
        if min(feature_dim, action_count, action_dim) <= 0:
            raise ValueError("predictive dimensions must be positive")
        self.feature_dim = feature_dim
        self.action_count = action_count
        self.action_embedding = nn.Embedding(action_count, action_dim)
        self.target_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.feedback = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.transition = nn.Sequential(
            nn.Linear(feature_dim + action_dim + 8, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> ModuleState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        tensors = {
            "forecast": torch.zeros(batch_size, self.feature_dim, device=device, dtype=dtype),
            "forecast_valid": torch.zeros(batch_size, device=device, dtype=torch.bool),
            "source_step": torch.full((batch_size,), -1, device=device, dtype=torch.long),
            "persistence": torch.zeros(batch_size, self.feature_dim, device=device, dtype=dtype),
        }
        return ModuleState(self.module_id, self.state_version, tensors)

    @staticmethod
    def _zero(reference: Tensor) -> Tensor:
        return reference.sum() * 0.0

    @staticmethod
    def _selected_action(goal_context: Tensor) -> Tensor:
        copied = goal_context[..., _ACTION_COPY]
        selected = copied.argmax(dim=-1)
        return torch.where(copied.sum(dim=-1).gt(0), selected, torch.full_like(selected, -1))

    def _reset(self, state: ModuleState, reset_mask: Tensor) -> ModuleState:
        tensors = {
            "forecast": reset_tensor_rows(state.tensors["forecast"], reset_mask),
            "forecast_valid": reset_tensor_rows(state.tensors["forecast_valid"], reset_mask, False),
            "source_step": reset_tensor_rows(state.tensors["source_step"], reset_mask, -1),
            "persistence": reset_tensor_rows(state.tensors["persistence"], reset_mask),
        }
        return ModuleState(self.module_id, self.state_version, tensors)

    def consume(
        self,
        packet: BrainPacket,
        state: ModuleState,
        context: ModuleContext,
        *,
        feedback_enabled: bool = True,
        shuffle_forecast: bool = False,
    ) -> PredictiveConsumeResult:
        """Consume only a forecast produced by the immediately preceding valid step.

        This method never stages the next forecast.  That is deliberately deferred
        to :meth:`commit`, after the action selector has made its current decision.
        """

        validate_inputs(
            packet, state, context, module_id=self.module_id, version=self.state_version
        )
        if packet.representation.shape[1] != 1:
            raise ValueError(
                "predictive_adapter.v2 consumes exactly one step; use "
                "ModularBrainNetworkV2.forward_batch for causal sequences"
            )
        if packet.representation.shape[-1] != self.feature_dim:
            raise ValueError(f"representation feature size must be {self.feature_dim}")
        current = state
        outputs: list[Tensor] = []
        masks: list[Tensor] = []
        forecast_errors: list[Tensor] = []
        persistence_errors: list[Tensor] = []
        deltas: list[Tensor] = []
        losses: list[Tensor] = []
        for step in range(packet.representation.shape[1]):
            current = self._reset(current, context.reset_mask[:, step])
            representation = packet.representation[:, step]
            target = self.target_norm(representation)
            valid = packet.valid_mask[:, step]
            step_index = packet.step_index[:, step].to(torch.long)
            forecast = current.tensors["forecast"]
            persistence = current.tensors["persistence"]
            forecast_valid = current.tensors["forecast_valid"]
            source_step = current.tensors["source_step"]
            if shuffle_forecast and forecast.shape[0] > 1:
                forecast = forecast.roll(1, dims=0)
                forecast_valid = forecast_valid.roll(1, dims=0)
                source_step = source_step.roll(1, dims=0)
            consecutive = forecast_valid & valid & step_index.eq(source_step + 1)
            error = target - forecast
            proposed_delta = 0.25 * torch.tanh(self.feedback(error))
            delta = proposed_delta * (consecutive & feedback_enabled).unsqueeze(-1)
            enriched = torch.where(valid.unsqueeze(-1), representation + delta, representation)

            smooth = F.smooth_l1_loss(forecast, target.detach(), reduction="none").mean(dim=-1)
            cosine = 1.0 - F.cosine_similarity(forecast, target.detach(), dim=-1)
            transition_loss = smooth + 0.1 * cosine
            persistence_error = F.smooth_l1_loss(
                persistence, target.detach(), reduction="none"
            ).mean(dim=-1)
            zero = torch.zeros_like(smooth)
            outputs.append(enriched)
            masks.append(consecutive)
            forecast_errors.append(torch.where(consecutive, smooth, zero))
            persistence_errors.append(torch.where(consecutive, persistence_error, zero))
            deltas.append(delta)
            losses.append(torch.where(consecutive, transition_loss, zero))

        transition_mask = torch.stack(masks, dim=1)
        loss_values = torch.stack(losses, dim=1)
        transition_loss = (
            loss_values[transition_mask].mean()
            if torch.any(transition_mask).item()
            else self._zero(packet.representation)
        )
        output = ModuleOutput(
            packet_from(packet, torch.stack(outputs, dim=1), self.module_id),
            current,
            prediction_logits=current.tensors["forecast"]
            .unsqueeze(1)
            .expand(-1, packet.representation.shape[1], -1),
            auxiliary_losses={"predictive_transition": transition_loss},
        )
        return PredictiveConsumeResult(
            output=output,
            transition_mask=transition_mask,
            forecast_error=torch.stack(forecast_errors, dim=1),
            persistence_error=torch.stack(persistence_errors, dim=1),
            feedback_delta=torch.stack(deltas, dim=1),
        )

    def commit(
        self,
        packet: BrainPacket,
        state: ModuleState,
        context: ModuleContext,
        selected_action: Tensor,
    ) -> ModuleState:
        """Stage one action-conditioned forecast for the next valid time step."""

        if selected_action.shape != packet.valid_mask.shape:
            raise ValueError("selected_action must have shape [B, T]")
        if selected_action.dtype != torch.long:
            raise TypeError("selected_action must use torch.long")
        validate_inputs(
            packet, state, context, module_id=self.module_id, version=self.state_version
        )
        if packet.representation.shape[1] != 1:
            raise ValueError(
                "predictive_adapter.v2 commits exactly one step; use "
                "ModularBrainNetworkV2.forward_batch for causal sequences"
            )
        goal = require_goal_context(packet, minimum_features=72)
        current = state
        for step in range(packet.representation.shape[1]):
            current = self._reset(current, context.reset_mask[:, step])
            representation = packet.representation[:, step]
            valid = packet.valid_mask[:, step]
            step_index = packet.step_index[:, step].to(torch.long)
            action = selected_action[:, step]
            safe_action = action.clamp(0, self.action_count - 1)
            action_valid = action.ge(0) & action.lt(self.action_count) & valid
            action_features = self.action_embedding(safe_action) * action_valid.unsqueeze(-1)
            goal_features = torch.cat((goal[:, step, _EVENT], goal[:, step, _TASK]), dim=-1)
            target = self.target_norm(representation)
            next_forecast = self.transition(
                torch.cat((representation, action_features, goal_features), dim=-1)
            )
            current = ModuleState(
                self.module_id,
                self.state_version,
                {
                    "forecast": torch.where(
                        valid.unsqueeze(-1), next_forecast, current.tensors["forecast"]
                    ),
                    "forecast_valid": torch.where(
                        valid, torch.ones_like(valid), current.tensors["forecast_valid"]
                    ),
                    "source_step": torch.where(valid, step_index, current.tensors["source_step"]),
                    "persistence": torch.where(
                        valid.unsqueeze(-1), target.detach(), current.tensors["persistence"]
                    ),
                },
            )
        return current

    def forward_with_action(
        self,
        packet: BrainPacket,
        state: ModuleState,
        context: ModuleContext,
        selected_action: Tensor,
    ) -> ModuleOutput:
        consumed = self.consume(packet, state, context)
        committed = self.commit(packet, consumed.output.state, context, selected_action)
        return ModuleOutput(
            consumed.output.packet,
            committed,
            prediction_logits=consumed.output.prediction_logits,
            auxiliary_losses=consumed.output.auxiliary_losses,
        )

    def forward(
        self, packet: BrainPacket, state: ModuleState, context: ModuleContext
    ) -> ModuleOutput:
        goal = require_goal_context(packet)
        return self.forward_with_action(packet, state, context, self._selected_action(goal))

    def reset_state(self, state: ModuleState, reset_mask: Tensor) -> ModuleState:
        if state.module_id != self.module_id or state.version != self.state_version:
            raise ValueError("invalid predictive adapter state")
        if reset_mask.ndim != 1 or reset_mask.dtype is not torch.bool:
            raise ValueError("reset_mask must be a boolean [B] tensor")
        return self._reset(state, reset_mask)


__all__ = ["PredictiveAdapterV2", "PredictiveConsumeResult"]
