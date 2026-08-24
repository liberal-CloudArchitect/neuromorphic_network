"""Residual latent forecasting whose surprise modulates routing, not sensation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neuromorphic.core.contracts import BrainPacket, ModuleContext, ModuleOutput, ModuleState
from neuromorphic.core.registry import PREDICTIVE_ADAPTER_V3
from neuromorphic.modules._utils import packet_from, require_goal_context, validate_inputs
from neuromorphic.modules.predictive_adapter_v2 import PredictiveAdapterV2, PredictiveConsumeResult

_EVENT = slice(0, 5)
_TASK = slice(69, 72)


@dataclass(frozen=True, slots=True)
class PredictiveConsumeResultV3(PredictiveConsumeResult):
    """Causal forecast evidence without direct representation mutation."""

    surprise: Tensor


class PredictiveAdapterV3(PredictiveAdapterV2):
    """Predict the next latent as a bounded delta from persistence.

    Version 2 learned an absolute forecast and transformed the resulting error
    into an additive correction of an already-observed sensory representation.
    Version 3 instead starts at the persistence baseline and exposes detached
    forecast surprise to the router.  The sensory packet remains unchanged.
    """

    module_id = PREDICTIVE_ADAPTER_V3
    state_version = "predictive-state-v3"

    def __init__(
        self, feature_dim: int = 128, action_count: int = 32, action_dim: int = 32
    ) -> None:
        nn.Module.__init__(self)
        if min(feature_dim, action_count, action_dim) <= 0:
            raise ValueError("predictive dimensions must be positive")
        self.feature_dim = feature_dim
        self.action_count = action_count
        self.action_embedding = nn.Embedding(action_count, action_dim)
        self.target_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.transition_delta = nn.Sequential(
            nn.Linear(feature_dim + action_dim + 8, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        final = self.transition_delta[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("transition_delta must end in a Linear layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def consume(
        self,
        packet: BrainPacket,
        state: ModuleState,
        context: ModuleContext,
        *,
        feedback_enabled: bool = True,
        shuffle_forecast: bool = False,
    ) -> PredictiveConsumeResultV3:
        validate_inputs(
            packet, state, context, module_id=self.module_id, version=self.state_version
        )
        if packet.representation.shape[1] != 1:
            raise ValueError("predictive_adapter.v3 consumes exactly one causal step")
        if packet.representation.shape[-1] != self.feature_dim:
            raise ValueError(f"representation feature size must be {self.feature_dim}")

        current = self._reset(state, context.reset_mask[:, 0])
        representation = packet.representation[:, 0]
        target = self.target_norm(representation)
        valid = packet.valid_mask[:, 0]
        step_index = packet.step_index[:, 0].to(torch.long)
        forecast = current.tensors["forecast"]
        persistence = current.tensors["persistence"]
        forecast_valid = current.tensors["forecast_valid"]
        source_step = current.tensors["source_step"]
        if shuffle_forecast and forecast.shape[0] > 1:
            forecast = forecast.roll(1, dims=0)
            forecast_valid = forecast_valid.roll(1, dims=0)
            source_step = source_step.roll(1, dims=0)

        consecutive = forecast_valid & valid & step_index.eq(source_step + 1)
        smooth = F.smooth_l1_loss(forecast, target.detach(), reduction="none").mean(dim=-1)
        cosine = 1.0 - F.cosine_similarity(forecast, target.detach(), dim=-1)
        persistence_error = F.smooth_l1_loss(persistence, target.detach(), reduction="none").mean(
            dim=-1
        )
        transition_loss_values = smooth + 0.1 * cosine
        transition_loss = (
            transition_loss_values[consecutive].mean()
            if torch.any(consecutive).item()
            else self._zero(packet.representation)
        )
        zero = torch.zeros_like(smooth)
        surprise = torch.where(
            consecutive & feedback_enabled,
            torch.log1p(smooth.detach()),
            zero,
        ).view(-1, 1, 1)
        feedback_delta = torch.zeros_like(packet.representation)
        output = ModuleOutput(
            packet_from(packet, packet.representation, self.module_id),
            current,
            prediction_logits=current.tensors["forecast"].unsqueeze(1),
            auxiliary_losses={"predictive_transition": transition_loss},
        )
        return PredictiveConsumeResultV3(
            output=output,
            transition_mask=consecutive.view(-1, 1),
            forecast_error=torch.where(consecutive, smooth, zero).view(-1, 1),
            persistence_error=torch.where(consecutive, persistence_error, zero).view(-1, 1),
            feedback_delta=feedback_delta,
            surprise=surprise,
        )

    def commit(
        self,
        packet: BrainPacket,
        state: ModuleState,
        context: ModuleContext,
        selected_action: Tensor,
    ) -> ModuleState:
        if selected_action.shape != packet.valid_mask.shape:
            raise ValueError("selected_action must have shape [B, T]")
        if selected_action.dtype != torch.long:
            raise TypeError("selected_action must use torch.long")
        validate_inputs(
            packet, state, context, module_id=self.module_id, version=self.state_version
        )
        if packet.representation.shape[1] != 1:
            raise ValueError("predictive_adapter.v3 commits exactly one causal step")

        goal = require_goal_context(packet, minimum_features=72)
        current = self._reset(state, context.reset_mask[:, 0])
        representation = packet.representation[:, 0]
        target = self.target_norm(representation)
        valid = packet.valid_mask[:, 0]
        step_index = packet.step_index[:, 0].to(torch.long)
        action = selected_action[:, 0]
        safe_action = action.clamp(0, self.action_count - 1)
        action_valid = action.ge(0) & action.lt(self.action_count) & valid
        action_features = self.action_embedding(safe_action) * action_valid.unsqueeze(-1)
        goal_features = torch.cat((goal[:, 0, _EVENT], goal[:, 0, _TASK]), dim=-1)
        predicted_delta = 0.25 * torch.tanh(
            self.transition_delta(torch.cat((target, action_features, goal_features), dim=-1))
        )
        next_forecast = target.detach() + predicted_delta
        return ModuleState(
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


__all__ = ["PredictiveAdapterV3", "PredictiveConsumeResultV3"]
