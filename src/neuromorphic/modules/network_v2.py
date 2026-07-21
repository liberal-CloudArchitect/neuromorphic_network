"""P4 composition graph with causal forecasting and semantic top-1 memory routing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn

from neuromorphic.core.contracts import (
    BrainPacket,
    ModuleContext,
    ModuleState,
    Phase,
    TelemetryRecord,
    trusted_internal_execution,
)
from neuromorphic.core.module_registry import ModuleRegistry
from neuromorphic.core.network_state import NetworkState
from neuromorphic.core.registry import (
    ACTION_SELECTOR,
    EPISODIC_MEMORY,
    P4_MODULE_IDS,
    P4_OPTIONAL_EXPERT_IDS,
    PREDICTIVE_ADAPTER_V2,
    SENSORY_ENCODER,
    SPARSE_ROUTER_V2,
    WORKING_MEMORY,
)
from neuromorphic.modules.action_selector import ActionSelector
from neuromorphic.modules.episodic_memory import EpisodicMemory
from neuromorphic.modules.network import TaskBoundaryAdapters
from neuromorphic.modules.predictive_adapter_v2 import PredictiveAdapterV2
from neuromorphic.modules.sensory_encoder import SensoryEncoder
from neuromorphic.modules.sparse_router_v2 import RoutingDecisionV2, SparseRouterV2
from neuromorphic.modules.working_memory import WorkingMemory
from neuromorphic.tasks.base import TaskBatch
from neuromorphic.tasks.control import (
    GOAL_CONTEXT_DIM,
    TASK_CLASS_COUNTS,
    TaskControl,
    task_control_from_batch,
)

type RoutingModeV2 = Literal["learned", "dense", "no_reservation", "legacy_capacity"]
type PredictorModeV2 = Literal[
    "full",
    "off",
    "loss_zero",
    "feedback_zero",
    "acute_feedback_off",
    "shuffle_forecast",
]

_LOSS_NAMES = {
    "episodic_retrieval": "episodic.retrieval",
    "episodic_separation": "episodic.separation",
    "working_consistency": "working.state_consistency",
    "working_gate_regularization": "working.gate_regularization",
    "predictive_transition": "predictive.temporal",
}


def _merge_losses(target: dict[str, Tensor], source: Mapping[str, Tensor]) -> None:
    for name, value in source.items():
        target[_LOSS_NAMES.get(name, name)] = value


@dataclass(frozen=True, slots=True)
class ModularBrainOutputV2:
    """Outputs and causal diagnostics from the P4 modular graph."""

    packet: BrainPacket
    action_logits: Tensor
    state: NetworkState
    auxiliary_losses: Mapping[str, Tensor]
    routing_trace: tuple[RoutingDecisionV2, ...]
    module_metrics: Mapping[str, Tensor]
    cost_statistics: Mapping[str, Tensor]
    forecast_logits: Tensor
    forecast_transition_mask: Tensor
    forecast_error: Tensor
    persistence_error: Tensor
    feedback_delta: Tensor
    telemetry_events: tuple[TelemetryRecord, ...] = ()

    @property
    def logits(self) -> Tensor:
        """Compatibility alias for shared training helpers."""

        return self.action_logits

    def __post_init__(self) -> None:
        batch, steps, features = self.packet.representation.shape
        if self.action_logits.shape[:2] != (batch, steps):
            raise ValueError("action_logits must align with packet [B, T]")
        if self.forecast_logits.shape != (batch, steps, features):
            raise ValueError("forecast_logits must align with packet [B, T, F]")
        for name, value in (
            ("forecast_transition_mask", self.forecast_transition_mask),
            ("forecast_error", self.forecast_error),
            ("persistence_error", self.persistence_error),
        ):
            if value.shape != (batch, steps):
                raise ValueError(f"{name} must have shape [B, T]")
        if self.forecast_transition_mask.dtype is not torch.bool:
            raise TypeError("forecast_transition_mask must use torch.bool")
        if self.feedback_delta.shape != (batch, steps, features):
            raise ValueError("feedback_delta must have shape [B, T, F]")
        if self.state.batch_size != batch:
            raise ValueError("state batch size must align with packet")


def _default_registry_v2(
    feature_dim: int,
    *,
    episodic_slots: int,
    working_slots: int,
    working_slot_dim: int,
    action_embedding_dim: int,
    task_embedding_dim: int,
) -> ModuleRegistry:
    return ModuleRegistry(
        (
            SensoryEncoder(feature_dim=feature_dim),
            EpisodicMemory(feature_dim=feature_dim, slots=episodic_slots),
            WorkingMemory(
                feature_dim=feature_dim,
                slots=working_slots,
                slot_dim=working_slot_dim,
            ),
            PredictiveAdapterV2(
                feature_dim=feature_dim,
                action_dim=action_embedding_dim,
            ),
            ActionSelector(feature_dim=feature_dim),
            SparseRouterV2(
                feature_dim=feature_dim,
                task_embedding_dim=task_embedding_dim,
            ),
        )
    )


def _slice_packet(packet: BrainPacket, indices: Tensor) -> BrainPacket:
    goal = None if packet.goal_context is None else packet.goal_context.index_select(0, indices)
    return BrainPacket(
        representation=packet.representation.index_select(0, indices),
        valid_mask=packet.valid_mask.index_select(0, indices),
        modality=packet.modality,
        step_index=packet.step_index.index_select(0, indices),
        source_module=packet.source_module,
        goal_context=goal,
        metadata=packet.metadata,
    )


def _slice_state(state: ModuleState, indices: Tensor) -> ModuleState:
    return ModuleState(
        state.module_id,
        state.version,
        {
            name: tensor if tensor.ndim == 0 else tensor.index_select(0, indices)
            for name, tensor in state.tensors.items()
        },
    )


def _scatter_state(base: ModuleState, update: ModuleState, indices: Tensor) -> ModuleState:
    return ModuleState(
        base.module_id,
        base.version,
        {
            name: tensor
            if tensor.ndim == 0
            else tensor.index_copy(0, indices, update.tensors[name])
            for name, tensor in base.tensors.items()
        },
    )


def _scatter_packet(base: BrainPacket, update: BrainPacket, indices: Tensor) -> BrainPacket:
    return BrainPacket(
        representation=torch.zeros_like(base.representation).index_copy(
            0, indices, update.representation
        ),
        valid_mask=base.valid_mask,
        modality=base.modality,
        step_index=base.step_index,
        source_module=update.source_module,
        goal_context=base.goal_context,
        metadata=base.metadata,
    )


def _slice_context(context: ModuleContext, indices: Tensor) -> ModuleContext:
    return ModuleContext(
        task_id=context.task_id,
        phase=context.phase,
        reset_mask=context.reset_mask.index_select(0, indices),
        eligible_modules=context.eligible_modules,
        telemetry_enabled=context.telemetry_enabled,
    )


class ModularBrainNetworkV2(nn.Module):
    """Execute the P4 causal graph without changing the P1--P3 network."""

    def __init__(
        self,
        *,
        feature_dim: int = 128,
        episodic_slots: int = 16,
        working_slots: int = 4,
        working_slot_dim: int = 32,
        action_embedding_dim: int = 32,
        task_embedding_dim: int = 16,
        registry: ModuleRegistry | None = None,
        tbptt_interval: int = 32,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or tbptt_interval <= 0:
            raise ValueError("feature_dim and tbptt_interval must be positive")
        self.feature_dim = feature_dim
        self.tbptt_interval = tbptt_interval
        self.boundary_adapters = TaskBoundaryAdapters(feature_dim)
        self.registry = registry or _default_registry_v2(
            feature_dim,
            episodic_slots=episodic_slots,
            working_slots=working_slots,
            working_slot_dim=working_slot_dim,
            action_embedding_dim=action_embedding_dim,
            task_embedding_dim=task_embedding_dim,
        )
        self.registry.require_complete(P4_MODULE_IDS)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> NetworkState:
        return NetworkState.initial(self.registry, batch_size, device=device, dtype=dtype)

    def _reset_state(self, state: NetworkState, reset_mask: Tensor) -> NetworkState:
        updated = state.reset_counts(reset_mask)
        for module_id in self.registry.ids:
            reset = self.registry.get(module_id).reset_state(updated.get(module_id), reset_mask)
            updated = updated.replace(reset)
        return updated

    def forward_step(
        self,
        inputs: Tensor,
        control: TaskControl,
        state: NetworkState,
        context: ModuleContext,
        *,
        routing_mode: RoutingModeV2 = "learned",
        predictor_mode: PredictorModeV2 = "full",
        episodic_off: bool = False,
        working_reset_every_step: bool = False,
        encoder_mode: Literal["full", "shallow"] = "full",
        selector_mode: Literal["integrated", "direct"] = "integrated",
        terminal_mask: Tensor | None = None,
    ) -> ModularBrainOutputV2:
        """Execute a single causal step; all inputs retain a singleton T axis."""

        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("forward_step inputs must have shape [B, 1, D]")
        if control.goal_context.shape[:2] != inputs.shape[:2]:
            raise ValueError("control must align with step inputs")
        if context.reset_mask.shape != inputs.shape[:2]:
            raise ValueError("context.reset_mask must align with step inputs")
        if state.batch_size != inputs.shape[0]:
            raise ValueError("state batch size must align with inputs")
        if control.task_id != context.task_id:
            raise ValueError("TaskControl and ModuleContext task IDs must match")
        if terminal_mask is None:
            terminal_mask = torch.zeros(inputs.shape[0], dtype=torch.bool, device=inputs.device)
        if terminal_mask.shape != (inputs.shape[0],) or terminal_mask.dtype is not torch.bool:
            raise ValueError("terminal_mask must be a boolean [B] tensor")

        valid_rows = control.valid_mask[:, 0]
        reset_rows = context.reset_mask[:, 0] & valid_rows
        state = self._reset_state(state, reset_rows)
        eligible_transitions = valid_rows & state.valid_step_counts.gt(0)
        if working_reset_every_step:
            working = cast(WorkingMemory, self.registry.get(WORKING_MEMORY))
            state = state.replace(working.reset_state(state.get(WORKING_MEMORY), valid_rows))

        step_index = state.valid_step_counts.unsqueeze(1)
        adapted = self.boundary_adapters(inputs, control.task_id)
        sensory = cast(SensoryEncoder, self.registry.get(SENSORY_ENCODER))
        sensory_output = sensory.forward_inputs(
            adapted,
            control.valid_mask,
            step_index,
            control.goal_context,
            context,
            state=state.get(SENSORY_ENCODER),
            mode=encoder_mode,
        )
        state = state.replace(sensory_output.state)

        zero = adapted.sum() * 0.0
        losses: dict[str, Tensor] = {
            "episodic.retrieval": zero,
            "episodic.separation": zero,
            "working.state_consistency": zero,
            "working.gate_regularization": zero,
            "predictive.temporal": zero,
        }
        telemetry: list[TelemetryRecord] = list(sensory_output.telemetry_events)

        predictor = cast(PredictiveAdapterV2, self.registry.get(PREDICTIVE_ADAPTER_V2))
        if predictor_mode == "off":
            state = state.replace(
                predictor.reset_state(state.get(PREDICTIVE_ADAPTER_V2), valid_rows)
            )
            predictive_packet = sensory_output.packet
            transition_mask = torch.zeros_like(control.valid_mask)
            forecast_error = torch.zeros_like(control.valid_mask, dtype=inputs.dtype)
            persistence_error = torch.zeros_like(forecast_error)
            feedback_delta = torch.zeros_like(sensory_output.packet.representation)
            forecast_logits = torch.zeros_like(sensory_output.packet.representation)
        else:
            consume = predictor.consume(
                sensory_output.packet,
                state.get(PREDICTIVE_ADAPTER_V2),
                context,
                feedback_enabled=predictor_mode not in {"feedback_zero", "acute_feedback_off"},
                shuffle_forecast=predictor_mode == "shuffle_forecast",
            )
            predictive_packet = consume.output.packet
            state = state.replace(consume.output.state)
            transition_mask = consume.transition_mask
            forecast_error = consume.forecast_error
            persistence_error = consume.persistence_error
            feedback_delta = consume.feedback_delta
            forecast_logits = cast(Tensor, consume.output.prediction_logits)
            if predictor_mode != "loss_zero":
                _merge_losses(losses, consume.output.auxiliary_losses)
            telemetry.extend(consume.output.telemetry_events)

        router = cast(SparseRouterV2, self.registry.get(SPARSE_ROUTER_V2))
        decision = router.route(predictive_packet, mode=routing_mode)
        routing_probabilities = torch.softmax(decision.scores, dim=-1)
        unreserved = control.valid_mask.unsqueeze(-1) & ~decision.reserved_mask.any(
            dim=-1, keepdim=True
        )
        balance_weight = unreserved.to(routing_probabilities.dtype)
        balance_count = balance_weight.sum()
        mean_probability = (routing_probabilities * balance_weight).sum(dim=(0, 1)) / (
            balance_count.clamp_min(1.0)
        )
        imbalance = (mean_probability - 1.0 / len(P4_OPTIONAL_EXPERT_IDS)).square().mean()
        losses["router.load_balance"] = torch.where(balance_count.gt(0), imbalance, zero)
        routing_valid = control.valid_mask.unsqueeze(-1).to(routing_probabilities.dtype)
        losses["router.communication_cost"] = (
            decision.executed_mask.to(routing_probabilities.dtype)
            * routing_probabilities
            * routing_valid
        ).sum() / routing_valid.sum().clamp_min(1.0)
        expert_packets: dict[str, BrainPacket] = {}
        selected_counts: dict[str, Tensor] = {}
        for expert_index, module_id in enumerate(P4_OPTIONAL_EXPERT_IDS):
            selected = decision.executed_mask[:, 0, expert_index] & valid_rows
            selected_counts[f"selected.{module_id}"] = selected.sum()
            if module_id == EPISODIC_MEMORY and episodic_off:
                continue
            indices = torch.nonzero(selected, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            module = self.registry.get(module_id)
            base_state = state.get(module_id)
            output = module.forward(
                _slice_packet(predictive_packet, indices),
                _slice_state(base_state, indices),
                _slice_context(context, indices),
            )
            state = state.replace(_scatter_state(base_state, output.state, indices))
            expert_packets[module_id] = _scatter_packet(predictive_packet, output.packet, indices)
            _merge_losses(losses, output.auxiliary_losses)
            telemetry.extend(output.telemetry_events)

        router_output = router.combine(
            predictive_packet,
            expert_packets,
            decision,
            state.get(SPARSE_ROUTER_V2),
            context,
        )
        state = state.replace(router_output.state)
        selector = cast(ActionSelector, self.registry.get(ACTION_SELECTOR))
        selector_output = selector.forward_with_mode(
            router_output.packet,
            state.get(ACTION_SELECTOR),
            context,
            mode=selector_mode,
        )
        if selector_output.action_logits is None:
            raise RuntimeError("action selector did not return action logits")
        state = state.replace(selector_output.state)
        selected_action = selector_output.action_logits.argmax(dim=-1)

        if predictor_mode != "off":
            state = state.replace(
                predictor.commit(
                    sensory_output.packet,
                    state.get(PREDICTIVE_ADAPTER_V2),
                    context,
                    selected_action,
                )
            )
        episodic = cast(EpisodicMemory, self.registry.get(EPISODIC_MEMORY))
        state = state.replace(episodic.commit_pending(state.get(EPISODIC_MEMORY)))
        if torch.any(terminal_mask).item():
            state = state.replace(
                predictor.reset_state(state.get(PREDICTIVE_ADAPTER_V2), terminal_mask)
            )
        state, detach_mask = state.advance(valid_rows, interval=self.tbptt_interval)

        reserved = decision.reserved_mask[:, 0]
        reserved_count = reserved[..., 0].sum()
        reserved_executed = (reserved[..., 0] & decision.executed_mask[:, 0, 0]).sum()
        non_reserved = valid_rows & ~reserved.any(dim=-1)
        active_calls = decision.executed_mask[:, 0].to(inputs.dtype).sum()
        if episodic_off:
            active_calls = active_calls - decision.executed_mask[:, 0, 0].to(inputs.dtype).sum()
        dense_calls = valid_rows.to(inputs.dtype).sum() * len(P4_OPTIONAL_EXPERT_IDS)
        metrics = {
            **selected_counts,
            "tbptt.detached_rows": detach_mask.sum(),
            "routing.capacity_drops": torch.as_tensor(
                decision.capacity_drops, device=inputs.device
            ),
            "routing.reserved_tokens": reserved_count,
            "routing.reserved_executed": reserved_executed,
            "routing.learned_tokens": non_reserved.sum(),
            "forecast.transitions": transition_mask.sum(),
            "forecast.feedback_nonzero": feedback_delta.abs().sum(dim=-1).gt(0).sum(),
            "predictive.transition_count": transition_mask.sum(),
            "predictive.eligible_transition_count": eligible_transitions.sum(),
            "predictive.forecast_error_sum": forecast_error.sum(),
            "predictive.persistence_error_sum": persistence_error.sum(),
            "predictive.feedback_latent_delta_sum": feedback_delta.abs().sum(),
        }
        costs = {
            "optional.active_calls": active_calls,
            "optional.dense_calls": dense_calls,
        }
        return ModularBrainOutputV2(
            packet=selector_output.packet,
            action_logits=selector_output.action_logits,
            state=state,
            auxiliary_losses=losses,
            routing_trace=(decision,),
            module_metrics=metrics,
            cost_statistics=costs,
            forecast_logits=forecast_logits,
            forecast_transition_mask=transition_mask,
            forecast_error=forecast_error,
            persistence_error=persistence_error,
            feedback_delta=feedback_delta,
            telemetry_events=tuple(telemetry),
        )

    def forward_batch(
        self,
        batch: TaskBatch,
        state: NetworkState | None = None,
        *,
        phase: Phase = "train",
        telemetry_enabled: bool = False,
        routing_mode: RoutingModeV2 = "learned",
        predictor_mode: PredictorModeV2 = "full",
        episodic_off: bool = False,
        working_reset_every_step: bool = False,
        encoder_mode: Literal["full", "shallow"] = "full",
        selector_mode: Literal["integrated", "direct"] = "integrated",
    ) -> ModularBrainOutputV2:
        """Execute a padded batch sequentially with episode-local causal state."""

        batch.validate()
        with trusted_internal_execution():
            control = task_control_from_batch(batch)
            if state is None:
                state = self.initial_state(
                    batch.batch_size,
                    device=batch.inputs.device,
                    dtype=batch.inputs.dtype,
                )
            elif state.batch_size != batch.batch_size:
                raise ValueError("state and TaskBatch batch sizes must match")

            packets: list[BrainPacket] = []
            logits: list[Tensor] = []
            forecasts: list[Tensor] = []
            transition_masks: list[Tensor] = []
            forecast_errors: list[Tensor] = []
            persistence_errors: list[Tensor] = []
            feedback_deltas: list[Tensor] = []
            traces: list[RoutingDecisionV2] = []
            losses: dict[str, list[Tensor]] = {}
            metrics: dict[str, list[Tensor]] = {}
            costs: dict[str, list[Tensor]] = {}
            telemetry: list[TelemetryRecord] = []
            previous_episode = torch.full(
                (batch.batch_size,), -1, dtype=torch.long, device=batch.inputs.device
            )
            for step in range(batch.sequence_length):
                valid = batch.valid_mask[:, step]
                episode = batch.episode_ids[:, step]
                reset = valid & (previous_episode.ne(episode) | previous_episode.lt(0))
                next_valid = (
                    batch.valid_mask[:, step + 1]
                    if step + 1 < batch.sequence_length
                    else torch.zeros_like(valid)
                )
                next_episode = (
                    batch.episode_ids[:, step + 1] if step + 1 < batch.sequence_length else episode
                )
                terminal = valid & (~next_valid | next_episode.ne(episode))
                context = ModuleContext(
                    task_id=control.task_id,
                    phase=phase,
                    reset_mask=reset.unsqueeze(1),
                    eligible_modules=P4_OPTIONAL_EXPERT_IDS,
                    telemetry_enabled=telemetry_enabled,
                )
                output = self.forward_step(
                    batch.inputs[:, step : step + 1],
                    control.at_step(step),
                    state,
                    context,
                    routing_mode=routing_mode,
                    predictor_mode=predictor_mode,
                    episodic_off=episodic_off,
                    working_reset_every_step=working_reset_every_step,
                    encoder_mode=encoder_mode,
                    selector_mode=selector_mode,
                    terminal_mask=terminal,
                )
                state = output.state
                packets.append(output.packet)
                logits.append(output.action_logits)
                forecasts.append(output.forecast_logits)
                transition_masks.append(output.forecast_transition_mask)
                forecast_errors.append(output.forecast_error)
                persistence_errors.append(output.persistence_error)
                feedback_deltas.append(output.feedback_delta)
                traces.extend(output.routing_trace)
                telemetry.extend(output.telemetry_events)
                for name, value in output.auxiliary_losses.items():
                    losses.setdefault(name, []).append(value)
                for name, value in output.module_metrics.items():
                    metrics.setdefault(name, []).append(value)
                for name, value in output.cost_statistics.items():
                    costs.setdefault(name, []).append(value)
                previous_episode = torch.where(valid, episode, previous_episode)

            first = packets[0]
            packet = BrainPacket(
                representation=torch.cat([item.representation for item in packets], dim=1),
                valid_mask=batch.valid_mask,
                modality=first.modality,
                step_index=torch.cat([item.step_index for item in packets], dim=1),
                source_module=ACTION_SELECTOR,
                goal_context=control.goal_context,
                metadata=first.metadata,
            )
            combined_losses = {name: torch.stack(values).mean() for name, values in losses.items()}
            temporal_values = losses.get("predictive.temporal")
            if temporal_values is not None:
                counts = [mask.sum().to(batch.inputs.dtype) for mask in transition_masks]
                denominator = torch.stack(counts).sum()
                if denominator.item() > 0:
                    combined_losses["predictive.temporal"] = (
                        sum(
                            (
                                value * count
                                for value, count in zip(temporal_values, counts, strict=True)
                            ),
                            batch.inputs.sum() * 0.0,
                        )
                        / denominator
                    )
            return ModularBrainOutputV2(
                packet=packet,
                action_logits=torch.cat(logits, dim=1),
                state=state,
                auxiliary_losses=combined_losses,
                routing_trace=tuple(traces),
                module_metrics={
                    name: torch.stack(values).sum() for name, values in metrics.items()
                },
                cost_statistics={name: torch.stack(values).sum() for name, values in costs.items()},
                forecast_logits=torch.cat(forecasts, dim=1),
                forecast_transition_mask=torch.cat(transition_masks, dim=1),
                forecast_error=torch.cat(forecast_errors, dim=1),
                persistence_error=torch.cat(persistence_errors, dim=1),
                feedback_delta=torch.cat(feedback_deltas, dim=1),
                telemetry_events=tuple(telemetry),
            )

    def forward(self, batch: TaskBatch) -> ModularBrainOutputV2:
        return self.forward_batch(batch)


__all__ = [
    "GOAL_CONTEXT_DIM",
    "TASK_CLASS_COUNTS",
    "ModularBrainNetworkV2",
    "ModularBrainOutputV2",
    "PredictorModeV2",
    "RoutingModeV2",
]
