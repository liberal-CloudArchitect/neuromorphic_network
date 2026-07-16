"""Composition graph for the six artificial P2 computation modules."""

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
    OPTIONAL_EXPERT_IDS,
    PREDICTIVE_ADAPTER,
    SENSORY_ENCODER,
    SPARSE_ROUTER,
)
from neuromorphic.modules.action_selector import ActionSelector
from neuromorphic.modules.episodic_memory import EpisodicMemory
from neuromorphic.modules.predictive_adapter import PredictiveAdapter
from neuromorphic.modules.sensory_encoder import SensoryEncoder
from neuromorphic.modules.sparse_router import RoutingDecision, SparseRouter
from neuromorphic.modules.working_memory import WorkingMemory
from neuromorphic.tasks.base import TaskBatch
from neuromorphic.tasks.control import (
    GOAL_CONTEXT_DIM,
    SMALL_GRAPH,
    TASK_CLASS_COUNTS,
    TASK_IDS,
    TASK_INPUT_DIMS,
    TaskControl,
    task_control_from_batch,
)

_LOSS_NAMES = {
    "episodic_retrieval": "episodic.retrieval",
    "episodic_separation": "episodic.separation",
    "working_consistency": "working.state_consistency",
    "working_gate_regularization": "working.gate_regularization",
    "predictive_next_state": "predictive.next_state",
}


def _merge_losses(target: dict[str, Tensor], source: Mapping[str, Tensor]) -> None:
    for name, value in source.items():
        target[_LOSS_NAMES.get(name, name)] = value


@dataclass(frozen=True, slots=True)
class ModularBrainOutput:
    """Result of one step or sequence through the modular computation graph."""

    packet: BrainPacket
    action_logits: Tensor
    prediction_logits: Tensor | None
    prediction_targets: Tensor
    prediction_mask: Tensor
    state: NetworkState
    auxiliary_losses: Mapping[str, Tensor]
    routing_trace: tuple[RoutingDecision, ...]
    module_metrics: Mapping[str, Tensor]
    cost_statistics: Mapping[str, Tensor]
    telemetry_events: tuple[TelemetryRecord, ...] = ()

    @property
    def logits(self) -> Tensor:
        """Compatibility alias consumed by the shared P1/P2 loss helpers."""

        return self.action_logits

    def __post_init__(self) -> None:
        batch, steps, _ = self.packet.representation.shape
        if self.action_logits.shape[:2] != (batch, steps):
            raise ValueError("action_logits must align with packet [B, T]")
        if self.prediction_logits is not None and self.prediction_logits.shape[:2] != (
            batch,
            steps,
        ):
            raise ValueError("prediction_logits must align with packet [B, T]")
        if self.prediction_targets.shape != (batch, steps):
            raise ValueError("prediction_targets must have shape [B, T]")
        if self.prediction_mask.shape != (batch, steps):
            raise ValueError("prediction_mask must have shape [B, T]")
        if self.prediction_targets.dtype is not torch.long:
            raise TypeError("prediction_targets must use torch.long")
        if self.prediction_mask.dtype is not torch.bool:
            raise TypeError("prediction_mask must use torch.bool")
        if self.state.batch_size != batch:
            raise ValueError("state batch size must align with output packet")


class TaskBoundaryAdapters(nn.Module):
    """Non-biological projections from each frozen task input into F=128."""

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.adapters = nn.ModuleList(
            nn.Linear(TASK_INPUT_DIMS[task_id], feature_dim) for task_id in TASK_IDS
        )

    def forward(self, inputs: Tensor, task_id: str) -> Tensor:
        if task_id not in TASK_IDS:
            raise ValueError(f"unsupported task_id: {task_id}")
        if inputs.ndim != 3 or inputs.shape[-1] != TASK_INPUT_DIMS[task_id]:
            raise ValueError(f"{task_id} inputs must have shape [B, T, {TASK_INPUT_DIMS[task_id]}]")
        return cast(Tensor, self.adapters[TASK_IDS.index(task_id)](inputs))


def _default_registry(
    feature_dim: int,
    *,
    episodic_slots: int,
    working_slots: int,
    working_slot_dim: int,
    action_embedding_dim: int,
    task_embedding_dim: int,
    router_top_k: int,
    router_capacity_factor: float,
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
            PredictiveAdapter(feature_dim=feature_dim, action_dim=action_embedding_dim),
            ActionSelector(feature_dim=feature_dim),
            SparseRouter(
                feature_dim=feature_dim,
                task_embedding_dim=task_embedding_dim,
                top_k=router_top_k,
                capacity_factor=router_capacity_factor,
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
    tensors: dict[str, Tensor] = {}
    for name, tensor in base.tensors.items():
        replacement = update.tensors[name]
        tensors[name] = tensor if tensor.ndim == 0 else tensor.index_copy(0, indices, replacement)
    return ModuleState(base.module_id, base.version, tensors)


def _scatter_packet(base: BrainPacket, update: BrainPacket, indices: Tensor) -> BrainPacket:
    representation = torch.zeros_like(base.representation).index_copy(
        0, indices, update.representation
    )
    return BrainPacket(
        representation=representation,
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


class ModularBrainNetwork(nn.Module):
    """Execute the frozen P2 graph with true row-sliced optional experts."""

    def __init__(
        self,
        *,
        feature_dim: int = 128,
        episodic_slots: int = 16,
        working_slots: int = 4,
        working_slot_dim: int = 32,
        action_embedding_dim: int = 32,
        task_embedding_dim: int = 16,
        router_top_k: int = 2,
        router_capacity_factor: float = 1.25,
        registry: ModuleRegistry | None = None,
        tbptt_interval: int = 32,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if tbptt_interval <= 0:
            raise ValueError("tbptt_interval must be positive")
        self.feature_dim = feature_dim
        self.tbptt_interval = tbptt_interval
        self.boundary_adapters = TaskBoundaryAdapters(feature_dim)
        self.registry = (
            _default_registry(
                feature_dim,
                episodic_slots=episodic_slots,
                working_slots=working_slots,
                working_slot_dim=working_slot_dim,
                action_embedding_dim=action_embedding_dim,
                task_embedding_dim=task_embedding_dim,
                router_top_k=router_top_k,
                router_capacity_factor=router_capacity_factor,
            )
            if registry is None
            else registry
        )
        self.registry.require_complete()

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
            module_state = self.registry.get(module_id).reset_state(
                updated.get(module_id), reset_mask
            )
            updated = updated.replace(module_state)
        return updated

    def forward_step(
        self,
        inputs: Tensor,
        control: TaskControl,
        state: NetworkState,
        context: ModuleContext,
        *,
        forced_experts: tuple[str, ...] | None = None,
        routing_mode: Literal["learned", "fixed", "random", "dense"] = "learned",
        disabled_experts: tuple[str, ...] = (),
    ) -> ModularBrainOutput:
        """Execute a single time step; ``inputs`` and controls use T=1."""

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

        reset_rows = context.reset_mask[:, 0] & control.valid_mask[:, 0]
        state = self._reset_state(state, reset_rows)
        valid_rows = control.valid_mask[:, 0]
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
        )
        state = state.replace(sensory_output.state)
        router = cast(SparseRouter, self.registry.get(SPARSE_ROUTER))
        unknown_disabled = set(disabled_experts).difference(OPTIONAL_EXPERT_IDS)
        if unknown_disabled:
            raise ValueError(f"unknown disabled experts: {sorted(unknown_disabled)}")
        decision = router.route(
            sensory_output.packet,
            forced_experts=forced_experts,
            mode=routing_mode,
        )

        expert_packets: dict[str, BrainPacket] = {}
        zero = adapted.sum() * 0.0
        losses: dict[str, Tensor] = {
            "episodic.retrieval": zero,
            "episodic.separation": zero,
            "working.state_consistency": zero,
            "working.gate_regularization": zero,
            "predictive.next_state": zero,
        }
        _merge_losses(losses, sensory_output.auxiliary_losses)
        routing_probabilities = torch.softmax(decision.scores, dim=-1)
        routing_valid = control.valid_mask.unsqueeze(-1).to(routing_probabilities.dtype)
        mean_probability = (routing_probabilities * routing_valid).sum(
            dim=(0, 1)
        ) / routing_valid.sum().clamp_min(1.0)
        losses["router.load_balance"] = (
            (mean_probability - 1.0 / len(OPTIONAL_EXPERT_IDS)).square().mean()
        )
        losses["router.communication_cost"] = (
            decision.executed_mask.to(routing_probabilities.dtype)
            * routing_probabilities
            * routing_valid
        ).sum() / routing_valid.sum().clamp_min(1.0)
        telemetry: list[TelemetryRecord] = list(sensory_output.telemetry_events)
        selected_counts: dict[str, Tensor] = {}
        for expert_index, module_id in enumerate(OPTIONAL_EXPERT_IDS):
            selected = decision.executed_mask[:, 0, expert_index] & valid_rows
            selected_counts[f"selected.{module_id}"] = selected.sum()
            # Predictive adaptation is action-conditioned and therefore runs
            # only after the action selector below.
            if module_id == PREDICTIVE_ADAPTER:
                continue
            if module_id in disabled_experts:
                continue
            indices = torch.nonzero(selected, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            module = self.registry.get(module_id)
            base_state = state.get(module_id)
            output = module.forward(
                _slice_packet(sensory_output.packet, indices),
                _slice_state(base_state, indices),
                _slice_context(context, indices),
            )
            state = state.replace(_scatter_state(base_state, output.state, indices))
            expert_packets[module_id] = _scatter_packet(
                sensory_output.packet, output.packet, indices
            )
            _merge_losses(losses, output.auxiliary_losses)
            telemetry.extend(output.telemetry_events)

        router_output = router.combine(
            sensory_output.packet,
            expert_packets,
            decision,
            state.get(SPARSE_ROUTER),
            context,
        )
        state = state.replace(router_output.state)
        _merge_losses(losses, router_output.auxiliary_losses)
        telemetry.extend(router_output.telemetry_events)

        selector = cast(ActionSelector, self.registry.get(ACTION_SELECTOR))
        selector_output = selector.forward(
            router_output.packet, state.get(ACTION_SELECTOR), context
        )
        if selector_output.action_logits is None:
            raise RuntimeError("action selector did not return action logits")
        state = state.replace(selector_output.state)
        _merge_losses(losses, selector_output.auxiliary_losses)
        telemetry.extend(selector_output.telemetry_events)
        selected_action = selector_output.action_logits.argmax(dim=-1)
        action_control = control.with_selected_action(selected_action)

        prediction_logits: Tensor | None = None
        prediction_targets = torch.full_like(selected_action, -100)
        predictive_selected = (
            decision.executed_mask[:, 0, OPTIONAL_EXPERT_IDS.index(PREDICTIVE_ADAPTER)] & valid_rows
        )
        prediction_mask = torch.zeros_like(control.valid_mask)
        indices = torch.nonzero(predictive_selected, as_tuple=False).flatten()
        if indices.numel() > 0 and PREDICTIVE_ADAPTER not in disabled_experts:
            predictor = cast(PredictiveAdapter, self.registry.get(PREDICTIVE_ADAPTER))
            base_state = state.get(PREDICTIVE_ADAPTER)
            predictor_packet = BrainPacket(
                representation=selector_output.packet.representation,
                valid_mask=selector_output.packet.valid_mask,
                modality=selector_output.packet.modality,
                step_index=selector_output.packet.step_index,
                source_module=selector_output.packet.source_module,
                goal_context=action_control.goal_context,
                metadata=selector_output.packet.metadata,
            )
            predictor_output = predictor.forward_with_action(
                _slice_packet(predictor_packet, indices),
                _slice_state(base_state, indices),
                _slice_context(context, indices),
                selected_action.index_select(0, indices),
                action_control.action_nodes.index_select(0, indices),
            )
            state = state.replace(_scatter_state(base_state, predictor_output.state, indices))
            if predictor_output.prediction_logits is not None:
                shape = (
                    inputs.shape[0],
                    1,
                    predictor_output.prediction_logits.shape[-1],
                )
                prediction_logits = torch.zeros(
                    shape,
                    device=inputs.device,
                    dtype=predictor_output.prediction_logits.dtype,
                ).index_copy(0, indices, predictor_output.prediction_logits)
            _merge_losses(losses, predictor_output.auxiliary_losses)
            telemetry.extend(predictor_output.telemetry_events)

            if control.task_id == SMALL_GRAPH:
                dynamic = predictor.dynamic_targets(
                    action_control.action_nodes.index_select(0, indices),
                    selected_action.index_select(0, indices),
                )
                prediction_targets = prediction_targets.index_copy(0, indices, dynamic)
                prediction_mask[:, 0] = (
                    predictive_selected & control.loss_mask[:, 0] & prediction_targets[:, 0].ge(0)
                )

        # The episodic write becomes visible only after action formation, so a
        # store event can never retrieve itself in the same step.
        episodic = cast(EpisodicMemory, self.registry.get(EPISODIC_MEMORY))
        state = state.replace(episodic.commit_pending(state.get(EPISODIC_MEMORY)))
        state, detach_mask = state.advance(valid_rows, interval=self.tbptt_interval)

        metrics = {
            **selected_counts,
            "tbptt.detached_rows": detach_mask.sum(),
            "routing.rerouted": decision.rerouted_mask.sum(),
            "routing.capacity_drops": torch.as_tensor(
                decision.capacity_drops,
                device=inputs.device,
            ),
        }
        active_optional = decision.executed_mask.to(inputs.dtype).sum()
        dense_optional = valid_rows.to(inputs.dtype).sum() * len(OPTIONAL_EXPERT_IDS)
        costs = {
            "optional.active_calls": active_optional,
            "optional.dense_calls": dense_optional,
        }
        return ModularBrainOutput(
            packet=selector_output.packet,
            action_logits=selector_output.action_logits,
            prediction_logits=prediction_logits,
            prediction_targets=prediction_targets,
            prediction_mask=prediction_mask,
            state=state,
            auxiliary_losses=losses,
            routing_trace=(decision,),
            module_metrics=metrics,
            cost_statistics=costs,
            telemetry_events=tuple(telemetry),
        )

    def forward_batch(
        self,
        batch: TaskBatch,
        state: NetworkState | None = None,
        *,
        phase: Phase = "train",
        telemetry_enabled: bool = False,
        forced_experts: tuple[str, ...] | None = None,
        routing_mode: Literal["learned", "fixed", "random", "dense"] = "learned",
        disabled_experts: tuple[str, ...] = (),
    ) -> ModularBrainOutput:
        """Execute a padded task batch sequentially with episode-local state."""

        batch.validate()
        with trusted_internal_execution():
            return self._forward_batch_trusted(
                batch,
                state,
                phase=phase,
                telemetry_enabled=telemetry_enabled,
                forced_experts=forced_experts,
                routing_mode=routing_mode,
                disabled_experts=disabled_experts,
            )

    def _forward_batch_trusted(
        self,
        batch: TaskBatch,
        state: NetworkState | None,
        *,
        phase: Phase,
        telemetry_enabled: bool,
        forced_experts: tuple[str, ...] | None,
        routing_mode: Literal["learned", "fixed", "random", "dense"],
        disabled_experts: tuple[str, ...],
    ) -> ModularBrainOutput:
        control = task_control_from_batch(batch)
        if state is None:
            state = self.initial_state(
                batch.batch_size, device=batch.inputs.device, dtype=batch.inputs.dtype
            )
        elif state.batch_size != batch.batch_size:
            raise ValueError("state and TaskBatch batch sizes must match")

        packets: list[BrainPacket] = []
        action_logits: list[Tensor] = []
        prediction_logits: list[Tensor | None] = []
        targets: list[Tensor] = []
        masks: list[Tensor] = []
        traces: list[RoutingDecision] = []
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
            context = ModuleContext(
                task_id=control.task_id,
                phase=phase,
                reset_mask=reset.unsqueeze(1),
                eligible_modules=OPTIONAL_EXPERT_IDS,
                telemetry_enabled=telemetry_enabled,
            )
            output = self.forward_step(
                batch.inputs[:, step : step + 1],
                control.at_step(step),
                state,
                context,
                forced_experts=forced_experts,
                routing_mode=routing_mode,
                disabled_experts=disabled_experts,
            )
            state = output.state
            packets.append(output.packet)
            action_logits.append(output.action_logits)
            prediction_logits.append(output.prediction_logits)
            targets.append(output.prediction_targets)
            masks.append(output.prediction_mask)
            traces.extend(output.routing_trace)
            telemetry.extend(output.telemetry_events)
            for name, value in output.auxiliary_losses.items():
                losses.setdefault(name, []).append(value)
            for name, value in output.module_metrics.items():
                metrics.setdefault(name, []).append(value)
            for name, value in output.cost_statistics.items():
                costs.setdefault(name, []).append(value)
            previous_episode = torch.where(valid, episode, previous_episode)

        first_packet = packets[0]
        packet = BrainPacket(
            representation=torch.cat([item.representation for item in packets], dim=1),
            valid_mask=batch.valid_mask,
            modality=first_packet.modality,
            step_index=torch.cat([item.step_index for item in packets], dim=1),
            source_module=ACTION_SELECTOR,
            goal_context=control.goal_context,
            metadata=first_packet.metadata,
        )
        combined_losses = {name: torch.stack(values).mean() for name, values in losses.items()}
        combined_metrics = {name: torch.stack(values).sum() for name, values in metrics.items()}
        combined_costs = {name: torch.stack(values).sum() for name, values in costs.items()}
        prediction = None
        available_predictions = [item for item in prediction_logits if item is not None]
        if available_predictions:
            classes = available_predictions[0].shape[-1]
            prediction = torch.cat(
                [
                    item
                    if item is not None
                    else torch.zeros(
                        (batch.batch_size, 1, classes),
                        device=batch.inputs.device,
                        dtype=batch.inputs.dtype,
                    )
                    for item in prediction_logits
                ],
                dim=1,
            )
        return ModularBrainOutput(
            packet=packet,
            action_logits=torch.cat(action_logits, dim=1),
            prediction_logits=prediction,
            prediction_targets=torch.cat(targets, dim=1),
            prediction_mask=torch.cat(masks, dim=1),
            state=state,
            auxiliary_losses=combined_losses,
            routing_trace=tuple(traces),
            module_metrics=combined_metrics,
            cost_statistics=combined_costs,
            telemetry_events=tuple(telemetry),
        )

    def forward(self, batch: TaskBatch) -> ModularBrainOutput:
        return self.forward_batch(batch)


__all__ = [
    "GOAL_CONTEXT_DIM",
    "TASK_CLASS_COUNTS",
    "ModularBrainNetwork",
    "ModularBrainOutput",
    "TaskBoundaryAdapters",
]
