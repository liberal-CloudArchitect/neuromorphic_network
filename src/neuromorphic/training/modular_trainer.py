"""Deterministic P2 curriculum and joint-training primitives."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from neuromorphic.core.contracts import ModuleContext, ModuleState
from neuromorphic.core.network_state import NetworkState
from neuromorphic.core.registry import (
    ACTION_SELECTOR,
    EPISODIC_MEMORY,
    MODULE_IDS,
    OPTIONAL_EXPERT_IDS,
    PREDICTIVE_ADAPTER,
    SENSORY_ENCODER,
    WORKING_MEMORY,
)
from neuromorphic.modules.network import ModularBrainNetwork, ModularBrainOutput
from neuromorphic.tasks import create_task
from neuromorphic.tasks.base import DatasetSplit, SequenceTask, TaskBatch
from neuromorphic.tasks.control import task_control_from_batch
from neuromorphic.tasks.small_graph import SmallGraphTask
from neuromorphic.training.modular_checkpoint import save_modular_checkpoint
from neuromorphic.training.modular_cost import profile_modular_execution
from neuromorphic.training.modular_metrics import modular_task_metrics, modular_training_loss
from neuromorphic.training.modular_monitoring import (
    capture_gradients,
    gradient_cosine_similarity,
    routing_statistics,
    state_dynamics,
)
from neuromorphic.training.modular_telemetry import build_step_events
from neuromorphic.training.p2_config import P2_TASK_ORDER, P2SuiteConfig
from neuromorphic.training.trainer import IndexSampler

PRETRAIN_STAGES = ("sensory", "episodic", "working", "predictive")

_STAGE_TASKS: Mapping[str, tuple[str, ...]] = {
    "sensory": P2_TASK_ORDER,
    "episodic": ("associative_recall.v1",),
    "working": ("delayed_rule_switch.v1",),
    "predictive": ("small_graph.v1",),
}

_STAGE_MODULES: Mapping[str, tuple[str, ...]] = {
    "sensory": (SENSORY_ENCODER, ACTION_SELECTOR),
    "episodic": (EPISODIC_MEMORY,),
    "working": (WORKING_MEMORY,),
    "predictive": (PREDICTIVE_ADAPTER,),
    "joint": MODULE_IDS,
}

_FORCED_ROUTES: Mapping[str, tuple[str, str]] = {
    "associative_recall.v1": (EPISODIC_MEMORY, WORKING_MEMORY),
    "delayed_rule_switch.v1": (WORKING_MEMORY, PREDICTIVE_ADAPTER),
    "small_graph.v1": (PREDICTIVE_ADAPTER, WORKING_MEMORY),
}


@dataclass(frozen=True, slots=True)
class TrainingBranchResult:
    """Serializable evidence returned by one pretrain or joint branch."""

    name: str
    loss_history: Mapping[str, tuple[float, ...]]
    evaluations: Mapping[str, Mapping[str, Mapping[str, float]]]
    routing: Mapping[str, object]
    state_dynamics: Mapping[str, object]
    gradient_cosines: tuple[float | None, ...]
    wall_clock_seconds: float
    checkpoint: str
    telemetry_event_count: int
    cost_profile: Mapping[str, object]
    gradient_coverage: Mapping[str, bool]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_modular_network(config: P2SuiteConfig) -> ModularBrainNetwork:
    model = config.model
    return ModularBrainNetwork(
        feature_dim=model.feature_dim,
        episodic_slots=model.episodic_slots,
        working_slots=model.working_slots,
        working_slot_dim=model.working_slot_dim,
        action_embedding_dim=model.action_embedding_dim,
        task_embedding_dim=model.task_embedding_dim,
        router_top_k=model.router_top_k,
        router_capacity_factor=model.router_capacity_factor,
        tbptt_interval=config.budget.tbptt_steps,
    )


def _set_trainable(model: ModularBrainNetwork, stage: str) -> tuple[str, ...]:
    if stage not in _STAGE_MODULES:
        raise ValueError(f"unknown P2 training stage: {stage}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_ids = _STAGE_MODULES[stage]
    if stage in {"sensory", "joint"}:
        for parameter in model.boundary_adapters.parameters():
            parameter.requires_grad_(True)
    for module_id in trainable_ids:
        module = model.registry.get(module_id)
        if not isinstance(module, nn.Module):
            raise TypeError("registered module is not a torch.nn.Module")
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return tuple(module_id for module_id in MODULE_IDS if module_id not in trainable_ids)


def _optimizer(model: nn.Module, config: P2SuiteConfig) -> torch.optim.AdamW:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("P2 stage has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )


def _stage_weights(config: P2SuiteConfig, stage: str) -> tuple[dict[str, float], bool]:
    weights = {name: 0.0 for name in config.losses.by_loss_name()}
    weights["primary"] = 1.0
    include_primary = True
    if stage == "episodic":
        weights["episodic.retrieval"] = config.losses.episodic_retrieval
        weights["episodic.separation"] = config.losses.episodic_separation
    elif stage == "working":
        weights["working.state_consistency"] = config.losses.working_consistency
        weights["working.gate_regularization"] = config.losses.working_gate
    elif stage == "predictive":
        weights["primary"] = 0.0
        weights["predictive.next_state"] = 1.0
        include_primary = False
    elif stage == "joint":
        weights = config.losses.by_loss_name()
    return weights, include_primary


def _append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _samplers(size: int, seed: int) -> dict[str, IndexSampler]:
    return {
        task_id: IndexSampler.create(size, seed + index * 10_000)
        for index, task_id in enumerate(P2_TASK_ORDER)
    }


def _state_snapshot(state: NetworkState) -> dict[str, ModuleState]:
    return {
        module_id: type(module_state)(
            module_state.module_id,
            module_state.version,
            {name: tensor.detach().clone() for name, tensor in module_state.tensors.items()},
        )
        for module_id, module_state in state.module_states.items()
    }


def _loss_decreased(history: Sequence[float]) -> bool:
    if len(history) < 2:
        return True
    window = min(10, len(history) // 2)
    return sum(history[-window:]) / window < sum(history[:window]) / window


def _record_gradient_coverage(model: ModularBrainNetwork, coverage: dict[str, bool]) -> None:
    for module_id in MODULE_IDS:
        module = model.registry.get(module_id)
        if not isinstance(module, nn.Module):
            raise TypeError("registered module is not a torch.nn.Module")
        coverage[module_id] = coverage[module_id] or any(
            parameter.grad is not None for parameter in module.parameters()
        )


def train_one_update(
    *,
    model: ModularBrainNetwork,
    optimizer: torch.optim.Optimizer,
    batch: TaskBatch,
    weights: Mapping[str, float],
    include_primary: bool,
    gradient_clip_norm: float,
    forced_experts: tuple[str, ...] | None,
    telemetry_enabled: bool,
) -> tuple[ModularBrainOutput, Tensor, dict[str, float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model.forward_batch(
        batch,
        phase="train",
        telemetry_enabled=telemetry_enabled,
        forced_experts=forced_experts,
    )
    loss, parts = modular_training_loss(
        output,
        batch,
        weights=weights,
        include_primary=include_primary,
    )
    loss.backward()  # type: ignore[no-untyped-call]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    torch.nn.utils.clip_grad_norm_(
        trainable,
        gradient_clip_norm,
        error_if_nonfinite=True,
    )
    optimizer.step()
    if not torch.isfinite(torch.nn.utils.parameters_to_vector(trainable)).all().item():
        raise FloatingPointError("non-finite parameter after modular optimizer step")
    return output, loss, parts


def evaluate_modular(
    model: ModularBrainNetwork,
    task: SequenceTask,
    *,
    split: DatasetSplit,
    size: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    weighted: dict[str, float] = {}
    total = 0
    with torch.no_grad():
        for start in range(0, size, batch_size):
            indices = list(range(start, min(start + batch_size, size)))
            batch = task.generate(split, indices, device=device)
            output = model.forward_batch(batch, phase="evaluate")
            metrics = modular_task_metrics(output, batch)
            count = len(indices)
            total += count
            for name, value in metrics.items():
                weighted[name] = weighted.get(name, 0.0) + value * count
    return {name: value / max(total, 1) for name, value in weighted.items()}


def evaluate_small_graph_rollout(
    model: ModularBrainNetwork,
    task: SmallGraphTask,
    *,
    split: DatasetSplit,
    size: int,
    device: torch.device,
) -> dict[str, float]:
    """Run the modular policy against action-dependent live graph states."""

    state: NetworkState | None = None
    graph_signature: bytes | None = None

    def policy(observation: Tensor) -> Tensor:
        nonlocal state, graph_signature
        observation = observation.to(device)
        signature = (
            observation[:256].detach().cpu().numpy().tobytes()
            + observation[272:288].detach().cpu().numpy().tobytes()
        )
        reset = state is None or signature != graph_signature
        graph_signature = signature
        if state is None:
            state = model.initial_state(1, device=device, dtype=observation.dtype)
        batch = TaskBatch(
            inputs=observation.reshape(1, 1, -1),
            targets=torch.full((1, 1), -100, dtype=torch.long, device=device),
            valid_mask=torch.ones((1, 1), dtype=torch.bool, device=device),
            loss_mask=torch.ones((1, 1), dtype=torch.bool, device=device),
            episode_ids=torch.zeros((1, 1), dtype=torch.long, device=device),
            metadata={
                "task_id": task.task_id,
                "task_version": task.task_version,
                "split": split,
            },
            auxiliary_targets={},
        )
        control = task_control_from_batch(batch)
        context = ModuleContext(
            task_id=task.task_id,
            phase="evaluate",
            reset_mask=torch.tensor([[reset]], dtype=torch.bool, device=device),
            eligible_modules=OPTIONAL_EXPERT_IDS,
            telemetry_enabled=False,
        )
        output = model.forward_step(batch.inputs, control, state, context)
        state = output.state
        return output.action_logits[0, 0]

    with torch.no_grad():
        metrics = task.rollout(policy, split, list(range(size)), device=device)
    return {f"live_{name}": float(value) for name, value in metrics.items()}


def run_pretraining(
    *,
    model: ModularBrainNetwork,
    config: P2SuiteConfig,
    device: torch.device,
    directory: Path,
) -> TrainingBranchResult:
    started = time.perf_counter()
    tasks = {task_id: create_task(task_id, profile="smoke") for task_id in P2_TASK_ORDER}
    samplers = _samplers(config.budget.train_size, config.seed)
    histories: dict[str, tuple[float, ...]] = {}
    evaluations: dict[str, dict[str, dict[str, float]]] = {}
    gradient_cosines: list[float | None] = []
    gradient_coverage = {module_id: False for module_id in MODULE_IDS}
    last_output: ModularBrainOutput | None = None
    checkpoint = directory / "shared.pt"
    for stage in PRETRAIN_STAGES:
        previous_gradients: dict[str, Tensor | None] | None = None
        frozen = _set_trainable(model, stage)
        optimizer = _optimizer(model, config)
        weights, include_primary = _stage_weights(config, stage)
        values: list[float] = []
        stage_tasks = _STAGE_TASKS[stage]
        for stage_step in range(config.budget.pretrain_steps_per_stage):
            task_id = stage_tasks[stage_step % len(stage_tasks)]
            task = tasks[task_id]
            indices = samplers[task_id].next(config.budget.batch_size)
            batch = task.generate("train", indices, device=device)
            output, loss, parts = train_one_update(
                model=model,
                optimizer=optimizer,
                batch=batch,
                weights=weights,
                include_primary=include_primary,
                gradient_clip_norm=config.optimizer.gradient_clip_norm,
                forced_experts=_FORCED_ROUTES[task_id],
                telemetry_enabled=False,
            )
            last_output = output
            _record_gradient_coverage(model, gradient_coverage)
            values.append(float(loss.detach().cpu()))
            _append_jsonl(
                directory / "metrics.jsonl",
                {
                    "phase": "pretrain",
                    "stage": stage,
                    "stage_step": stage_step + 1,
                    "task_id": task_id,
                    **parts,
                },
            )
            if (stage_step + 1) % config.budget.validation_interval_per_task == 0:
                evaluations.setdefault(stage, {})[task_id] = evaluate_modular(
                    model,
                    task,
                    split="validation",
                    size=config.budget.validation_size,
                    batch_size=config.budget.batch_size,
                    device=device,
                )
            if (stage_step + 1) % 25 == 0:
                gradients = capture_gradients(model)
                if previous_gradients is not None:
                    gradient_cosines.append(
                        gradient_cosine_similarity(previous_gradients, gradients)
                    )
                previous_gradients = gradients
        if config.profile == "gate" and not _loss_decreased(values):
            raise ValueError(f"P2 pretraining loss did not decrease for stage {stage}")
        histories[stage] = tuple(values)
        if last_output is None:
            raise RuntimeError("P2 pretraining produced no output")
        save_modular_checkpoint(
            directory / f"stage-{stage}.pt",
            model=model,
            optimizer=optimizer,
            module_states=last_output.state.module_states,
            curriculum_stage=stage,
            stage_step=config.budget.pretrain_steps_per_stage,
            sampler_states={name: sampler.state_dict() for name, sampler in samplers.items()},
            tbptt_counters=last_output.state.valid_step_counts,
            frozen_module_ids=frozen,
            config=config.checkpoint_compatible_dict(),
        )
        if stage == PRETRAIN_STAGES[-1]:
            save_modular_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                module_states=last_output.state.module_states,
                curriculum_stage=stage,
                stage_step=config.budget.pretrain_steps_per_stage,
                sampler_states={name: sampler.state_dict() for name, sampler in samplers.items()},
                tbptt_counters=last_output.state.valid_step_counts,
                frozen_module_ids=frozen,
                config=config.checkpoint_compatible_dict(),
            )
    return TrainingBranchResult(
        name="pretrain",
        loss_history=histories,
        evaluations=evaluations,
        routing={},
        state_dynamics={},
        gradient_cosines=tuple(gradient_cosines),
        wall_clock_seconds=time.perf_counter() - started,
        checkpoint=str(checkpoint),
        telemetry_event_count=0,
        cost_profile={},
        gradient_coverage=gradient_coverage,
    )


def run_joint_branch(
    *,
    name: str,
    initial_model_state: Mapping[str, Tensor],
    config: P2SuiteConfig,
    device: torch.device,
    directory: Path,
    telemetry_enabled: bool,
) -> tuple[TrainingBranchResult, dict[str, Tensor]]:
    started = time.perf_counter()
    model = build_modular_network(config).to(device)
    model.load_state_dict(initial_model_state)
    _set_trainable(model, "joint")
    optimizer = _optimizer(model, config)
    weights, include_primary = _stage_weights(config, "joint")
    tasks = {task_id: create_task(task_id, profile="smoke") for task_id in P2_TASK_ORDER}
    samplers = _samplers(config.budget.train_size, config.seed)
    histories: dict[str, list[float]] = {task_id: [] for task_id in P2_TASK_ORDER}
    evaluations: dict[str, dict[str, dict[str, float]]] = {task_id: {} for task_id in P2_TASK_ORDER}
    raw_masks: list[Tensor] = []
    executed_masks: list[Tensor] = []
    valid_masks: list[Tensor] = []
    previous_gradients: dict[str, Tensor | None] | None = None
    gradient_cosines: list[float | None] = []
    initial_states: dict[str, ModuleState] | None = None
    final_states: dict[str, ModuleState] | None = None
    telemetry_count = 0
    task_token_counts = {task_id: 0 for task_id in P2_TASK_ORDER}
    expert_active_counts = {
        module_id: 0 for module_id in (EPISODIC_MEMORY, WORKING_MEMORY, PREDICTIVE_ADAPTER)
    }
    gradient_coverage = {module_id: False for module_id in MODULE_IDS}
    last_output: ModularBrainOutput | None = None
    total_steps = config.budget.joint_steps_per_task * len(P2_TASK_ORDER)
    for global_step in range(total_steps):
        task_id = P2_TASK_ORDER[global_step % len(P2_TASK_ORDER)]
        task_update = global_step // len(P2_TASK_ORDER) + 1
        task = tasks[task_id]
        indices = samplers[task_id].next(config.budget.batch_size)
        batch = task.generate("train", indices, device=device)
        output, loss, parts = train_one_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            weights=weights,
            include_primary=include_primary,
            gradient_clip_norm=config.optimizer.gradient_clip_norm,
            forced_experts=None,
            telemetry_enabled=telemetry_enabled,
        )
        last_output = output
        _record_gradient_coverage(model, gradient_coverage)
        histories[task_id].append(float(loss.detach().cpu()))
        if initial_states is None:
            initial_states = _state_snapshot(output.state)
        final_states = _state_snapshot(output.state)
        raw_batch = torch.cat(
            [trace.raw_top2_mask.detach() for trace in output.routing_trace], dim=1
        ).cpu()
        executed_batch = torch.cat(
            [trace.executed_mask.detach() for trace in output.routing_trace], dim=1
        ).cpu()
        valid_batch = batch.valid_mask.detach().cpu()
        raw_masks.append(raw_batch)
        executed_masks.append(executed_batch)
        valid_masks.append(valid_batch)
        task_token_counts[task_id] += int(valid_batch.sum().item())
        for expert_index, module_id in enumerate(
            (EPISODIC_MEMORY, WORKING_MEMORY, PREDICTIVE_ADAPTER)
        ):
            expert_active_counts[module_id] += int(executed_batch[..., expert_index].sum().item())
        _append_jsonl(
            directory / "metrics.jsonl",
            {
                "phase": "joint",
                "branch": name,
                "global_step": global_step + 1,
                "task_step": task_update,
                "task_id": task_id,
                **parts,
            },
        )
        if task_update % config.budget.validation_interval_per_task == 0:
            evaluations[task_id]["validation"] = evaluate_modular(
                model,
                task,
                split="validation",
                size=config.budget.validation_size,
                batch_size=config.budget.batch_size,
                device=device,
            )
        if task_update % 25 == 0:
            gradients = capture_gradients(model)
            if previous_gradients is not None:
                gradient_cosines.append(gradient_cosine_similarity(previous_gradients, gradients))
            previous_gradients = gradients
        if telemetry_enabled:
            valid_tokens = max(int(batch.valid_mask.sum().item()), 1)
            selected = {
                module_id: float(output.module_metrics[f"selected.{module_id}"].detach().cpu())
                / valid_tokens
                for module_id in (EPISODIC_MEMORY, WORKING_MEMORY, PREDICTIVE_ADAPTER)
            }
            confidence = float(
                output.logits.softmax(dim=-1).amax(dim=-1)[batch.valid_mask].mean().detach().cpu()
            )
            module_metrics: dict[str, dict[str, float | bool | None]] = {
                module_id: {"compute_gate": True, "activity_raw": 0.0} for module_id in MODULE_IDS
            }
            for module_id, mass in selected.items():
                module_metrics[module_id]["routing_mass"] = mass
            module_metrics[ACTION_SELECTOR]["confidence"] = confidence
            events = build_step_events(
                run_id=name,
                step=global_step,
                phase="train",
                split="train",
                module_metrics=module_metrics,
                reducer_version=config.telemetry.reducer_version,
                baseline_version=config.telemetry.baseline_version,
            )
            telemetry_count += len(events)
            for event in events:
                _append_jsonl(directory / "telemetry.jsonl", event.to_dict())
        if task_update % config.budget.checkpoint_interval_per_task == 0:
            save_modular_checkpoint(
                directory / "latest.pt",
                model=model,
                optimizer=optimizer,
                module_states=output.state.module_states,
                curriculum_stage="joint",
                stage_step=global_step + 1,
                sampler_states={key: sampler.state_dict() for key, sampler in samplers.items()},
                tbptt_counters=output.state.valid_step_counts,
                frozen_module_ids=(),
                config=config.checkpoint_compatible_dict(),
            )
    if last_output is None or initial_states is None or final_states is None:
        raise RuntimeError("joint branch produced no updates")
    for task_id, values in histories.items():
        if config.profile == "gate" and not _loss_decreased(values):
            raise ValueError(f"P2 joint loss did not decrease for task {task_id}")
        task = tasks[task_id]
        for split, size in (
            ("validation", config.budget.validation_size),
            ("test", config.budget.test_size),
            ("ood", config.budget.ood_size),
        ):
            metrics = evaluate_modular(
                model,
                task,
                split=split,  # type: ignore[arg-type]
                size=size,
                batch_size=config.budget.batch_size,
                device=device,
            )
            if isinstance(task, SmallGraphTask):
                metrics.update(
                    evaluate_small_graph_rollout(
                        model,
                        task,
                        split=split,  # type: ignore[arg-type]
                        size=size,
                        device=device,
                    )
                )
            evaluations[task_id][split] = metrics
    raw = torch.cat([item.reshape(-1, item.shape[-1]) for item in raw_masks], dim=0)
    executed = torch.cat([item.reshape(-1, item.shape[-1]) for item in executed_masks], dim=0)
    valid = torch.cat([item.reshape(-1) for item in valid_masks], dim=0)
    route = routing_statistics(raw, executed, valid_mask=valid, top_k=2).to_dict()
    raw_shares = cast(tuple[float, ...], route["raw_shares"])
    route["raw_health_guard"] = all(share >= 0.05 for share in raw_shares)
    dynamics = {
        module_id: asdict(value)
        for module_id, value in state_dynamics(initial_states, final_states).items()
    }
    checkpoint = directory / "final.pt"
    save_modular_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        module_states=last_output.state.module_states,
        curriculum_stage="joint",
        stage_step=total_steps,
        sampler_states={key: sampler.state_dict() for key, sampler in samplers.items()},
        tbptt_counters=last_output.state.valid_step_counts,
        frozen_module_ids=(),
        config=config.checkpoint_compatible_dict(),
    )
    frozen_history = {key: tuple(values) for key, values in histories.items()}
    result = TrainingBranchResult(
        name=name,
        loss_history=frozen_history,
        evaluations=evaluations,
        routing=route,
        state_dynamics=dynamics,
        gradient_cosines=tuple(gradient_cosines),
        wall_clock_seconds=time.perf_counter() - started,
        checkpoint=str(checkpoint),
        telemetry_event_count=telemetry_count,
        cost_profile=profile_modular_execution(
            model,
            task_token_counts=task_token_counts,
            expert_active_counts=expert_active_counts,
        ).to_dict(),
        gradient_coverage=gradient_coverage,
    )
    final_model_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    return result, final_model_state


def clone_model_state(model: nn.Module) -> dict[str, Tensor]:
    """Return a device-neutral clone shared by paired joint branches."""

    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def compare_branch_models(
    first: Mapping[str, Tensor],
    second: Mapping[str, Tensor],
    *,
    device: torch.device,
) -> float:
    """Return the maximum absolute paired parameter difference."""

    if set(first) != set(second):
        raise ValueError("paired branch model keys differ")
    maximum = 0.0
    for name in first:
        left = first[name]
        right = second[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"paired branch tensor contract differs: {name}")
        difference = float((left.to(torch.float64) - right.to(torch.float64)).abs().max().item())
        maximum = max(maximum, difference)
        if device.type == "cpu" and not torch.equal(left, right):
            raise ValueError(f"CPU telemetry branches are not bitwise equal: {name}")
        if device.type == "mps" and not torch.allclose(left, right, rtol=1e-5, atol=1e-6):
            raise ValueError(f"MPS telemetry branches exceed tolerance: {name}")
    return maximum


__all__ = [
    "PRETRAIN_STAGES",
    "TrainingBranchResult",
    "build_modular_network",
    "clone_model_state",
    "compare_branch_models",
    "evaluate_modular",
    "evaluate_small_graph_rollout",
    "run_joint_branch",
    "run_pretraining",
    "train_one_update",
]
