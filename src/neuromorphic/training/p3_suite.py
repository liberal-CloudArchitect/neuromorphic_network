"""Execute the qualification or formal P3 experiment matrix."""

from __future__ import annotations

import hashlib
import json
import math
import resource
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from neuromorphic.core.registry import (
    ACTION_SELECTOR,
    EPISODIC_MEMORY,
    OPTIONAL_EXPERT_IDS,
    PREDICTIVE_ADAPTER,
    WORKING_MEMORY,
)
from neuromorphic.evaluation.p3_records import (
    linear_cka,
    linear_probe_accuracy,
    p3_sample_records,
    rsa_spearman,
)
from neuromorphic.evaluation.p3_statistics import normalized_aulc
from neuromorphic.inference.bundle import create_network_mvp_bundle
from neuromorphic.tasks import SmallGraphTask, TaskBatch, create_task
from neuromorphic.tasks.base import SequenceTask
from neuromorphic.training.baselines import (
    GRUBaseline,
    TransformerBaseline,
    select_parameter_matched_baseline,
)
from neuromorphic.training.config import resolve_device
from neuromorphic.training.metrics import (
    ensure_finite_training_state,
    masked_task_loss,
    task_metrics,
)
from neuromorphic.training.modular_cost import profile_modular_execution
from neuromorphic.training.modular_metrics import modular_task_metrics, modular_training_loss
from neuromorphic.training.modular_trainer import build_modular_network
from neuromorphic.training.p2_config import P2LossWeights, P2SuiteConfig
from neuromorphic.training.p3_baselines import (
    P3_TASK_DIMS,
    SharedGRUBaseline,
    SharedTransformerBaseline,
    SingleTaskTransformerV2,
    select_shared_parameter_match,
)
from neuromorphic.training.p3_checkpoint import (
    P3CheckpointState,
    load_p3_checkpoint,
    save_p3_checkpoint,
)
from neuromorphic.training.p3_config import (
    P3_TASK_ORDER,
    P3ExperimentCell,
    P3SuiteConfig,
)
from neuromorphic.training.reproducibility import set_global_seed
from neuromorphic.training.trainer import IndexSampler

GIB = 1024**3
PROTOCOL_HASH = hashlib.sha256(b"p3-protocol-v2").hexdigest()
_SHARED_MATCH_CACHE: dict[tuple[str, int], tuple[int, int | None, float]] = {}
_SINGLE_MATCH_CACHE: dict[tuple[str, str, int], tuple[int, int | None, float]] = {}
_PILOT_PRESETS: dict[str, dict[str, dict[str, object]]] = {
    "modular": {
        "preset-0": {"learning_rate": 1e-4, "weight_decay": 1e-2, "auxiliary_scale": 1.0},
        "preset-1": {"learning_rate": 3e-4, "weight_decay": 1e-2, "auxiliary_scale": 1.0},
        "preset-2": {
            "learning_rate": 3e-4,
            "weight_decay": 1e-2,
            "auxiliary_scale": 0.0,
            "mode": "joint-only",
        },
        "preset-3": {"learning_rate": 3e-4, "weight_decay": 1e-2, "auxiliary_scale": 0.5},
    },
    "gru": {
        "preset-0": {"learning_rate": 1e-4, "weight_decay": 1e-2, "auxiliary_scale": 1.0},
        "preset-1": {"learning_rate": 3e-4, "weight_decay": 1e-2, "auxiliary_scale": 1.0},
        "preset-2": {"learning_rate": 1e-3, "weight_decay": 1e-2, "auxiliary_scale": 1.0},
        "preset-3": {"learning_rate": 3e-4, "weight_decay": 0.0, "auxiliary_scale": 1.0},
    },
    "transformer": {
        "preset-0": {"learning_rate": 1e-4, "weight_decay": 1e-2, "auxiliary_scale": 1.0},
        "preset-1": {"learning_rate": 3e-4, "weight_decay": 1e-2, "auxiliary_scale": 1.0},
        "preset-2": {"learning_rate": 1e-3, "weight_decay": 1e-2, "auxiliary_scale": 1.0},
        "preset-3": {"learning_rate": 3e-4, "weight_decay": 0.0, "auxiliary_scale": 1.0},
    },
}


class _P3ResourceLimit(RuntimeError):
    """Stop the matrix without converting an expected resource limit into a failed cell."""


def _guard_suite_resources(directory: Path, deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise _P3ResourceLimit("P3 wall-clock budget was exhausted")
    if shutil.disk_usage(directory).free < 20 * GIB:
        raise _P3ResourceLimit("P3 free disk space fell below 20 GiB")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_checkpoint(cell_directory: Path) -> Path:
    best = cell_directory / "best.pt"
    return best if best.is_file() else cell_directory / "checkpoint.pt"


def _repository_state() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        check=True,
        capture_output=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            check=True,
            capture_output=True,
        ).stdout.strip()
    )
    return commit, dirty


def _task_profile(config: P3SuiteConfig) -> Literal["smoke", "qualification"]:
    return "smoke" if config.profile == "qualification" else "qualification"


def _task(config: P3SuiteConfig, task_id: str, distribution: str = "v1") -> SequenceTask:
    return create_task(task_id, profile=_task_profile(config), distribution=distribution)


def _build_modular(config: P3SuiteConfig, seed: int) -> nn.Module:
    p2 = P2SuiteConfig(profile="ci", seed=seed, device=config.device)
    return build_modular_network(p2)


def _freeze_unused_task_paths(model: nn.Module, task_id: str) -> None:
    modular = cast(Any, model)
    active_index = P3_TASK_ORDER.index(task_id)
    for index, adapter in enumerate(modular.boundary_adapters.adapters):
        if index != active_index:
            for parameter in adapter.parameters():
                parameter.requires_grad_(False)
    selector = modular.registry.get(ACTION_SELECTOR)
    active_head = task_id.removesuffix(".v1")
    for name, head in selector.heads.items():
        if name != active_head:
            for parameter in head.parameters():
                parameter.requires_grad_(False)


def _build_model(
    config: P3SuiteConfig,
    cell: P3ExperimentCell,
    device: torch.device,
) -> tuple[nn.Module, str]:
    modular = _build_modular(config, cell.seed)
    if cell.regime == "per_task":
        if cell.task_id is None:
            raise ValueError("per-task cell requires a task_id")
        _freeze_unused_task_paths(modular, cell.task_id)
    target = sum(parameter.numel() for parameter in modular.parameters() if parameter.requires_grad)
    if cell.model_id == "modular":
        model = modular
        target_label = "modular"
    elif cell.regime in {"shared", "continual"}:
        shared_cache_key = (cell.model_id, target)
        cached = _SHARED_MATCH_CACHE.get(shared_cache_key)
        if cached is None:
            match = select_shared_parameter_match(cell.model_id, target)
            cached = (match.hidden_size, match.feedforward_size, match.relative_error)
            _SHARED_MATCH_CACHE[shared_cache_key] = cached
            model = match.model
        elif cell.model_id == "gru":
            model = SharedGRUBaseline(hidden_size=cached[0])
        else:
            model = SharedTransformerBaseline(
                hidden_size=cached[0], feedforward_size=cast(int, cached[1])
            )
        target_label = f"parameter-match:{cached[2]:.6f}"
    else:
        if cell.task_id is None:
            raise ValueError("per-task baseline cell requires a task_id")
        task = _task(config, cell.task_id)
        auxiliary = 16 if cell.task_id == "small_graph.v1" else None
        single_cache_key = (cell.model_id, cell.task_id, target)
        cached = _SINGLE_MATCH_CACHE.get(single_cache_key)
        if cached is None:
            selected, single_match = select_parameter_matched_baseline(
                kind=cell.model_id,
                input_dim=task.input_dim,
                num_classes=task.num_classes,
                target=target,
                auxiliary_classes=auxiliary,
            )
            cached = (
                single_match.hidden_size,
                single_match.feedforward_size,
                single_match.relative_error,
            )
            _SINGLE_MATCH_CACHE[single_cache_key] = cached
        elif cell.model_id == "gru":
            selected = GRUBaseline(
                input_dim=task.input_dim,
                num_classes=task.num_classes,
                hidden_size=cached[0],
                auxiliary_classes=auxiliary,
            )
        else:
            selected = TransformerBaseline(
                input_dim=task.input_dim,
                num_classes=task.num_classes,
                hidden_size=cached[0],
                layers=2,
                heads=4,
                feedforward_size=cast(int, cached[1]),
                auxiliary_classes=auxiliary,
            )
        if cell.model_id == "transformer":
            selected = SingleTaskTransformerV2(
                input_dim=task.input_dim,
                num_classes=task.num_classes,
                hidden_size=cached[0],
                layers=2,
                heads=4,
                feedforward_size=cast(int, cached[1]),
                auxiliary_classes=auxiliary,
            )
        model = selected
        target_label = f"parameter-match:{cached[2]:.6f}"
    model.to(device)
    if cell.model_id == "modular":
        registry = cast(Any, model).registry
        if cell.intervention.target == EPISODIC_MEMORY:
            for parameter in registry.get(EPISODIC_MEMORY).parameters():
                parameter.requires_grad_(False)
    return model, target_label


def _intervention_arguments(
    cell: P3ExperimentCell,
) -> tuple[
    str,
    tuple[str, ...],
    tuple[str, ...],
    Literal["full", "shallow"],
    Literal["integrated", "direct"],
]:
    routing = (
        cell.intervention.operation if cell.intervention.target == "sparse_router.v1" else "learned"
    )
    disabled: tuple[str, ...] = ()
    if cell.intervention.target == EPISODIC_MEMORY:
        disabled = (cell.intervention.target,)
    if cell.intervention.target == PREDICTIVE_ADAPTER and not cell.max_steps:
        disabled = (cell.intervention.target,)
    reset = (WORKING_MEMORY,) if cell.intervention.target == WORKING_MEMORY else ()
    encoder_mode: Literal["full", "shallow"] = (
        "shallow" if cell.intervention.operation == "shallow" else "full"
    )
    selector_mode: Literal["integrated", "direct"] = (
        "direct" if cell.intervention.operation == "direct_head" else "integrated"
    )
    return routing, disabled, reset, encoder_mode, selector_mode


def _forward(model: nn.Module, cell: P3ExperimentCell, batch: TaskBatch) -> object:
    if cell.model_id == "modular":
        routing, disabled, reset, encoder_mode, selector_mode = _intervention_arguments(cell)
        return cast(Any, model).forward_batch(
            batch,
            phase="train" if model.training else "evaluate",
            routing_mode=routing,
            disabled_experts=disabled,
            reset_experts_every_step=reset,
            encoder_mode=encoder_mode,
            selector_mode=selector_mode,
        )
    if cell.regime in {"shared", "continual"}:
        return model(batch)
    return model(batch.inputs, batch.valid_mask)


def _training_loss(
    model: nn.Module,
    cell: P3ExperimentCell,
    output: object,
    batch: TaskBatch,
    *,
    auxiliary_scale: float,
) -> tuple[Tensor, dict[str, float]]:
    if cell.model_id == "modular":
        weights = {"primary": 1.0, **P2LossWeights().by_loss_name()}
        weights = {
            name: value if name == "primary" else value * auxiliary_scale
            for name, value in weights.items()
        }
        if cell.intervention.target == "predictive_adapter.v1":
            weights["predictive.next_state"] = 0.0
        return modular_training_loss(cast(Any, output), batch, weights=weights)
    return masked_task_loss(cast(Any, output), batch, auxiliary_weight=0.1 * auxiliary_scale)


def _training_settings(config: P3SuiteConfig, cell: P3ExperimentCell) -> dict[str, object]:
    preset: str | None = None
    if cell.cell_type == "pilot":
        preset = cell.variant_id
    elif config.profile == "full":
        if config.selected_presets is None:
            raise ValueError("full P3 execution requires a frozen pilot preset selection")
        preset = str(getattr(config.selected_presets, cell.model_id))
    if preset is not None:
        return {"preset_id": preset, **_PILOT_PRESETS[cell.model_id][preset]}
    return {
        "preset_id": None,
        "learning_rate": config.optimizer.learning_rate,
        "weight_decay": config.optimizer.weight_decay,
        "auxiliary_scale": 1.0,
    }


def _metrics(cell: P3ExperimentCell, output: object, batch: TaskBatch) -> dict[str, float]:
    if cell.model_id == "modular":
        return modular_task_metrics(cast(Any, output), batch)
    return task_metrics(cast(Any, output), batch)


def _task_sequence(cell: P3ExperimentCell, step: int) -> str:
    if cell.task_id is not None:
        return cell.task_id
    if cell.cell_type == "continual":
        orders = {
            17: P3_TASK_ORDER,
            29: (P3_TASK_ORDER[1], P3_TASK_ORDER[2], P3_TASK_ORDER[0]),
            43: (P3_TASK_ORDER[2], P3_TASK_ORDER[0], P3_TASK_ORDER[1]),
            7: P3_TASK_ORDER,
        }
        order = orders[cell.seed]
        per_stage = max(cell.max_steps // len(order), 1)
        return order[min(step // per_stage, len(order) - 1)]
    return P3_TASK_ORDER[step % len(P3_TASK_ORDER)]


def _early_stop_reached(cell: P3ExperimentCell, *, stale: int, patience: int) -> bool:
    """Keep pre-registered pilot and continual budgets exact."""

    return cell.cell_type not in {"continual", "pilot"} and stale >= patience


def _evaluation_metric(
    model: nn.Module,
    cell: P3ExperimentCell,
    task: SequenceTask,
    config: P3SuiteConfig,
    device: torch.device,
    *,
    split: str,
    size: int,
) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for start in range(0, size, config.budget.batch_size):
            indices = list(range(start, min(start + config.budget.batch_size, size)))
            batch = task.generate(cast(Any, split), indices, device=device)
            metrics = _metrics(cell, _forward(model, cell, batch), batch)
            name = "optimal_action_rate" if task.task_id == "small_graph.v1" else "accuracy"
            values.append(metrics[name] * len(indices))
    model.train()
    return sum(values) / max(size, 1)


def _validation_macro(
    model: nn.Module,
    cell: P3ExperimentCell,
    config: P3SuiteConfig,
    device: torch.device,
    split: str = "validation",
) -> dict[str, float]:
    task_ids = (cell.task_id,) if cell.task_id is not None else P3_TASK_ORDER
    size = config.data.validation if split == "validation" else config.data.analysis
    return {
        task_id: _evaluation_metric(
            model,
            cell,
            _task(config, task_id),
            config,
            device,
            split=split,
            size=size,
        )
        for task_id in task_ids
    }


def _state(
    cell: P3ExperimentCell,
    config: P3SuiteConfig,
    *,
    global_step: int,
    task_steps: Mapping[str, int],
    samplers: Mapping[str, IndexSampler],
    best_metrics: Mapping[str, float],
    stale: int,
    matrix_cursor: int,
    analysis_curves: Mapping[str, Sequence[tuple[int, float]]],
    validation_curve: Sequence[tuple[int, float]],
    last_loss: float | None,
) -> P3CheckpointState:
    return P3CheckpointState(
        cell_id=cell.cell_id,
        global_step=global_step,
        task_steps=dict(task_steps),
        sampler_states={name: sampler.state_dict() for name, sampler in samplers.items()},
        best_metrics=dict(best_metrics),
        stale_evaluations=stale,
        matrix_cursor=matrix_cursor,
        config_hash=config.config_hash(),
        protocol_hash=PROTOCOL_HASH,
        analysis_curves={name: tuple(points) for name, points in analysis_curves.items()},
        validation_curve=tuple(validation_curve),
        last_loss=last_loss,
    )


def _train_cell(
    model: nn.Module,
    cell: P3ExperimentCell,
    config: P3SuiteConfig,
    device: torch.device,
    directory: Path,
    matrix_cursor: int,
    suite_directory: Path,
    deadline: float,
) -> dict[str, object]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    settings = _training_settings(config, cell)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cast(float, settings["learning_rate"]),
        weight_decay=cast(float, settings["weight_decay"]),
    )
    task_ids = (cell.task_id,) if cell.task_id is not None else P3_TASK_ORDER
    samplers = {
        task_id: IndexSampler.create(config.data.train, cell.seed + index * 10_000)
        for index, task_id in enumerate(task_ids)
    }
    sampler_signatures = {
        task_id: (sampler.size, sampler.seed) for task_id, sampler in samplers.items()
    }
    checkpoint = directory / "checkpoint.pt"
    best_checkpoint = directory / "best.pt"
    step = 0
    task_steps = {task_id: 0 for task_id in task_ids}
    best_metrics: dict[str, float] = {}
    stale = 0
    curves: dict[str, list[tuple[int, float]]] = {task_id: [] for task_id in task_ids}
    validation_curve: list[tuple[int, float]] = []
    last_loss: float | None = None
    exit_reason: Literal["stopped", "resource_limit"] | None = None
    if checkpoint.exists():
        restored = load_p3_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            expected_cell_id=cell.cell_id,
            expected_config_hash=config.config_hash(),
            expected_protocol_hash=PROTOCOL_HASH,
            expected_matrix_cursor=matrix_cursor,
            expected_sampler_signatures=sampler_signatures,
        )
        step = restored.global_step
        task_steps = dict(restored.task_steps)
        best_metrics = dict(restored.best_metrics)
        stale = restored.stale_evaluations
        curves = {name: list(points) for name, points in restored.analysis_curves.items()}
        validation_curve = list(restored.validation_curve)
        last_loss = restored.last_loss
        for task_id, sampler_state in restored.sampler_states.items():
            samplers[task_id].load_state_dict(sampler_state)

    def save_latest() -> None:
        save_p3_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            state=_state(
                cell,
                config,
                global_step=step,
                task_steps=task_steps,
                samplers=samplers,
                best_metrics=best_metrics,
                stale=stale,
                matrix_cursor=matrix_cursor,
                analysis_curves=curves,
                validation_curve=validation_curve,
                last_loss=last_loss,
            ),
        )

    def write_heartbeat(task_id: str | None) -> None:
        remaining_seconds = max(deadline - time.perf_counter(), 0.0)
        _write_json(
            suite_directory / "heartbeat.json",
            {
                "cell_id": cell.cell_id,
                "seed": cell.seed,
                "task_id": task_id,
                "step": step,
                "max_steps": cell.max_steps,
                "updated_at": datetime.now(UTC).isoformat(),
                "free_bytes": shutil.disk_usage(suite_directory).free,
                "remaining_wall_clock_seconds": remaining_seconds,
                "suite_elapsed_seconds": max(
                    config.budget.wall_clock_hours * 3600.0 - remaining_seconds,
                    0.0,
                ),
            },
        )

    started = time.perf_counter()
    last_heartbeat = started
    while step < cell.max_steps:
        if (suite_directory / "STOP").exists():
            save_latest()
            write_heartbeat(None)
            exit_reason = "stopped"
            break
        if time.perf_counter() >= deadline:
            save_latest()
            write_heartbeat(None)
            exit_reason = "resource_limit"
            break
        task_id = _task_sequence(cell, step)
        task = _task(config, task_id)
        indices = samplers[task_id].next(min(config.budget.batch_size, config.data.train))
        batch = task.generate("train", indices, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = _forward(model, cell, batch)
        loss, parts = _training_loss(
            model,
            cell,
            output,
            batch,
            auxiliary_scale=cast(float, settings["auxiliary_scale"]),
        )
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(
            trainable, config.optimizer.gradient_clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        ensure_finite_training_state(loss=loss, model=model, metrics=parts)
        last_loss = float(loss.detach().cpu())
        step += 1
        task_steps[task_id] += 1
        if step % config.budget.validation_interval == 0 or step == cell.max_steps:
            current = _validation_macro(model, cell, config, device)
            analysis = (
                {}
                if config.profile == "pilot"
                else _validation_macro(model, cell, config, device, split="analysis")
            )
            for name, value in analysis.items():
                curve_step = step if cell.cell_type == "continual" else task_steps.get(name, step)
                point = (curve_step, value)
                curve = curves.setdefault(name, [])
                if curve and curve[-1][0] == point[0]:
                    curve[-1] = point
                else:
                    curve.append(point)
            macro = sum(current.values()) / len(current)
            validation_curve.append((step, macro))
            previous = best_metrics.get("macro", -math.inf)
            if macro > previous + config.budget.min_delta:
                best_metrics = {"macro": macro, **current}
                stale = 0
                save_p3_checkpoint(
                    best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    state=_state(
                        cell,
                        config,
                        global_step=step,
                        task_steps=task_steps,
                        samplers=samplers,
                        best_metrics=best_metrics,
                        stale=stale,
                        matrix_cursor=matrix_cursor,
                        analysis_curves=curves,
                        validation_curve=validation_curve,
                        last_loss=last_loss,
                    ),
                )
            else:
                stale += 1
        if step % config.budget.checkpoint_interval == 0 or step == cell.max_steps:
            save_latest()
            write_heartbeat(task_id)
            last_heartbeat = time.perf_counter()
            if shutil.disk_usage(suite_directory).free < 20 * GIB:
                exit_reason = "resource_limit"
                break
        elif time.perf_counter() - last_heartbeat >= 60.0:
            write_heartbeat(task_id)
            last_heartbeat = time.perf_counter()
        if _early_stop_reached(cell, stale=stale, patience=config.budget.patience):
            break
    if exit_reason is None and best_checkpoint.exists():
        load_p3_checkpoint(
            best_checkpoint,
            model=model,
            optimizer=optimizer,
            expected_cell_id=cell.cell_id,
            expected_config_hash=config.config_hash(),
            expected_protocol_hash=PROTOCOL_HASH,
            expected_matrix_cursor=matrix_cursor,
            expected_sampler_signatures=sampler_signatures,
        )
    aulc = {
        task_id: normalized_aulc(
            points or [(0, 0.0)],
            maximum_step=max(
                (
                    cell.max_steps
                    if cell.task_id is not None or cell.cell_type == "continual"
                    else cell.max_steps // 3
                ),
                1,
            ),
        )
        for task_id, points in curves.items()
    }
    forgetting = {
        task_id: (max(value for _, value in points) - points[-1][1] if points else None)
        for task_id, points in curves.items()
    }
    return {
        "steps": step,
        "task_steps": task_steps,
        "best_metrics": best_metrics,
        "stale_evaluations": stale,
        "analysis_aulc": aulc,
        "analysis_forgetting": forgetting,
        "validation_macro_aulc": normalized_aulc(
            validation_curve or [(0, 0.0)], maximum_step=max(cell.max_steps, 1)
        ),
        "last_loss": last_loss,
        "wall_clock_seconds": time.perf_counter() - started,
        "stopped": exit_reason == "stopped",
        "resource_limited": exit_reason == "resource_limit",
        "training_settings": settings,
    }


def _apply_control_after_parent_load(model: nn.Module, cell: P3ExperimentCell) -> None:
    """Apply controls whose semantics require the trained parent to be loaded first."""

    if cell.model_id != "modular" or cell.cell_type != "control":
        return
    registry = cast(Any, model).registry
    if cell.intervention.target != "sensory_encoder.v1":
        return
    encoder = registry.get("sensory_encoder.v1")
    if cell.intervention.operation == "frozen_random":
        set_global_seed(cell.intervention.seed + 50_000)
        for child in encoder.modules():
            reset_parameters = getattr(child, "reset_parameters", None)
            if child is not encoder and callable(reset_parameters):
                reset_parameters()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of no values")
    ordered = sorted(values)
    index = min(math.ceil(probability * len(ordered)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def _supported_parameter_coverage(model: nn.Module) -> tuple[int, int, float]:
    supported_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.GRU, nn.MultiheadAttention)):
            supported_ids.update(id(parameter) for parameter in module.parameters(recurse=False))
    parameters = tuple(model.parameters())
    supported = sum(parameter.numel() for parameter in parameters if id(parameter) in supported_ids)
    total = sum(parameter.numel() for parameter in parameters)
    return supported, total, supported / total if total else 1.0


def _baseline_batch_macs(model: nn.Module, task_id: str, batch: TaskBatch) -> int:
    """Estimate supported MACs per valid token while retaining padded attention width."""

    length = batch.sequence_length
    valid_tokens = int(batch.valid_mask.sum().item())
    if isinstance(model, SharedGRUBaseline):
        input_dim, classes = P3_TASK_DIMS[task_id]
        hidden = model.hidden_size
        recurrent = model.layers * 3 * (2 * hidden * hidden)
        head = hidden * classes + (hidden * 16 if task_id == "small_graph.v1" else 0)
        return valid_tokens * (input_dim * hidden + recurrent + head)
    if isinstance(model, SharedTransformerBaseline):
        input_dim, classes = P3_TASK_DIMS[task_id]
        hidden = model.hidden_size
        per_token = (
            input_dim * hidden
            + model.layers
            * (4 * hidden * hidden + 2 * length * hidden + 2 * hidden * model.feedforward_size)
            + hidden * classes
            + (hidden * 16 if task_id == "small_graph.v1" else 0)
        )
        return valid_tokens * per_token
    if isinstance(model, GRUBaseline):
        recurrent = model.layers * 3 * (2 * model.hidden_size * model.hidden_size)
        heads = model.hidden_size * model.output_head.out_features
        if model.next_state_head is not None:
            heads += model.hidden_size * model.next_state_head.out_features
        return valid_tokens * (model.input_dim * model.hidden_size + recurrent + heads)
    if isinstance(model, TransformerBaseline):
        hidden = model.hidden_size
        heads = hidden * model.output_head.out_features
        if model.next_state_head is not None:
            heads += hidden * model.next_state_head.out_features
        per_token = (
            model.input_dim * hidden
            + model.layers
            * (4 * hidden * hidden + 2 * length * hidden + 2 * hidden * model.feedforward_size)
            + heads
        )
        return valid_tokens * per_token
    raise TypeError(f"unsupported P3 baseline for MAC profiling: {type(model).__name__}")


def _active_path_parameters(model: nn.Module, cell: P3ExperimentCell) -> int:
    total = sum(parameter.numel() for parameter in model.parameters())
    if cell.model_id != "modular" or cell.task_id is None:
        return total
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _profile_cell_cost(
    model: nn.Module,
    cell: P3ExperimentCell,
    config: P3SuiteConfig,
    device: torch.device,
    suite_directory: Path,
    deadline: float,
) -> dict[str, object]:
    """Record auditable supported MACs and observed end-to-end inference latency."""

    task_ids = (cell.task_id,) if cell.task_id is not None else P3_TASK_ORDER
    task_token_counts = {task_id: 0 for task_id in P3_TASK_ORDER}
    expert_active_counts = {module_id: 0 for module_id in OPTIONAL_EXPERT_IDS}
    baseline_macs = 0
    latencies: list[float] = []
    if config.profile == "full":
        warmup_count, measurement_count = 10, 50
    elif config.profile == "pilot":
        warmup_count, measurement_count = 3, 10
    else:
        warmup_count, measurement_count = 1, 3
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for task_id in task_ids:
            _guard_suite_resources(suite_directory, deadline)
            batch = _task(config, task_id).generate("test", [0], device=device)
            output = _forward(model, cell, batch)
            for _ in range(warmup_count - 1):
                output = _forward(model, cell, batch)
            _synchronize(device)
            for _ in range(measurement_count):
                _guard_suite_resources(suite_directory, deadline)
                started = time.perf_counter()
                output = _forward(model, cell, batch)
                _synchronize(device)
                latencies.append((time.perf_counter() - started) * 1_000.0)
            if cell.model_id == "modular":
                task_token_counts[task_id] += int(batch.valid_mask.sum().item())
                for trace in cast(Any, output).routing_trace:
                    for index, module_id in enumerate(OPTIONAL_EXPERT_IDS):
                        expert_active_counts[module_id] += int(
                            trace.executed_mask[..., index].sum().item()
                        )
            else:
                baseline_macs += _baseline_batch_macs(model, task_id, batch)
    model.train(was_training)
    supported, total, coverage = _supported_parameter_coverage(model)
    if cell.model_id == "modular":
        profile = profile_modular_execution(
            cast(Any, model),
            task_token_counts=task_token_counts,
            expert_active_counts=expert_active_counts,
        ).to_dict()
    else:
        profile = {
            "active_total_macs": baseline_macs,
            "dense_total_macs": baseline_macs,
            "active_optional_macs": 0,
            "dense_optional_macs": 0,
            "parameter_coverage": coverage,
            "supported_parameters": supported,
            "total_parameters": total,
            "records": [],
        }
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_memory = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
    if device.type == "mps":
        peak_memory = max(peak_memory, int(torch.mps.current_allocated_memory()))
    return {
        "mac_accounting": "effective_valid_token_calls",
        "active_path_parameters": _active_path_parameters(model, cell),
        "profile": profile,
        "latency_ms": {
            "warmup_per_task": warmup_count,
            "measurements_per_task": measurement_count,
            "samples": len(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "peak_memory_bytes": peak_memory,
        "peak_memory_method": "process_max_rss_and_device_allocator",
    }


def _history_policy(
    model: nn.Module, cell: P3ExperimentCell, task: SmallGraphTask, device: torch.device
) -> Any:
    history: list[Tensor] = []

    def policy(observation: Tensor, reset: bool) -> Tensor:
        if reset:
            history.clear()
        history.append(observation.detach())
        inputs = torch.stack(history).unsqueeze(0).to(device)
        length = inputs.shape[1]
        batch = TaskBatch(
            inputs=inputs,
            targets=torch.full((1, length), -100, dtype=torch.long, device=device),
            valid_mask=torch.ones((1, length), dtype=torch.bool, device=device),
            loss_mask=torch.ones((1, length), dtype=torch.bool, device=device),
            episode_ids=torch.zeros((1, length), dtype=torch.long, device=device),
            metadata={
                "task_id": task.task_id,
                "task_version": task.task_version,
                "split": "test",
                "distribution": task.distribution,
            },
            auxiliary_targets={},
        )
        with torch.no_grad():
            output = _forward(model, cell, batch)
        logits = getattr(output, "logits", None)
        if not isinstance(logits, Tensor):
            raise TypeError("P3 rollout output does not expose logits")
        return logits[0, -1]

    return policy


def _representation(
    model: nn.Module,
    cell: P3ExperimentCell,
    batch: TaskBatch,
    output: object,
) -> Tensor:
    if cell.model_id == "modular":
        packet = getattr(output, "packet", None)
        value = getattr(packet, "representation", None)
        if not isinstance(value, Tensor):
            raise TypeError("modular output does not expose a representation packet")
        return value
    if isinstance(model, (SharedGRUBaseline, SharedTransformerBaseline)):
        return model.encode_representation(batch)
    if isinstance(model, GRUBaseline):
        encoded, _ = model.encoder(model.input_projection(batch.inputs))
        return cast(Tensor, encoded)
    if isinstance(model, SingleTaskTransformerV2):
        return model.encode_representation(batch.inputs, batch.valid_mask)
    if isinstance(model, TransformerBaseline):
        length = batch.sequence_length
        causal = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=batch.inputs.device), diagonal=1
        )
        return cast(
            Tensor,
            model.encoder(
                model.input_projection(batch.inputs),
                mask=causal,
                src_key_padding_mask=~batch.valid_mask,
            ),
        )
    raise TypeError(f"unsupported P3 representation model: {type(model).__name__}")


def _representation_metrics(
    train_features: Sequence[Tensor],
    train_inputs: Sequence[Tensor],
    train_labels: Sequence[Tensor],
    test_features: Sequence[Tensor],
    test_inputs: Sequence[Tensor],
    test_labels: Sequence[Tensor],
) -> dict[str, object]:
    if not all(
        values
        for values in (
            train_features,
            train_inputs,
            train_labels,
            test_features,
            test_inputs,
            test_labels,
        )
    ):
        return {"status": "undefined", "reason": "analysis split produced no probe events"}
    train_x = torch.cat(tuple(train_features), dim=0).detach().cpu()
    train_raw = torch.cat(tuple(train_inputs), dim=0).detach().cpu()
    train_y = torch.cat(tuple(train_labels), dim=0).detach().cpu()
    test_x = torch.cat(tuple(test_features), dim=0).detach().cpu()
    test_raw = torch.cat(tuple(test_inputs), dim=0).detach().cpu()
    test_y = torch.cat(tuple(test_labels), dim=0).detach().cpu()
    combined_x = torch.cat((train_x, test_x), dim=0)[:512]
    combined_raw = torch.cat((train_raw, test_raw), dim=0)[:512]
    return {
        "status": "ok",
        "comparison": "input_vs_hidden",
        "event_count": int(train_x.shape[0] + test_x.shape[0]),
        "train_event_count": int(train_x.shape[0]),
        "test_event_count": int(test_x.shape[0]),
        "linear_cka": linear_cka(combined_raw, combined_x),
        "rsa_spearman": rsa_spearman(combined_raw, combined_x),
        "linear_probe_accuracy": linear_probe_accuracy(train_x, train_y, test_x, test_y),
    }


def _evaluate_cell(
    model: nn.Module,
    cell: P3ExperimentCell,
    config: P3SuiteConfig,
    device: torch.device,
    directory: Path,
    suite_directory: Path,
    deadline: float,
) -> dict[str, object]:
    if cell.cell_type == "pilot":
        return {"validation_only": _validation_macro(model, cell, config, device)}
    task_ids = (cell.task_id,) if cell.task_id is not None else P3_TASK_ORDER
    records: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    representation_summaries: dict[str, object] = {}
    distribution_map = {
        "associative_recall.v1": ("capacity", "interference", "joint"),
        "delayed_rule_switch.v1": ("delay", "composition", "joint"),
        "small_graph.v1": ("scale", "topology", "joint"),
    }
    model.eval()
    for task_id in task_ids:
        task_summary: dict[str, object] = {}
        train_features: list[Tensor] = []
        train_inputs: list[Tensor] = []
        train_labels: list[Tensor] = []
        test_features: list[Tensor] = []
        test_inputs: list[Tensor] = []
        test_labels: list[Tensor] = []
        views = (("test", "v1", config.data.test), ("analysis", "v1", config.data.analysis))
        for split, distribution, size in views:
            task = _task(config, task_id, distribution)
            view_records: list[dict[str, object]] = []
            for start in range(0, size, config.budget.batch_size):
                _guard_suite_resources(suite_directory, deadline)
                indices = list(range(start, min(start + config.budget.batch_size, size)))
                batch = task.generate(cast(Any, split), indices, device=device)
                with torch.no_grad():
                    output = _forward(model, cell, batch)
                    if split == "analysis":
                        hidden = _representation(model, cell, batch, output)
                        event_mask = batch.loss_mask & batch.targets.ge(0)
                        midpoint = size // 2
                        train_rows = torch.tensor(
                            [index < midpoint for index in indices],
                            dtype=torch.bool,
                            device=device,
                        ).unsqueeze(1)
                        for row_mask, feature_store, input_store, label_store in (
                            (train_rows, train_features, train_inputs, train_labels),
                            (~train_rows, test_features, test_inputs, test_labels),
                        ):
                            selected = event_mask & row_mask
                            if selected.any():
                                feature_store.append(hidden[selected].detach())
                                input_store.append(batch.inputs[selected].detach())
                                label_store.append(batch.targets[selected].detach())
                view_records.extend(
                    p3_sample_records(
                        output,
                        batch,
                        run_seed=cell.seed,
                        model_id=cell.model_id,
                        variant_id=cell.variant_id,
                    )
                )
            records.extend(view_records)
            task_summary[f"{split}:{distribution}"] = len(view_records)
        for distribution in distribution_map[task_id]:
            task = _task(config, task_id, distribution)
            size = config.data.ood
            for start in range(0, size, config.budget.batch_size):
                _guard_suite_resources(suite_directory, deadline)
                indices = list(range(start, min(start + config.budget.batch_size, size)))
                batch = task.generate("ood", indices, device=device)
                with torch.no_grad():
                    output = _forward(model, cell, batch)
                records.extend(
                    p3_sample_records(
                        output,
                        batch,
                        run_seed=cell.seed,
                        model_id=cell.model_id,
                        variant_id=cell.variant_id,
                    )
                )
            task_summary[f"ood:{distribution}"] = size
        if task_id == "small_graph.v1":
            _guard_suite_resources(suite_directory, deadline)
            graph_task = cast(SmallGraphTask, _task(config, task_id))
            rollout = graph_task.rollout_records(
                _history_policy(model, cell, graph_task, device),
                "test",
                list(range(config.data.test)),
                device=device,
            )
            for record in rollout:
                record.update(
                    {
                        "model_id": cell.model_id,
                        "variant_id": cell.variant_id,
                        "seed": cell.seed,
                        "stratum": record["bootstrap_stratum"],
                    }
                )
            records.extend(rollout)
            task_summary["test:live_rollout"] = len(rollout)
            _guard_suite_resources(suite_directory, deadline)
        summaries[task_id] = task_summary
        representation_summaries[task_id] = _representation_metrics(
            train_features,
            train_inputs,
            train_labels,
            test_features,
            test_inputs,
            test_labels,
        )
    _write_jsonl(directory / "sample_records.jsonl", records)
    representation_report = {
        "schema_version": "p3-representation-analysis-v1",
        "split": "analysis",
        "selection_use": False,
        "tasks": representation_summaries,
    }
    _write_json(directory / "representation-analysis.json", representation_report)
    return {
        "record_count": len(records),
        "views": summaries,
        "representation_analysis": representation_report,
    }


def _initial_registry(config: P3SuiteConfig, run_id: str) -> dict[str, object]:
    return {
        "schema_version": "p3-suite-registry-v1",
        "run_id": run_id,
        "profile": config.profile,
        "qualification_only": config.qualification_only,
        "protocol_version": config.protocol_version,
        "config_hash": config.config_hash(),
        "matrix_hash": config.matrix_hash(),
        "status": "running",
        "cells": [
            {
                **cell.model_dump(mode="json"),
                "status": "PENDING",
                "artifact_dir": None,
                "error": None,
            }
            for cell in config.matrix()
        ],
    }


def _write_pilot_selection(
    config: P3SuiteConfig,
    directory: Path,
    *,
    git_commit: str,
    git_dirty: bool,
    qualification_fixture: bool = False,
) -> dict[str, object]:
    candidates: dict[str, list[dict[str, object]]] = {model: [] for model in _PILOT_PRESETS}
    for cell in (item for item in config.matrix() if item.cell_type == "pilot"):
        summary = json.loads(
            (directory / "cells" / cell.cell_id / "summary.json").read_text(encoding="utf-8")
        )
        training = cast(dict[str, object], summary["training"])
        aulc = training.get("validation_macro_aulc")
        last_loss = training.get("last_loss")
        if not isinstance(aulc, (int, float)) or not isinstance(last_loss, (int, float)):
            raise ValueError(f"pilot cell is missing finite selection metrics: {cell.cell_id}")
        candidates[cell.model_id].append(
            {
                "preset_id": cell.variant_id,
                "validation_macro_aulc": float(aulc),
                "final_loss": float(last_loss),
                "training_settings": training["training_settings"],
            }
        )
    selected: dict[str, str] = {}
    for model, values in candidates.items():
        ordered = sorted(
            values,
            key=lambda value: (
                -cast(float, value["validation_macro_aulc"]),
                cast(float, value["final_loss"]),
                str(value["preset_id"]),
            ),
        )
        selected[model] = str(ordered[0]["preset_id"])
    report: dict[str, object] = {
        "schema_version": "p3-pilot-selection-v1",
        "status": "QUALIFICATION_ONLY" if qualification_fixture else "PASSED",
        "qualification_only": qualification_fixture,
        "selection_rule": "validation_macro_aulc_desc,final_loss_asc,preset_id_asc",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_hash": config.config_hash(),
        "matrix_hash": config.matrix_hash(),
        "selected_presets": selected,
        "candidates": candidates,
    }
    filename = "pilot-selection-fixture.json" if qualification_fixture else "pilot-selection.json"
    _write_json(directory / filename, report)
    return report


def _qualification_diagnostics(
    config: P3SuiteConfig, device: torch.device, directory: Path
) -> dict[str, object]:
    """Exercise telemetry equality and bounded analysis paths outside scientific reports."""

    set_global_seed(7)
    first = cast(Any, _build_modular(config, 7)).to(device)
    second = cast(Any, _build_modular(config, 7)).to(device)
    second.load_state_dict(first.state_dict())
    task = _task(config, "delayed_rule_switch.v1")
    batch = task.generate("train", [0, 1], device=device)
    first.train()
    second.train()
    first_optimizer = torch.optim.AdamW(first.parameters(), lr=3e-4)
    second_optimizer = torch.optim.AdamW(second.parameters(), lr=3e-4)
    outputs = []
    gradients: list[dict[str, Tensor]] = []
    weights = {"primary": 1.0, **P2LossWeights().by_loss_name()}
    for model, optimizer, telemetry in (
        (first, first_optimizer, False),
        (second, second_optimizer, True),
    ):
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_batch(batch, phase="train", telemetry_enabled=telemetry)
        loss, _ = modular_training_loss(output, batch, weights=weights)
        loss.backward()  # type: ignore[no-untyped-call]
        gradients.append(
            {
                name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        )
        optimizer.step()
        outputs.append(output)
    tolerance = 1e-5 if device.type == "mps" else 0.0
    logits_difference = float(
        (outputs[0].logits.detach() - outputs[1].logits.detach()).abs().max().cpu()
    )
    gradient_difference = max(
        float((gradients[0][name] - gradients[1][name]).abs().max().cpu()) for name in gradients[0]
    )
    parameter_difference = max(
        float((left - right).abs().max().cpu())
        for left, right in zip(
            first.state_dict().values(), second.state_dict().values(), strict=True
        )
        if left.is_floating_point()
    )
    if max(logits_difference, gradient_difference, parameter_difference) > tolerance:
        raise ValueError("P3 telemetry qualification exceeded numerical tolerance")

    generator = torch.Generator(device="cpu").manual_seed(7)
    first_representation = torch.randn((32, 8), generator=generator)
    second_representation = first_representation @ torch.randn((8, 8), generator=generator)
    labels = torch.arange(32).remainder(2)
    analysis = {
        "linear_cka": linear_cka(first_representation, second_representation),
        "rsa_spearman": rsa_spearman(first_representation, second_representation),
        "linear_probe_accuracy": linear_probe_accuracy(
            first_representation[:16],
            labels[:16],
            first_representation[16:],
            labels[16:],
        ),
    }
    result = {
        "schema_version": "p3-qualification-diagnostics-v1",
        "qualification_only": True,
        "device": str(device),
        "telemetry_equivalence": {
            "logits_max_abs": logits_difference,
            "gradient_max_abs": gradient_difference,
            "parameter_max_abs": parameter_difference,
            "tolerance": tolerance,
        },
        "bounded_analysis": analysis,
    }
    _write_json(directory / "qualification-diagnostics.json", result)
    return result


def execute_p3_suite(config: P3SuiteConfig) -> dict[str, object]:
    """Run or resume every pre-registered cell without skipping failures."""

    set_global_seed(config.seeds[0])
    device = resolve_device(config.device)
    git_commit, git_dirty = _repository_state()
    if config.expected_git_commit is not None and config.expected_git_commit != git_commit:
        raise ValueError("P3 expected_git_commit does not match the checked-out repository")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = config.run_id or f"p3-{config.profile}-{timestamp}"
    directory = config.output_root / run_id
    registry_path = directory / "registry.json"
    if directory.exists() and not registry_path.exists():
        raise FileExistsError(f"P3 run directory exists without a registry: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if registry.get("config_hash") != config.config_hash():
            raise ValueError("existing P3 registry configuration hash does not match")
        if registry.get("matrix_hash") != config.matrix_hash():
            raise ValueError("existing P3 experiment matrix hash does not match")
    else:
        registry = _initial_registry(config, run_id)
        _write_json(directory / "config.json", config.model_dump(mode="json"))
        _write_json(registry_path, registry)
    started = time.perf_counter()
    prior_wall_clock = float(registry.get("wall_clock_seconds", 0.0))
    heartbeat_path = directory / "heartbeat.json"
    if heartbeat_path.is_file():
        heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
        heartbeat_elapsed = heartbeat.get("suite_elapsed_seconds")
        if isinstance(heartbeat_elapsed, (int, float)):
            prior_wall_clock = max(prior_wall_clock, float(heartbeat_elapsed))
    if not math.isfinite(prior_wall_clock) or prior_wall_clock < 0.0:
        raise ValueError("existing P3 registry wall-clock value is invalid")
    limit_seconds = config.budget.wall_clock_hours * 3600.0
    deadline = started + max(limit_seconds - prior_wall_clock, 0.0)
    registry["status"] = "running"
    registry.pop("resource_limit_reason", None)

    def persist_registry() -> None:
        registry["wall_clock_seconds"] = prior_wall_clock + time.perf_counter() - started
        _write_json(registry_path, registry)

    persist_registry()
    cells = config.matrix()
    registry_cells = cast(list[dict[str, object]], registry["cells"])
    for cursor, cell in enumerate(cells):
        entry = registry_cells[cursor]
        if entry["cell_id"] != cell.cell_id:
            raise ValueError("P3 registry cell ordering does not match the frozen matrix")
        if entry["status"] == "COMPLETED":
            continue
        if prior_wall_clock + time.perf_counter() - started >= limit_seconds:
            entry["status"] = "RESOURCE_LIMIT"
            registry["status"] = "resource_limit"
            persist_registry()
            break
        if shutil.disk_usage(directory).free < 20 * GIB:
            entry["status"] = "RESOURCE_LIMIT"
            registry["status"] = "resource_limit"
            registry["resource_limit_reason"] = "free disk space fell below 20 GiB"
            persist_registry()
            break
        if (directory / "STOP").exists():
            registry["status"] = "stopped"
            persist_registry()
            break
        cell_directory = directory / "cells" / cell.cell_id
        cell_directory.mkdir(parents=True, exist_ok=True)
        entry["status"] = "RUNNING"
        entry["artifact_dir"] = str(cell_directory.relative_to(directory))
        persist_registry()
        try:
            cell_started = time.perf_counter()
            set_global_seed(cell.seed)
            model, matching = _build_model(config, cell, device)
            if cell.cell_type == "control":
                parent_directory = (
                    directory / "cells" / f"shared__shared__modular__full__s{cell.seed}__all"
                )
                parent = _selected_checkpoint(parent_directory)
                if parent.exists():
                    payload = torch.load(parent, map_location=device, weights_only=False)
                    model.load_state_dict(payload["model_state"])
                else:
                    raise FileNotFoundError(f"P3 control parent checkpoint is missing: {parent}")
                _apply_control_after_parent_load(model, cell)
            training = (
                _train_cell(
                    model,
                    cell,
                    config,
                    device,
                    cell_directory,
                    cursor,
                    directory,
                    deadline,
                )
                if cell.max_steps
                else {"steps": 0, "wall_clock_seconds": 0.0}
            )
            if training.get("stopped"):
                entry["status"] = "PENDING"
                registry["status"] = "stopped"
                persist_registry()
                break
            if training.get("resource_limited"):
                entry["status"] = "RESOURCE_LIMIT"
                registry["status"] = "resource_limit"
                registry["resource_limit_reason"] = (
                    "wall-clock or free-disk limit reached after a safe checkpoint"
                )
                persist_registry()
                break
            evaluation = _evaluate_cell(
                model,
                cell,
                config,
                device,
                cell_directory,
                directory,
                deadline,
            )
            cost = _profile_cell_cost(model, cell, config, device, directory, deadline)
            summary = {
                "schema_version": "p3-cell-summary-v1",
                "cell": cell.model_dump(mode="json"),
                "matching": matching,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
                "active_path_parameters": cost["active_path_parameters"],
                "training": training,
                "evaluation": evaluation,
                "cost": cost,
                "wall_clock_seconds": time.perf_counter() - cell_started,
            }
            _write_json(cell_directory / "summary.json", summary)
            entry["status"] = "COMPLETED"
            entry["error"] = None
        except _P3ResourceLimit as error:
            entry["status"] = "RESOURCE_LIMIT"
            entry["error"] = {"type": type(error).__name__, "message": str(error)}
            registry["status"] = "resource_limit"
            registry["resource_limit_reason"] = str(error)
            persist_registry()
            break
        except Exception as error:
            entry["status"] = "FAILED"
            entry["error"] = {"type": type(error).__name__, "message": str(error)}
            _write_json(
                cell_directory / "failure.json",
                {
                    "schema_version": "p3-failure-v1",
                    "cell_id": cell.cell_id,
                    "error": entry["error"],
                    "git_commit": git_commit,
                    "config_hash": config.config_hash(),
                    "protocol_hash": PROTOCOL_HASH,
                },
            )
        persist_registry()
    statuses = [str(entry["status"]) for entry in registry_cells]
    if all(status == "COMPLETED" for status in statuses):
        if config.profile == "qualification":
            registry["status"] = "qualification_passed"
        elif config.profile == "pilot":
            registry["status"] = "pilot_passed"
        else:
            registry["status"] = "completed"
    elif registry.get("status") == "running":
        if config.profile == "qualification":
            registry["status"] = "qualification_failed"
        elif config.profile == "pilot":
            registry["status"] = "pilot_failed"
        else:
            registry["status"] = "completed_with_failures"
    if config.profile == "qualification" and registry["status"] == "qualification_passed":
        registry["qualification_diagnostics"] = _qualification_diagnostics(
            config, device, directory
        )
        registry["pilot_selection_fixture"] = _write_pilot_selection(
            config,
            directory,
            git_commit=git_commit,
            git_dirty=git_dirty,
            qualification_fixture=True,
        )
        fixture_path = directory / "network-mvp-fixture"
        if not fixture_path.exists():
            fixture_model = cast(Any, _build_modular(config, 7)).to(device)
            parent = directory / "cells" / "shared__shared__modular__full__s7__all"
            parent = _selected_checkpoint(parent)
            payload = torch.load(parent, map_location=device, weights_only=False)
            fixture_model.load_state_dict(payload["model_state"])
            create_network_mvp_bundle(
                fixture_path,
                model=fixture_model,
                source_commit=config.expected_git_commit or "qualification-fixture",
                gate_status="QUALIFICATION_ONLY",
                qualification_only=True,
            )
    if config.profile == "pilot" and registry["status"] == "pilot_passed":
        registry["pilot_selection"] = _write_pilot_selection(
            config,
            directory,
            git_commit=git_commit,
            git_dirty=git_dirty,
        )
    registry["wall_clock_seconds"] = prior_wall_clock + time.perf_counter() - started
    registry["completed_cells"] = statuses.count("COMPLETED")
    registry["total_cells"] = len(statuses)
    if config.profile == "qualification":
        _write_json(
            directory / "qualification-report.json",
            {
                "schema_version": "p3-qualification-report-v1",
                "qualification_only": True,
                "status": "PASSED" if registry["status"] == "qualification_passed" else "FAILED",
                "run_id": run_id,
                "config_hash": config.config_hash(),
                "matrix_hash": config.matrix_hash(),
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "device": str(device),
                "completed_cells": registry["completed_cells"],
                "total_cells": registry["total_cells"],
            },
        )
    artifacts = {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"registry.json", "heartbeat.json"}
    }
    registry["artifacts"] = artifacts
    _write_json(registry_path, registry)
    return {
        "run_id": run_id,
        "artifact_dir": str(directory),
        "status": registry["status"],
        "completed_cells": registry["completed_cells"],
        "total_cells": registry["total_cells"],
    }


def verify_p3_run(directory: Path) -> dict[str, object]:
    registry_path = directory / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    config = P3SuiteConfig.model_validate(
        json.loads((directory / "config.json").read_text(encoding="utf-8"))
    )
    if registry.get("config_hash") != config.config_hash():
        raise ValueError("P3 registry config hash does not match frozen config")
    if registry.get("matrix_hash") != config.matrix_hash():
        raise ValueError("P3 registry matrix hash does not match frozen config")
    cells = cast(list[dict[str, object]], registry.get("cells"))
    ids = [str(cell["cell_id"]) for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("P3 registry contains duplicate cells")
    expected_ids = [cell.cell_id for cell in config.matrix()]
    if ids != expected_ids:
        raise ValueError("P3 registry does not match the pre-registered matrix")
    artifacts = cast(dict[str, str], registry.get("artifacts", {}))
    for relative, expected in artifacts.items():
        path = directory / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"P3 artifact checksum mismatch: {relative}")
    actual_artifacts = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"registry.json", "heartbeat.json"}
    }
    if actual_artifacts != set(artifacts):
        raise ValueError("P3 run contains missing or unregistered artifacts")
    sample_keys: set[tuple[object, ...]] = set()
    sample_count = 0
    for cell in cells:
        artifact_dir = cell.get("artifact_dir")
        if not isinstance(artifact_dir, str):
            continue
        cell_directory = directory / artifact_dir
        summary_path = cell_directory / "summary.json"
        if cell.get("status") == "COMPLETED":
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("schema_version") != "p3-cell-summary-v1":
                raise ValueError(f"P3 cell summary schema is invalid: {cell['cell_id']}")
        records_path = cell_directory / "sample_records.jsonl"
        if not records_path.is_file():
            continue
        for line in records_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            key = (
                cell["cell_id"],
                record.get("schema_version"),
                record.get("task_id"),
                record.get("split"),
                record.get("distribution"),
                record.get("sample_index"),
                record.get("stratum"),
            )
            if key in sample_keys:
                raise ValueError(f"P3 sample record is duplicated: {key}")
            sample_keys.add(key)
            sample_count += 1
    missing = [cell["cell_id"] for cell in cells if cell["status"] != "COMPLETED"]
    return {
        "run_id": registry["run_id"],
        "status": registry["status"],
        "cells": len(cells),
        "missing_cells": missing,
        "checksums_ok": True,
        "sample_records": sample_count,
        "registered_artifacts": len(artifacts),
    }


__all__ = ["execute_p3_suite", "verify_p3_run"]
