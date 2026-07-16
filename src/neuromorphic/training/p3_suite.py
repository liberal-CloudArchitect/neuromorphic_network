"""Execute the qualification or formal P3 experiment matrix."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from neuromorphic.core.registry import EPISODIC_MEMORY, WORKING_MEMORY
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
from neuromorphic.training.modular_metrics import modular_task_metrics, modular_training_loss
from neuromorphic.training.modular_trainer import build_modular_network
from neuromorphic.training.p2_config import P2LossWeights, P2SuiteConfig
from neuromorphic.training.p3_baselines import (
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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_profile(config: P3SuiteConfig) -> Literal["smoke", "qualification"]:
    return "smoke" if config.profile == "qualification" else "qualification"


def _task(config: P3SuiteConfig, task_id: str, distribution: str = "v1") -> SequenceTask:
    return create_task(task_id, profile=_task_profile(config), distribution=distribution)


def _build_modular(config: P3SuiteConfig, seed: int) -> nn.Module:
    p2 = P2SuiteConfig(profile="ci", seed=seed, device=config.device)
    return build_modular_network(p2)


def _build_model(
    config: P3SuiteConfig,
    cell: P3ExperimentCell,
    device: torch.device,
) -> tuple[nn.Module, str]:
    modular = _build_modular(config, cell.seed)
    target = sum(parameter.numel() for parameter in modular.parameters())
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
        elif cell.intervention.target == WORKING_MEMORY:
            for parameter in registry.get(WORKING_MEMORY).parameters():
                parameter.requires_grad_(False)
        elif cell.intervention.target == "sensory_encoder.v1" and cell.intervention.operation in {
            "frozen_random",
            "shallow",
        }:
            for parameter in registry.get("sensory_encoder.v1").parameters():
                parameter.requires_grad_(False)
    return model, target_label


def _intervention_arguments(cell: P3ExperimentCell) -> tuple[str, tuple[str, ...]]:
    routing = (
        cell.intervention.operation if cell.intervention.target == "sparse_router.v1" else "learned"
    )
    disabled: tuple[str, ...] = ()
    if cell.intervention.target in {
        "episodic_memory.v1",
        "working_memory.v1",
    }:
        disabled = (cell.intervention.target,)
    return routing, disabled


def _forward(model: nn.Module, cell: P3ExperimentCell, batch: TaskBatch) -> object:
    if cell.model_id == "modular":
        routing, disabled = _intervention_arguments(cell)
        return cast(Any, model).forward_batch(
            batch,
            phase="train" if model.training else "evaluate",
            routing_mode=routing,
            disabled_experts=disabled,
        )
    if cell.regime in {"shared", "continual"}:
        return model(batch)
    return model(batch.inputs, batch.valid_mask)


def _training_loss(
    model: nn.Module, cell: P3ExperimentCell, output: object, batch: TaskBatch
) -> tuple[Tensor, dict[str, float]]:
    if cell.model_id == "modular":
        weights = {"primary": 1.0, **P2LossWeights().by_loss_name()}
        if cell.intervention.target == "predictive_adapter.v1":
            weights["predictive.next_state"] = 0.0
        return modular_training_loss(cast(Any, output), batch, weights=weights)
    return masked_task_loss(cast(Any, output), batch, auxiliary_weight=0.1)


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
    )


def _train_cell(
    model: nn.Module,
    cell: P3ExperimentCell,
    config: P3SuiteConfig,
    device: torch.device,
    directory: Path,
    matrix_cursor: int,
    suite_directory: Path,
) -> dict[str, object]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    task_ids = (cell.task_id,) if cell.task_id is not None else P3_TASK_ORDER
    samplers = {
        task_id: IndexSampler.create(config.data.train, cell.seed + index * 10_000)
        for index, task_id in enumerate(task_ids)
    }
    checkpoint = directory / "checkpoint.pt"
    best_checkpoint = directory / "best.pt"
    step = 0
    task_steps = {task_id: 0 for task_id in task_ids}
    best_metrics: dict[str, float] = {}
    stale = 0
    curves: dict[str, list[tuple[int, float]]] = {task_id: [] for task_id in task_ids}
    if checkpoint.exists():
        restored = load_p3_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            expected_cell_id=cell.cell_id,
            expected_config_hash=config.config_hash(),
            expected_protocol_hash=PROTOCOL_HASH,
        )
        step = restored.global_step
        task_steps = dict(restored.task_steps)
        best_metrics = dict(restored.best_metrics)
        stale = restored.stale_evaluations
        for task_id, sampler_state in restored.sampler_states.items():
            samplers[task_id].load_state_dict(sampler_state)
    started = time.perf_counter()
    while step < cell.max_steps:
        if (suite_directory / "STOP").exists():
            break
        task_id = _task_sequence(cell, step)
        task = _task(config, task_id)
        indices = samplers[task_id].next(min(config.budget.batch_size, config.data.train))
        batch = task.generate("train", indices, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = _forward(model, cell, batch)
        loss, parts = _training_loss(model, cell, output, batch)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(
            trainable, config.optimizer.gradient_clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        ensure_finite_training_state(loss=loss, model=model, metrics=parts)
        step += 1
        task_steps[task_id] += 1
        if step % config.budget.validation_interval == 0 or step == cell.max_steps:
            current = _validation_macro(model, cell, config, device)
            analysis = _validation_macro(model, cell, config, device, split="analysis")
            for name, value in analysis.items():
                point = (task_steps.get(name, step), value)
                curve = curves.setdefault(name, [])
                if curve and curve[-1][0] == point[0]:
                    curve[-1] = point
                else:
                    curve.append(point)
            macro = sum(current.values()) / len(current)
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
                    ),
                )
            else:
                stale += 1
        if step % config.budget.checkpoint_interval == 0 or step == cell.max_steps:
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
                ),
            )
            _write_json(
                suite_directory / "heartbeat.json",
                {
                    "cell_id": cell.cell_id,
                    "step": step,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "free_bytes": shutil.disk_usage(suite_directory).free,
                },
            )
            if shutil.disk_usage(suite_directory).free < 20 * GIB:
                raise OSError("P3 free disk space fell below 20 GiB")
        if stale >= config.budget.patience:
            break
    if best_checkpoint.exists():
        load_p3_checkpoint(
            best_checkpoint,
            model=model,
            optimizer=optimizer,
            expected_cell_id=cell.cell_id,
            expected_config_hash=config.config_hash(),
            expected_protocol_hash=PROTOCOL_HASH,
        )
    aulc = {
        task_id: normalized_aulc(
            points or [(0, 0.0)],
            maximum_step=max(
                cell.max_steps if cell.task_id is not None else cell.max_steps // 3,
                1,
            ),
        )
        for task_id, points in curves.items()
    }
    return {
        "steps": step,
        "task_steps": task_steps,
        "best_metrics": best_metrics,
        "stale_evaluations": stale,
        "analysis_aulc": aulc,
        "wall_clock_seconds": time.perf_counter() - started,
        "stopped": step < cell.max_steps and (suite_directory / "STOP").exists(),
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


def _evaluate_cell(
    model: nn.Module,
    cell: P3ExperimentCell,
    config: P3SuiteConfig,
    device: torch.device,
    directory: Path,
) -> dict[str, object]:
    if cell.cell_type == "pilot":
        return {"validation_only": _validation_macro(model, cell, config, device)}
    task_ids = (cell.task_id,) if cell.task_id is not None else P3_TASK_ORDER
    records: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    distribution_map = {
        "associative_recall.v1": ("capacity", "interference", "joint"),
        "delayed_rule_switch.v1": ("delay", "composition", "joint"),
        "small_graph.v1": ("scale", "topology", "joint"),
    }
    model.eval()
    for task_id in task_ids:
        task_summary: dict[str, object] = {}
        views = (("test", "v1", config.data.test), ("analysis", "v1", config.data.analysis))
        for split, distribution, size in views:
            task = _task(config, task_id, distribution)
            view_records: list[dict[str, object]] = []
            for start in range(0, size, config.budget.batch_size):
                indices = list(range(start, min(start + config.budget.batch_size, size)))
                batch = task.generate(cast(Any, split), indices, device=device)
                with torch.no_grad():
                    output = _forward(model, cell, batch)
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
        summaries[task_id] = task_summary
    _append_jsonl(directory / "sample_records.jsonl", records)
    return {"record_count": len(records), "views": summaries}


def _initial_registry(config: P3SuiteConfig, run_id: str) -> dict[str, object]:
    return {
        "schema_version": "p3-suite-registry-v1",
        "run_id": run_id,
        "profile": config.profile,
        "qualification_only": config.qualification_only,
        "protocol_version": config.protocol_version,
        "config_hash": config.config_hash(),
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
    else:
        registry = _initial_registry(config, run_id)
        _write_json(directory / "config.json", config.model_dump(mode="json"))
        _write_json(registry_path, registry)
    started = time.perf_counter()
    limit_seconds = config.budget.wall_clock_hours * 3600.0
    cells = config.matrix()
    registry_cells = cast(list[dict[str, object]], registry["cells"])
    for cursor, cell in enumerate(cells):
        entry = registry_cells[cursor]
        if entry["cell_id"] != cell.cell_id:
            raise ValueError("P3 registry cell ordering does not match the frozen matrix")
        if entry["status"] == "COMPLETED":
            continue
        if time.perf_counter() - started >= limit_seconds:
            entry["status"] = "RESOURCE_LIMIT"
            registry["status"] = "resource_limit"
            _write_json(registry_path, registry)
            break
        if (directory / "STOP").exists():
            registry["status"] = "stopped"
            _write_json(registry_path, registry)
            break
        cell_directory = directory / "cells" / cell.cell_id
        cell_directory.mkdir(parents=True, exist_ok=True)
        entry["status"] = "RUNNING"
        entry["artifact_dir"] = str(cell_directory.relative_to(directory))
        _write_json(registry_path, registry)
        try:
            set_global_seed(cell.seed)
            model, matching = _build_model(config, cell, device)
            if cell.cell_type == "control":
                parent = (
                    directory
                    / "cells"
                    / f"shared__shared__modular__full__s{cell.seed}__all"
                    / "checkpoint.pt"
                )
                if parent.exists():
                    payload = torch.load(parent, map_location=device, weights_only=False)
                    model.load_state_dict(payload["model_state"])
            training = (
                _train_cell(
                    model,
                    cell,
                    config,
                    device,
                    cell_directory,
                    cursor,
                    directory,
                )
                if cell.max_steps
                else {"steps": 0, "wall_clock_seconds": 0.0}
            )
            if training.get("stopped"):
                registry["status"] = "stopped"
                _write_json(registry_path, registry)
                break
            evaluation = _evaluate_cell(model, cell, config, device, cell_directory)
            summary = {
                "schema_version": "p3-cell-summary-v1",
                "cell": cell.model_dump(mode="json"),
                "matching": matching,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
                "training": training,
                "evaluation": evaluation,
            }
            _write_json(cell_directory / "summary.json", summary)
            entry["status"] = "COMPLETED"
            entry["error"] = None
        except BaseException as error:
            entry["status"] = "FAILED"
            entry["error"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(registry_path, registry)
    statuses = [str(entry["status"]) for entry in registry_cells]
    if all(status == "COMPLETED" for status in statuses):
        registry["status"] = "qualification_passed" if config.qualification_only else "completed"
    elif registry.get("status") == "running":
        registry["status"] = (
            "qualification_failed" if config.qualification_only else "completed_with_failures"
        )
    if config.qualification_only and registry["status"] == "qualification_passed":
        registry["qualification_diagnostics"] = _qualification_diagnostics(
            config, device, directory
        )
        fixture_path = directory / "network-mvp-fixture"
        if not fixture_path.exists():
            fixture_model = cast(Any, _build_modular(config, 7)).to(device)
            parent = (
                directory / "cells" / "shared__shared__modular__full__s7__all" / "checkpoint.pt"
            )
            payload = torch.load(parent, map_location=device, weights_only=False)
            fixture_model.load_state_dict(payload["model_state"])
            create_network_mvp_bundle(
                fixture_path,
                model=fixture_model,
                source_commit=config.expected_git_commit or "qualification-fixture",
                gate_status="QUALIFICATION_ONLY",
                qualification_only=True,
            )
    artifacts = {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"registry.json", "heartbeat.json"}
    }
    registry["artifacts"] = artifacts
    registry["wall_clock_seconds"] = time.perf_counter() - started
    registry["completed_cells"] = statuses.count("COMPLETED")
    registry["total_cells"] = len(statuses)
    _write_json(registry_path, registry)
    if config.qualification_only:
        _write_json(
            directory / "qualification-report.json",
            {
                "schema_version": "p3-qualification-report-v1",
                "qualification_only": True,
                "status": "PASSED" if registry["status"] == "qualification_passed" else "FAILED",
                "run_id": run_id,
                "config_hash": config.config_hash(),
                "completed_cells": registry["completed_cells"],
                "total_cells": registry["total_cells"],
            },
        )
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
    cells = cast(list[dict[str, object]], registry.get("cells"))
    ids = [str(cell["cell_id"]) for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("P3 registry contains duplicate cells")
    for relative, expected in cast(dict[str, str], registry.get("artifacts", {})).items():
        path = directory / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"P3 artifact checksum mismatch: {relative}")
    missing = [cell["cell_id"] for cell in cells if cell["status"] != "COMPLETED"]
    return {
        "run_id": registry["run_id"],
        "status": registry["status"],
        "cells": len(cells),
        "missing_cells": missing,
        "checksums_ok": True,
    }


__all__ = ["execute_p3_suite", "verify_p3_run"]
