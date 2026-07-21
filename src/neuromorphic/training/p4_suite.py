"""Execute and verify the pre-registered P4 experiment suites.

The runner deliberately keeps scientific selection, process control, and artifact
verification separate.  It never invents a result for a matrix cell: a cell is
``COMPLETED`` only after training (when registered), evaluation, and an atomic
summary write have all succeeded.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neuromorphic.core.contracts import ModuleContext, trusted_internal_execution
from neuromorphic.core.network_state import NetworkState
from neuromorphic.core.registry import (
    P4_OPTIONAL_EXPERT_IDS,
    PREDICTIVE_ADAPTER_V2,
    SPARSE_ROUTER_V2,
)
from neuromorphic.evaluation.p3_records import as_baseline_output, p3_sample_records
from neuromorphic.evaluation.p3_statistics import (
    adjust_family,
    normalized_aulc,
    paired_hierarchical_bootstrap,
)
from neuromorphic.evaluation.p4_statistics import small_graph_chance_level
from neuromorphic.tasks import SmallGraphTask, TaskBatch, create_task
from neuromorphic.tasks.base import SequenceTask
from neuromorphic.tasks.control import task_control_from_batch
from neuromorphic.telemetry.events_v2 import TelemetryV2Event
from neuromorphic.training.baselines import GRUBaseline, TransformerBaseline
from neuromorphic.training.config import resolve_device
from neuromorphic.training.metrics import (
    ensure_finite_training_state,
    masked_task_loss,
    task_metrics,
)
from neuromorphic.training.p3_baselines import (
    SharedGRUBaseline,
    SharedTransformerBaseline,
    SingleTaskTransformerV2,
    select_shared_parameter_match,
)
from neuromorphic.training.p4_checkpoint import (
    P4CheckpointState,
    load_p4_checkpoint,
    save_p4_checkpoint,
)
from neuromorphic.training.p4_config import (
    P4_PILOT_PRESETS,
    P4_TASK_ORDER,
    P4ExperimentCell,
    P4SuiteConfig,
)
from neuromorphic.training.reproducibility import set_global_seed
from neuromorphic.training.trainer import IndexSampler

GIB = 1024**3
_PROTOCOL_PATH = Path(__file__).resolve().parents[3] / "docs" / "p4_implementation_spec.md"
PROTOCOL_HASH = hashlib.sha256(_PROTOCOL_PATH.read_bytes()).hexdigest()
_OOD_DISTRIBUTIONS = {
    "associative_recall.v1": ("capacity", "interference", "joint"),
    "delayed_rule_switch.v1": ("delay", "composition", "joint"),
    "small_graph.v1": ("scale", "topology", "joint"),
}
_SHARED_MATCH: dict[tuple[str, int], tuple[nn.Module, str]] = {}


class P4ResourceLimit(RuntimeError):
    """A safe, resumable suite resource stop rather than a failed experiment."""


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
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _numeric(mapping: Mapping[str, object], key: str) -> float:
    """Read a numeric summary field without accepting booleans or strings."""

    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _repository_state() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    )
    return commit, dirty


def _task(config: P4SuiteConfig, task_id: str, distribution: str = "v1") -> SequenceTask:
    profile: Literal["smoke", "qualification"] = (
        "smoke" if config.profile == "qualification" else "qualification"
    )
    return create_task(task_id, profile=profile, distribution=distribution, namespace="p4")


def _freeze_unused_modular_paths(model: nn.Module, task_id: str) -> None:
    adapters = getattr(getattr(model, "boundary_adapters", None), "adapters", ())
    active = P4_TASK_ORDER.index(task_id)
    for index, adapter in enumerate(adapters):
        if index != active:
            for parameter in adapter.parameters():
                parameter.requires_grad_(False)
    registry = getattr(model, "registry", None)
    if registry is not None:
        selector = registry.get("action_selector.v1")
        name = task_id.removesuffix(".v1")
        for head_name, head in selector.heads.items():
            if head_name != name:
                for parameter in head.parameters():
                    parameter.requires_grad_(False)


def _build_model(
    config: P4SuiteConfig, cell: P4ExperimentCell, device: torch.device
) -> tuple[nn.Module, str]:
    from neuromorphic.modules.network_v2 import ModularBrainNetworkV2

    set_global_seed(cell.seed)
    modular = ModularBrainNetworkV2()
    if cell.regime == "per_task":
        if cell.task_id is None:
            raise ValueError("per-task P4 cell requires task_id")
        _freeze_unused_modular_paths(modular, cell.task_id)
    target = sum(parameter.numel() for parameter in modular.parameters() if parameter.requires_grad)
    if cell.model_id == "modular-v2":
        model: nn.Module = modular
        match = "modular-v2"
    elif cell.regime in {"shared", "continual"}:
        key = (cell.model_id, target)
        cached = _SHARED_MATCH.get(key)
        if cached is None:
            shared_match = select_shared_parameter_match(cell.model_id, target)
            cached = (
                shared_match.model,
                f"parameter-match:{shared_match.relative_error:.6f}",
            )
            _SHARED_MATCH[key] = cached
        template, match = cached
        if isinstance(template, SharedGRUBaseline):
            model = SharedGRUBaseline(template.hidden_size, template.layers)
        else:
            transformer = cast(SharedTransformerBaseline, template)
            model = SharedTransformerBaseline(
                transformer.hidden_size,
                transformer.layers,
                transformer.heads,
                transformer.feedforward_size,
            )
    else:
        if cell.task_id is None:
            raise ValueError("per-task baseline requires task_id")
        task = _task(config, cell.task_id)
        # Formal fairness is checked against the active modular path.  Reuse the
        # deterministic P3 search rather than introducing another matching rule.
        from neuromorphic.training.baselines import select_parameter_matched_baseline

        selected_module, result = select_parameter_matched_baseline(
            kind=cell.model_id,
            input_dim=task.input_dim,
            num_classes=task.num_classes,
            target=target,
            auxiliary_classes=16 if cell.task_id == "small_graph.v1" else None,
        )
        if cell.model_id == "transformer":
            selected_module = SingleTaskTransformerV2(
                input_dim=task.input_dim,
                num_classes=task.num_classes,
                hidden_size=result.hidden_size,
                layers=2,
                heads=4,
                feedforward_size=cast(int, result.feedforward_size),
                auxiliary_classes=16 if cell.task_id == "small_graph.v1" else None,
            )
        model = cast(GRUBaseline | TransformerBaseline, selected_module)
        match = f"parameter-match:{result.relative_error:.6f}"
    model.to(device)
    return model, match


def _intervention_kwargs(cell: P4ExperimentCell, *, evaluation: bool) -> dict[str, object]:
    spec = cell.intervention
    if (spec.phase == "train" and evaluation) or (spec.phase == "evaluate" and not evaluation):
        operation = "none"
    else:
        operation = spec.operation
    result: dict[str, object] = {
        "predictor_mode": "full",
        "routing_mode": "learned",
        "episodic_off": False,
        "working_reset_every_step": False,
        "encoder_mode": "full",
        "selector_mode": "integrated",
    }
    mapping: dict[str, tuple[str, object]] = {
        "predictor_off": ("predictor_mode", "off"),
        "loss_zero": ("predictor_mode", "loss_zero"),
        "feedback_zero": ("predictor_mode", "feedback_zero"),
        "acute_feedback_off": ("predictor_mode", "acute_feedback_off"),
        "shuffle_forecast": ("predictor_mode", "shuffle_forecast"),
        "dense_memory": ("routing_mode", "dense"),
        "legacy_capacity": ("routing_mode", "legacy_capacity"),
        "episodic_off": ("episodic_off", True),
        "working_reset": ("working_reset_every_step", True),
        "direct_head": ("selector_mode", "direct"),
        "frozen_random_encoder": ("encoder_mode", "full"),
        "shallow_encoder": ("encoder_mode", "shallow"),
    }
    if operation != "none":
        try:
            name, value = mapping[operation]
        except KeyError as error:
            raise ValueError(f"unsupported P4 intervention: {operation}") from error
        result[name] = value
    return result


def _forward(
    model: nn.Module,
    cell: P4ExperimentCell,
    batch: TaskBatch,
    *,
    evaluation: bool,
    state: NetworkState | None = None,
) -> object:
    if cell.model_id == "modular-v2":
        return cast(Any, model).forward_batch(
            batch,
            state=state,
            phase="evaluate" if evaluation else "train",
            telemetry_enabled=False,
            **_intervention_kwargs(cell, evaluation=evaluation),
        )
    if cell.regime in {"shared", "continual"}:
        return model(batch)
    return model(batch.inputs, batch.valid_mask)


def _primary_loss(logits: Tensor, batch: TaskBatch) -> Tensor:
    supervised = batch.loss_mask
    if not supervised.any():
        raise ValueError("batch has no supervised positions")
    optimal = batch.auxiliary_targets.get("optimal_action_mask")
    if optimal is None:
        return F.cross_entropy(logits[supervised], batch.targets[supervised])
    allowed = F.log_softmax(logits, dim=-1).masked_fill(~optimal, -torch.inf)
    return -torch.logsumexp(allowed[supervised], dim=-1).mean()


def _loss(
    model: nn.Module,
    cell: P4ExperimentCell,
    output: object,
    batch: TaskBatch,
    *,
    temporal_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    if cell.model_id != "modular-v2":
        return masked_task_loss(as_baseline_output(output), batch, auxiliary_weight=0.1)
    logits = cast(Any, output).logits
    primary = _primary_loss(logits, batch)
    auxiliary = getattr(output, "auxiliary_losses", {})
    total = primary
    parts = {"loss/primary": float(primary.detach().cpu())}
    weights = {
        "episodic.retrieval": 0.1,
        "episodic.separation": 0.01,
        "working.state_consistency": 0.05,
        "working.gate_regularization": 0.001,
        "router.load_balance": 0.01,
        "router.communication_cost": 0.001,
        "router_load_balance": 0.01,
        "router_communication_cost": 0.001,
        "predictive.temporal": temporal_weight,
    }
    for name, value in cast(Mapping[str, Tensor], auxiliary).items():
        if value.ndim != 0 or not torch.isfinite(value):
            raise FloatingPointError(f"invalid P4 auxiliary loss: {name}")
        weight = weights.get(name, 0.0)
        total = total + weight * value
        parts[f"loss/{name}"] = float(value.detach().cpu())
    parts["loss/total"] = float(total.detach().cpu())
    return total, parts


def _metric(output: object, batch: TaskBatch) -> float:
    metrics = task_metrics(as_baseline_output(output), batch)
    key = "optimal_action_rate" if batch.metadata["task_id"] == "small_graph.v1" else "accuracy"
    return metrics[key]


def _mean_primary_records(task_id: str, records: Sequence[Mapping[str, object]]) -> float:
    key = {
        "associative_recall.v1": "query_accuracy",
        "delayed_rule_switch.v1": "response_accuracy",
        "small_graph.v1": "optimal_action_rate",
    }[task_id]
    if not records:
        raise ValueError(f"P4 {task_id} score requires non-empty records")
    return sum(_numeric(record, key) for record in records) / len(records)


def _prediction_stats(output: object) -> dict[str, float]:
    mask = getattr(output, "forecast_transition_mask", None)
    error = getattr(output, "forecast_error", None)
    persistence = getattr(output, "persistence_error", None)
    delta = getattr(output, "feedback_delta", None)
    if not all(isinstance(value, Tensor) for value in (mask, error, persistence, delta)):
        return {
            "eligible": 0.0,
            "covered": 0.0,
            "error_sum": 0.0,
            "persistence_sum": 0.0,
            "latent_feedback_sum": 0.0,
            "logits_feedback_sum": 0.0,
        }
    mask = cast(Tensor, mask).bool()
    error = cast(Tensor, error)
    persistence = cast(Tensor, persistence)
    delta = cast(Tensor, delta)
    if (
        error.shape != mask.shape
        or persistence.shape != mask.shape
        or delta.shape[:2] != mask.shape
    ):
        raise ValueError("P4 forecast evidence tensors do not align")
    covered = mask & torch.isfinite(error) & torch.isfinite(persistence)
    module_metrics = cast(Mapping[str, Tensor], getattr(output, "module_metrics", {}))
    eligible_value = module_metrics.get("predictive.eligible_transition_count")
    eligible = (
        float(eligible_value.detach().cpu()) if eligible_value is not None else float(mask.numel())
    )
    return {
        "eligible": eligible,
        "covered": float(covered.sum().item()),
        "error_sum": float(error[covered].sum().detach().cpu()) if covered.any() else 0.0,
        "persistence_sum": (
            float(persistence[covered].sum().detach().cpu()) if covered.any() else 0.0
        ),
        "latent_feedback_sum": (
            float(delta[covered].abs().sum().detach().cpu()) if covered.any() else 0.0
        ),
        "logits_feedback_sum": 0.0,
    }


def _merge_prediction(target: dict[str, float], update: Mapping[str, float]) -> None:
    for name, value in update.items():
        target[name] = target.get(name, 0.0) + value


def _prediction_summary(total: Mapping[str, float]) -> dict[str, float]:
    covered = total.get("covered", 0.0)
    eligible = total.get("eligible", 0.0)
    error = total.get("error_sum", 0.0) / max(covered, 1.0)
    persistence = total.get("persistence_sum", 0.0) / max(covered, 1.0)
    return {
        **dict(total),
        "coverage": covered / max(eligible, 1.0),
        "forecast_error": error,
        "persistence_error": persistence,
        "relative_improvement": (persistence - error) / max(persistence, 1e-12),
        "feedback_nonzero": float(total.get("logits_feedback_sum", 0.0) > 0.0),
    }


def _task_sequence(cell: P4ExperimentCell, step: int) -> str:
    if cell.task_id is not None:
        return cell.task_id
    if cell.cell_type == "continual":
        orders = {
            17: P4_TASK_ORDER,
            29: (P4_TASK_ORDER[1], P4_TASK_ORDER[2], P4_TASK_ORDER[0]),
            43: (P4_TASK_ORDER[2], P4_TASK_ORDER[0], P4_TASK_ORDER[1]),
            7: P4_TASK_ORDER,
        }
        order = orders[cell.seed]
        stage = max(cell.max_steps // 3, 1)
        return order[min(step // stage, 2)]
    return P4_TASK_ORDER[step % 3]


def _settings(config: P4SuiteConfig, cell: P4ExperimentCell) -> tuple[float, float, float]:
    preset = cell.variant_id if cell.cell_type == "pilot" else config.selected_preset
    if preset is not None:
        return P4_PILOT_PRESETS[preset]
    return (
        config.optimizer.learning_rate,
        config.optimizer.weight_decay,
        config.optimizer.temporal_loss_weight,
    )


def _checkpoint_state(
    config: P4SuiteConfig,
    cell: P4ExperimentCell,
    cursor: int,
    step: int,
    task_steps: Mapping[str, int],
    samplers: Mapping[str, IndexSampler],
    best: Mapping[str, float],
    stale: int,
    curves: Mapping[str, Sequence[tuple[int, float]]],
    last_loss: float | None,
    elapsed: float,
    transitions: int,
    pilot_hash: str | None,
    mechanism_hash: str | None,
    network_state: NetworkState | None,
    prediction_totals: Mapping[str, float],
    validation_prediction_totals: Mapping[str, float],
) -> P4CheckpointState:
    return P4CheckpointState(
        cell_id=cell.cell_id,
        global_step=step,
        task_steps=dict(task_steps),
        sampler_states={name: sampler.state_dict() for name, sampler in samplers.items()},
        best_metrics=dict(best),
        stale_evaluations=stale,
        matrix_cursor=cursor,
        config_hash=config.config_hash(),
        protocol_hash=PROTOCOL_HASH,
        matrix_hash=config.matrix_hash(),
        pilot_lock_hash=pilot_hash,
        mechanism_lock_hash=mechanism_hash,
        analysis_curves={name: tuple(points) for name, points in curves.items()},
        last_loss=last_loss,
        cumulative_wall_clock_seconds=elapsed,
        transition_count=transitions,
        prediction_totals=dict(prediction_totals),
        validation_prediction_totals=dict(validation_prediction_totals),
        network_state=network_state,
    )


def _score_split(
    model: nn.Module,
    cell: P4ExperimentCell,
    config: P4SuiteConfig,
    device: torch.device,
    *,
    split: Literal["validation", "analysis"],
) -> tuple[dict[str, float], dict[str, float]]:
    task_ids = (cell.task_id,) if cell.task_id else P4_TASK_ORDER
    scores: dict[str, float] = {}
    prediction: dict[str, float] = {}
    size = config.data.validation if split == "validation" else config.data.analysis
    model.eval()
    with torch.no_grad():
        for task_id in task_ids:
            task = _task(config, task_id)
            weighted = 0.0
            for start in range(0, size, config.budget.batch_size):
                indices = list(range(start, min(start + config.budget.batch_size, size)))
                batch = task.generate(split, indices, device=device)
                output = _forward(model, cell, batch, evaluation=True)
                weighted += _metric(output, batch) * len(indices)
                _merge_prediction(prediction, _prediction_stats(output))
                if cell.model_id == "modular-v2" and cell.intervention.operation == "none":
                    paired = cast(Any, model).forward_batch(
                        batch,
                        phase="evaluate",
                        telemetry_enabled=False,
                        predictor_mode="feedback_zero",
                        routing_mode="learned",
                    )
                    prediction["logits_feedback_sum"] = prediction.get(
                        "logits_feedback_sum", 0.0
                    ) + float((cast(Any, output).logits - paired.logits).abs().sum().detach().cpu())
            scores[task_id] = weighted / size
    model.train()
    return scores, prediction


def _train_cell(
    model: nn.Module,
    cell: P4ExperimentCell,
    config: P4SuiteConfig,
    device: torch.device,
    directory: Path,
    suite_directory: Path,
    cursor: int,
    deadline: float,
    pilot_hash: str | None,
    mechanism_hash: str | None,
) -> dict[str, object]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    learning_rate, weight_decay, temporal_weight = _settings(config, cell)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    task_ids = (cell.task_id,) if cell.task_id else P4_TASK_ORDER
    samplers = {
        task_id: IndexSampler.create(config.data.train, cell.seed + 10_000 * index)
        for index, task_id in enumerate(task_ids)
    }
    task_steps = {task_id: 0 for task_id in task_ids}
    curves: dict[str, list[tuple[int, float]]] = {task_id: [] for task_id in task_ids}
    best: dict[str, float] = {}
    stale = 0
    step = 0
    transitions = 0
    prediction: dict[str, float] = {}
    validation_prediction_total: dict[str, float] = {}
    last_loss: float | None = None
    checkpoint = directory / "checkpoint.pt"
    best_checkpoint = directory / "best.pt"
    started = time.perf_counter()
    prior_cell_elapsed = 0.0
    network_state = (
        cast(Any, model).initial_state(
            min(config.budget.batch_size, config.data.train),
            device=device,
            dtype=torch.float32,
        )
        if cell.model_id == "modular-v2"
        else None
    )
    sampler_signatures = {
        task_id: (sampler.size, sampler.seed) for task_id, sampler in samplers.items()
    }
    if checkpoint.is_file():
        restored = load_p4_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            expected_cell_id=cell.cell_id,
            expected_config_hash=config.config_hash(),
            expected_protocol_hash=PROTOCOL_HASH,
            expected_matrix_hash=config.matrix_hash(),
            expected_matrix_cursor=cursor,
            expected_pilot_lock_hash=pilot_hash,
            expected_mechanism_lock_hash=mechanism_hash,
            expected_sampler_signatures=sampler_signatures,
            expected_network_state=network_state,
        )
        step = restored.global_step
        task_steps = dict(restored.task_steps)
        best = dict(restored.best_metrics)
        stale = restored.stale_evaluations
        curves = {name: list(points) for name, points in restored.analysis_curves.items()}
        last_loss = restored.last_loss
        transitions = restored.transition_count
        prediction = dict(restored.prediction_totals)
        validation_prediction_total = dict(restored.validation_prediction_totals)
        prior_cell_elapsed = restored.cumulative_wall_clock_seconds
        network_state = restored.network_state
        for task_id, state in restored.sampler_states.items():
            samplers[task_id].load_state_dict(state)

    def save(path: Path) -> None:
        save_p4_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            state=_checkpoint_state(
                config,
                cell,
                cursor,
                step,
                task_steps,
                samplers,
                best,
                stale,
                curves,
                last_loss,
                prior_cell_elapsed + time.perf_counter() - started,
                transitions,
                pilot_hash,
                mechanism_hash,
                network_state,
                prediction,
                validation_prediction_total,
            ),
        )

    while step < cell.max_steps:
        if (suite_directory / "STOP").is_file():
            save(checkpoint)
            return {"stopped": True, "steps": step}
        if time.perf_counter() >= deadline or shutil.disk_usage(suite_directory).free < 20 * GIB:
            save(checkpoint)
            return {"resource_limited": True, "steps": step}
        task_id = _task_sequence(cell, step)
        task = _task(config, task_id)
        indices = samplers[task_id].next(min(config.budget.batch_size, config.data.train))
        batch = task.generate("train", indices, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if cell.model_id == "modular-v2":
            # Every generated batch contains complete, independent episodes.  A
            # recurrent state may therefore be restored for validation and audit,
            # but must not be carried across optimizer updates into a new episode
            # batch (doing so would also retain a graph over mutated parameters).
            network_state = cast(Any, model).initial_state(
                batch.batch_size, device=device, dtype=batch.inputs.dtype
            )
        output = _forward(model, cell, batch, evaluation=False, state=network_state)
        if cell.model_id == "modular-v2":
            network_state = cast(Any, output).state
        loss, parts = _loss(model, cell, output, batch, temporal_weight=temporal_weight)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(
            parameters, config.optimizer.gradient_clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        if network_state is not None:
            network_state = network_state.detach_rows(
                torch.ones(network_state.batch_size, dtype=torch.bool, device=device)
            )
        ensure_finite_training_state(loss=loss, model=model, metrics=parts)
        last_loss = float(loss.detach().cpu())
        observed = _prediction_stats(output)
        _merge_prediction(prediction, observed)
        transitions += int(observed["covered"])
        step += 1
        task_steps[task_id] += 1
        if step % config.budget.validation_interval == 0 or step == cell.max_steps:
            scores, validation_prediction = _score_split(
                model, cell, config, device, split="validation"
            )
            _merge_prediction(prediction, validation_prediction)
            _merge_prediction(validation_prediction_total, validation_prediction)
            curve_scores = (
                scores
                if config.profile == "pilot"
                else _score_split(model, cell, config, device, split="analysis")[0]
            )
            for name, value in curve_scores.items():
                point = (task_steps.get(name, step), value)
                if curves[name] and curves[name][-1][0] == point[0]:
                    curves[name][-1] = point
                else:
                    curves[name].append(point)
            macro = sum(scores.values()) / len(scores)
            if macro > best.get("macro", -math.inf) + config.budget.min_delta:
                best = {"macro": macro, **scores}
                stale = 0
                save(best_checkpoint)
            else:
                stale += 1
        if step % config.budget.checkpoint_interval == 0 or step == cell.max_steps:
            save(checkpoint)
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
                    "remaining_wall_clock_seconds": max(deadline - time.perf_counter(), 0.0),
                },
            )
    if best_checkpoint.is_file():
        load_p4_checkpoint(
            best_checkpoint,
            model=model,
            optimizer=optimizer,
            expected_cell_id=cell.cell_id,
            expected_config_hash=config.config_hash(),
            expected_protocol_hash=PROTOCOL_HASH,
            expected_matrix_hash=config.matrix_hash(),
            expected_matrix_cursor=cursor,
            expected_pilot_lock_hash=pilot_hash,
            expected_mechanism_lock_hash=mechanism_hash,
            expected_sampler_signatures=sampler_signatures,
            expected_network_state=network_state,
        )
    aulc = {
        name: normalized_aulc(points or [(0, 0.0)], maximum_step=max(task_steps[name], 1))
        for name, points in curves.items()
    }
    return {
        "steps": step,
        "task_steps": task_steps,
        "best_metrics": best,
        "analysis_aulc": aulc,
        "validation_macro_aulc": sum(aulc.values()) / len(aulc),
        "last_loss": last_loss,
        "prediction": _prediction_summary(
            validation_prediction_total if config.profile == "pilot" else prediction
        ),
        "settings": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "temporal_loss_weight": temporal_weight,
        },
        "wall_clock_seconds": prior_cell_elapsed + time.perf_counter() - started,
    }


def _rollout_policy(
    model: nn.Module, cell: P4ExperimentCell, task: SmallGraphTask, device: torch.device
) -> Any:
    history: list[Tensor] = []
    modular_state: NetworkState | None = None

    def policy(observation: Tensor, reset: bool) -> Tensor:
        nonlocal modular_state
        if reset:
            history.clear()
        history.append(observation.detach())
        inputs = (
            observation.detach().unsqueeze(0).unsqueeze(0).to(device)
            if cell.model_id == "modular-v2"
            else torch.stack(history).unsqueeze(0).to(device)
        )
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
                "namespace": "p4",
                "split": "rollout",
                "distribution": task.distribution,
            },
            auxiliary_targets={},
        )
        with torch.no_grad():
            if cell.model_id == "modular-v2":
                if reset or modular_state is None:
                    modular_state = cast(Any, model).initial_state(
                        1, device=device, dtype=inputs.dtype
                    )
                control = task_control_from_batch(batch)
                context = ModuleContext(
                    task_id=task.task_id,
                    phase="evaluate",
                    reset_mask=torch.tensor([[reset]], dtype=torch.bool, device=device),
                    eligible_modules=P4_OPTIONAL_EXPERT_IDS,
                    telemetry_enabled=False,
                )
                with trusted_internal_execution():
                    output = cast(Any, model).forward_step(
                        inputs,
                        control,
                        modular_state,
                        context,
                        terminal_mask=torch.zeros(1, dtype=torch.bool, device=device),
                        **_intervention_kwargs(cell, evaluation=True),
                    )
                modular_state = cast(Any, output).state
            else:
                output = _forward(model, cell, batch, evaluation=True)
        return cast(Tensor, cast(Any, output).logits)[0, -1]

    return policy


def _rollout_records_with_chance(
    model: nn.Module,
    cell: P4ExperimentCell,
    task: SmallGraphTask,
    split: Literal["test", "ood"],
    size: int,
    device: torch.device,
) -> list[dict[str, object]]:
    records = task.rollout_records(
        _rollout_policy(model, cell, task, device), split, list(range(size)), device=device
    )
    for record in records:
        index = cast(int, record["sample_index"])
        batch = task.generate(split, [index], device=torch.device("cpu"))
        nodes = int(batch.auxiliary_targets["node_count"][0, 0])
        shortest = int(batch.auxiliary_targets["optimal_distance"][0, 0])
        horizon = max(nodes * 2, shortest * 2)
        chance = small_graph_chance_level(
            batch.auxiliary_targets["adjacency"][0, 0].numpy(),
            start=int(batch.auxiliary_targets["start_node"][0, 0]),
            goal=int(batch.auxiliary_targets["goal_node"][0, 0]),
            node_count=nodes,
            horizon=horizon,
        )
        record.update(
            {
                "schema_version": "p4-small-graph-rollout-v1",
                "model_id": cell.model_id,
                "variant_id": cell.variant_id,
                "seed": cell.seed,
                "namespace": "p4",
                "horizon": horizon,
                "chance": chance,
                "stratum": record["bootstrap_stratum"],
            }
        )
    return records


def _routing_stats(output: object, batch: TaskBatch) -> dict[str, float]:
    totals = {
        "valid_tokens": float(batch.valid_mask.sum().item()),
        "capacity_drops": 0.0,
        "reserved_episodic": 0.0,
        "reserved_executed": 0.0,
        "reserved_total": 0.0,
        "raw_optional": 0.0,
        "executed_optional": 0.0,
        "rerouted_tokens": 0.0,
        "dense_optional": float(batch.valid_mask.sum().item() * 2),
        **{f"executed.{module_id}": 0.0 for module_id in P4_OPTIONAL_EXPERT_IDS},
        **{f"capacity.{module_id}": 0.0 for module_id in P4_OPTIONAL_EXPERT_IDS},
    }
    for trace in getattr(output, "routing_trace", ()):
        totals["capacity_drops"] += float(trace.capacity_drops)
        totals["reserved_episodic"] += float(trace.reserved_mask[..., 0].sum().item())
        totals["reserved_executed"] += float(
            (trace.reserved_mask[..., 0] & trace.executed_mask[..., 0]).sum().item()
        )
        totals["reserved_total"] += float(trace.reserved_mask.sum().item())
        totals["raw_optional"] += float(trace.raw_mask.sum().item())
        totals["executed_optional"] += float(trace.executed_mask.sum().item())
        totals["rerouted_tokens"] += float(
            (trace.raw_mask != trace.executed_mask).any(dim=-1).sum().item()
        )
        for index, module_id in enumerate(P4_OPTIONAL_EXPERT_IDS):
            totals[f"executed.{module_id}"] += float(trace.executed_mask[..., index].sum().item())
            totals[f"capacity.{module_id}"] += float(trace.capacity[..., index].sum().item())
    return totals


def _optional_mac_profile(model: nn.Module, routing: Mapping[str, float]) -> dict[str, object]:
    """Convert observed expert calls into supported Linear MAC accounting."""

    registry = getattr(model, "registry", None)
    if registry is None:
        return {
            "mac_applicable": False,
            "active_optional_macs": 0,
            "dense_optional_macs": 0,
            "optional_mac_coverage": 0.0,
            "optional_mac_records": [],
        }
    valid = int(routing.get("valid_tokens", 0.0))
    active_macs = 0
    dense_macs = 0
    supported = 0
    total = 0
    records: list[dict[str, object]] = []
    for module_id in P4_OPTIONAL_EXPERT_IDS:
        module = registry.get(module_id)
        if not isinstance(module, nn.Module):
            raise TypeError("P4 optional registry entry must be torch.nn.Module")
        linears = tuple(layer for layer in module.modules() if isinstance(layer, nn.Linear))
        macs_per_call = sum(layer.in_features * layer.out_features for layer in linears)
        supported_parameters = sum(
            parameter.numel() for layer in linears for parameter in layer.parameters(recurse=False)
        )
        total_parameters = sum(parameter.numel() for parameter in module.parameters())
        active_calls = int(routing.get(f"executed.{module_id}", 0.0))
        active = macs_per_call * active_calls
        dense = macs_per_call * valid
        active_macs += active
        dense_macs += dense
        supported += supported_parameters
        total += total_parameters
        records.append(
            {
                "module_id": module_id,
                "operator": "Linear",
                "macs_per_call": macs_per_call,
                "active_calls": active_calls,
                "dense_calls": valid,
                "active_macs": active,
                "dense_macs": dense,
                "supported_parameters": supported_parameters,
                "total_parameters": total_parameters,
            }
        )
    return {
        "mac_applicable": True,
        "active_optional_macs": active_macs,
        "dense_optional_macs": dense_macs,
        "optional_mac_coverage": 1.0 if total == 0 else supported / total,
        "optional_mac_records": records,
    }


def _evaluate_cell(
    model: nn.Module,
    cell: P4ExperimentCell,
    config: P4SuiteConfig,
    device: torch.device,
    directory: Path,
    deadline: float,
) -> dict[str, object]:
    if cell.cell_type == "pilot":
        pilot_scores, pilot_prediction = _score_split(
            model, cell, config, device, split="validation"
        )
        return {
            "validation": pilot_scores,
            "prediction": _prediction_summary(pilot_prediction),
        }
    task_ids = (cell.task_id,) if cell.task_id else P4_TASK_ORDER
    records: list[dict[str, object]] = []
    views: dict[str, object] = {}
    routing: dict[str, dict[str, Any]] = {}
    prediction: dict[str, dict[str, float]] = {}
    scores: dict[str, float] = {}
    telemetry_records: list[dict[str, object]] = []
    model.eval()
    for task_id in task_ids:
        task_views: dict[str, int] = {}
        route_total: dict[str, float] = {}
        prediction_total: dict[str, float] = {}
        for split, distribution, size in (
            ("analysis", "v1", config.data.analysis),
            ("test", "v1", config.data.test),
        ):
            task = _task(config, task_id, distribution)
            view: list[dict[str, object]] = []
            for start in range(0, size, config.budget.batch_size):
                if time.perf_counter() >= deadline:
                    raise P4ResourceLimit("P4 wall-clock budget exhausted during evaluation")
                indices = list(range(start, min(start + config.budget.batch_size, size)))
                batch = task.generate(cast(Any, split), indices, device=device)
                with torch.no_grad():
                    output = _forward(model, cell, batch, evaluation=True)
                for item in p3_sample_records(
                    output,
                    batch,
                    run_seed=cell.seed,
                    model_id=cell.model_id,
                    variant_id=cell.variant_id,
                ):
                    item["schema_version"] = "p4-evaluation-sample-v1"
                    item["namespace"] = "p4"
                    view.append(item)
                current_routing = _routing_stats(output, batch)
                for name, value in current_routing.items():
                    route_total[name] = route_total.get(name, 0.0) + value
                _merge_prediction(prediction_total, _prediction_stats(output))
            records.extend(view)
            task_views[f"{split}:{distribution}"] = len(view)
            if split == "test":
                scores[task_id] = _mean_primary_records(task_id, view)
        for distribution in _OOD_DISTRIBUTIONS[task_id]:
            task = _task(config, task_id, distribution)
            for start in range(0, config.data.ood, config.budget.batch_size):
                indices = list(range(start, min(start + config.budget.batch_size, config.data.ood)))
                batch = task.generate("ood", indices, device=device)
                with torch.no_grad():
                    output = _forward(model, cell, batch, evaluation=True)
                for item in p3_sample_records(
                    output,
                    batch,
                    run_seed=cell.seed,
                    model_id=cell.model_id,
                    variant_id=cell.variant_id,
                ):
                    item["schema_version"] = "p4-evaluation-sample-v1"
                    item["namespace"] = "p4"
                    records.append(item)
            task_views[f"ood:{distribution}"] = config.data.ood
            if task_id == "small_graph.v1":
                rollout = _rollout_records_with_chance(
                    model, cell, cast(SmallGraphTask, task), "ood", config.data.ood, device
                )
                records.extend(rollout)
                task_views[f"ood:{distribution}:live_rollout"] = len(rollout)
        if task_id == "small_graph.v1":
            task = cast(SmallGraphTask, _task(config, task_id))
            rollout = _rollout_records_with_chance(
                model, cell, task, "test", config.data.test, device
            )
            records.extend(rollout)
            task_views["test:v1:live_rollout"] = len(rollout)
            scores[task_id] = sum(_numeric(item, "success_rate") for item in rollout) / len(rollout)
        views[task_id] = task_views
        routing_summary = {**route_total, **_optional_mac_profile(model, route_total)}
        prediction_summary = _prediction_summary(prediction_total)
        routing[task_id] = routing_summary
        prediction[task_id] = prediction_summary
        if routing_summary["mac_applicable"]:
            reserved_total = _numeric(routing_summary, "reserved_total")
            valid_tokens = _numeric(routing_summary, "valid_tokens")
            executed_optional = _numeric(routing_summary, "executed_optional")
            capacity_drops = _numeric(routing_summary, "capacity_drops")
            reserved_executed = _numeric(routing_summary, "reserved_executed")
            telemetry_records.extend(
                (
                    TelemetryV2Event(
                        event_id=f"{cell.cell_id}:{task_id}:route",
                        run_id=cell.cell_id,
                        global_step=cell.max_steps,
                        task=task_id,
                        event_type="summary",
                        phase="evaluate",
                        module_id=SPARSE_ROUTER_V2,
                        compute_gate=True,
                        reserved_count=int(reserved_total),
                        learned_count=int(valid_tokens - reserved_total),
                        raw_count=int(_numeric(routing_summary, "raw_optional")),
                        executed_count=int(executed_optional),
                        capacity=int(valid_tokens),
                        drop_count=int(capacity_drops),
                        reserved_coverage=(
                            1.0 if reserved_total == 0 else reserved_executed / reserved_total
                        ),
                        active_mac=_numeric(routing_summary, "active_optional_macs"),
                        dense_mac=_numeric(routing_summary, "dense_optional_macs"),
                        metadata={
                            "rerouted_tokens": _numeric(routing_summary, "rerouted_tokens"),
                            "reroute_rate": _numeric(routing_summary, "rerouted_tokens")
                            / max(valid_tokens, 1.0),
                            **{
                                f"capacity.{index}": _numeric(
                                    routing_summary, f"capacity.{module_id}"
                                )
                                for index, module_id in enumerate(P4_OPTIONAL_EXPERT_IDS)
                            },
                        },
                    ).to_dict(),
                    TelemetryV2Event(
                        event_id=f"{cell.cell_id}:{task_id}:predict",
                        run_id=cell.cell_id,
                        global_step=cell.max_steps,
                        task=task_id,
                        event_type="summary",
                        phase="evaluate",
                        module_id=PREDICTIVE_ADAPTER_V2,
                        compute_gate=cell.intervention.operation != "predictor_off",
                        forecast_coverage=prediction_summary["coverage"],
                        forecast_error=prediction_summary["forecast_error"],
                        persistence_error=prediction_summary["persistence_error"],
                        metadata={
                            "feedback_nonzero": bool(prediction_summary["feedback_nonzero"]),
                        },
                    ).to_dict(),
                )
            )
    _write_jsonl(directory / "sample_records.jsonl", records)
    _write_jsonl(directory / "telemetry-v2.jsonl", telemetry_records)
    return {
        "record_count": len(records),
        "views": views,
        "routing": routing,
        "prediction": prediction,
        "scores": scores,
        "telemetry_v2_events": len(telemetry_records),
    }


def _parent_cell_id(cell: P4ExperimentCell) -> str:
    return f"shared__shared__modular-v2__full__s{cell.seed}__all"


def _load_parent(
    model: nn.Module,
    cell: P4ExperimentCell,
    directory: Path,
    device: torch.device,
) -> None:
    parent = directory / "cells" / _parent_cell_id(cell)
    checkpoint = parent / ("best.pt" if (parent / "best.pt").is_file() else "checkpoint.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"P4 frozen-checkpoint control parent is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if cell.intervention.operation == "frozen_random_encoder":
        set_global_seed(cell.seed + 50_000)
        encoder = cast(Any, model).registry.get("sensory_encoder.v1")
        for child in encoder.modules():
            reset = getattr(child, "reset_parameters", None)
            if child is not encoder and callable(reset):
                reset()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)


def _lock_hash(
    path: Path | None, *, required_status: str, expected_commit: str | None = None
) -> str | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"P4 prerequisite lock is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != required_status:
        raise ValueError(f"P4 prerequisite lock is not {required_status}: {path}")
    lock_commit = value.get("git_commit", value.get("expected_git_commit"))
    if expected_commit is not None and lock_commit != expected_commit:
        raise ValueError(f"P4 prerequisite lock belongs to another commit: {path}")
    root = Path(__file__).resolve().parents[3]
    for artifact_key, checksum_key in (
        ("qualification_report", "qualification_report_sha256"),
        ("pilot_selection", "pilot_selection_sha256"),
        ("mechanism_report", "mechanism_report_sha256"),
    ):
        artifact = value.get(artifact_key)
        checksum = value.get(checksum_key)
        if artifact is None and checksum is None:
            continue
        if not isinstance(artifact, str) or not isinstance(checksum, str):
            raise ValueError(f"P4 prerequisite lock has incomplete evidence: {path}")
        evidence = (root / artifact).resolve()
        try:
            evidence.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"P4 prerequisite evidence escapes repository: {path}") from error
        if not evidence.is_file() or _sha256(evidence) != checksum:
            raise ValueError(f"P4 prerequisite evidence checksum mismatch: {path}")
    return _sha256(path)


def _initial_registry(config: P4SuiteConfig, run_id: str) -> dict[str, object]:
    return {
        "schema_version": "p4-suite-registry-v1",
        "run_id": run_id,
        "profile": config.profile,
        "qualification_only": config.qualification_only,
        "protocol_version": config.protocol_version,
        "protocol_hash": PROTOCOL_HASH,
        "config_hash": config.config_hash(),
        "matrix_hash": config.matrix_hash(),
        "status": "running",
        "cells": [
            {
                **cell.model_dump(mode="json"),
                "status": "PENDING",
                "artifact_dir": None,
                "error": None,
                "attempts": [],
            }
            for cell in config.matrix()
        ],
    }


def _all_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, Sequence):
        return all(_all_finite(item) for item in value)
    return True


def _validate_telemetry_v2_file(path: Path) -> int:
    required = {
        "schema_version",
        "scientific_disclaimer",
        "event_id",
        "run_id",
        "global_step",
        "task",
        "event_type",
        "phase",
        "module_id",
        "compute_gate",
        "reserved_count",
        "learned_count",
        "raw_count",
        "executed_count",
        "capacity",
        "drop_count",
        "reserved_coverage",
        "forecast_coverage",
        "forecast_error",
        "persistence_error",
        "active_mac",
        "dense_mac",
        "metadata",
    }
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if not isinstance(record, dict) or set(record) != required:
            raise ValueError("telemetry-v2 event fields do not match the frozen schema")
        arguments = dict(record)
        arguments.pop("schema_version")
        arguments.pop("scientific_disclaimer")
        rebuilt = TelemetryV2Event(**cast(Any, arguments)).to_dict()
        if rebuilt != record:
            raise ValueError("telemetry-v2 event does not round-trip through validation")
        count += 1
    return count


def _qualification_evidence(config: P4SuiteConfig, directory: Path) -> dict[str, object]:
    violations: list[str] = []
    forecast_path_seen = False
    feedback_nonzero = False
    restored_checkpoints = 0
    completed_variants: set[str] = set()
    for cell in config.matrix():
        cell_directory = directory / "cells" / cell.cell_id
        summary_path = cell_directory / "summary.json"
        if not summary_path.is_file():
            violations.append(f"missing summary: {cell.cell_id}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        completed_variants.add(cell.variant_id)
        if not _all_finite(summary):
            violations.append(f"non-finite summary: {cell.cell_id}")
        if cell.max_steps:
            checkpoint = cell_directory / "checkpoint.pt"
            if checkpoint.is_file():
                restored_checkpoints += 1
            else:
                violations.append(f"missing checkpoint-v4: {cell.cell_id}")
        evaluation = cast(Mapping[str, Any], summary.get("evaluation", {}))
        telemetry_path = cell_directory / "telemetry-v2.jsonl"
        if not telemetry_path.is_file():
            violations.append(f"missing telemetry-v2: {cell.cell_id}")
        else:
            try:
                if _validate_telemetry_v2_file(telemetry_path) != 6:
                    violations.append(f"unexpected telemetry-v2 count: {cell.cell_id}")
            except (TypeError, ValueError) as error:
                violations.append(f"invalid telemetry-v2: {cell.cell_id}: {error}")
        routing = cast(Mapping[str, Mapping[str, float]], evaluation.get("routing", {}))
        for task_id, values in routing.items():
            if values.get("capacity_drops", 0.0) != 0.0:
                violations.append(f"capacity drops: {cell.cell_id}/{task_id}")
            valid = values.get("valid_tokens", 0.0)
            active = values.get("executed_optional", 0.0)
            dense = values.get("dense_optional", 0.0)
            if cell.variant_id == "dense-memory":
                if valid > 0 and active != dense:
                    violations.append(f"dense routing mismatch: {cell.cell_id}/{task_id}")
            elif valid > 0 and active != valid:
                violations.append(f"top-1 routing mismatch: {cell.cell_id}/{task_id}")
            if cell.variant_id != "dense-memory" and valid > 0 and not active < dense:
                violations.append(f"sparse calls not below dense: {cell.cell_id}/{task_id}")
            if values.get("mac_applicable"):
                active_macs = values.get("active_optional_macs", 0.0)
                dense_macs = values.get("dense_optional_macs", 0.0)
                if cell.variant_id == "dense-memory":
                    if dense_macs > 0 and active_macs != dense_macs:
                        violations.append(f"dense MAC mismatch: {cell.cell_id}/{task_id}")
                elif dense_macs > 0 and not active_macs < dense_macs:
                    violations.append(f"sparse MAC not below dense: {cell.cell_id}/{task_id}")
            if (
                task_id == "associative_recall.v1"
                and cell.variant_id != "legacy-capacity"
                and values.get("reserved_total", 0.0) > 0
                and values.get("reserved_executed") != values.get("reserved_total")
            ):
                violations.append(f"AR reservation coverage below 100%: {cell.cell_id}")
        if cell.variant_id == "full":
            training = cast(Mapping[str, Any], summary.get("training", {}))
            prediction = cast(Mapping[str, float], training.get("prediction", {}))
            feedback_nonzero = feedback_nonzero or prediction.get("feedback_nonzero", 0.0) == 1.0
            forecast_path_seen = forecast_path_seen or (
                prediction.get("eligible", 0.0) > 0
                and prediction.get("covered", 0.0) > 0
                and prediction.get("feedback_nonzero", 0.0) == 1.0
            )
    if not forecast_path_seen:
        violations.append("full predictor path produced no covered transition/logits effect")
    if restored_checkpoints == 0:
        violations.append("checkpoint-v4 path was not exercised")
    return {
        "status": "PASSED" if not violations else "FAILED",
        "violations": violations,
        "finite": not any("non-finite" in item for item in violations),
        "checkpoint_count": restored_checkpoints,
        "forecast_path_seen": forecast_path_seen,
        "feedback_nonzero": feedback_nonzero,
        "all_cells_completed": len(completed_variants) == len(config.matrix()),
        "all_cells_covered": completed_variants == {cell.variant_id for cell in config.matrix()},
        "cpu_micro_required": config.device == "cpu",
        "mps_required": config.device == "mps",
        "sparse_mac_less_than_dense": not any(
            "sparse MAC not below dense" in item or "sparse calls not below dense" in item
            for item in violations
        ),
        "dense_control_matches_dense_macs": not any(
            "dense MAC mismatch" in item or "dense routing mismatch" in item for item in violations
        ),
    }


def _repository_relative(path: Path) -> str:
    root = Path(__file__).resolve().parents[3]
    return path.resolve().relative_to(root.resolve()).as_posix()


def _is_repository_path(path: Path) -> bool:
    root = Path(__file__).resolve().parents[3]
    return path.resolve().is_relative_to(root.resolve())


def _write_external_lock(path: Path, value: Mapping[str, object], *, dirty: bool) -> None:
    if dirty:
        return
    _write_json(path, value)


def _write_pilot_lock(
    config: P4SuiteConfig, directory: Path, *, git_commit: str, git_dirty: bool
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for cell in config.matrix():
        summary = json.loads(
            (directory / "cells" / cell.cell_id / "summary.json").read_text(encoding="utf-8")
        )
        training = cast(dict[str, Any], summary["training"])
        prediction = cast(dict[str, float], training["prediction"])
        eligible = (
            prediction["coverage"] >= 0.95
            and prediction["relative_improvement"] >= 0.05
            and prediction["feedback_nonzero"] == 1.0
        )
        candidates.append(
            {
                "preset_id": cell.variant_id,
                "eligible": eligible,
                "validation_macro_aulc": training["validation_macro_aulc"],
                "final_loss": training["last_loss"],
                "prediction": prediction,
                "settings": training["settings"],
            }
        )
    valid = [candidate for candidate in candidates if candidate["eligible"]]
    if not valid:
        raise ValueError("no P4 pilot candidate met frozen predictive eligibility")
    selected = min(
        valid,
        key=lambda item: (
            -cast(float, item["validation_macro_aulc"]),
            cast(float, item["final_loss"]),
            str(item["preset_id"]),
        ),
    )
    selected_settings = cast(Mapping[str, object], selected["settings"])
    result = {
        "schema_version": "p4-pilot-lock-v1",
        "status": "PASSED",
        "qualification_only": False,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_hash": config.config_hash(),
        "matrix_hash": config.matrix_hash(),
        "selection_rule": "eligible,validation_macro_aulc_desc,final_loss_asc,preset_id_asc",
        "selected_preset": selected["preset_id"],
        "optimizer": {
            "learning_rate": selected_settings["learning_rate"],
            "weight_decay": selected_settings["weight_decay"],
            "temporal_loss_weight": selected_settings["temporal_loss_weight"],
            "gradient_clip_norm": 1.0,
        },
        "candidates": candidates,
    }
    _write_json(directory / "pilot-selection.json", result)
    return result


def _mechanism_gate_evidence(config: P4SuiteConfig, directory: Path) -> dict[str, object]:
    """Apply the frozen P4 mechanism thresholds to the completed 24-cell matrix."""

    summaries: dict[tuple[int, str], Mapping[str, object]] = {}
    for cell in config.matrix():
        path = directory / "cells" / cell.cell_id / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"P4 mechanism summary is missing: {path}")
        summaries[(cell.seed, cell.variant_id)] = cast(
            Mapping[str, object], json.loads(path.read_text(encoding="utf-8"))
        )

    relative_aulc: list[dict[str, object]] = []
    zero_aulc: list[dict[str, object]] = []
    relative_forecast: list[dict[str, object]] = []
    zero_forecast: list[dict[str, object]] = []
    sparse_margin: list[dict[str, object]] = []
    zero_sparse_margin: list[dict[str, object]] = []
    final_score_deltas: dict[str, float] = {}
    prediction_totals = {
        task_id: {"eligible": 0.0, "covered": 0.0, "error_sum": 0.0, "persistence_sum": 0.0}
        for task_id in P4_TASK_ORDER
    }
    active_optional_macs = 0.0
    dense_optional_macs = 0.0
    capacity_drops = 0.0
    reserved_total = 0.0
    reserved_executed = 0.0
    sparse_score_deltas: dict[str, float] = {}
    aulc_denominators_valid = True
    forecast_denominators_valid = True

    for seed in config.seeds:
        full = summaries[(seed, "full")]
        predictor_off = summaries[(seed, "predictor-off")]
        dense_memory = summaries[(seed, "dense-memory")]
        full_training = cast(Mapping[str, object], full["training"])
        off_training = cast(Mapping[str, object], predictor_off["training"])
        full_aulc = cast(Mapping[str, object], full_training["analysis_aulc"])
        off_aulc = cast(Mapping[str, object], off_training["analysis_aulc"])
        full_evaluation = cast(Mapping[str, object], full["evaluation"])
        off_evaluation = cast(Mapping[str, object], predictor_off["evaluation"])
        dense_evaluation = cast(Mapping[str, object], dense_memory["evaluation"])
        full_scores = cast(Mapping[str, object], full_evaluation["scores"])
        off_scores = cast(Mapping[str, object], off_evaluation["scores"])
        dense_scores = cast(Mapping[str, object], dense_evaluation["scores"])
        full_prediction = cast(Mapping[str, object], full_evaluation["prediction"])
        full_routing = cast(Mapping[str, object], full_evaluation["routing"])

        for task_index, task_id in enumerate(P4_TASK_ORDER):
            full_value = _numeric(full_aulc, task_id)
            off_value = _numeric(off_aulc, task_id)
            if off_value <= 0.0:
                aulc_denominators_valid = False
                relative = 0.0
            else:
                relative = (full_value - off_value) / off_value
            common = {
                "task_id": task_id,
                "split": "analysis",
                "distribution": "v1",
                "stratum": task_id,
                "seed": seed,
                "sample_index": task_index,
            }
            relative_aulc.append(
                {
                    **common,
                    "model_id": "modular-v2",
                    "variant_id": "full-relative-aulc",
                    "value": relative,
                }
            )
            zero_aulc.append(
                {
                    **common,
                    "model_id": "modular-v2",
                    "variant_id": "predictor-off-reference",
                    "value": 0.0,
                }
            )
            final_score_deltas[f"s{seed}:{task_id}"] = _numeric(full_scores, task_id) - _numeric(
                off_scores, task_id
            )
            sparse_score_deltas[f"s{seed}:{task_id}"] = _numeric(full_scores, task_id) - _numeric(
                dense_scores, task_id
            )

            prediction = cast(Mapping[str, object], full_prediction[task_id])
            covered = _numeric(prediction, "covered")
            forecast_error = _numeric(prediction, "error_sum") / max(covered, 1.0)
            persistence_error = _numeric(prediction, "persistence_sum") / max(covered, 1.0)
            if covered <= 0.0 or persistence_error <= 0.0:
                forecast_denominators_valid = False
                forecast_relative = 0.0
            else:
                forecast_relative = (persistence_error - forecast_error) / persistence_error
            forecast_common = {**common, "stratum": f"forecast:{task_id}"}
            relative_forecast.append(
                {
                    **forecast_common,
                    "model_id": "modular-v2",
                    "variant_id": "forecast-relative-improvement",
                    "value": forecast_relative,
                }
            )
            zero_forecast.append(
                {
                    **forecast_common,
                    "model_id": "modular-v2",
                    "variant_id": "persistence-reference",
                    "value": 0.0,
                }
            )
            sparse_common = {
                **common,
                "split": "test",
                "stratum": f"sparse:{task_id}",
            }
            sparse_margin.append(
                {
                    **sparse_common,
                    "model_id": "modular-v2",
                    "variant_id": "sparse-noninferiority-margin",
                    "value": sparse_score_deltas[f"s{seed}:{task_id}"] + 0.02,
                }
            )
            zero_sparse_margin.append(
                {
                    **sparse_common,
                    "model_id": "modular-v2",
                    "variant_id": "dense-memory-reference",
                    "value": 0.0,
                }
            )
            for key in prediction_totals[task_id]:
                prediction_totals[task_id][key] += _numeric(prediction, key)

            routing = cast(Mapping[str, object], full_routing[task_id])
            active_optional_macs += _numeric(routing, "active_optional_macs")
            dense_optional_macs += _numeric(routing, "dense_optional_macs")
            capacity_drops += _numeric(routing, "capacity_drops")
            if task_id == "associative_recall.v1":
                reserved_total += _numeric(routing, "reserved_total")
                reserved_executed += _numeric(routing, "reserved_executed")

    bootstrap = paired_hierarchical_bootstrap(
        relative_aulc,
        zero_aulc,
        samples=config.budget.bootstrap_samples,
        rng_seed=20_260_715,
    )
    forecast_bootstrap = paired_hierarchical_bootstrap(
        relative_forecast,
        zero_forecast,
        samples=config.budget.bootstrap_samples,
        rng_seed=20_260_715,
    )
    sparse_bootstrap = paired_hierarchical_bootstrap(
        sparse_margin,
        zero_sparse_margin,
        samples=config.budget.bootstrap_samples,
        rng_seed=20_260_715,
    )
    adjusted_p = adjust_family([bootstrap, forecast_bootstrap, sparse_bootstrap])
    aulc_pass = (
        aulc_denominators_valid
        and bootstrap.estimate >= 0.05
        and bootstrap.lower > 0.0
        and adjusted_p[0] <= 0.05
        and min(final_score_deltas.values()) >= -0.02
    )

    prediction_by_task: dict[str, dict[str, float]] = {}
    for task_id, total in prediction_totals.items():
        covered = total["covered"]
        error = total["error_sum"] / max(covered, 1.0)
        persistence = total["persistence_sum"] / max(covered, 1.0)
        prediction_by_task[task_id] = {
            "coverage": covered / max(total["eligible"], 1.0),
            "forecast_error": error,
            "persistence_error": persistence,
            "relative_improvement": (persistence - error) / max(persistence, 1.0e-12),
        }
    total_covered = sum(item["covered"] for item in prediction_totals.values())
    total_eligible = sum(item["eligible"] for item in prediction_totals.values())
    total_error = sum(item["error_sum"] for item in prediction_totals.values()) / max(
        total_covered, 1.0
    )
    total_persistence = sum(item["persistence_sum"] for item in prediction_totals.values()) / max(
        total_covered, 1.0
    )
    forecast_improvement = (total_persistence - total_error) / max(total_persistence, 1.0e-12)
    positive_prediction_tasks = sum(
        item["relative_improvement"] > 0.0 for item in prediction_by_task.values()
    )
    prediction_pass = (
        forecast_denominators_valid
        and total_covered / max(total_eligible, 1.0) >= 0.95
        and forecast_improvement >= 0.05
        and positive_prediction_tasks >= 2
        and forecast_bootstrap.lower > 0.0
        and adjusted_p[1] <= 0.05
    )

    mac_reduction = 1.0 - active_optional_macs / max(dense_optional_macs, 1.0)
    reserved_coverage = reserved_executed / max(reserved_total, 1.0)
    sparse_pass = (
        dense_optional_macs > 0.0
        and mac_reduction >= 0.20
        and min(sparse_score_deltas.values()) >= -0.02
        and capacity_drops == 0.0
        and reserved_total > 0.0
        and reserved_coverage == 1.0
        and sparse_bootstrap.lower > 0.0
        and adjusted_p[2] <= 0.05
    )
    violations: list[str] = []
    if not aulc_pass:
        violations.append("predictive causal AULC gate failed")
    if not prediction_pass:
        violations.append("forecast quality gate failed")
    if not sparse_pass:
        violations.append("semantic sparse-routing gate failed")
    return {
        "status": "PASSED" if not violations else "FAILED",
        "violations": violations,
        "predictive_causality": {
            "relative_macro_aulc": bootstrap.estimate,
            "ci95": [bootstrap.lower, bootstrap.upper],
            "p_value": bootstrap.p_value,
            "holm_adjusted_p": adjusted_p[0],
            "bootstrap_samples": bootstrap.samples,
            "rng_seed": 20_260_715,
            "denominators_valid": aulc_denominators_valid,
            "minimum_final_score_delta": min(final_score_deltas.values()),
            "per_seed_task_final_score_delta": final_score_deltas,
            "passed": aulc_pass,
        },
        "prediction_quality": {
            "coverage": total_covered / max(total_eligible, 1.0),
            "forecast_error": total_error,
            "persistence_error": total_persistence,
            "relative_improvement": forecast_improvement,
            "paired_relative_improvement": forecast_bootstrap.estimate,
            "ci95": [forecast_bootstrap.lower, forecast_bootstrap.upper],
            "p_value": forecast_bootstrap.p_value,
            "holm_adjusted_p": adjusted_p[1],
            "bootstrap_samples": forecast_bootstrap.samples,
            "denominators_valid": forecast_denominators_valid,
            "positive_tasks": positive_prediction_tasks,
            "by_task": prediction_by_task,
            "passed": prediction_pass,
        },
        "sparse_routing": {
            "active_optional_macs": active_optional_macs,
            "dense_optional_macs": dense_optional_macs,
            "mac_reduction": mac_reduction,
            "minimum_dense_score_delta": min(sparse_score_deltas.values()),
            "per_seed_task_dense_score_delta": sparse_score_deltas,
            "capacity_drops": capacity_drops,
            "reserved_total": reserved_total,
            "reserved_coverage": reserved_coverage,
            "noninferiority_margin": sparse_bootstrap.estimate,
            "noninferiority_ci95": [sparse_bootstrap.lower, sparse_bootstrap.upper],
            "noninferiority_p_value": sparse_bootstrap.p_value,
            "holm_adjusted_p": adjusted_p[2],
            "bootstrap_samples": sparse_bootstrap.samples,
            "passed": sparse_pass,
        },
    }


def run_p4_suite(config: P4SuiteConfig) -> dict[str, object]:
    """Run or resume P4, preserving every failure and completed matrix cell."""

    set_global_seed(config.seeds[0])
    device = resolve_device(config.device)
    commit, dirty = _repository_state()
    if config.expected_git_commit is not None and config.expected_git_commit != commit:
        raise ValueError("P4 expected_git_commit does not match HEAD")
    if config.profile != "qualification":
        _lock_hash(
            config.qualification_report,
            required_status="PASSED",
            expected_commit=commit,
        )
    pilot_hash = (
        _lock_hash(config.pilot_lock, required_status="PASSED", expected_commit=commit)
        if config.profile in {"mechanism", "full"}
        else None
    )
    mechanism_hash = (
        _lock_hash(config.mechanism_report, required_status="PASSED", expected_commit=commit)
        if config.profile == "full"
        else None
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = config.run_id or f"p4-{config.profile}-{timestamp}"
    directory = config.resume or config.output_root / run_id
    registry_path = directory / "registry.json"
    if directory.exists() and not registry_path.is_file():
        raise FileExistsError(f"P4 run directory exists without registry: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if registry.get("run_id") != run_id:
            raise ValueError("P4 resume run_id does not match registry")
        if (
            registry.get("config_hash") != config.config_hash()
            or registry.get("matrix_hash") != config.matrix_hash()
        ):
            raise ValueError("P4 resume configuration or matrix hash does not match")
    else:
        registry = _initial_registry(config, run_id)
        _write_json(directory / "config.json", config.model_dump(mode="json"))
        _write_json(registry_path, registry)
    started = time.perf_counter()
    prior = float(registry.get("wall_clock_seconds", 0.0))
    if not math.isfinite(prior) or prior < 0:
        raise ValueError("P4 prior wall-clock ledger is invalid")
    deadline = started + max(config.budget.wall_clock_hours * 3600.0 - prior, 0.0)
    registry["status"] = "running"
    entries = cast(list[dict[str, Any]], registry["cells"])

    def persist() -> None:
        registry["wall_clock_seconds"] = prior + time.perf_counter() - started
        _write_json(registry_path, registry)

    persist()
    for cursor, cell in enumerate(config.matrix()):
        entry = entries[cursor]
        if entry["cell_id"] != cell.cell_id:
            raise ValueError("P4 registry order differs from frozen matrix")
        if entry["status"] == "COMPLETED":
            continue
        if (directory / "STOP").is_file():
            registry["status"] = "stopped"
            break
        if time.perf_counter() >= deadline or shutil.disk_usage(directory).free < 20 * GIB:
            entry["status"] = "RESOURCE_LIMIT"
            registry["status"] = "resource_limit"
            break
        cell_directory = directory / "cells" / cell.cell_id
        cell_directory.mkdir(parents=True, exist_ok=True)
        entry["status"] = "RUNNING"
        entry["artifact_dir"] = cell_directory.relative_to(directory).as_posix()
        entry["attempts"].append({"started_at": datetime.now(UTC).isoformat()})
        persist()
        try:
            cell_started = time.perf_counter()
            model, matching = _build_model(config, cell, device)
            if cell.max_steps == 0:
                _load_parent(model, cell, directory, device)
            training = (
                _train_cell(
                    model,
                    cell,
                    config,
                    device,
                    cell_directory,
                    directory,
                    cursor,
                    deadline,
                    pilot_hash,
                    mechanism_hash,
                )
                if cell.max_steps
                else {"steps": 0, "wall_clock_seconds": 0.0}
            )
            if training.get("stopped"):
                entry["status"] = "PENDING"
                registry["status"] = "stopped"
                break
            if training.get("resource_limited"):
                entry["status"] = "RESOURCE_LIMIT"
                registry["status"] = "resource_limit"
                break
            if config.profile == "pilot":
                # Pilot selection is validation-only.  Do not instantiate or
                # access analysis/test/OOD before the preset lock exists.
                evaluation = {
                    "record_count": 0,
                    "views": {"selection": "validation_only"},
                    "routing": {},
                    "prediction": training["prediction"],
                    "telemetry_v2_events": 0,
                }
            else:
                evaluation = _evaluate_cell(model, cell, config, device, cell_directory, deadline)
            summary = {
                "schema_version": "p4-cell-summary-v1",
                "cell": cell.model_dump(mode="json"),
                "matching": matching,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
                "training": training,
                "evaluation": evaluation,
                "wall_clock_seconds": time.perf_counter() - cell_started,
            }
            _write_json(cell_directory / "summary.json", summary)
            entry["status"] = "COMPLETED"
            entry["error"] = None
            entry["attempts"][-1]["completed_at"] = datetime.now(UTC).isoformat()
            entry["attempts"][-1]["result"] = "COMPLETED"
        except P4ResourceLimit as error:
            entry["status"] = "RESOURCE_LIMIT"
            entry["error"] = {"type": type(error).__name__, "message": str(error)}
            registry["status"] = "resource_limit"
            break
        except Exception as error:
            entry["status"] = "FAILED"
            entry["error"] = {"type": type(error).__name__, "message": str(error)}
            entry["attempts"][-1]["completed_at"] = datetime.now(UTC).isoformat()
            entry["attempts"][-1]["result"] = "FAILED"
            _write_json(
                cell_directory / "failure.json",
                {
                    "schema_version": "p4-failure-v1",
                    "cell_id": cell.cell_id,
                    "error": entry["error"],
                    "git_commit": commit,
                    "config_hash": config.config_hash(),
                    "protocol_hash": PROTOCOL_HASH,
                },
            )
        persist()
    statuses = [entry["status"] for entry in entries]
    if all(status == "COMPLETED" for status in statuses):
        registry["status"] = (
            f"{config.profile}_passed"
            if config.profile in {"qualification", "pilot"}
            else "completed"
        )
    elif registry.get("status") == "running":
        registry["status"] = f"{config.profile}_failed"
    registry["completed_cells"] = statuses.count("COMPLETED")
    registry["failed_cells"] = statuses.count("FAILED")
    registry["resource_limited_cells"] = statuses.count("RESOURCE_LIMIT")
    registry["total_cells"] = len(statuses)
    registry["wall_clock_seconds"] = prior + time.perf_counter() - started
    if config.profile == "qualification":
        evidence = _qualification_evidence(config, directory)
        registry["qualification_evidence"] = evidence
        if evidence["status"] != "PASSED":
            registry["status"] = "qualification_failed"
        report_path = directory / "qualification-report.json"
        _write_json(
            report_path,
            {
                "schema_version": "p4-qualification-report-v1",
                "qualification_only": True,
                "status": "PASSED" if registry["status"] == "qualification_passed" else "FAILED",
                "run_id": run_id,
                "git_commit": commit,
                "git_dirty": dirty,
                "device": str(device),
                "config_hash": config.config_hash(),
                "matrix_hash": config.matrix_hash(),
                "completed_cells": registry["completed_cells"],
                "total_cells": registry["total_cells"],
                "all_cells_completed": evidence["all_cells_completed"],
                "all_cells_covered": evidence["all_cells_covered"],
                "cpu_micro_required": evidence["cpu_micro_required"],
                "mps_required": evidence["mps_required"],
                "forecast_path_seen": evidence["forecast_path_seen"],
                "feedback_nonzero": evidence["feedback_nonzero"],
                "sparse_mac_less_than_dense": evidence["sparse_mac_less_than_dense"],
                "dense_control_matches_dense_macs": evidence["dense_control_matches_dense_macs"],
                "registry_checksum": None,
                "evidence": evidence,
            },
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["registry_checksum_basis"] = "canonical registry without artifacts"
        report["registry_checksum"] = _canonical_sha256(cast(Mapping[str, object], registry))
        _write_json(report_path, report)
        if (
            registry["status"] == "qualification_passed"
            and not dirty
            and _is_repository_path(report_path)
        ):
            _write_external_lock(
                config.control_root.parent / "qualification-lock.json",
                {
                    "schema_version": "p4-qualification-lock-v1",
                    "status": "PASSED",
                    "git_commit": commit,
                    "qualification_report": _repository_relative(report_path),
                    "qualification_report_sha256": _sha256(report_path),
                    "config_hash": config.config_hash(),
                    "matrix_hash": config.matrix_hash(),
                },
                dirty=False,
            )
    elif config.profile == "pilot" and registry["status"] == "pilot_passed":
        selection = _write_pilot_lock(config, directory, git_commit=commit, git_dirty=dirty)
        registry["pilot_selection"] = selection
        selection_path = directory / "pilot-selection.json"
        if not dirty and _is_repository_path(selection_path):
            _write_external_lock(
                config.control_root.parent / "pilot-lock.json",
                {
                    "schema_version": "p4-pilot-lock-v1",
                    "status": "PASSED",
                    "git_commit": commit,
                    "pilot_selection": _repository_relative(selection_path),
                    "pilot_selection_sha256": _sha256(selection_path),
                    "selected_preset": selection["selected_preset"],
                    "optimizer": selection["optimizer"],
                    "config_hash": config.config_hash(),
                    "matrix_hash": config.matrix_hash(),
                },
                dirty=False,
            )
    elif config.profile == "mechanism" and registry["status"] == "completed":
        evidence = _mechanism_gate_evidence(config, directory)
        report_path = directory / "mechanism-report.json"
        report = {
            "schema_version": "p4-mechanism-report-v1",
            "status": evidence["status"],
            "run_id": run_id,
            "git_commit": commit,
            "git_dirty": dirty,
            "protocol_hash": PROTOCOL_HASH,
            "config_hash": config.config_hash(),
            "matrix_hash": config.matrix_hash(),
            "completed_cells": registry["completed_cells"],
            "total_cells": registry["total_cells"],
            "evidence": evidence,
        }
        _write_json(report_path, report)
        registry["mechanism_evidence"] = evidence
        registry["status"] = (
            "mechanism_passed" if evidence["status"] == "PASSED" else "mechanism_failed"
        )
        if (
            registry["status"] == "mechanism_passed"
            and not dirty
            and _is_repository_path(report_path)
        ):
            _write_external_lock(
                config.control_root.parent / "mechanism-lock.json",
                {
                    "schema_version": "p4-mechanism-lock-v1",
                    "status": "PASSED",
                    "git_commit": commit,
                    "mechanism_report": _repository_relative(report_path),
                    "mechanism_report_sha256": _sha256(report_path),
                    "protocol_hash": PROTOCOL_HASH,
                    "config_hash": config.config_hash(),
                    "matrix_hash": config.matrix_hash(),
                },
                dirty=False,
            )
    registry["artifacts"] = {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"registry.json", "heartbeat.json"}
    }
    _write_json(registry_path, registry)
    return {
        "run_id": run_id,
        "artifact_dir": str(directory),
        "status": registry["status"],
        "completed_cells": registry["completed_cells"],
        "total_cells": registry["total_cells"],
    }


def verify_p4_run(directory: Path) -> dict[str, object]:
    """Verify frozen config, registry completeness, checksums, and sample identity."""

    registry = json.loads((directory / "registry.json").read_text(encoding="utf-8"))
    config = P4SuiteConfig.model_validate(
        json.loads((directory / "config.json").read_text(encoding="utf-8"))
    )
    if registry.get("schema_version") != "p4-suite-registry-v1":
        raise ValueError("invalid P4 registry schema")
    if (
        registry.get("config_hash") != config.config_hash()
        or registry.get("matrix_hash") != config.matrix_hash()
    ):
        raise ValueError("P4 registry differs from frozen config or matrix")
    cells = cast(list[dict[str, Any]], registry.get("cells"))
    ids = [cell["cell_id"] for cell in cells]
    expected = [cell.cell_id for cell in config.matrix()]
    if ids != expected or len(ids) != len(set(ids)):
        raise ValueError("P4 registry cells are missing, duplicated, or reordered")
    artifacts = cast(dict[str, str], registry.get("artifacts", {}))
    for relative, checksum in artifacts.items():
        path = directory / relative
        if not path.is_file() or _sha256(path) != checksum:
            raise ValueError(f"P4 artifact checksum mismatch: {relative}")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {"registry.json", "heartbeat.json"}
    }
    if actual != set(artifacts):
        raise ValueError("P4 artifact inventory differs from registry")
    if config.profile == "qualification":
        report = json.loads((directory / "qualification-report.json").read_text(encoding="utf-8"))
        core_registry = dict(registry)
        core_registry.pop("artifacts", None)
        if report.get("registry_checksum") != _canonical_sha256(core_registry):
            raise ValueError("P4 qualification registry checksum mismatch")
    sample_keys: set[tuple[object, ...]] = set()
    sample_count = 0
    for cell in cells:
        artifact_dir = cell.get("artifact_dir")
        if cell["status"] == "COMPLETED":
            if not isinstance(artifact_dir, str):
                raise ValueError(f"completed P4 cell lacks artifact_dir: {cell['cell_id']}")
            summary = json.loads(
                (directory / artifact_dir / "summary.json").read_text(encoding="utf-8")
            )
            if summary.get("schema_version") != "p4-cell-summary-v1":
                raise ValueError(f"invalid P4 cell summary: {cell['cell_id']}")
        records = directory / str(artifact_dir) / "sample_records.jsonl"
        if not records.is_file():
            continue
        for line in records.read_text(encoding="utf-8").splitlines():
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
                raise ValueError(f"duplicated P4 sample record: {key}")
            sample_keys.add(key)
            sample_count += 1
    return {
        "run_id": registry["run_id"],
        "status": registry["status"],
        "cells": len(cells),
        "missing_cells": [cell["cell_id"] for cell in cells if cell["status"] != "COMPLETED"],
        "checksums_ok": True,
        "sample_records": sample_count,
        "registered_artifacts": len(artifacts),
    }


execute_p4_suite = run_p4_suite

__all__ = ["P4ResourceLimit", "execute_p4_suite", "run_p4_suite", "verify_p4_run"]
