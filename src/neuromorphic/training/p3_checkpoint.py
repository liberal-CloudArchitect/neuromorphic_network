"""Fully validated checkpoint-v3 for P3 matrix cells."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from neuromorphic.training.checkpoint import CheckpointCompatibilityError
from neuromorphic.training.reproducibility import capture_rng_state, restore_rng_state

P3_CHECKPOINT_VERSION = "p3-checkpoint-v3"


@dataclass(frozen=True, slots=True)
class P3CheckpointState:
    cell_id: str
    global_step: int
    task_steps: Mapping[str, int]
    sampler_states: Mapping[str, Mapping[str, Any]]
    best_metrics: Mapping[str, float]
    stale_evaluations: int
    matrix_cursor: int
    config_hash: str
    protocol_hash: str
    analysis_curves: Mapping[str, tuple[tuple[int, float], ...]]
    validation_curve: tuple[tuple[int, float], ...]
    last_loss: float | None


def _validate_curve(points: object, *, name: str) -> tuple[tuple[int, float], ...]:
    if not isinstance(points, (list, tuple)):
        raise CheckpointCompatibilityError(f"P3 checkpoint {name} curve is invalid")
    parsed: list[tuple[int, float]] = []
    previous = -1
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise CheckpointCompatibilityError(f"P3 checkpoint {name} point is invalid")
        step, value = point
        if isinstance(step, bool) or not isinstance(step, int) or step < previous:
            raise CheckpointCompatibilityError(f"P3 checkpoint {name} steps are invalid")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise CheckpointCompatibilityError(f"P3 checkpoint {name} values are invalid")
        parsed.append((step, float(value)))
        previous = step
    return tuple(parsed)


def _validate_sampler_state(
    state: object,
    *,
    task_id: str,
    expected_size: int | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise CheckpointCompatibilityError(f"P3 checkpoint sampler is invalid: {task_id}")
    size = state.get("size")
    seed = state.get("seed")
    cursor = state.get("cursor")
    epoch = state.get("epoch")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(cursor, bool)
        or not isinstance(cursor, int)
        or not 0 <= cursor < size
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
    ):
        raise CheckpointCompatibilityError(f"P3 checkpoint sampler counters are invalid: {task_id}")
    if expected_size is not None and size != expected_size:
        raise CheckpointCompatibilityError(f"P3 checkpoint sampler size does not match: {task_id}")
    if expected_seed is not None and seed != expected_seed:
        raise CheckpointCompatibilityError(f"P3 checkpoint sampler seed does not match: {task_id}")
    permutation = state.get("permutation")
    if not isinstance(permutation, (list, tuple)) or sorted(permutation) != list(range(size)):
        raise CheckpointCompatibilityError(
            f"P3 checkpoint sampler permutation is invalid: {task_id}"
        )
    generator_state = state.get("generator_state")
    if not isinstance(generator_state, Mapping):
        raise CheckpointCompatibilityError(f"P3 checkpoint sampler generator is invalid: {task_id}")
    try:
        generator = np.random.default_rng(seed)
        generator.bit_generator.state = copy.deepcopy(dict(generator_state))
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointCompatibilityError(
            f"P3 checkpoint sampler generator is incompatible: {task_id}"
        ) from error
    return copy.deepcopy(dict(state))


def _validate_rng_state(state: Mapping[str, Any]) -> None:
    current = capture_rng_state()
    try:
        restore_rng_state(copy.deepcopy(dict(state)))
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise CheckpointCompatibilityError("P3 checkpoint RNG state is incompatible") from error
    finally:
        restore_rng_state(current)


def _parameter_names(model: nn.Module, optimizer: Optimizer) -> tuple[tuple[str, ...], ...]:
    by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    groups: list[tuple[str, ...]] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            try:
                names.append(by_id[id(parameter)])
            except KeyError as error:
                raise ValueError("optimizer contains a parameter not owned by the model") from error
        groups.append(tuple(names))
    return tuple(groups)


def save_p3_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    state: P3CheckpointState,
) -> None:
    if not state.cell_id or state.global_step < 0 or state.matrix_cursor < 0:
        raise ValueError("invalid P3 checkpoint cursor")
    if any(value < 0 for value in state.task_steps.values()) or state.stale_evaluations < 0:
        raise ValueError("invalid P3 checkpoint counters")
    if state.last_loss is not None and not math.isfinite(state.last_loss):
        raise ValueError("invalid P3 checkpoint last loss")
    if any(not math.isfinite(value) for value in state.best_metrics.values()):
        raise ValueError("invalid P3 checkpoint best metrics")
    for task_id, sampler_state in state.sampler_states.items():
        _validate_sampler_state(sampler_state, task_id=task_id)
    for name, curve in state.analysis_curves.items():
        _validate_curve(curve, name=f"analysis:{name}")
    _validate_curve(state.validation_curve, name="validation")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": P3_CHECKPOINT_VERSION,
            "cell_id": state.cell_id,
            "global_step": state.global_step,
            "task_steps": dict(state.task_steps),
            "sampler_states": copy.deepcopy(dict(state.sampler_states)),
            "best_metrics": dict(state.best_metrics),
            "stale_evaluations": state.stale_evaluations,
            "matrix_cursor": state.matrix_cursor,
            "config_hash": state.config_hash,
            "protocol_hash": state.protocol_hash,
            "analysis_curves": {
                name: tuple(points) for name, points in state.analysis_curves.items()
            },
            "validation_curve": tuple(state.validation_curve),
            "last_loss": state.last_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "optimizer_parameter_names": _parameter_names(model, optimizer),
            "rng_state": capture_rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def load_p3_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    expected_cell_id: str,
    expected_config_hash: str,
    expected_protocol_hash: str,
    expected_matrix_cursor: int | None = None,
    expected_sampler_signatures: Mapping[str, tuple[int, int]] | None = None,
) -> P3CheckpointState:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != P3_CHECKPOINT_VERSION:
        raise CheckpointCompatibilityError("unsupported P3 checkpoint schema")
    checks = {
        "cell_id": expected_cell_id,
        "config_hash": expected_config_hash,
        "protocol_hash": expected_protocol_hash,
    }
    for name, expected in checks.items():
        if payload.get(name) != expected:
            raise CheckpointCompatibilityError(f"P3 checkpoint {name} does not match")
    model_state = payload.get("model_state")
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(model_state, Mapping) or not isinstance(optimizer_state, Mapping):
        raise CheckpointCompatibilityError("P3 checkpoint parameter state is invalid")
    live = model.state_dict()
    if live.keys() != model_state.keys():
        raise CheckpointCompatibilityError("P3 checkpoint model keys do not match")
    for name, value in model_state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != live[name].shape
            or value.dtype != live[name].dtype
        ):
            raise CheckpointCompatibilityError(f"P3 checkpoint tensor is incompatible: {name}")
    expected_names = _parameter_names(model, optimizer)
    saved_names = payload.get("optimizer_parameter_names")
    if saved_names != expected_names:
        raise CheckpointCompatibilityError("P3 checkpoint optimizer groups do not match")
    required_mappings = (
        "task_steps",
        "sampler_states",
        "best_metrics",
        "rng_state",
        "analysis_curves",
    )
    if any(not isinstance(payload.get(name), Mapping) for name in required_mappings):
        raise CheckpointCompatibilityError("P3 checkpoint metadata mapping is invalid")
    integer_fields = ("global_step", "stale_evaluations", "matrix_cursor")
    if any(
        isinstance(payload.get(name), bool)
        or not isinstance(payload.get(name), int)
        or int(payload[name]) < 0
        for name in integer_fields
    ):
        raise CheckpointCompatibilityError("P3 checkpoint cursor is invalid")
    if expected_matrix_cursor is not None and payload["matrix_cursor"] != expected_matrix_cursor:
        raise CheckpointCompatibilityError("P3 checkpoint matrix cursor does not match")
    task_steps: dict[str, int] = {}
    for key, value in payload["task_steps"].items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise CheckpointCompatibilityError("P3 checkpoint task counters are invalid")
        task_steps[key] = value
    sampler_signatures = expected_sampler_signatures or {}
    if sampler_signatures and (
        set(task_steps) != set(sampler_signatures)
        or set(payload["sampler_states"]) != set(sampler_signatures)
    ):
        raise CheckpointCompatibilityError("P3 checkpoint task or sampler IDs do not match")
    sampler_states: dict[str, Mapping[str, Any]] = {}
    for key, value in payload["sampler_states"].items():
        if not isinstance(key, str) or not key:
            raise CheckpointCompatibilityError("P3 checkpoint sampler ID is invalid")
        signature = sampler_signatures.get(key)
        sampler_states[key] = _validate_sampler_state(
            value,
            task_id=key,
            expected_size=None if signature is None else signature[0],
            expected_seed=None if signature is None else signature[1],
        )
    best_metrics: dict[str, float] = {}
    for key, value in payload["best_metrics"].items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise CheckpointCompatibilityError("P3 checkpoint best metrics are invalid")
        best_metrics[key] = float(value)
    analysis_curves = {
        str(name): _validate_curve(points, name=f"analysis:{name}")
        for name, points in payload["analysis_curves"].items()
    }
    validation_curve = _validate_curve(payload.get("validation_curve"), name="validation")
    last_loss = payload.get("last_loss")
    if last_loss is not None and (
        isinstance(last_loss, bool)
        or not isinstance(last_loss, (int, float))
        or not math.isfinite(float(last_loss))
    ):
        raise CheckpointCompatibilityError("P3 checkpoint last_loss is invalid")
    _validate_rng_state(payload["rng_state"])
    # No live object is mutated before every compatibility check above succeeds.
    model.load_state_dict(model_state)
    optimizer.load_state_dict(dict(optimizer_state))
    device = next(model.parameters()).device
    for parameter_state in optimizer.state.values():
        for name, value in parameter_state.items():
            if isinstance(value, torch.Tensor):
                parameter_state[name] = value.to(device)
    restore_rng_state(dict(payload["rng_state"]))
    return P3CheckpointState(
        cell_id=expected_cell_id,
        global_step=int(payload["global_step"]),
        task_steps=task_steps,
        sampler_states=sampler_states,
        best_metrics=best_metrics,
        stale_evaluations=int(payload["stale_evaluations"]),
        matrix_cursor=int(payload["matrix_cursor"]),
        config_hash=expected_config_hash,
        protocol_hash=expected_protocol_hash,
        analysis_curves=analysis_curves,
        validation_curve=validation_curve,
        last_loss=None if last_loss is None else float(last_loss),
    )


__all__ = ["P3CheckpointState", "load_p3_checkpoint", "save_p3_checkpoint"]
