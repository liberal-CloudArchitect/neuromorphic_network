"""Strict atomic checkpoint-v6 persistence for P5 qualification and experiment runs."""

from __future__ import annotations

import copy
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer

from neuromorphic.core.network_state import NetworkState
from neuromorphic.training.checkpoint import CheckpointCompatibilityError
from neuromorphic.training.p4_checkpoint import (
    _cpu_clone,
    _deserialize_network_state,
    _move_optimizer_state,
    _object_id,
    _parameter_names,
    _serialize_network_state,
    _validate_model_state,
    _validate_optimizer_state,
    _validate_rng_state,
)
from neuromorphic.training.reproducibility import capture_rng_state, restore_rng_state

P5_CHECKPOINT_VERSION = "p5-checkpoint-v6"


@dataclass(frozen=True, slots=True)
class P5CheckpointState:
    """All cursors required to resume one P5 qualification, pilot, or mechanism cell."""

    profile: Literal["qualification", "pilot", "mechanism"]
    candidate_id: str
    candidate_index: int
    global_step: int
    task_steps: Mapping[str, int]
    config_hash: str
    protocol_hash: str
    best_metrics: Mapping[str, float]
    validation_macro: tuple[float, ...]
    router_gradient_seen: bool
    predictor_gradient_seen: bool
    stale_evaluations: int
    last_loss: float | None
    network_state: NetworkState | None = None
    analysis_macro_curve: tuple[tuple[int, float], ...] = ()


def _validate_state(state: P5CheckpointState) -> None:
    if state.profile not in {"qualification", "pilot", "mechanism"}:
        raise ValueError("invalid P5 checkpoint profile")
    if not state.candidate_id:
        raise ValueError("candidate_id cannot be empty")
    if min(state.candidate_index, state.global_step, state.stale_evaluations) < 0:
        raise ValueError("P5 checkpoint cursors must be non-negative")
    if not state.config_hash or not state.protocol_hash:
        raise ValueError("P5 checkpoint hashes cannot be empty")
    for task_id, step in state.task_steps.items():
        if not task_id or isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("P5 checkpoint task cursor is invalid")
    for name, metric in state.best_metrics.items():
        if not name or not math.isfinite(float(metric)):
            raise ValueError("P5 checkpoint best metric is invalid")
    if any(not math.isfinite(value) for value in state.validation_macro):
        raise ValueError("P5 checkpoint validation curve is invalid")
    previous_step = -1
    for point in state.analysis_macro_curve:
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or isinstance(point[0], bool)
            or not isinstance(point[0], int)
            or point[0] < 0
            or point[0] <= previous_step
            or isinstance(point[1], bool)
            or not isinstance(point[1], int | float)
            or not math.isfinite(float(point[1]))
        ):
            raise ValueError("P5 checkpoint analysis curve is invalid")
        previous_step = point[0]
    if state.last_loss is not None and not math.isfinite(state.last_loss):
        raise ValueError("P5 checkpoint loss must be finite")


def save_p5_checkpoint(
    path: Path, *, model: nn.Module, optimizer: Optimizer, state: P5CheckpointState
) -> None:
    """Atomically save a complete P5 candidate boundary."""

    _validate_state(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": P5_CHECKPOINT_VERSION,
        **{
            field.name: getattr(state, field.name)
            for field in fields(state)
            if field.name != "network_state"
        },
        "task_steps": dict(state.task_steps),
        "best_metrics": dict(state.best_metrics),
        "network_state": _serialize_network_state(state.network_state),
        "model_id": _object_id(model),
        "optimizer_id": _object_id(optimizer),
        "model_state": _cpu_clone(model.state_dict()),
        "optimizer_state": _cpu_clone(optimizer.state_dict()),
        "optimizer_parameter_names": _parameter_names(model, optimizer),
        "rng_state": capture_rng_state(),
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_p5_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    expected_profile: Literal["qualification", "pilot", "mechanism"],
    expected_candidate_id: str,
    expected_candidate_index: int,
    expected_config_hash: str,
    expected_protocol_hash: str,
    expected_network_state: NetworkState | None = None,
    restore_rng: bool = True,
) -> P5CheckpointState:
    """Validate the full payload before mutating live model, optimizer, or RNG."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != P5_CHECKPOINT_VERSION:
        raise CheckpointCompatibilityError("unsupported P5 checkpoint schema")
    expected: dict[str, object] = {
        "profile": expected_profile,
        "candidate_id": expected_candidate_id,
        "candidate_index": expected_candidate_index,
        "config_hash": expected_config_hash,
        "protocol_hash": expected_protocol_hash,
        "model_id": _object_id(model),
        "optimizer_id": _object_id(optimizer),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise CheckpointCompatibilityError(f"P5 checkpoint {name} does not match")
    model_state = _validate_model_state(model, payload.get("model_state"))
    optimizer_state = _validate_optimizer_state(optimizer, payload.get("optimizer_state"))
    if payload.get("optimizer_parameter_names") != _parameter_names(model, optimizer):
        raise CheckpointCompatibilityError("P5 checkpoint optimizer groups do not match")
    if not isinstance(payload.get("task_steps"), Mapping) or not isinstance(
        payload.get("best_metrics"), Mapping
    ):
        raise CheckpointCompatibilityError("P5 checkpoint metadata mapping is invalid")
    curve = payload.get("validation_macro")
    if not isinstance(curve, tuple) or any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        for value in curve
    ):
        raise CheckpointCompatibilityError("P5 checkpoint validation curve is invalid")
    analysis_curve = payload.get("analysis_macro_curve")
    if not isinstance(analysis_curve, tuple):
        raise CheckpointCompatibilityError("P5 checkpoint analysis curve is invalid")
    normalized_analysis_curve: list[tuple[int, float]] = []
    previous_step = -1
    for point in analysis_curve:
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or isinstance(point[0], bool)
            or not isinstance(point[0], int)
            or point[0] < 0
            or point[0] <= previous_step
            or isinstance(point[1], bool)
            or not isinstance(point[1], int | float)
            or not math.isfinite(float(point[1]))
        ):
            raise CheckpointCompatibilityError("P5 checkpoint analysis curve is invalid")
        normalized_analysis_curve.append((point[0], float(point[1])))
        previous_step = point[0]
    for name in ("router_gradient_seen", "predictor_gradient_seen"):
        if not isinstance(payload.get(name), bool):
            raise CheckpointCompatibilityError(f"P5 checkpoint {name} is invalid")
    task_steps: dict[str, int] = {}
    for task_id, step in payload["task_steps"].items():
        if (
            not isinstance(task_id, str)
            or not task_id
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
        ):
            raise CheckpointCompatibilityError("P5 checkpoint task cursor is invalid")
        task_steps[task_id] = step
    best_metrics: dict[str, float] = {}
    for name, metric in payload["best_metrics"].items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(metric, bool)
            or not isinstance(metric, int | float)
            or not math.isfinite(float(metric))
        ):
            raise CheckpointCompatibilityError("P5 checkpoint best metric is invalid")
        best_metrics[name] = float(metric)
    for name in ("global_step", "stale_evaluations"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CheckpointCompatibilityError(f"P5 checkpoint {name} is invalid")
    last_loss = payload.get("last_loss")
    if last_loss is not None and (
        isinstance(last_loss, bool)
        or not isinstance(last_loss, int | float)
        or not math.isfinite(float(last_loss))
    ):
        raise CheckpointCompatibilityError("P5 checkpoint loss is invalid")
    network_state = _deserialize_network_state(payload.get("network_state"), expected_network_state)
    rng_state = _validate_rng_state(payload.get("rng_state"))
    restored = P5CheckpointState(
        profile=expected_profile,
        candidate_id=expected_candidate_id,
        candidate_index=expected_candidate_index,
        global_step=int(payload["global_step"]),
        task_steps=task_steps,
        config_hash=expected_config_hash,
        protocol_hash=expected_protocol_hash,
        best_metrics=best_metrics,
        validation_macro=tuple(float(value) for value in curve),
        router_gradient_seen=bool(payload["router_gradient_seen"]),
        predictor_gradient_seen=bool(payload["predictor_gradient_seen"]),
        stale_evaluations=int(payload["stale_evaluations"]),
        last_loss=None if last_loss is None else float(last_loss),
        network_state=network_state,
        analysis_macro_curve=tuple(normalized_analysis_curve),
    )
    try:
        _validate_state(restored)
        staged_model, staged_optimizer = copy.deepcopy((model, optimizer))
        staged_model.load_state_dict(model_state)
        staged_optimizer.load_state_dict(dict(optimizer_state))
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise CheckpointCompatibilityError("P5 checkpoint cannot be applied") from error
    model.load_state_dict(model_state)
    optimizer.load_state_dict(dict(optimizer_state))
    _move_optimizer_state(optimizer, next(model.parameters()).device)
    if restore_rng:
        restore_rng_state(rng_state)
    return restored


__all__ = [
    "P5_CHECKPOINT_VERSION",
    "P5CheckpointState",
    "load_p5_checkpoint",
    "save_p5_checkpoint",
]
