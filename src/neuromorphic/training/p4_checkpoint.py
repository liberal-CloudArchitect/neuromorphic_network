"""Strict, atomic checkpoint-v4 persistence for P4 matrix cells.

The loader treats a checkpoint as untrusted input.  It validates every model,
optimizer, sampler, recurrent-state, cursor, hash, and RNG boundary before it
mutates the live training objects.
"""

from __future__ import annotations

import copy
import math
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from neuromorphic.core.contracts import ModuleState
from neuromorphic.core.network_state import NetworkState
from neuromorphic.training.checkpoint import CheckpointCompatibilityError
from neuromorphic.training.reproducibility import capture_rng_state, restore_rng_state

P4_CHECKPOINT_VERSION = "p4-checkpoint-v4"


@dataclass(frozen=True, slots=True)
class P4CheckpointState:
    """All non-parameter state needed to resume one exact P4 matrix cell."""

    cell_id: str
    global_step: int
    task_steps: Mapping[str, int]
    sampler_states: Mapping[str, Mapping[str, Any]]
    best_metrics: Mapping[str, float]
    stale_evaluations: int
    matrix_cursor: int
    config_hash: str
    protocol_hash: str
    matrix_hash: str
    pilot_lock_hash: str | None
    mechanism_lock_hash: str | None
    analysis_curves: Mapping[str, tuple[tuple[int, float], ...]]
    last_loss: float | None
    cumulative_wall_clock_seconds: float
    transition_count: int
    prediction_totals: Mapping[str, float]
    validation_prediction_totals: Mapping[str, float]
    network_state: NetworkState | None = None


def _object_id(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _parameter_names(model: nn.Module, optimizer: Optimizer) -> tuple[tuple[str, ...], ...]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    try:
        return tuple(
            tuple(names[id(parameter)] for parameter in group["params"])
            for group in optimizer.param_groups
        )
    except KeyError as error:
        raise ValueError("optimizer contains a parameter not owned by the P4 model") from error


def _validate_curve(value: object, *, name: str) -> tuple[tuple[int, float], ...]:
    if not isinstance(value, (list, tuple)):
        raise CheckpointCompatibilityError(f"P4 checkpoint curve is invalid: {name}")
    parsed: list[tuple[int, float]] = []
    previous = -1
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise CheckpointCompatibilityError(f"P4 checkpoint curve point is invalid: {name}")
        step, metric = point
        if isinstance(step, bool) or not isinstance(step, int) or step < previous:
            raise CheckpointCompatibilityError(f"P4 checkpoint curve steps are invalid: {name}")
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
        ):
            raise CheckpointCompatibilityError(f"P4 checkpoint curve values are invalid: {name}")
        parsed.append((step, float(metric)))
        previous = step
    return tuple(parsed)


def _validate_sampler_state(
    value: object,
    *,
    task_id: str,
    expected_signature: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError(f"P4 checkpoint sampler is invalid: {task_id}")
    size = value.get("size")
    seed = value.get("seed")
    cursor = value.get("cursor")
    epoch = value.get("epoch")
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
        raise CheckpointCompatibilityError(f"P4 checkpoint sampler counters are invalid: {task_id}")
    if expected_signature is not None and (size, seed) != expected_signature:
        raise CheckpointCompatibilityError(
            f"P4 checkpoint sampler signature does not match: {task_id}"
        )
    permutation = value.get("permutation")
    if not isinstance(permutation, (list, tuple)) or sorted(permutation) != list(range(size)):
        raise CheckpointCompatibilityError(
            f"P4 checkpoint sampler permutation is invalid: {task_id}"
        )
    generator_state = value.get("generator_state")
    if not isinstance(generator_state, Mapping):
        raise CheckpointCompatibilityError(f"P4 checkpoint sampler generator is invalid: {task_id}")
    try:
        generator = np.random.default_rng(seed)
        generator.bit_generator.state = copy.deepcopy(dict(generator_state))
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointCompatibilityError(
            f"P4 checkpoint sampler generator is incompatible: {task_id}"
        ) from error
    return copy.deepcopy(dict(value))


def _validate_state(state: P4CheckpointState) -> None:
    if (
        not state.cell_id
        or isinstance(state.global_step, bool)
        or not isinstance(state.global_step, int)
        or state.global_step < 0
        or isinstance(state.matrix_cursor, bool)
        or not isinstance(state.matrix_cursor, int)
        or state.matrix_cursor < 0
    ):
        raise ValueError("invalid P4 checkpoint cursor")
    hashes = (state.config_hash, state.protocol_hash, state.matrix_hash)
    if any(not value or value.isspace() for value in hashes):
        raise ValueError("P4 checkpoint hashes must be non-empty")
    if (
        isinstance(state.stale_evaluations, bool)
        or not isinstance(state.stale_evaluations, int)
        or state.stale_evaluations < 0
        or isinstance(state.transition_count, bool)
        or not isinstance(state.transition_count, int)
        or state.transition_count < 0
    ):
        raise ValueError("invalid P4 checkpoint counters")
    if state.cumulative_wall_clock_seconds < 0 or not math.isfinite(
        state.cumulative_wall_clock_seconds
    ):
        raise ValueError("invalid P4 checkpoint wall-clock ledger")
    if state.last_loss is not None and not math.isfinite(state.last_loss):
        raise ValueError("invalid P4 checkpoint loss")
    if any(
        not isinstance(name, str)
        or not name
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for name, value in state.task_steps.items()
    ):
        raise ValueError("invalid P4 task cursor")
    if any(
        not isinstance(name, str)
        or not name
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for name, value in state.best_metrics.items()
    ):
        raise ValueError("invalid P4 best metric")
    for totals in (state.prediction_totals, state.validation_prediction_totals):
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for name, value in totals.items()
        ):
            raise ValueError("invalid P4 prediction accumulator")
    for task_id, sampler in state.sampler_states.items():
        _validate_sampler_state(sampler, task_id=task_id)
    for name, points in state.analysis_curves.items():
        _validate_curve(points, name=name)
    if state.network_state is not None:
        # Construction already validates owner/version/device/batch invariants.
        NetworkState(state.network_state.module_states, state.network_state.valid_step_counts)
        for module_state in state.network_state.module_states.values():
            for tensor in module_state.tensors.values():
                if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
                    raise ValueError("P4 network state tensors must be finite")


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def _serialize_network_state(state: NetworkState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "valid_step_counts": _cpu_clone(state.valid_step_counts),
        "module_states": {
            module_id: {
                "owner": module_state.module_id,
                "version": module_state.version,
                "tensors": _cpu_clone(module_state.tensors),
            }
            for module_id, module_state in state.module_states.items()
        },
    }


def _deserialize_network_state(value: object, expected: NetworkState | None) -> NetworkState | None:
    if value is None:
        if expected is not None:
            raise CheckpointCompatibilityError("P4 checkpoint network state is missing")
        return None
    if expected is None:
        raise CheckpointCompatibilityError("P4 checkpoint has unexpected network state")
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError("P4 checkpoint network state is invalid")
    counters = value.get("valid_step_counts")
    modules = value.get("module_states")
    if not isinstance(counters, Tensor) or not isinstance(modules, Mapping):
        raise CheckpointCompatibilityError("P4 checkpoint network state fields are invalid")
    if counters.shape != expected.valid_step_counts.shape or counters.dtype != torch.long:
        raise CheckpointCompatibilityError("P4 checkpoint TBPTT counters are incompatible")
    if torch.any(counters < 0).item():
        raise CheckpointCompatibilityError("P4 checkpoint TBPTT counters are invalid")
    if set(modules) != set(expected.module_states):
        raise CheckpointCompatibilityError("P4 checkpoint module state IDs do not match")
    restored: dict[str, ModuleState] = {}
    device = expected.valid_step_counts.device
    for module_id, expected_module in expected.module_states.items():
        section = modules[module_id]
        if not isinstance(section, Mapping) or section.get("owner") != module_id:
            raise CheckpointCompatibilityError(
                f"P4 checkpoint module state owner does not match: {module_id}"
            )
        if section.get("version") != expected_module.version:
            raise CheckpointCompatibilityError(
                f"P4 checkpoint module state version does not match: {module_id}"
            )
        tensors = section.get("tensors")
        if not isinstance(tensors, Mapping) or set(tensors) != set(expected_module.tensors):
            raise CheckpointCompatibilityError(
                f"P4 checkpoint module tensor keys do not match: {module_id}"
            )
        checked: dict[str, Tensor] = {}
        for name, expected_tensor in expected_module.tensors.items():
            tensor = tensors[name]
            if (
                not isinstance(tensor, Tensor)
                or tensor.shape != expected_tensor.shape
                or tensor.dtype != expected_tensor.dtype
            ):
                raise CheckpointCompatibilityError(
                    f"P4 checkpoint module tensor is incompatible: {module_id}.{name}"
                )
            if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
                raise CheckpointCompatibilityError(
                    f"P4 checkpoint module tensor is non-finite: {module_id}.{name}"
                )
            checked[name] = tensor.to(device)
        restored[module_id] = ModuleState(module_id, expected_module.version, checked)
    return NetworkState(restored, counters.to(device))


def _validate_model_state(model: nn.Module, value: object) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError("P4 checkpoint model state is invalid")
    live = model.state_dict()
    if live.keys() != value.keys():
        raise CheckpointCompatibilityError("P4 checkpoint model keys do not match")
    checked: dict[str, Tensor] = {}
    for name, expected in live.items():
        tensor = value[name]
        if (
            not isinstance(tensor, Tensor)
            or tensor.shape != expected.shape
            or tensor.dtype != expected.dtype
        ):
            raise CheckpointCompatibilityError(f"P4 checkpoint tensor is incompatible: {name}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise CheckpointCompatibilityError(f"P4 checkpoint tensor is non-finite: {name}")
        checked[name] = tensor
    return checked


def _validate_optimizer_state(optimizer: Optimizer, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError("P4 checkpoint optimizer state is invalid")
    saved_groups = value.get("param_groups")
    saved_state = value.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_state, Mapping):
        raise CheckpointCompatibilityError("P4 checkpoint optimizer fields are invalid")
    if len(saved_groups) != len(optimizer.param_groups):
        raise CheckpointCompatibilityError("P4 checkpoint optimizer group count does not match")
    parameter_by_saved_id: dict[int, Tensor] = {}
    for saved_group, live_group in zip(saved_groups, optimizer.param_groups, strict=True):
        if not isinstance(saved_group, Mapping) or not isinstance(saved_group.get("params"), list):
            raise CheckpointCompatibilityError("P4 checkpoint optimizer group is invalid")
        saved_ids = saved_group["params"]
        live_parameters = live_group["params"]
        if set(saved_group) != set(live_group):
            raise CheckpointCompatibilityError("P4 checkpoint optimizer group keys do not match")
        for name, live_value in live_group.items():
            if name != "params" and saved_group[name] != live_value:
                raise CheckpointCompatibilityError(
                    f"P4 checkpoint optimizer group setting does not match: {name}"
                )
        if len(saved_ids) != len(live_parameters):
            raise CheckpointCompatibilityError(
                "P4 checkpoint optimizer parameter count does not match"
            )
        for saved_id, parameter in zip(saved_ids, live_parameters, strict=True):
            if isinstance(saved_id, bool) or not isinstance(saved_id, int):
                raise CheckpointCompatibilityError("P4 checkpoint optimizer ID is invalid")
            if saved_id in parameter_by_saved_id:
                raise CheckpointCompatibilityError("P4 checkpoint optimizer IDs are duplicated")
            parameter_by_saved_id[saved_id] = parameter
    if not set(saved_state).issubset(parameter_by_saved_id):
        raise CheckpointCompatibilityError("P4 checkpoint optimizer state has unknown IDs")
    for saved_id, parameter_state in saved_state.items():
        if not isinstance(parameter_state, Mapping):
            raise CheckpointCompatibilityError("P4 checkpoint optimizer parameter state is invalid")
        parameter = parameter_by_saved_id[saved_id]
        for tensor in parameter_state.values():
            if not isinstance(tensor, Tensor):
                continue
            if tensor.numel() != 1 and (
                tensor.shape != parameter.shape or tensor.dtype != parameter.dtype
            ):
                raise CheckpointCompatibilityError(
                    "P4 checkpoint optimizer tensor is incompatible with its parameter"
                )
            if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
                raise CheckpointCompatibilityError("P4 checkpoint optimizer tensor is non-finite")
    return value


def _validate_rng_state(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError("P4 checkpoint RNG state is invalid")
    state = copy.deepcopy(dict(value))
    missing = {"python", "numpy", "torch_cpu"}.difference(state)
    if missing:
        raise CheckpointCompatibilityError(f"P4 checkpoint RNG state is missing: {sorted(missing)}")
    try:
        random.Random().setstate(state["python"])
        numpy_rng = np.random.RandomState()
        numpy_rng.set_state(state["numpy"])
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.set_state(state["torch_cpu"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise CheckpointCompatibilityError("P4 checkpoint RNG state is incompatible") from error
    if "torch_mps" in state:
        if not torch.backends.mps.is_available():
            raise CheckpointCompatibilityError("P4 checkpoint requires unavailable MPS RNG")
        mps_state = state["torch_mps"]
        current_mps_state = torch.mps.get_rng_state()
        if (
            not isinstance(mps_state, Tensor)
            or mps_state.shape != current_mps_state.shape
            or mps_state.dtype != current_mps_state.dtype
        ):
            raise CheckpointCompatibilityError("P4 checkpoint MPS RNG state is incompatible")
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise CheckpointCompatibilityError("P4 checkpoint requires unavailable CUDA RNG")
        cuda_states = state["torch_cuda"]
        current_cuda_states = torch.cuda.get_rng_state_all()
        if (
            not isinstance(cuda_states, (list, tuple))
            or len(cuda_states) != len(current_cuda_states)
            or any(
                not isinstance(saved, Tensor)
                or saved.shape != current.shape
                or saved.dtype != current.dtype
                for saved, current in zip(cuda_states, current_cuda_states, strict=True)
            )
        ):
            raise CheckpointCompatibilityError("P4 checkpoint CUDA RNG state is incompatible")
    return state


def _move_optimizer_state(optimizer: Optimizer, device: torch.device) -> None:
    for parameter_state in optimizer.state.values():
        for name, value in parameter_state.items():
            if isinstance(value, Tensor):
                parameter_state[name] = value.to(device)


def save_p4_checkpoint(
    path: Path, *, model: nn.Module, optimizer: Optimizer, state: P4CheckpointState
) -> None:
    """Atomically save one complete P4 cell resume point."""

    _validate_state(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": P4_CHECKPOINT_VERSION,
        **{
            field.name: getattr(state, field.name)
            for field in fields(state)
            if field.name != "network_state"
        },
        "sampler_states": copy.deepcopy(dict(state.sampler_states)),
        "analysis_curves": {name: tuple(points) for name, points in state.analysis_curves.items()},
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


def load_p4_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    expected_cell_id: str,
    expected_config_hash: str,
    expected_protocol_hash: str,
    expected_matrix_hash: str,
    expected_matrix_cursor: int,
    expected_pilot_lock_hash: str | None,
    expected_mechanism_lock_hash: str | None,
    expected_sampler_signatures: Mapping[str, tuple[int, int]] | None = None,
    expected_network_state: NetworkState | None = None,
    restore_rng: bool = True,
) -> P4CheckpointState:
    """Validate a complete checkpoint, then restore model, optimizer, and RNG."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != P4_CHECKPOINT_VERSION:
        raise CheckpointCompatibilityError("unsupported P4 checkpoint schema")
    expected = {
        "cell_id": expected_cell_id,
        "config_hash": expected_config_hash,
        "protocol_hash": expected_protocol_hash,
        "matrix_hash": expected_matrix_hash,
        "matrix_cursor": expected_matrix_cursor,
        "pilot_lock_hash": expected_pilot_lock_hash,
        "mechanism_lock_hash": expected_mechanism_lock_hash,
        "model_id": _object_id(model),
        "optimizer_id": _object_id(optimizer),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise CheckpointCompatibilityError(f"P4 checkpoint {name} does not match")
    model_state = _validate_model_state(model, payload.get("model_state"))
    optimizer_state = _validate_optimizer_state(optimizer, payload.get("optimizer_state"))
    if payload.get("optimizer_parameter_names") != _parameter_names(model, optimizer):
        raise CheckpointCompatibilityError("P4 checkpoint optimizer groups do not match")

    required_mappings = (
        "task_steps",
        "sampler_states",
        "best_metrics",
        "analysis_curves",
        "prediction_totals",
        "validation_prediction_totals",
    )
    if any(not isinstance(payload.get(name), Mapping) for name in required_mappings):
        raise CheckpointCompatibilityError("P4 checkpoint metadata mapping is invalid")
    integer_fields = (
        "global_step",
        "stale_evaluations",
        "matrix_cursor",
        "transition_count",
    )
    if any(
        isinstance(payload.get(name), bool)
        or not isinstance(payload.get(name), int)
        or int(payload[name]) < 0
        for name in integer_fields
    ):
        raise CheckpointCompatibilityError("P4 checkpoint cursor is invalid")
    task_steps: dict[str, int] = {}
    for task_id, step in payload["task_steps"].items():
        if (
            not isinstance(task_id, str)
            or not task_id
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
        ):
            raise CheckpointCompatibilityError("P4 checkpoint task cursor is invalid")
        task_steps[task_id] = step
    signatures = dict(expected_sampler_signatures or {})
    if signatures and (
        set(signatures) != set(task_steps) or set(signatures) != set(payload["sampler_states"])
    ):
        raise CheckpointCompatibilityError("P4 checkpoint sampler or task IDs do not match")
    sampler_states: dict[str, Mapping[str, Any]] = {}
    for task_id, sampler in payload["sampler_states"].items():
        if not isinstance(task_id, str) or not task_id:
            raise CheckpointCompatibilityError("P4 checkpoint sampler ID is invalid")
        sampler_states[task_id] = _validate_sampler_state(
            sampler, task_id=task_id, expected_signature=signatures.get(task_id)
        )
    best_metrics: dict[str, float] = {}
    for name, metric in payload["best_metrics"].items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
        ):
            raise CheckpointCompatibilityError("P4 checkpoint best metric is invalid")
        best_metrics[name] = float(metric)
    prediction_totals: dict[str, float] = {}
    validation_prediction_totals: dict[str, float] = {}
    for source, target in (
        (payload["prediction_totals"], prediction_totals),
        (payload["validation_prediction_totals"], validation_prediction_totals),
    ):
        for name, metric in source.items():
            if (
                not isinstance(name, str)
                or not name
                or isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric))
            ):
                raise CheckpointCompatibilityError(
                    "P4 checkpoint prediction accumulator is invalid"
                )
            target[name] = float(metric)
    analysis_curves = {
        str(name): _validate_curve(curve, name=str(name))
        for name, curve in payload["analysis_curves"].items()
    }
    last_loss = payload.get("last_loss")
    if last_loss is not None and (
        isinstance(last_loss, bool)
        or not isinstance(last_loss, (int, float))
        or not math.isfinite(float(last_loss))
    ):
        raise CheckpointCompatibilityError("P4 checkpoint loss is invalid")
    wall_clock = payload.get("cumulative_wall_clock_seconds")
    if (
        isinstance(wall_clock, bool)
        or not isinstance(wall_clock, (int, float))
        or not math.isfinite(float(wall_clock))
        or float(wall_clock) < 0
    ):
        raise CheckpointCompatibilityError("P4 checkpoint wall-clock ledger is invalid")
    network_state = _deserialize_network_state(payload.get("network_state"), expected_network_state)
    rng_state = _validate_rng_state(payload.get("rng_state"))

    restored = P4CheckpointState(
        cell_id=expected_cell_id,
        global_step=int(payload["global_step"]),
        task_steps=task_steps,
        sampler_states=sampler_states,
        best_metrics=best_metrics,
        stale_evaluations=int(payload["stale_evaluations"]),
        matrix_cursor=int(payload["matrix_cursor"]),
        config_hash=expected_config_hash,
        protocol_hash=expected_protocol_hash,
        matrix_hash=expected_matrix_hash,
        pilot_lock_hash=expected_pilot_lock_hash,
        mechanism_lock_hash=expected_mechanism_lock_hash,
        analysis_curves=analysis_curves,
        last_loss=None if last_loss is None else float(last_loss),
        cumulative_wall_clock_seconds=float(wall_clock),
        transition_count=int(payload["transition_count"]),
        prediction_totals=prediction_totals,
        validation_prediction_totals=validation_prediction_totals,
        network_state=network_state,
    )
    try:
        _validate_state(restored)
    except (TypeError, ValueError) as error:
        raise CheckpointCompatibilityError("P4 checkpoint metadata is invalid") from error

    # Exercise both framework load paths on isolated copies.  This catches
    # optimizer-specific incompatibilities without touching live parameters.
    try:
        staged_model, staged_optimizer = copy.deepcopy((model, optimizer))
        staged_model.load_state_dict(model_state)
        staged_optimizer.load_state_dict(dict(optimizer_state))
        _move_optimizer_state(staged_optimizer, next(staged_model.parameters()).device)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise CheckpointCompatibilityError(
            "P4 checkpoint cannot be applied to the model or optimizer"
        ) from error

    # Mutation starts only after all compatibility checks above have succeeded.
    model.load_state_dict(model_state)
    optimizer.load_state_dict(dict(optimizer_state))
    device = next(model.parameters()).device
    _move_optimizer_state(optimizer, device)
    if restore_rng:
        restore_rng_state(rng_state)
    return restored


__all__ = [
    "P4_CHECKPOINT_VERSION",
    "P4CheckpointState",
    "load_p4_checkpoint",
    "save_p4_checkpoint",
]
