"""Fully validated checkpoint-v3 for P3 matrix cells."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        if not isinstance(value, torch.Tensor) or value.shape != live[name].shape:
            raise CheckpointCompatibilityError(f"P3 checkpoint tensor is incompatible: {name}")
    expected_names = _parameter_names(model, optimizer)
    saved_names = payload.get("optimizer_parameter_names")
    if saved_names != expected_names:
        raise CheckpointCompatibilityError("P3 checkpoint optimizer groups do not match")
    required_mappings = ("task_steps", "sampler_states", "best_metrics", "rng_state")
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
        task_steps={str(key): int(value) for key, value in payload["task_steps"].items()},
        sampler_states={str(key): dict(value) for key, value in payload["sampler_states"].items()},
        best_metrics={str(key): float(value) for key, value in payload["best_metrics"].items()},
        stale_evaluations=int(payload["stale_evaluations"]),
        matrix_cursor=int(payload["matrix_cursor"]),
        config_hash=expected_config_hash,
        protocol_hash=expected_protocol_hash,
    )


__all__ = ["P3CheckpointState", "load_p3_checkpoint", "save_p3_checkpoint"]
