"""Deterministic checkpoint-v2 persistence for modular P2 training.

The v2 format is intentionally separate from :mod:`neuromorphic.training.checkpoint`.
Loading validates every compatibility boundary before it mutates the live model,
optimizer, or random-number generators.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from neuromorphic.core.contracts import ModuleState
from neuromorphic.core.registry import MODULE_IDS
from neuromorphic.training.checkpoint import CheckpointCompatibilityError
from neuromorphic.training.reproducibility import capture_rng_state, restore_rng_state

MODULAR_CHECKPOINT_SCHEMA_VERSION = "modular-checkpoint-v2"
TASK_IDS = (
    "associative_recall.v1",
    "delayed_rule_switch.v1",
    "small_graph.v1",
)

_VOLATILE_CONFIG_KEYS = frozenset(
    {
        "output_path",
        "output_root",
        "resume",
        "resume_path",
        "run_directory",
        "run_id",
        "telemetry_enabled",
    }
)


@dataclass(frozen=True, slots=True)
class ModularCheckpointState:
    """Validated non-parameter state returned by a v2 restore."""

    module_states: Mapping[str, ModuleState]
    curriculum_stage: str
    stage_step: int
    sampler_states: Mapping[str, Mapping[str, Any]]
    tbptt_counters: Tensor
    frozen_module_ids: tuple[str, ...]
    optimizer_groups: tuple[Mapping[str, Any], ...]
    config_hash: str


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _VOLATILE_CONFIG_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_modular_config_hash(config: Mapping[str, Any]) -> str:
    """Hash structural, curriculum, and loss settings while ignoring run-local fields."""

    payload = json.dumps(
        _canonical_value(config), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _serialize_module_states(states: Mapping[str, ModuleState]) -> dict[str, dict[str, Any]]:
    _validate_module_state_set(states)
    return {
        module_id: {
            "owner": state.module_id,
            "version": state.version,
            "tensors": {
                name: tensor.detach().to("cpu").clone() for name, tensor in state.tensors.items()
            },
        }
        for module_id, state in states.items()
    }


def _validate_module_state_set(states: Mapping[str, ModuleState]) -> None:
    if set(states) != set(MODULE_IDS):
        raise ValueError("module_states must contain exactly the six registered modules")
    for module_id, state in states.items():
        if not isinstance(state, ModuleState):
            raise TypeError("module_states values must be ModuleState instances")
        if state.module_id != module_id:
            raise ValueError("module state mapping key must match its owner")


def _optimizer_group_metadata(optimizer: Optimizer) -> tuple[dict[str, Any], ...]:
    groups: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        metadata = {name: copy.deepcopy(value) for name, value in group.items() if name != "params"}
        metadata["index"] = index
        metadata["parameter_count"] = len(group["params"])
        groups.append(metadata)
    return tuple(groups)


def _validate_common_state(
    *,
    curriculum_stage: str,
    stage_step: int,
    sampler_states: Mapping[str, Mapping[str, Any]],
    tbptt_counters: Tensor,
    frozen_module_ids: Sequence[str],
) -> None:
    if not curriculum_stage or curriculum_stage.isspace():
        raise ValueError("curriculum_stage must be non-empty")
    if not isinstance(stage_step, int) or isinstance(stage_step, bool) or stage_step < 0:
        raise ValueError("stage_step must be a non-negative integer")
    if set(sampler_states) != set(TASK_IDS):
        raise ValueError("sampler_states must contain exactly the three P1 tasks")
    if any(not isinstance(state, Mapping) for state in sampler_states.values()):
        raise TypeError("each sampler state must be a mapping")
    if tbptt_counters.ndim != 1 or tbptt_counters.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("tbptt_counters must be a rank-one int32/int64 tensor")
    if torch.any(tbptt_counters < 0).item():
        raise ValueError("tbptt_counters must be non-negative")
    if len(set(frozen_module_ids)) != len(frozen_module_ids):
        raise ValueError("frozen_module_ids must not contain duplicates")
    if unknown := set(frozen_module_ids).difference(MODULE_IDS):
        raise ValueError(f"unknown frozen module identifiers: {sorted(unknown)}")


def save_modular_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    module_states: Mapping[str, ModuleState],
    curriculum_stage: str,
    stage_step: int,
    sampler_states: Mapping[str, Mapping[str, Any]],
    tbptt_counters: Tensor,
    frozen_module_ids: Sequence[str],
    config: Mapping[str, Any],
) -> None:
    """Atomically save all state needed to resume a P2 curriculum."""

    _validate_module_state_set(module_states)
    _validate_common_state(
        curriculum_stage=curriculum_stage,
        stage_step=stage_step,
        sampler_states=sampler_states,
        tbptt_counters=tbptt_counters,
        frozen_module_ids=frozen_module_ids,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": MODULAR_CHECKPOINT_SCHEMA_VERSION,
        "config_hash": canonical_modular_config_hash(config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_groups": _optimizer_group_metadata(optimizer),
        "module_states": _serialize_module_states(module_states),
        "curriculum": {"stage": curriculum_stage, "step": stage_step},
        "sampler_states": copy.deepcopy(dict(sampler_states)),
        "tbptt_counters": tbptt_counters.detach().to("cpu").clone(),
        "frozen_module_ids": tuple(frozen_module_ids),
        "rng_state": capture_rng_state(),
    }
    torch.save(payload, temporary)
    temporary.replace(path)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError(f"checkpoint {name} is invalid")
    return value


def _validate_model_state(model: nn.Module, saved: Mapping[str, Any]) -> None:
    current = model.state_dict()
    if set(saved) != set(current):
        raise CheckpointCompatibilityError("checkpoint model parameter keys do not match")
    for name, expected in current.items():
        value = saved[name]
        if not isinstance(value, Tensor):
            raise CheckpointCompatibilityError(f"checkpoint model value is not a tensor: {name}")
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise CheckpointCompatibilityError(f"checkpoint model tensor is incompatible: {name}")


def _validate_optimizer_state(optimizer: Optimizer, saved: Mapping[str, Any]) -> None:
    saved_groups = saved.get("param_groups")
    saved_state = saved.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_state, Mapping):
        raise CheckpointCompatibilityError("checkpoint optimizer state is invalid")
    if len(saved_groups) != len(optimizer.param_groups):
        raise CheckpointCompatibilityError("checkpoint optimizer group count does not match")
    for saved_group, current_group in zip(saved_groups, optimizer.param_groups, strict=True):
        if not isinstance(saved_group, Mapping) or not isinstance(saved_group.get("params"), list):
            raise CheckpointCompatibilityError("checkpoint optimizer group is invalid")
        saved_ids = saved_group["params"]
        current_parameters = current_group["params"]
        if len(saved_ids) != len(current_parameters):
            raise CheckpointCompatibilityError(
                "checkpoint optimizer parameter count does not match"
            )
        for saved_id, parameter in zip(saved_ids, current_parameters, strict=True):
            parameter_state = saved_state.get(saved_id, {})
            if not isinstance(parameter_state, Mapping):
                raise CheckpointCompatibilityError(
                    "checkpoint optimizer parameter state is invalid"
                )
            for value in parameter_state.values():
                if (
                    isinstance(value, Tensor)
                    and value.numel() != 1
                    and value.shape != parameter.shape
                ):
                    raise CheckpointCompatibilityError(
                        "checkpoint optimizer tensor shape does not match its parameter"
                    )


def _deserialize_module_states(
    payload: Mapping[str, Any], expected: Mapping[str, ModuleState]
) -> dict[str, ModuleState]:
    _validate_module_state_set(expected)
    if set(payload) != set(MODULE_IDS):
        raise CheckpointCompatibilityError("checkpoint does not contain all six module states")
    restored: dict[str, ModuleState] = {}
    for module_id in MODULE_IDS:
        section = _require_mapping(payload[module_id], f"module state {module_id}")
        if section.get("owner") != module_id:
            raise CheckpointCompatibilityError("checkpoint module state owner does not match")
        version = section.get("version")
        if version != expected[module_id].version:
            raise CheckpointCompatibilityError(
                f"checkpoint module state version does not match: {module_id}"
            )
        tensors = _require_mapping(section.get("tensors"), f"module tensors {module_id}")
        expected_tensors = expected[module_id].tensors
        if set(tensors) != set(expected_tensors):
            raise CheckpointCompatibilityError(
                f"checkpoint module tensor keys do not match: {module_id}"
            )
        checked: dict[str, Tensor] = {}
        for name, expected_tensor in expected_tensors.items():
            value = tensors[name]
            if not isinstance(value, Tensor):
                raise CheckpointCompatibilityError("checkpoint module state value is not a tensor")
            if value.shape != expected_tensor.shape or value.dtype != expected_tensor.dtype:
                raise CheckpointCompatibilityError(
                    f"checkpoint module tensor is incompatible: {module_id}.{name}"
                )
            checked[name] = value.to(expected_tensor.device)
        restored[module_id] = ModuleState(module_id, str(version), checked)
    return restored


def _validate_rng_state(value: Any) -> dict[str, Any]:
    state = dict(_require_mapping(value, "RNG state"))
    if missing := {"python", "numpy", "torch_cpu"}.difference(state):
        raise CheckpointCompatibilityError(f"checkpoint RNG state is missing: {sorted(missing)}")
    if "torch_mps" in state and not torch.backends.mps.is_available():
        raise CheckpointCompatibilityError("checkpoint requires unavailable MPS RNG state")
    if "torch_cuda" in state and not torch.cuda.is_available():
        raise CheckpointCompatibilityError("checkpoint requires unavailable CUDA RNG state")
    return state


def _move_optimizer_state(optimizer: Optimizer, device: torch.device) -> None:
    for parameter_state in optimizer.state.values():
        for name, value in parameter_state.items():
            if isinstance(value, Tensor):
                parameter_state[name] = value.to(device)


def load_modular_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    expected_module_states: Mapping[str, ModuleState],
    config: Mapping[str, Any],
    restore_rng: bool = True,
) -> ModularCheckpointState:
    """Validate then restore a compatible P2 checkpoint.

    ``expected_module_states`` supplies the live owner/version/shape contract. No
    live object is changed until the full payload passes validation.
    """

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != (
        MODULAR_CHECKPOINT_SCHEMA_VERSION
    ):
        raise CheckpointCompatibilityError("unsupported modular checkpoint schema")
    expected_hash = canonical_modular_config_hash(config)
    if payload.get("config_hash") != expected_hash:
        raise CheckpointCompatibilityError("checkpoint configuration hash does not match")

    model_state = _require_mapping(payload.get("model_state"), "model state")
    optimizer_state = _require_mapping(payload.get("optimizer_state"), "optimizer state")
    _validate_model_state(model, model_state)
    _validate_optimizer_state(optimizer, optimizer_state)
    module_states = _deserialize_module_states(
        _require_mapping(payload.get("module_states"), "module states"), expected_module_states
    )
    curriculum = _require_mapping(payload.get("curriculum"), "curriculum")
    stage = curriculum.get("stage")
    step = curriculum.get("step")
    sampler_states = _require_mapping(payload.get("sampler_states"), "sampler states")
    counters = payload.get("tbptt_counters")
    frozen = payload.get("frozen_module_ids")
    groups = payload.get("optimizer_groups")
    if not isinstance(stage, str) or not isinstance(step, int) or isinstance(step, bool):
        raise CheckpointCompatibilityError("checkpoint curriculum is invalid")
    if not isinstance(counters, Tensor):
        raise CheckpointCompatibilityError("checkpoint TBPTT counters are invalid")
    if not isinstance(frozen, tuple | list) or any(not isinstance(item, str) for item in frozen):
        raise CheckpointCompatibilityError("checkpoint frozen module identifiers are invalid")
    if not isinstance(groups, tuple | list) or any(
        not isinstance(item, Mapping) for item in groups
    ):
        raise CheckpointCompatibilityError("checkpoint optimizer group metadata is invalid")
    typed_samplers: dict[str, Mapping[str, Any]] = {}
    for task_id, sampler_state in sampler_states.items():
        if not isinstance(task_id, str) or not isinstance(sampler_state, Mapping):
            raise CheckpointCompatibilityError("checkpoint sampler state is invalid")
        typed_samplers[task_id] = sampler_state
    try:
        _validate_common_state(
            curriculum_stage=stage,
            stage_step=step,
            sampler_states=typed_samplers,
            tbptt_counters=counters,
            frozen_module_ids=frozen,
        )
    except (TypeError, ValueError) as error:
        raise CheckpointCompatibilityError(str(error)) from error
    rng_state = _validate_rng_state(payload.get("rng_state"))

    # Mutation begins only after every compatibility check above has succeeded.
    model.load_state_dict(model_state)
    optimizer.load_state_dict(dict(optimizer_state))
    device = next(model.parameters()).device
    _move_optimizer_state(optimizer, device)
    if restore_rng:
        restore_rng_state(rng_state)
    return ModularCheckpointState(
        module_states=module_states,
        curriculum_stage=stage,
        stage_step=step,
        sampler_states=typed_samplers,
        tbptt_counters=counters.to(device),
        frozen_module_ids=tuple(frozen),
        optimizer_groups=tuple(dict(item) for item in groups),
        config_hash=expected_hash,
    )


__all__ = [
    "MODULAR_CHECKPOINT_SCHEMA_VERSION",
    "ModularCheckpointState",
    "canonical_modular_config_hash",
    "load_modular_checkpoint",
    "save_modular_checkpoint",
]
