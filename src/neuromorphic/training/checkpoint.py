"""Versioned training checkpoint persistence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from neuromorphic.training.reproducibility import capture_rng_state, restore_rng_state

CHECKPOINT_SCHEMA_VERSION = "checkpoint-v1"


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint cannot safely resume the requested run."""


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible run configuration deterministically."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    training_state: Mapping[str, Any],
    sampler_state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Atomically save all state required for deterministic continuation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_hash": canonical_config_hash(config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "training_state": dict(training_state),
        "sampler_state": dict(sampler_state),
        "rng_state": capture_rng_state(),
    }
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    config: Mapping[str, Any],
    restore_rng: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a compatible checkpoint and return training and sampler state."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError("unsupported checkpoint schema")
    expected_hash = canonical_config_hash(config)
    if payload.get("config_hash") != expected_hash:
        raise CheckpointCompatibilityError("checkpoint configuration hash does not match")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    model_device = next(model.parameters()).device
    for optimizer_state in optimizer.state.values():
        for name, value in optimizer_state.items():
            if isinstance(value, torch.Tensor):
                optimizer_state[name] = value.to(model_device)
    scheduler_state = payload.get("scheduler_state")
    if scheduler_state is not None:
        if scheduler is None:
            raise CheckpointCompatibilityError("checkpoint requires a scheduler")
        scheduler.load_state_dict(scheduler_state)
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    training_state = payload.get("training_state")
    sampler_state = payload.get("sampler_state")
    if not isinstance(training_state, dict) or not isinstance(sampler_state, dict):
        raise CheckpointCompatibilityError("checkpoint state sections are invalid")
    return training_state, sampler_state
