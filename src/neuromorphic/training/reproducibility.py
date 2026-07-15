"""Random-number state management for reproducible experiments."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and every available PyTorch backend."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """Capture serializable RNG state for every available backend."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore an RNG snapshot created by :func:`capture_rng_state`."""
    required = {"python", "numpy", "torch_cpu"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"RNG state is missing keys: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_mps" in state:
        if not torch.backends.mps.is_available():
            raise RuntimeError("checkpoint contains MPS RNG state but MPS is unavailable")
        torch.mps.set_rng_state(state["torch_mps"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
