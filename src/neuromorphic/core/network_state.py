"""Functional state container for the modular artificial network."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from neuromorphic.core.contracts import (
    ModuleState,
    internal_execution_is_trusted,
    validate_module_state,
)
from neuromorphic.core.module_registry import ModuleRegistry
from neuromorphic.core.registry import MODULE_IDS


@dataclass(frozen=True, slots=True)
class NetworkState:
    """Module-owned states plus per-item valid-step TBPTT counters."""

    module_states: Mapping[str, ModuleState]
    valid_step_counts: Tensor

    def __post_init__(self) -> None:
        if self.valid_step_counts.ndim != 1:
            raise ValueError("valid_step_counts must have shape [B]")
        if self.valid_step_counts.dtype != torch.long:
            raise TypeError("valid_step_counts must use torch.long")
        if not internal_execution_is_trusted() and torch.any(self.valid_step_counts < 0).item():
            raise ValueError("valid_step_counts must be non-negative")
        unknown = set(self.module_states).difference(MODULE_IDS)
        if unknown:
            raise ValueError(f"network state contains unknown owners: {sorted(unknown)}")
        for module_id, state in self.module_states.items():
            if module_id != state.module_id:
                raise ValueError("network state key must equal ModuleState.module_id")
            validate_module_state(state, device=self.valid_step_counts.device)
            for tensor in state.tensors.values():
                if tensor.ndim > 0 and tensor.shape[0] != self.batch_size:
                    raise ValueError("batched state tensors must start with the network batch size")

    @property
    def batch_size(self) -> int:
        return int(self.valid_step_counts.shape[0])

    @classmethod
    def initial(
        cls,
        registry: ModuleRegistry,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> NetworkState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        states = {
            module_id: registry.get(module_id).initial_state(batch_size, device=device, dtype=dtype)
            for module_id in registry.ids
        }
        return cls(
            module_states=states,
            valid_step_counts=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def get(self, module_id: str) -> ModuleState:
        try:
            return self.module_states[module_id]
        except KeyError as error:
            raise KeyError(f"network state has no owner: {module_id}") from error

    def replace(self, state: ModuleState) -> NetworkState:
        if state.module_id not in self.module_states:
            raise KeyError(f"network state has no owner: {state.module_id}")
        updated = dict(self.module_states)
        updated[state.module_id] = state
        return NetworkState(updated, self.valid_step_counts)

    def detach_rows(self, detach_mask: Tensor) -> NetworkState:
        """Detach selected batch rows without severing other rows' graphs."""

        self._validate_batch_mask(detach_mask, "detach_mask")
        updated: dict[str, ModuleState] = {}
        for module_id, state in self.module_states.items():
            tensors: dict[str, Tensor] = {}
            for name, tensor in state.tensors.items():
                if tensor.ndim == 0:
                    tensors[name] = tensor
                    continue
                selector = detach_mask.reshape((self.batch_size,) + (1,) * (tensor.ndim - 1))
                tensors[name] = torch.where(selector, tensor.detach(), tensor)
            updated[module_id] = ModuleState(module_id, state.version, tensors)
        return NetworkState(updated, self.valid_step_counts)

    def advance(self, valid_mask: Tensor, *, interval: int = 32) -> tuple[NetworkState, Tensor]:
        """Advance valid rows and detach each item exactly at interval boundaries."""

        self._validate_batch_mask(valid_mask, "valid_mask")
        if interval <= 0:
            raise ValueError("interval must be positive")
        counts = self.valid_step_counts + valid_mask.to(torch.long)
        detach_mask = valid_mask & counts.remainder(interval).eq(0)
        advanced = NetworkState(self.module_states, counts).detach_rows(detach_mask)
        return advanced, detach_mask

    def reset_counts(self, reset_mask: Tensor) -> NetworkState:
        self._validate_batch_mask(reset_mask, "reset_mask")
        counts = torch.where(
            reset_mask, torch.zeros_like(self.valid_step_counts), self.valid_step_counts
        )
        return NetworkState(self.module_states, counts)

    def _validate_batch_mask(self, mask: Tensor, name: str) -> None:
        if mask.shape != (self.batch_size,):
            raise ValueError(f"{name} must have shape [B]")
        if mask.dtype is not torch.bool:
            raise TypeError(f"{name} must use torch.bool")
        if mask.device != self.valid_step_counts.device:
            raise ValueError(f"{name} and network state must share a device")


__all__ = ["NetworkState"]
