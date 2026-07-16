"""Internal helpers shared by P2 computation modules."""

from __future__ import annotations

import torch
from torch import Tensor

from neuromorphic.core.contracts import BrainPacket, ModuleContext, ModuleState


def validate_inputs(
    packet: BrainPacket,
    state: ModuleState,
    context: ModuleContext,
    *,
    module_id: str,
    version: str,
) -> None:
    if state.module_id != module_id or state.version != version:
        raise ValueError(f"expected {module_id} state version {version}")
    batch, steps, _ = packet.representation.shape
    if context.reset_mask.shape != (batch, steps):
        raise ValueError("context reset_mask must align with packet [B, T]")
    if context.reset_mask.device != packet.representation.device:
        raise ValueError("context and packet must share a device")


def packet_from(
    source: BrainPacket,
    representation: Tensor,
    module_id: str,
    *,
    goal_context: Tensor | None = None,
) -> BrainPacket:
    return BrainPacket(
        representation=representation,
        valid_mask=source.valid_mask,
        modality=source.modality,
        step_index=source.step_index,
        source_module=module_id,
        goal_context=source.goal_context if goal_context is None else goal_context,
        metadata=source.metadata,
    )


def reset_tensor_rows(tensor: Tensor, reset_mask: Tensor, fill: float | int | bool = 0) -> Tensor:
    selector = reset_mask.reshape((reset_mask.shape[0],) + (1,) * (tensor.ndim - 1))
    replacement = torch.full_like(tensor, fill)
    return torch.where(selector, replacement, tensor)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    weights = expanded.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def require_goal_context(packet: BrainPacket, minimum_features: int = 104) -> Tensor:
    goal = packet.goal_context
    if goal is None or goal.shape[-1] < minimum_features:
        raise ValueError(f"module requires goal_context with at least {minimum_features} features")
    return goal
