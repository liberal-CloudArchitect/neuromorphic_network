"""Device-neutral monitoring summaries for modular training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from neuromorphic.core.contracts import ModuleState


@dataclass(frozen=True, slots=True)
class StateDynamics:
    """Norm and step-to-step change for one module-owned state."""

    module_id: str
    state_norm: float
    change_norm: float
    relative_change: float | None


@dataclass(frozen=True, slots=True)
class RoutingStatistics:
    """Raw and capacity-adjusted sparse-routing diagnostics."""

    valid_tokens: int
    expert_count: int
    top_k: int
    raw_assignments: int
    executed_assignments: int
    raw_shares: tuple[float, ...]
    executed_shares: tuple[float, ...]
    reroute_rate: float
    executed_entropy: float
    executed_coefficient_of_variation: float
    capacity_drops: int
    exact_top_k: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return asdict(self)


def capture_gradients(model: nn.Module) -> dict[str, Tensor | None]:
    """Detach and clone current gradients without synchronizing scalar values."""

    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def gradient_cosine_similarity(
    previous: Mapping[str, Tensor | None], current: Mapping[str, Tensor | None]
) -> float | None:
    """Return cosine similarity across common finite gradient tensors."""

    if set(previous) != set(current):
        raise ValueError("gradient snapshots must contain identical parameter names")
    dot = torch.zeros((), dtype=torch.float64)
    previous_norm = torch.zeros((), dtype=torch.float64)
    current_norm = torch.zeros((), dtype=torch.float64)
    used = False
    for name in sorted(previous):
        left = previous[name]
        right = current[name]
        if left is None and right is None:
            continue
        if left is None or right is None or left.shape != right.shape:
            raise ValueError(f"gradient availability or shape changed: {name}")
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError(f"gradient contains a non-finite value: {name}")
        left_cpu = left.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        right_cpu = right.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        dot += torch.dot(left_cpu, right_cpu)
        previous_norm += torch.dot(left_cpu, left_cpu)
        current_norm += torch.dot(right_cpu, right_cpu)
        used = True
    if not used or previous_norm.item() == 0.0 or current_norm.item() == 0.0:
        return None
    value = dot / torch.sqrt(previous_norm * current_norm)
    return float(value.clamp(-1.0, 1.0).item())


def state_dynamics(
    previous: Mapping[str, ModuleState], current: Mapping[str, ModuleState]
) -> dict[str, StateDynamics]:
    """Compare two complete state snapshots without modifying either snapshot."""

    if set(previous) != set(current):
        raise ValueError("state snapshots must have identical module owners")
    result: dict[str, StateDynamics] = {}
    for module_id in sorted(previous):
        left = previous[module_id]
        right = current[module_id]
        if left.module_id != right.module_id or left.version != right.version:
            raise ValueError(f"state owner or version changed: {module_id}")
        if set(left.tensors) != set(right.tensors):
            raise ValueError(f"state tensor keys changed: {module_id}")
        norm_squared = 0.0
        change_squared = 0.0
        for name in sorted(left.tensors):
            before = left.tensors[name]
            after = right.tensors[name]
            if before.shape != after.shape or before.dtype != after.dtype:
                raise ValueError(f"state tensor contract changed: {module_id}.{name}")
            if not torch.isfinite(before).all() or not torch.isfinite(after).all():
                raise ValueError(f"state tensor is not finite: {module_id}.{name}")
            after_cpu = after.detach().to(device="cpu", dtype=torch.float64)
            before_cpu = before.detach().to(device="cpu", dtype=torch.float64)
            delta_cpu = after_cpu - before_cpu
            norm_squared += float(after_cpu.square().sum().item())
            change_squared += float(delta_cpu.square().sum().item())
        norm = math.sqrt(norm_squared)
        change = math.sqrt(change_squared)
        result[module_id] = StateDynamics(
            module_id=module_id,
            state_norm=norm,
            change_norm=change,
            relative_change=None if norm == 0.0 else change / norm,
        )
    return result


def routing_statistics(
    raw_topk_mask: Tensor,
    executed_mask: Tensor,
    *,
    valid_mask: Tensor,
    top_k: int,
) -> RoutingStatistics:
    """Summarize raw and executed masks over valid tokens only."""

    if raw_topk_mask.dtype is not torch.bool or executed_mask.dtype is not torch.bool:
        raise TypeError("routing masks must use torch.bool")
    if raw_topk_mask.shape != executed_mask.shape or raw_topk_mask.ndim < 2:
        raise ValueError("routing masks must share shape [..., experts]")
    if valid_mask.shape != raw_topk_mask.shape[:-1] or valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be boolean and match routing token dimensions")
    if raw_topk_mask.device != executed_mask.device or valid_mask.device != executed_mask.device:
        raise ValueError("routing and valid masks must share a device")
    expert_count = raw_topk_mask.shape[-1]
    if top_k <= 0 or top_k > expert_count:
        raise ValueError("top_k must be within the expert count")
    if raw_topk_mask[~valid_mask].any() or executed_mask[~valid_mask].any():
        raise ValueError("padding tokens must not be routed")
    valid_tokens = int(valid_mask.sum().item())
    raw = raw_topk_mask[valid_mask]
    executed = executed_mask[valid_mask]
    # MPS does not support float64 kernels. Count on the source device in
    # float32, then use CPU float64 for stable reporting reductions.
    raw_counts = raw.to(torch.float32).sum(dim=0).cpu().to(torch.float64)
    executed_counts = executed.to(torch.float32).sum(dim=0).cpu().to(torch.float64)
    raw_assignments = int(raw_counts.sum().item())
    executed_assignments = int(executed_counts.sum().item())
    raw_shares = tuple(
        float(value)
        for value in (
            raw_counts / raw_counts.sum() if raw_assignments else torch.zeros_like(raw_counts)
        ).tolist()
    )
    executed_shares_tensor = (
        executed_counts / executed_counts.sum()
        if executed_assignments
        else torch.zeros_like(executed_counts)
    )
    executed_shares = tuple(float(value) for value in executed_shares_tensor.tolist())
    nonzero = executed_shares_tensor[executed_shares_tensor > 0]
    entropy = float((-(nonzero * nonzero.log()).sum()).item()) if nonzero.numel() else 0.0
    mean_count = float(executed_counts.mean().item())
    coefficient_of_variation = (
        0.0
        if mean_count == 0.0
        else float((executed_counts.std(unbiased=False) / mean_count).item())
    )
    changed_tokens = int((raw != executed).any(dim=-1).sum().item())
    counts_per_token = executed.sum(dim=-1)
    drops = int(torch.clamp(top_k - counts_per_token, min=0).sum().item())
    return RoutingStatistics(
        valid_tokens=valid_tokens,
        expert_count=expert_count,
        top_k=top_k,
        raw_assignments=raw_assignments,
        executed_assignments=executed_assignments,
        raw_shares=raw_shares,
        executed_shares=executed_shares,
        reroute_rate=0.0 if valid_tokens == 0 else changed_tokens / valid_tokens,
        executed_entropy=entropy,
        executed_coefficient_of_variation=coefficient_of_variation,
        capacity_drops=drops,
        exact_top_k=bool(torch.all(counts_per_token == top_k).item()),
    )


__all__ = [
    "RoutingStatistics",
    "StateDynamics",
    "capture_gradients",
    "gradient_cosine_similarity",
    "routing_statistics",
    "state_dynamics",
]
