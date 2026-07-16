"""Losses and task metrics for the P2 modular network."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import functional as F

from neuromorphic.tasks.base import TaskBatch


class ModularOutputProtocol(Protocol):
    @property
    def logits(self) -> Tensor: ...

    @property
    def prediction_logits(self) -> Tensor | None: ...

    @property
    def prediction_targets(self) -> Tensor | None: ...

    @property
    def prediction_mask(self) -> Tensor | None: ...

    @property
    def auxiliary_losses(self) -> Mapping[str, Tensor]: ...


def _primary_loss(logits: Tensor, batch: TaskBatch) -> Tensor:
    supervised = batch.loss_mask
    if logits.shape[:2] != supervised.shape or not supervised.any():
        raise ValueError("modular logits must align with a non-empty loss mask")
    optimal = batch.auxiliary_targets.get("optimal_action_mask")
    if optimal is None:
        return F.cross_entropy(logits[supervised], batch.targets[supervised])
    if optimal.shape != logits.shape or optimal.dtype is not torch.bool:
        raise ValueError("optimal_action_mask must be boolean and match modular logits")
    allowed = F.log_softmax(logits, dim=-1).masked_fill(~optimal, -torch.inf)
    return -torch.logsumexp(allowed[supervised], dim=-1).mean()


def modular_training_loss(
    output: ModularOutputProtocol,
    batch: TaskBatch,
    *,
    weights: Mapping[str, float],
    include_primary: bool = True,
) -> tuple[Tensor, dict[str, float]]:
    """Combine the primary task objective with explicitly named auxiliaries."""

    losses: dict[str, Tensor] = {}
    if include_primary:
        losses["primary"] = _primary_loss(output.logits, batch)
    for name, value in output.auxiliary_losses.items():
        if value.ndim != 0 or not value.is_floating_point() or not torch.isfinite(value):
            raise ValueError(f"modular auxiliary loss must be a finite scalar: {name}")
        losses[name] = value
    prediction_mask = output.prediction_mask
    if prediction_mask is not None and prediction_mask.any():
        if output.prediction_logits is None or output.prediction_targets is None:
            raise ValueError("prediction mask requires logits and dynamic targets")
        if prediction_mask.shape != batch.valid_mask.shape:
            raise ValueError("prediction mask must have shape [B, T]")
        losses["predictive.next_state"] = F.cross_entropy(
            output.prediction_logits[prediction_mask],
            output.prediction_targets[prediction_mask],
        )
    if not losses:
        raise ValueError("modular training step produced no losses")
    total: Tensor | None = None
    parts: dict[str, float] = {}
    for name, value in losses.items():
        weight = float(weights.get(name, 0.0))
        weighted = value * weight
        total = weighted if total is None else total + weighted
        parts[f"loss/{name}"] = float(value.detach().cpu())
        parts[f"weight/{name}"] = weight
    if total is None or not torch.isfinite(total):
        raise FloatingPointError("modular total loss is not finite")
    parts["loss/total"] = float(total.detach().cpu())
    return total, parts


def modular_task_metrics(
    output: ModularOutputProtocol,
    batch: TaskBatch,
) -> dict[str, float]:
    """Return primary accuracy and actual-action prediction diagnostics."""

    supervised = batch.loss_mask
    predicted = output.logits.argmax(dim=-1)
    optimal = batch.auxiliary_targets.get("optimal_action_mask")
    if optimal is None:
        correct = predicted[supervised].eq(batch.targets[supervised])
        result = {"accuracy": float(correct.float().mean().detach().cpu())}
    else:
        correct = optimal.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)[supervised]
        result = {"optimal_action_rate": float(correct.float().mean().detach().cpu())}
    mask = output.prediction_mask
    if mask is not None:
        prediction_count = float(mask.sum().detach().cpu())
        supervised_count = max(float(supervised.sum().detach().cpu()), 1.0)
        coverage = prediction_count / supervised_count
        result["prediction_coverage"] = coverage
        if (
            mask.any()
            and output.prediction_logits is not None
            and output.prediction_targets is not None
        ):
            predicted_next = output.prediction_logits.argmax(dim=-1)
            result["actual_action_next_state_accuracy"] = float(
                predicted_next[mask]
                .eq(output.prediction_targets[mask])
                .float()
                .mean()
                .detach()
                .cpu()
            )
    return result


__all__ = [
    "ModularOutputProtocol",
    "modular_task_metrics",
    "modular_training_loss",
]
