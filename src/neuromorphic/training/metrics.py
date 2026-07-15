"""Masked task losses, metrics, and numerical safety checks."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neuromorphic.tasks.base import TaskBatch
from neuromorphic.training.baselines import BaselineOutput


def masked_task_loss(
    output: BaselineOutput, batch: TaskBatch, *, auxiliary_weight: float
) -> tuple[Tensor, dict[str, float]]:
    """Compute classification or set-valued action loss over supervised positions."""
    supervised = batch.loss_mask
    if not supervised.any():
        raise ValueError("batch has no supervised positions")
    optimal = batch.auxiliary_targets.get("optimal_action_mask")
    if optimal is None:
        primary = F.cross_entropy(output.logits[supervised], batch.targets[supervised])
    else:
        if optimal.shape != output.logits.shape or optimal.dtype != torch.bool:
            raise ValueError("optimal_action_mask must be boolean and match logits")
        log_probabilities = F.log_softmax(output.logits, dim=-1)
        allowed = log_probabilities.masked_fill(~optimal, -torch.inf)
        primary = -torch.logsumexp(allowed[supervised], dim=-1).mean()

    total = primary
    parts = {"loss/primary": float(primary.detach().cpu())}
    next_state = batch.auxiliary_targets.get("next_state")
    if next_state is not None:
        if output.next_state_logits is None:
            raise ValueError("task requires a next-state prediction head")
        auxiliary = F.cross_entropy(output.next_state_logits[supervised], next_state[supervised])
        total = total + auxiliary_weight * auxiliary
        parts["loss/next_state"] = float(auxiliary.detach().cpu())
    parts["loss/total"] = float(total.detach().cpu())
    return total, parts


def task_metrics(output: BaselineOutput, batch: TaskBatch) -> dict[str, float]:
    """Calculate deterministic episode-position metrics."""
    supervised = batch.loss_mask
    predictions = output.logits.argmax(dim=-1)
    optimal = batch.auxiliary_targets.get("optimal_action_mask")
    if optimal is None:
        correct = predictions[supervised].eq(batch.targets[supervised])
        name = "accuracy"
    else:
        correct = optimal.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)[supervised]
        name = "optimal_action_rate"
    metrics = {name: float(correct.float().mean().cpu())}
    next_state = batch.auxiliary_targets.get("next_state")
    if next_state is not None and output.next_state_logits is not None:
        next_predictions = output.next_state_logits.argmax(dim=-1)
        metrics["next_state_accuracy"] = float(
            next_predictions[supervised].eq(next_state[supervised]).float().mean().cpu()
        )
    return metrics


def ensure_finite_training_state(
    *, loss: Tensor, model: nn.Module, metrics: Mapping[str, float]
) -> None:
    """Fail immediately on NaN/Inf loss, gradient, parameter, or metric values."""
    if not torch.isfinite(loss):
        raise FloatingPointError("loss is not finite")
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"parameter is not finite: {name}")
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"gradient is not finite: {name}")
    if any(not torch.isfinite(torch.tensor(value)) for value in metrics.values()):
        raise FloatingPointError("metric is not finite")
