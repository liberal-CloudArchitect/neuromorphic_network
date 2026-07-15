"""Deterministic weak and oracle baselines used to qualify P1 tasks."""

from __future__ import annotations

import torch
from torch import Tensor

from neuromorphic.tasks.base import TaskBatch


def random_predictions(batch: TaskBatch, *, num_classes: int, seed: int) -> Tensor:
    """Generate deterministic random class predictions without touching global RNG."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    predictions = torch.randint(
        0,
        num_classes,
        batch.targets.shape,
        generator=generator,
        dtype=torch.long,
    )
    return predictions.to(batch.inputs.device)


def associative_key_value_predictions(batch: TaskBatch) -> Tensor:
    """Solve Associative Recall using an explicit per-episode key/value table."""
    predictions = torch.zeros_like(batch.targets)
    for batch_index in range(batch.batch_size):
        memory: dict[int, int] = {}
        for step in range(batch.sequence_length):
            if not bool(batch.valid_mask[batch_index, step].item()):
                continue
            event = batch.inputs[batch_index, step]
            if bool(event[0].item()):
                key = int(event[4:36].argmax().item())
                value = int(event[36:68].argmax().item())
                memory[key] = value
            elif bool(event[2].item()):
                key = int(event[4:36].argmax().item())
                predictions[batch_index, step] = memory[key]
    return predictions


def delayed_rule_predictions(batch: TaskBatch, *, update_on_switch: bool) -> Tensor:
    """Apply the initial rule only, or update rule state whenever a cue appears."""
    predictions = torch.zeros_like(batch.targets)
    for batch_index in range(batch.batch_size):
        active_rule = 0
        has_rule = False
        for step in range(batch.sequence_length):
            if not bool(batch.valid_mask[batch_index, step].item()):
                continue
            row = batch.inputs[batch_index, step]
            if bool(row[0].item()) and (update_on_switch or not has_rule):
                active_rule = int(row[3:7].argmax().item())
                has_rule = True
            if bool(batch.loss_mask[batch_index, step].item()):
                first = int(row[7].item())
                second = int(row[8].item())
                if active_rule == 0:
                    answer = first
                elif active_rule == 1:
                    answer = second
                elif active_rule == 2:
                    answer = first ^ second
                else:
                    answer = 1 - (first ^ second)
                predictions[batch_index, step] = answer
    return predictions


def masked_accuracy(predictions: Tensor, batch: TaskBatch) -> float:
    """Return class accuracy over the frozen loss mask."""
    if predictions.shape != batch.targets.shape:
        raise ValueError("predictions must match target shape")
    return float(predictions[batch.loss_mask].eq(batch.targets[batch.loss_mask]).float().mean())
