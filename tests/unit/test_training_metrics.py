from __future__ import annotations

import pytest
import torch

from neuromorphic.tasks.base import TaskBatch
from neuromorphic.training.baselines import BaselineOutput
from neuromorphic.training.metrics import (
    ensure_finite_training_state,
    masked_task_loss,
    task_metrics,
)


def _batch() -> TaskBatch:
    return TaskBatch(
        inputs=torch.zeros(1, 2, 3),
        targets=torch.tensor([[0, 1]]),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        loss_mask=torch.tensor([[False, True]]),
        episode_ids=torch.zeros(1, 2, dtype=torch.long),
        metadata={"task_version": "test-v1", "split": "train"},
        auxiliary_targets={},
    )


def test_masked_loss_and_metrics_use_loss_positions_only() -> None:
    logits = torch.tensor([[[10.0, -10.0], [-10.0, 10.0]]], requires_grad=True)
    output = BaselineOutput(logits=logits)
    loss, parts = masked_task_loss(output, _batch(), auxiliary_weight=0.1)
    assert loss < 1e-6
    assert parts["loss/total"] < 1e-6
    assert task_metrics(output, _batch()) == {"accuracy": 1.0}


def test_numerical_guard_detects_nonfinite_gradient() -> None:
    model = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("nan"))
    with pytest.raises(FloatingPointError, match="gradient"):
        ensure_finite_training_state(loss=torch.tensor(1.0), model=model, metrics={"accuracy": 1.0})
