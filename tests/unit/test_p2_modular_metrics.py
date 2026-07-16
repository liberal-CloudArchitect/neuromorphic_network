from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from neuromorphic.tasks import create_task
from neuromorphic.training.modular_metrics import (
    modular_task_metrics,
    modular_training_loss,
)


@dataclass
class _Output:
    logits: torch.Tensor
    prediction_logits: torch.Tensor | None = None
    prediction_targets: torch.Tensor | None = None
    prediction_mask: torch.Tensor | None = None
    auxiliary_losses: Mapping[str, torch.Tensor] = field(default_factory=dict)


def test_modular_loss_combines_named_primary_and_auxiliary_losses() -> None:
    batch = create_task("associative_recall.v1").generate("train", [0, 1])
    logits = torch.zeros((*batch.targets.shape, 32), requires_grad=True)
    auxiliary = torch.tensor(0.25, requires_grad=True)
    output = _Output(logits=logits, auxiliary_losses={"episodic.retrieval": auxiliary})

    loss, parts = modular_training_loss(
        output,
        batch,
        weights={"primary": 1.0, "episodic.retrieval": 0.1},
    )

    assert torch.isfinite(loss)
    assert parts["weight/episodic.retrieval"] == 0.1
    loss.backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    assert auxiliary.grad is not None


def test_small_graph_prediction_uses_dynamic_actual_action_target() -> None:
    batch = create_task("small_graph.v1").generate("train", [0])
    logits = torch.zeros((*batch.targets.shape, 4), requires_grad=True)
    prediction_mask = batch.loss_mask.clone()
    targets = torch.zeros_like(batch.targets)
    targets[prediction_mask] = 3
    prediction_logits = torch.zeros((*batch.targets.shape, 16), requires_grad=True)
    prediction_logits.data[..., 3] = 4.0
    output = _Output(
        logits=logits,
        prediction_logits=prediction_logits,
        prediction_targets=targets,
        prediction_mask=prediction_mask,
    )

    loss, _ = modular_training_loss(
        output,
        batch,
        weights={"primary": 1.0, "predictive.next_state": 0.1},
    )
    metrics = modular_task_metrics(output, batch)

    assert torch.isfinite(loss)
    assert metrics["actual_action_next_state_accuracy"] == 1.0
    assert metrics["prediction_coverage"] == 1.0


def test_prediction_mask_requires_dynamic_logits_and_targets() -> None:
    batch = create_task("small_graph.v1").generate("train", [0])
    output = _Output(
        logits=torch.zeros((*batch.targets.shape, 4)),
        prediction_mask=batch.loss_mask,
    )
    try:
        modular_training_loss(output, batch, weights={"primary": 1.0})
    except ValueError as error:
        assert "dynamic targets" in str(error)
    else:
        raise AssertionError("missing dynamic prediction targets were accepted")
