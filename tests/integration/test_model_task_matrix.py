from __future__ import annotations

import pytest
import torch

from neuromorphic.tasks import create_task
from neuromorphic.training.baselines import GRUBaseline, TransformerBaseline
from neuromorphic.training.metrics import masked_task_loss


@pytest.mark.parametrize("device_name", ["cpu", "mps"])
@pytest.mark.parametrize(
    "task_id",
    ["associative_recall.v1", "delayed_rule_switch.v1", "small_graph.v1"],
)
@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_model_task_forward_backward_matrix(device_name: str, task_id: str, kind: str) -> None:
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    device = torch.device(device_name)
    task = create_task(task_id)
    auxiliary_classes = 16 if task_id == "small_graph.v1" else None
    if kind == "gru":
        model: torch.nn.Module = GRUBaseline(
            input_dim=task.input_dim,
            num_classes=task.num_classes,
            hidden_size=16,
            auxiliary_classes=auxiliary_classes,
        )
    else:
        model = TransformerBaseline(
            input_dim=task.input_dim,
            num_classes=task.num_classes,
            hidden_size=16,
            layers=2,
            heads=4,
            feedforward_size=32,
            auxiliary_classes=auxiliary_classes,
        )
    model.to(device)
    batch = task.generate("train", [0, 1], device=device)
    output = model(batch.inputs, batch.valid_mask)
    loss, _ = masked_task_loss(output, batch, auxiliary_weight=0.1)
    assert torch.isfinite(loss)
    loss.backward()  # type: ignore[no-untyped-call]
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
