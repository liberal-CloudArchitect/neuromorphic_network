from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from neuromorphic.core.registry import P5_MODULE_IDS, PREDICTIVE_ADAPTER_V3
from neuromorphic.modules.network_v3 import ModularBrainNetworkV3
from neuromorphic.tasks.associative_recall import AssociativeRecallTask
from neuromorphic.tasks.delayed_rule_switch import DelayedRuleSwitchTask
from neuromorphic.tasks.small_graph import SmallGraphTask


@pytest.mark.parametrize(
    "task",
    [
        AssociativeRecallTask(namespace="p4"),
        DelayedRuleSwitchTask(namespace="p4"),
        SmallGraphTask(namespace="p4"),
    ],
)
def test_v3_network_runs_all_tasks_with_finite_backward(task: object) -> None:
    torch.manual_seed(7)
    batch = task.generate("train", [0, 1, 2, 3])  # type: ignore[attr-defined]
    model = ModularBrainNetworkV3()
    output = model.forward_batch(batch)

    assert set(model.registry.ids) == set(P5_MODULE_IDS)
    assert output.action_logits.shape[:2] == batch.inputs.shape[:2]
    assert not output.feedback_delta.any()
    assert "router.semantic_alignment" in output.auxiliary_losses
    primary = F.cross_entropy(output.action_logits[batch.loss_mask], batch.targets[batch.loss_mask])
    loss = primary + 0.1 * output.auxiliary_losses["predictive.temporal"]
    loss = loss + 0.01 * output.auxiliary_losses["router.semantic_alignment"]
    loss.backward()  # type: ignore[no-untyped-call]
    assert torch.isfinite(loss)


def test_v3_drs_working_coverage_and_dual_budget_are_exact() -> None:
    batch = DelayedRuleSwitchTask(namespace="p4").generate("train", list(range(8)))
    model = ModularBrainNetworkV3(dual_route_fraction=0.25)
    output = model.forward_batch(batch)

    for step, decision in enumerate(output.routing_trace):
        valid = batch.valid_mask[:, step]
        assert decision.reserved_mask[:, 0, 1][valid].all()
        assert decision.executed_mask[:, 0, 1][valid].all()
        dual = decision.executed_mask[:, 0].sum(dim=-1).gt(1) & valid
        assert int(dual.sum().item()) <= math.floor(int(valid.sum().item()) * 0.25)
    assert (
        output.cost_statistics["optional.active_calls"]
        < output.cost_statistics["optional.dense_calls"]
    )
    state = output.state.get(PREDICTIVE_ADAPTER_V3)
    assert not state.tensors["forecast_valid"].any()
