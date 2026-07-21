"""Integration coverage for the P4 causal modular graph."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch.nn import functional as F

from neuromorphic.core.registry import (
    ACTION_SELECTOR,
    EPISODIC_MEMORY,
    P4_MODULE_IDS,
    P4_OPTIONAL_EXPERT_IDS,
    PREDICTIVE_ADAPTER_V2,
    SPARSE_ROUTER_V2,
)
from neuromorphic.modules.network_v2 import ModularBrainNetworkV2
from neuromorphic.modules.sparse_router_v2 import SparseRouterV2
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
def test_v2_network_runs_p4_tasks_with_true_top1_memory_routing(task: object) -> None:
    torch.manual_seed(7)
    batch = task.generate("train", [0, 1, 2])  # type: ignore[attr-defined]
    model = ModularBrainNetworkV2()
    output = model.forward_batch(batch)

    assert set(model.registry.ids) == set(P4_MODULE_IDS)
    assert output.packet.source_module == ACTION_SELECTOR
    assert output.action_logits.shape == (
        batch.batch_size,
        batch.sequence_length,
        task.num_classes,  # type: ignore[attr-defined]
    )
    for step, decision in enumerate(output.routing_trace):
        expected = batch.valid_mask[:, step].to(torch.long)
        assert torch.equal(decision.executed_mask[:, 0].sum(dim=-1), expected)
        assert decision.capacity_drops == 0
    assert (
        output.cost_statistics["optional.active_calls"]
        < output.cost_statistics["optional.dense_calls"]
    )
    assert output.state.valid_step_counts.tolist() == batch.valid_mask.sum(dim=1).tolist()
    predictive_state = output.state.get(PREDICTIVE_ADAPTER_V2)
    assert not predictive_state.tensors["forecast_valid"].any()


def test_ar_store_and_query_are_always_reserved_for_episodic_memory() -> None:
    batch = AssociativeRecallTask(namespace="p4").generate("train", list(range(8)))
    model = ModularBrainNetworkV2()
    output = model.forward_batch(batch)

    reserved = torch.cat([item.reserved_mask for item in output.routing_trace], dim=1)
    executed = torch.cat([item.executed_mask for item in output.routing_trace], dim=1)
    events = batch.inputs[..., :3]
    semantic = batch.valid_mask & (events[..., 0].gt(0.5) | events[..., 2].gt(0.5))
    episodic = P4_OPTIONAL_EXPERT_IDS.index(EPISODIC_MEMORY)
    assert semantic.any()
    assert reserved[..., episodic][semantic].all()
    assert executed[..., episodic][semantic].all()
    assert output.module_metrics["routing.reserved_tokens"] == semantic.sum()
    assert output.module_metrics["routing.reserved_executed"] == semantic.sum()


def test_predictive_feedback_is_delayed_one_step_and_has_finite_backward() -> None:
    torch.manual_seed(7)
    batch = DelayedRuleSwitchTask(namespace="p4").generate("train", [0, 1, 2, 3])
    model = ModularBrainNetworkV2()
    full = model.forward_batch(batch, predictor_mode="full")
    no_feedback = model.forward_batch(batch, predictor_mode="feedback_zero")

    assert not full.forecast_transition_mask[:, 0].any()
    assert torch.equal(full.feedback_delta[:, 0], torch.zeros_like(full.feedback_delta[:, 0]))
    assert full.forecast_transition_mask.any()
    assert full.feedback_delta.abs().sum() > 0
    assert torch.all(full.feedback_delta.abs() <= 0.25)
    assert not no_feedback.feedback_delta.any()
    assert torch.allclose(full.action_logits[:, 0], no_feedback.action_logits[:, 0])

    primary = F.cross_entropy(full.action_logits[batch.loss_mask], batch.targets[batch.loss_mask])
    loss = primary + 0.1 * full.auxiliary_losses["predictive.temporal"]
    assert torch.isfinite(loss)
    loss.backward()  # type: ignore[no-untyped-call]
    predictor = cast(torch.nn.Module, model.registry.get(PREDICTIVE_ADAPTER_V2))
    gradients = [
        parameter.grad for parameter in predictor.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_predictor_off_and_loss_zero_are_distinct_interventions() -> None:
    batch = SmallGraphTask(namespace="p4").generate("train", [0, 1])
    model = ModularBrainNetworkV2()
    off = model.forward_batch(batch, predictor_mode="off")
    loss_zero = model.forward_batch(batch, predictor_mode="loss_zero")

    assert not off.forecast_transition_mask.any()
    assert off.auxiliary_losses["predictive.temporal"] == 0
    assert loss_zero.forecast_transition_mask.any()
    assert loss_zero.auxiliary_losses["predictive.temporal"] == 0
    assert loss_zero.feedback_delta.abs().sum() > 0


def test_dense_control_executes_both_memories_without_drops() -> None:
    batch = DelayedRuleSwitchTask(namespace="p4").generate("train", [0, 1, 2])
    output = ModularBrainNetworkV2().forward_batch(batch, routing_mode="dense")

    assert (
        output.cost_statistics["optional.active_calls"]
        == output.cost_statistics["optional.dense_calls"]
    )
    for decision in output.routing_trace:
        assert decision.capacity_drops == 0


def test_unselected_memory_is_not_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = DelayedRuleSwitchTask(namespace="p4").generate("train", [0, 1])
    model = ModularBrainNetworkV2()
    router = cast_router(model.registry.get(SPARSE_ROUTER_V2))
    with torch.no_grad():
        router.scorer.weight.zero_()
        router.scorer.bias.copy_(torch.tensor([-1.0, 1.0]))
    episodic = model.registry.get(EPISODIC_MEMORY)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("unselected episodic memory executed")

    monkeypatch.setattr(episodic, "forward", forbidden)
    output = model.forward_batch(batch, routing_mode="no_reservation")
    assert all(not decision.executed_mask[..., 0].any() for decision in output.routing_trace)


def test_telemetry_flag_is_numerically_inert() -> None:
    batch = DelayedRuleSwitchTask(namespace="p4").generate("validation", [0, 1, 2])
    model = ModularBrainNetworkV2().eval()
    with torch.no_grad():
        disabled = model.forward_batch(batch, phase="evaluate", telemetry_enabled=False)
        enabled = model.forward_batch(batch, phase="evaluate", telemetry_enabled=True)
    torch.testing.assert_close(enabled.logits, disabled.logits, rtol=0.0, atol=0.0)
    for name in disabled.auxiliary_losses:
        torch.testing.assert_close(
            enabled.auxiliary_losses[name],
            disabled.auxiliary_losses[name],
            rtol=0.0,
            atol=0.0,
        )
    for module_id, expected in disabled.state.module_states.items():
        actual = enabled.state.get(module_id)
        for name, tensor in expected.tensors.items():
            torch.testing.assert_close(actual.tensors[name], tensor, rtol=0.0, atol=0.0)


def cast_router(module: object) -> SparseRouterV2:
    assert isinstance(module, SparseRouterV2)
    return module
