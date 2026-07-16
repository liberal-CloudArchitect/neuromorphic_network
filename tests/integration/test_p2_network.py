"""Integration tests for the six-module P2 composition graph."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as functional

from neuromorphic.core.registry import (
    ACTION_SELECTOR,
    EPISODIC_MEMORY,
    OPTIONAL_EXPERT_IDS,
    PREDICTIVE_ADAPTER,
    WORKING_MEMORY,
)
from neuromorphic.modules.network import ModularBrainNetwork
from neuromorphic.tasks.associative_recall import AssociativeRecallTask
from neuromorphic.tasks.delayed_rule_switch import DelayedRuleSwitchTask
from neuromorphic.tasks.small_graph import SmallGraphTask


@pytest.mark.parametrize(
    "task",
    [AssociativeRecallTask(), DelayedRuleSwitchTask(), SmallGraphTask()],
)
def test_modular_network_runs_all_tasks_with_true_top2_routing(task: object) -> None:
    torch.manual_seed(7)
    batch = task.generate("train", [0, 1, 2])  # type: ignore[attr-defined]
    model = ModularBrainNetwork()
    output = model.forward_batch(batch)

    assert output.packet.source_module == ACTION_SELECTOR
    assert output.packet.representation.shape == (*batch.inputs.shape[:2], 128)
    assert output.action_logits.shape == (
        batch.batch_size,
        batch.sequence_length,
        task.num_classes,  # type: ignore[attr-defined]
    )
    assert len(output.routing_trace) == batch.sequence_length
    for step, decision in enumerate(output.routing_trace):
        executed = decision.executed_mask[:, 0]
        expected = batch.valid_mask[:, step].to(torch.long) * 2
        assert torch.equal(executed.sum(dim=-1), expected)
        assert decision.capacity_drops == 0
    assert (
        output.cost_statistics["optional.active_calls"]
        < output.cost_statistics["optional.dense_calls"]
    )
    selected_total = output.cost_statistics["optional.active_calls"].new_zeros(())
    for module_id in OPTIONAL_EXPERT_IDS:
        selected_total = selected_total + output.module_metrics[f"selected.{module_id}"]
    assert torch.equal(selected_total, output.cost_statistics["optional.active_calls"])
    assert output.state.valid_step_counts.tolist() == batch.valid_mask.sum(dim=1).tolist()


def test_network_loss_has_finite_backward_path() -> None:
    torch.manual_seed(7)
    batch = DelayedRuleSwitchTask().generate("train", [0, 1, 2, 3])
    model = ModularBrainNetwork()
    output = model.forward_batch(batch)
    primary = functional.cross_entropy(
        output.action_logits[batch.loss_mask], batch.targets[batch.loss_mask]
    )
    auxiliary = sum(output.auxiliary_losses.values(), primary.new_zeros(()))
    loss = primary + 0.01 * auxiliary

    assert torch.isfinite(loss)
    loss.backward()  # type: ignore[no-untyped-call]
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_small_graph_prediction_target_uses_model_action_slots() -> None:
    torch.manual_seed(7)
    batch = SmallGraphTask().generate("train", [5, 6, 7, 8])
    model = ModularBrainNetwork()
    output = model.forward_batch(batch)
    selected_action = output.action_logits.argmax(dim=-1)
    observed_action_nodes = (
        batch.inputs[..., 288:352]
        .reshape(batch.batch_size, batch.sequence_length, 4, 16)
        .argmax(-1)
    )
    observed_valid = batch.inputs[..., 352:356].bool()
    observed_action_nodes = torch.where(
        observed_valid, observed_action_nodes, torch.full_like(observed_action_nodes, -1)
    )
    expected = observed_action_nodes.gather(-1, selected_action.unsqueeze(-1)).squeeze(-1)

    assert output.prediction_logits is not None
    assert torch.equal(
        output.prediction_targets[output.prediction_mask], expected[output.prediction_mask]
    )
    assert torch.all(output.prediction_mask <= batch.loss_mask)


def test_padding_neither_routes_nor_advances_state() -> None:
    torch.manual_seed(7)
    batch = AssociativeRecallTask().generate("train", [0, 19])
    model = ModularBrainNetwork()
    output = model.forward_batch(batch)
    lengths = batch.valid_mask.sum(dim=1)

    assert torch.equal(output.state.valid_step_counts, lengths)
    for step, decision in enumerate(output.routing_trace):
        padded = ~batch.valid_mask[:, step]
        assert not decision.executed_mask[padded].any()
        assert not decision.raw_top2_mask[padded].any()


def test_unselected_expert_is_never_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = DelayedRuleSwitchTask().generate("train", [0, 1])
    model = ModularBrainNetwork()
    episodic = model.registry.get(EPISODIC_MEMORY)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("unselected expert executed")

    monkeypatch.setattr(episodic, "forward", forbidden)
    output = model.forward_batch(
        batch,
        forced_experts=(WORKING_MEMORY, PREDICTIVE_ADAPTER),
    )
    assert all(
        not decision.executed_mask[..., OPTIONAL_EXPERT_IDS.index(EPISODIC_MEMORY)].any()
        for decision in output.routing_trace
    )


def test_forced_routing_is_batch_reorder_equivalent() -> None:
    batch = AssociativeRecallTask().generate("train", [0, 1, 2])
    permutation = torch.tensor([2, 0, 1])
    reordered = type(batch)(
        inputs=batch.inputs[permutation],
        targets=batch.targets[permutation],
        valid_mask=batch.valid_mask[permutation],
        loss_mask=batch.loss_mask[permutation],
        episode_ids=batch.episode_ids[permutation],
        metadata=batch.metadata,
        auxiliary_targets={
            name: value[permutation] for name, value in batch.auxiliary_targets.items()
        },
    )
    model = ModularBrainNetwork()
    route = (EPISODIC_MEMORY, WORKING_MEMORY)
    first = model.forward_batch(batch, forced_experts=route)
    second = model.forward_batch(reordered, forced_experts=route)
    inverse = torch.argsort(permutation)
    assert torch.allclose(first.action_logits, second.action_logits[inverse], atol=1e-6)
    for module_id in first.state.module_states:
        for name, tensor in first.state.get(module_id).tensors.items():
            reordered_tensor = second.state.get(module_id).tensors[name][inverse]
            if tensor.is_floating_point():
                assert torch.allclose(tensor, reordered_tensor, atol=1e-6)
            else:
                assert torch.equal(tensor, reordered_tensor)
