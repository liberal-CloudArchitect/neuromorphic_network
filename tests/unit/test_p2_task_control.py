"""Tests for the leakage-safe P2 task boundary."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from neuromorphic.tasks.associative_recall import AssociativeRecallTask
from neuromorphic.tasks.control import (
    ASSOCIATIVE_RECALL,
    DELAYED_RULE_SWITCH,
    GOAL_CONTEXT_DIM,
    SMALL_GRAPH,
    TASK_INPUT_DIMS,
    task_control_from_batch,
)
from neuromorphic.tasks.delayed_rule_switch import DelayedRuleSwitchTask
from neuromorphic.tasks.small_graph import SmallGraphTask


@pytest.mark.parametrize(
    ("task", "task_id"),
    [
        (AssociativeRecallTask(), ASSOCIATIVE_RECALL),
        (DelayedRuleSwitchTask(), DELAYED_RULE_SWITCH),
        (SmallGraphTask(), SMALL_GRAPH),
    ],
)
def test_task_control_has_fixed_leakage_safe_shape(task: object, task_id: str) -> None:
    batch = task.generate("train", [0, 1])  # type: ignore[attr-defined]
    control = task_control_from_batch(batch)

    assert control.task_id == task_id
    assert batch.input_dim == TASK_INPUT_DIMS[task_id]
    assert control.goal_context.shape == (2, batch.sequence_length, GOAL_CONTEXT_DIM)
    assert torch.equal(control.valid_mask, batch.valid_mask)
    assert torch.equal(control.loss_mask, batch.loss_mask)
    assert torch.count_nonzero(control.goal_context[~batch.valid_mask]) == 0


def test_control_does_not_read_targets_or_auxiliary_targets() -> None:
    batch = SmallGraphTask().generate("train", [3, 4])
    first = task_control_from_batch(batch)
    changed = replace(
        batch,
        targets=torch.full_like(batch.targets, 1),
        auxiliary_targets={
            name: torch.zeros_like(value) for name, value in batch.auxiliary_targets.items()
        },
    )
    second = task_control_from_batch(changed)

    assert torch.equal(first.goal_context, second.goal_context)
    assert torch.equal(first.action_nodes, second.action_nodes)


def test_small_graph_action_slots_are_decoded_from_observation() -> None:
    batch = SmallGraphTask().generate("train", [5])
    control = task_control_from_batch(batch)
    expected_valid = batch.inputs[..., 352:356].bool()

    assert torch.equal(control.action_valid_mask[..., :4], expected_valid)
    assert not control.action_valid_mask[..., 4:].any()
    observed_slots = batch.inputs[..., 288:352].reshape(1, batch.sequence_length, 4, 16)
    expected_nodes = observed_slots.argmax(-1)
    available = expected_valid & batch.valid_mask.unsqueeze(-1)
    assert torch.equal(control.action_nodes[..., :4][available], expected_nodes[available])
    assert torch.all(control.action_nodes[..., :4][~available] == -1)


def test_selected_action_copy_is_model_owned_and_padding_safe() -> None:
    batch = AssociativeRecallTask().generate("train", [0, 1])
    control = task_control_from_batch(batch)
    actions = torch.zeros_like(batch.targets)
    actions[batch.valid_mask] = 7
    copied = control.with_selected_action(actions)

    assert not control.selected_action_copy.any()
    assert torch.all(copied.selected_action_copy[batch.valid_mask, 7] == 1)
    assert torch.count_nonzero(copied.selected_action_copy[~batch.valid_mask]) == 0


def test_invalid_task_dimension_and_invalid_selected_action_are_rejected() -> None:
    batch = AssociativeRecallTask().generate("train", [0])
    wrong = replace(batch, inputs=torch.zeros(1, batch.sequence_length, 67))
    with pytest.raises(ValueError, match="input dimension"):
        task_control_from_batch(wrong)

    control = task_control_from_batch(batch)
    with pytest.raises(ValueError, match=r"\[0, 31\]"):
        control.with_selected_action(torch.full_like(batch.targets, 32))
