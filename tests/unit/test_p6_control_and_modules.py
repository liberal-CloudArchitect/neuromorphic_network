"""P6 four-task control and module variants (CR-015, erratum E1)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from neuromorphic.core.contracts import BrainPacket, ModuleContext, trusted_internal_execution
from neuromorphic.core.registry import (
    EPISODIC_MEMORY,
    SENSORY_ENCODER,
    WORKING_MEMORY,
    WORKING_MEMORY_V2,
)
from neuromorphic.modules.action_selector import ActionSelector
from neuromorphic.modules.p6_modules import (
    P6ActionSelector,
    P6TaskBoundaryAdapters,
    PredictiveAdapterV3P6,
    SparseRouterV3P6,
)
from neuromorphic.modules.predictive_adapter_v3 import PredictiveAdapterV3
from neuromorphic.modules.sparse_router_v3 import SparseRouterV3
from neuromorphic.tasks.associative_recall import AssociativeRecallTask
from neuromorphic.tasks.control import TaskControl, task_control_from_batch
from neuromorphic.tasks.delayed_rule_switch import DelayedRuleSwitchTask
from neuromorphic.tasks.p6_control import (
    P6_GOAL_CONTEXT_DIM,
    UNCUED_RULE_SWITCH,
    UNCUED_TASK_CHANNEL,
    P6TaskControl,
    p6_task_control_from_batch,
    p6_task_one_hot,
)
from neuromorphic.tasks.small_graph import SmallGraphTask
from neuromorphic.tasks.uncued_rule_switch import FEEDBACK, UncuedRuleSwitchTask

_MAINLINE_TASKS = (
    AssociativeRecallTask(namespace="p6"),
    DelayedRuleSwitchTask(namespace="p6"),
    SmallGraphTask(namespace="p6"),
)


def _uncued_control() -> tuple[P6TaskControl, torch.Tensor]:
    batch = UncuedRuleSwitchTask().generate("train", list(range(4)))
    with trusted_internal_execution():
        return p6_task_control_from_batch(batch), batch.inputs


@pytest.mark.parametrize("task", _MAINLINE_TASKS, ids=lambda task: task.task_id)
def test_p6_control_keeps_the_mainline_layout_and_a_zero_uncued_bit(task: object) -> None:
    batch = task.generate("train", [0, 1, 2])  # type: ignore[attr-defined]
    with trusted_internal_execution():
        mainline = task_control_from_batch(batch)
        p6 = p6_task_control_from_batch(batch)

    assert isinstance(p6, P6TaskControl)
    assert p6.goal_context.shape[-1] == P6_GOAL_CONTEXT_DIM
    assert torch.equal(p6.goal_context[..., :UNCUED_TASK_CHANNEL], mainline.goal_context)
    assert torch.count_nonzero(p6.goal_context[..., UNCUED_TASK_CHANNEL]) == 0
    assert torch.equal(p6.action_nodes, mainline.action_nodes)


def test_uncued_control_sets_its_task_bit_and_drs_style_event_bits() -> None:
    control, inputs = _uncued_control()
    goal = control.goal_context
    valid = control.valid_mask

    assert control.task_id == UNCUED_RULE_SWITCH
    assert torch.equal(goal[..., UNCUED_TASK_CHANNEL].bool(), valid)
    assert torch.count_nonzero(goal[..., 69:72]) == 0
    assert torch.equal(goal[..., :3], inputs[..., :3] * valid.unsqueeze(-1))
    feedback = inputs[..., FEEDBACK].gt(0.5) & valid
    assert feedback.any() and torch.count_nonzero(goal[feedback][:, :3]) == 0
    assert torch.equal(goal[..., 3].bool(), control.loss_mask)
    assert torch.count_nonzero(goal[~valid]) == 0
    assert torch.equal(
        p6_task_one_hot(goal)[valid], torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(int(valid.sum()), 4)
    )


def test_p6_control_steps_and_action_copies_keep_the_uncued_bit() -> None:
    control, _ = _uncued_control()
    with trusted_internal_execution():
        step = control.at_step(2)
        copied = step.with_selected_action(torch.ones(step.valid_mask.shape, dtype=torch.long))

    assert isinstance(step, P6TaskControl) and isinstance(copied, P6TaskControl)
    assert torch.equal(
        copied.goal_context[..., UNCUED_TASK_CHANNEL], step.goal_context[..., UNCUED_TASK_CHANNEL]
    )
    assert torch.equal(copied.goal_context[..., 73], step.valid_mask.to(torch.float32))


def test_mainline_control_still_rejects_the_fourth_task_and_wide_context() -> None:
    control, _ = _uncued_control()
    with pytest.raises(ValueError, match="unsupported task_id"):
        TaskControl(
            task_id=UNCUED_RULE_SWITCH,
            goal_context=control.goal_context[..., :UNCUED_TASK_CHANNEL],
            valid_mask=control.valid_mask,
            loss_mask=control.loss_mask,
            action_nodes=control.action_nodes,
        )
    with pytest.raises(ValueError, match=r"\[B, T, 104\]"):
        TaskControl(
            task_id="delayed_rule_switch.v1",
            goal_context=control.goal_context,
            valid_mask=control.valid_mask,
            loss_mask=control.loss_mask,
            action_nodes=control.action_nodes,
        )


def _packet(goal: torch.Tensor, features: int = 8) -> BrainPacket:
    batch, steps, _ = goal.shape
    return BrainPacket(
        representation=torch.randn(batch, steps, features),
        valid_mask=torch.ones(batch, steps, dtype=torch.bool),
        modality="fixture",
        step_index=torch.arange(steps).repeat(batch, 1),
        source_module=SENSORY_ENCODER,
        goal_context=goal,
    )


def _narrowed(packet: BrainPacket) -> BrainPacket:
    """The same packet with the mainline 104-channel goal context."""

    assert packet.goal_context is not None
    return BrainPacket(
        representation=packet.representation,
        valid_mask=packet.valid_mask,
        modality=packet.modality,
        step_index=packet.step_index,
        source_module=packet.source_module,
        goal_context=packet.goal_context[..., :UNCUED_TASK_CHANNEL],
    )


def _linear(sequential: nn.Sequential, index: int) -> nn.Linear:
    layer = sequential[index]
    assert isinstance(layer, nn.Linear)
    return layer


def _goal(task: int, batch: int = 8, steps: int = 2) -> torch.Tensor:
    goal = torch.zeros(batch, steps, P6_GOAL_CONTEXT_DIM)
    goal[..., UNCUED_TASK_CHANNEL if task == 3 else 69 + task] = 1.0
    goal[..., 4] = 1.0
    goal[..., 2] = 1.0
    return goal


def _router_pair() -> tuple[SparseRouterV3, SparseRouterV3P6]:
    torch.manual_seed(3)
    v3 = SparseRouterV3(feature_dim=8, task_embedding_dim=4)
    p6 = SparseRouterV3P6(feature_dim=8, task_embedding_dim=4, working_module_id=WORKING_MEMORY)
    with torch.no_grad():
        p6.task_embedding.weight[:3].copy_(v3.task_embedding.weight)
        p6.scorer.load_state_dict(v3.scorer.state_dict())
        p6.fusion.load_state_dict(v3.fusion.state_dict())
    return v3, p6


@pytest.mark.parametrize("task", [0, 1, 2])
def test_p6_router_matches_v3_on_the_three_mainline_tasks(task: int) -> None:
    v3, p6 = _router_pair()
    packet = _packet(_goal(task))
    narrow = _narrowed(packet)
    surprise = torch.rand(8, 2, 1)
    expected = v3.route(narrow, surprise=surprise)
    actual = p6.route(packet, surprise=surprise)

    assert torch.equal(actual.scores, expected.scores)
    assert torch.equal(actual.executed_mask, expected.executed_mask)
    for name, loss in v3.routing_losses(narrow, expected).items():
        assert torch.equal(p6.routing_losses(packet, actual)[name], loss)


def test_p6_router_routes_uncued_steps_like_drs() -> None:
    _, p6 = _router_pair()
    drs = p6.route(_packet(_goal(1)), surprise=torch.zeros(8, 2, 1))
    uncued = p6.route(_packet(_goal(3)), surprise=torch.zeros(8, 2, 1))

    assert torch.equal(p6._task_index(_goal(3)), torch.full((8, 2), 3))
    for decision in (drs, uncued):
        assert decision.reserved_mask[..., 1].all() and not decision.reserved_mask[..., 0].any()
        assert decision.executed_mask[..., 1].all()
        assert int(decision.executed_mask.sum(dim=-1).gt(1).sum()) == 2 * 2
    assert p6.expert_ids == (EPISODIC_MEMORY, WORKING_MEMORY)
    assert SparseRouterV3P6(feature_dim=8).expert_ids == (EPISODIC_MEMORY, WORKING_MEMORY_V2)
    with pytest.raises(ValueError, match="working-memory expert"):
        SparseRouterV3P6(feature_dim=8, working_module_id=EPISODIC_MEMORY)


def test_p6_predictor_matches_v3_when_the_uncued_column_is_zero() -> None:
    torch.manual_seed(5)
    v3 = PredictiveAdapterV3(feature_dim=8, action_count=4, action_dim=2)
    p6 = PredictiveAdapterV3P6(feature_dim=8, action_count=4, action_dim=2)
    v3_first, v3_last = _linear(v3.transition_delta, 0), _linear(v3.transition_delta, -1)
    p6_first, p6_last = _linear(p6.transition_delta, 0), _linear(p6.transition_delta, -1)
    with torch.no_grad():
        p6_first.weight.zero_()
        p6_first.weight[:, :-1].copy_(v3_first.weight)
        p6_first.bias.copy_(v3_first.bias)
        v3_last.weight.normal_()
        p6_last.weight.copy_(v3_last.weight)
        p6_last.bias.copy_(v3_last.bias)
        p6.action_embedding.load_state_dict(v3.action_embedding.state_dict())

    wide = _packet(_goal(1, batch=3, steps=1))
    narrow = _narrowed(wide)
    context = ModuleContext(
        task_id="delayed_rule_switch.v1",
        phase="train",
        reset_mask=torch.zeros(3, 1, dtype=torch.bool),
        eligible_modules=(EPISODIC_MEMORY, WORKING_MEMORY),
    )
    action = torch.tensor([[0], [1], [3]])
    cpu, dtype = torch.device("cpu"), torch.float32
    expected = v3.commit(narrow, v3.initial_state(3, device=cpu, dtype=dtype), context, action)
    actual = p6.commit(wide, p6.initial_state(3, device=cpu, dtype=dtype), context, action)

    assert p6_first.in_features == v3_first.in_features + 1
    torch.testing.assert_close(actual.tensors["forecast"], expected.tensors["forecast"])
    assert not torch.equal(expected.tensors["forecast"], expected.tensors["persistence"])


def test_p6_selector_and_adapters_add_only_the_uncued_task() -> None:
    torch.manual_seed(9)
    selector = P6ActionSelector(feature_dim=8)
    assert set(selector.heads) == {*ActionSelector(feature_dim=8).heads, "uncued_rule_switch"}
    assert selector.heads["uncued_rule_switch"].out_features == 2

    adapters = P6TaskBoundaryAdapters(feature_dim=8)
    assert len(adapters.adapters) == 4
    assert adapters(torch.zeros(2, 3, 11), UNCUED_RULE_SWITCH).shape == (2, 3, 8)
    with pytest.raises(ValueError, match="shape"):
        adapters(torch.zeros(2, 3, 9), UNCUED_RULE_SWITCH)
