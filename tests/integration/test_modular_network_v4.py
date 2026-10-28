"""The P6 network over the four P6-namespace tasks (CR-015)."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from neuromorphic.core.contracts import BrainPacket, ModuleContext
from neuromorphic.core.registry import (
    EPISODIC_MEMORY,
    P6_MODULE_IDS,
    SENSORY_ENCODER,
    WORKING_MEMORY,
    WORKING_MEMORY_V2,
)
from neuromorphic.modules.network_v4 import ModularBrainNetworkV4
from neuromorphic.modules.working_memory_v2 import WorkingMemoryV2
from neuromorphic.tasks.associative_recall import AssociativeRecallTask
from neuromorphic.tasks.delayed_rule_switch import DelayedRuleSwitchTask
from neuromorphic.tasks.small_graph import SmallGraphTask
from neuromorphic.tasks.uncued_rule_switch import UncuedRuleSwitchTask

_TASKS = (
    AssociativeRecallTask(namespace="p6"),
    DelayedRuleSwitchTask(namespace="p6"),
    SmallGraphTask(namespace="p6"),
    UncuedRuleSwitchTask(),
)


@pytest.mark.parametrize("working_memory", ["v2", "v1"])
@pytest.mark.parametrize("task", _TASKS, ids=lambda task: task.task_id)
def test_v4_runs_all_four_tasks_with_finite_backward(task: object, working_memory: str) -> None:
    torch.manual_seed(7)
    batch = task.generate("train", [0, 1, 2, 3])  # type: ignore[attr-defined]
    model = ModularBrainNetworkV4(working_memory=working_memory)  # type: ignore[arg-type]
    output = model.forward_batch(batch)

    working_id = WORKING_MEMORY_V2 if working_memory == "v2" else WORKING_MEMORY
    assert set(model.registry.ids) == set(model.module_ids)
    assert working_id in model.module_ids and model.optional_expert_ids == (
        EPISODIC_MEMORY,
        working_id,
    )
    assert output.action_logits.shape[:2] == batch.inputs.shape[:2]
    primary = F.cross_entropy(output.action_logits[batch.loss_mask], batch.targets[batch.loss_mask])
    auxiliary = torch.stack(list(output.auxiliary_losses.values())).sum()
    loss = primary + 0.01 * auxiliary
    loss.backward()  # type: ignore[no-untyped-call]
    assert torch.isfinite(loss)


def test_v4_default_is_the_full_cell() -> None:
    model = ModularBrainNetworkV4()
    working = model.registry.get(WORKING_MEMORY_V2)
    assert model.module_ids == P6_MODULE_IDS
    assert isinstance(working, WorkingMemoryV2)
    assert working.write_source == "enriched" and working.surprise_gating


def test_uncued_steps_always_execute_working_memory_within_the_dual_budget() -> None:
    batch = UncuedRuleSwitchTask().generate("train", list(range(8)))
    model = ModularBrainNetworkV4(dual_route_fraction=0.25)
    output = model.forward_batch(batch)

    for step, decision in enumerate(output.routing_trace):
        valid = batch.valid_mask[:, step]
        assert decision.reserved_mask[:, 0, 1][valid].all()
        assert decision.executed_mask[:, 0, 1][valid].all()
        dual = decision.executed_mask[:, 0].sum(dim=-1).gt(1) & valid
        assert int(dual.sum().item()) <= math.floor(int(valid.sum().item()) * 0.25)
    assert output.module_metrics[f"selected.{WORKING_MEMORY_V2}"] == batch.valid_mask.sum()


def test_uncued_training_reaches_the_surprise_gate() -> None:
    torch.manual_seed(2)
    batch = UncuedRuleSwitchTask().generate("train", list(range(4)))
    model = ModularBrainNetworkV4()
    output = model.forward_batch(batch)
    assert output.module_metrics["predictive.surprise_sum"] > 0
    loss = F.cross_entropy(output.action_logits[batch.loss_mask], batch.targets[batch.loss_mask])
    loss.backward()  # type: ignore[no-untyped-call]

    working = model.registry.get(WORKING_MEMORY_V2)
    assert isinstance(working, WorkingMemoryV2)
    assert working.surprise_gate.grad is not None
    assert working.surprise_gate.grad.abs().sum() > 0


def _expert_call(model: ModularBrainNetworkV4, surprise: torch.Tensor) -> torch.Tensor:
    working = model.registry.get(WORKING_MEMORY_V2)
    assert isinstance(working, WorkingMemoryV2)
    indices = torch.tensor([0, 2, 3])
    selected = BrainPacket(
        representation=torch.randn(3, 1, model.feature_dim),
        valid_mask=torch.ones(3, 1, dtype=torch.bool),
        modality="fixture",
        step_index=torch.ones(3, 1, dtype=torch.long),
        source_module=SENSORY_ENCODER,
    )
    context = ModuleContext(
        task_id="uncued_rule_switch.v1",
        phase="evaluate",
        reset_mask=torch.zeros(3, 1, dtype=torch.bool),
        eligible_modules=model.optional_expert_ids,
    )
    state = working.initial_state(3, device=torch.device("cpu"), dtype=torch.float32)
    state.tensors["slots"].normal_()
    output = model._run_expert(
        WORKING_MEMORY_V2, working, selected, state, context, indices=indices, surprise=surprise
    )
    expected_surprise = {
        "full": surprise,
        "shuffle": surprise.roll(1, dims=0),
        "zero": torch.zeros_like(surprise),
    }[model.working_surprise_mode].index_select(0, indices)
    direct = working.forward(selected, state, context, surprise=expected_surprise)
    assert torch.equal(output.state.tensors["slots"], direct.state.tensors["slots"])
    return output.state.tensors["slots"]


def test_working_surprise_modes_feed_working_memory_only_the_intended_signal() -> None:
    torch.manual_seed(4)
    model = ModularBrainNetworkV4()
    surprise = torch.tensor([0.0, 0.9, 0.4, 1.3]).view(4, 1, 1)

    full = _expert_call(model, surprise)
    with model.working_surprise("shuffle"):
        shuffled = _expert_call(model, surprise)
    with model.working_surprise("zero"):
        zeroed = _expert_call(model, surprise)
    assert model.working_surprise_mode == "full"
    assert not torch.equal(full, shuffled) and not torch.equal(full, zeroed)
    with (
        pytest.raises(ValueError, match="working surprise mode"),
        model.working_surprise("drop"),  # type: ignore[arg-type]
    ):
        pass


def test_ablation_cells_build_the_frozen_variants() -> None:
    gate_off = ModularBrainNetworkV4(surprise_gating=False)
    enrich_off = ModularBrainNetworkV4(write_source="current")
    for model, source, gated in ((gate_off, "enriched", False), (enrich_off, "current", True)):
        working = model.registry.get(WORKING_MEMORY_V2)
        assert isinstance(working, WorkingMemoryV2)
        assert working.write_source == source and working.surprise_gating is gated

    batch = UncuedRuleSwitchTask().generate("train", list(range(3)))
    torch.manual_seed(1)
    with gate_off.working_surprise("zero"):
        zeroed = gate_off.forward_batch(batch).action_logits
    full = gate_off.forward_batch(batch).action_logits
    assert torch.equal(zeroed, full)
    with pytest.raises(ValueError, match=r"working_memory\.v2 only"):
        ModularBrainNetworkV4(working_memory="v1", surprise_gating=False)
