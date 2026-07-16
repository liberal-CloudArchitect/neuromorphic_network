"""Contract and behavior tests for P2 artificial computation modules."""

from __future__ import annotations

import pytest
import torch

from neuromorphic.core.contracts import BrainPacket, ModuleContext
from neuromorphic.core.registry import (
    ACTION_SELECTOR,
    EPISODIC_MEMORY,
    OPTIONAL_EXPERT_IDS,
    SENSORY_ENCODER,
    SPARSE_ROUTER,
)
from neuromorphic.modules.action_selector import ActionSelector
from neuromorphic.modules.episodic_memory import EpisodicMemory
from neuromorphic.modules.predictive_adapter import PredictiveAdapter
from neuromorphic.modules.sensory_encoder import SensoryEncoder
from neuromorphic.modules.sparse_router import SparseRouter
from neuromorphic.modules.working_memory import WorkingMemory


def _goal(batch: int, steps: int, task: int = 0) -> torch.Tensor:
    goal = torch.zeros(batch, steps, 104)
    goal[..., 69 + task] = 1.0
    return goal


def _packet(
    *, batch: int = 3, steps: int = 1, source: str = SENSORY_ENCODER, task: int = 0
) -> BrainPacket:
    return BrainPacket(
        representation=torch.randn(batch, steps, 128),
        valid_mask=torch.ones(batch, steps, dtype=torch.bool),
        modality="symbolic",
        step_index=torch.arange(steps).repeat(batch, 1),
        source_module=source,
        goal_context=_goal(batch, steps, task),
    )


def _context(packet: BrainPacket, task_id: str = "associative_recall.v1") -> ModuleContext:
    return ModuleContext(
        task_id=task_id,
        phase="train",
        reset_mask=torch.zeros_like(packet.valid_mask),
        eligible_modules=OPTIONAL_EXPERT_IDS,
    )


def test_sensory_encoder_creates_first_legal_packet_and_backpropagates() -> None:
    module = SensoryEncoder()
    values = torch.randn(2, 3, 128, requires_grad=True)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    context = ModuleContext(
        "associative_recall.v1",
        "train",
        torch.zeros(2, 3, dtype=torch.bool),
        OPTIONAL_EXPERT_IDS,
    )
    output = module.forward_inputs(
        values, valid, torch.arange(3).repeat(2, 1), _goal(2, 3), context
    )
    assert output.packet.source_module == SENSORY_ENCODER
    assert output.packet.representation.shape == (2, 3, 128)
    torch.autograd.backward(output.packet.representation.sum())
    assert values.grad is not None


def test_episodic_memory_stages_after_read_then_commits_for_next_step() -> None:
    module = EpisodicMemory()
    store = _packet(batch=1)
    assert store.goal_context is not None
    store.goal_context[..., 0] = 1.0
    store.goal_context[..., 5 + 7] = 1.0
    state = module.initial_state(1, device=torch.device("cpu"), dtype=torch.float32)
    first = module(store, state, _context(store))
    assert not first.state.tensors["occupied"].any()
    assert first.state.tensors["pending_valid"].tolist() == [True]
    committed = module.commit_pending(first.state)
    assert committed.tensors["occupied"].sum().item() == 1
    assert not committed.tensors["pending_valid"].any()

    query_goal = _goal(1, 1)
    query_goal[..., 2] = 1.0
    query_goal[..., 5 + 7] = 1.0
    query = BrainPacket(
        torch.zeros(1, 1, 128),
        torch.ones(1, 1, dtype=torch.bool),
        "symbolic",
        torch.ones(1, 1, dtype=torch.long),
        SENSORY_ENCODER,
        query_goal,
    )
    recalled = module(query, committed, _context(query))
    assert recalled.packet.source_module == EPISODIC_MEMORY
    assert recalled.packet.representation.abs().sum().item() > 0


def test_episodic_distractor_does_not_write_and_reset_is_row_local() -> None:
    module = EpisodicMemory()
    packet = _packet(batch=2)
    assert packet.goal_context is not None
    packet.goal_context[..., 1] = 1.0
    state = module.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    output = module(packet, state, _context(packet))
    assert not output.state.tensors["pending_valid"].any()
    filled = module.commit_pending(
        module(
            BrainPacket(
                packet.representation,
                packet.valid_mask,
                packet.modality,
                packet.step_index,
                packet.source_module,
                _goal(2, 1),
            ),
            state,
            _context(packet),
        ).state
    )
    reset = module.reset_state(filled, torch.tensor([True, False]))
    assert not reset.tensors["occupied"][0].any()


def test_working_memory_has_fixed_slots_padding_and_row_reset() -> None:
    module = WorkingMemory()
    packet = _packet(batch=2, steps=2)
    valid = torch.tensor([[True, False], [True, True]])
    packet = BrainPacket(
        packet.representation,
        valid,
        packet.modality,
        packet.step_index,
        packet.source_module,
        packet.goal_context,
    )
    state = module.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    output = module(packet, state, _context(packet))
    assert output.state.tensors["slots"].shape == (2, 4, 32)
    reset = module.reset_state(output.state, torch.tensor([True, False]))
    assert torch.count_nonzero(reset.tensors["slots"][0]) == 0
    assert torch.count_nonzero(reset.tensors["slots"][1]) > 0
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in output.auxiliary_losses.values())


def test_predictive_targets_follow_actual_action_slot_not_canonical_target() -> None:
    action_nodes = torch.tensor([[[4, 8, -1, -1]]])
    selected = torch.tensor([[1]])
    assert PredictiveAdapter.dynamic_targets(action_nodes, selected).item() == 8
    swapped = torch.tensor([[[8, 4, -1, -1]]])
    assert PredictiveAdapter.dynamic_targets(swapped, selected).item() == 4

    module = PredictiveAdapter()
    packet = _packet(batch=1, source=ACTION_SELECTOR, task=2)
    state = module.initial_state(1, device=torch.device("cpu"), dtype=torch.float32)
    output = module.forward_with_action(
        packet, state, _context(packet, "small_graph.v1"), selected, action_nodes
    )
    assert output.prediction_logits is not None
    assert output.prediction_logits.shape == (1, 1, 16)
    assert torch.isfinite(output.auxiliary_losses["predictive_next_state"])
    reset = module.reset_state(output.state, torch.tensor([True]))
    assert reset.tensors["pending_action"].item() == -1


def test_action_selector_isolates_heads_and_masks_invalid_graph_actions() -> None:
    module = ActionSelector()
    packet = _packet(batch=1, source=SPARSE_ROUTER, task=2)
    assert packet.goal_context is not None
    packet.goal_context[..., 37:39] = 1.0
    output = module(
        packet,
        module.initial_state(1, device=torch.device("cpu"), dtype=torch.float32),
        _context(packet, "small_graph.v1"),
    )
    assert output.action_logits is not None
    assert output.action_logits.shape == (1, 1, 4)
    assert output.action_logits[..., 2:].max() < -1.0e20


def test_router_exact_top2_capacity_ties_padding_and_combine_source() -> None:
    module = SparseRouter()
    torch.nn.init.zeros_(module.scorer.weight)
    torch.nn.init.zeros_(module.scorer.bias)
    packet = _packet(batch=7, steps=2)
    valid = torch.tensor([[True, False]] * 7)
    packet = BrainPacket(
        packet.representation,
        valid,
        packet.modality,
        packet.step_index,
        packet.source_module,
        packet.goal_context,
    )
    decision = module.route(packet)
    assert torch.all(decision.raw_top2_mask[:, 0].sum(-1) == 2)
    assert torch.all(decision.executed_mask[:, 0].sum(-1) == 2)
    assert not decision.executed_mask[:, 1].any()
    assert decision.raw_top2_mask[0, 0].tolist() == [True, True, False]
    loads = decision.executed_mask[:, 0].sum(0)
    assert torch.all(loads <= decision.capacity[0])
    assert decision.capacity_drops == 0

    state = module.initial_state(7, device=torch.device("cpu"), dtype=torch.float32)
    expert = BrainPacket(
        packet.representation + 1,
        packet.valid_mask,
        packet.modality,
        packet.step_index,
        EPISODIC_MEMORY,
        packet.goal_context,
    )
    combined = module.combine(packet, {EPISODIC_MEMORY: expert}, decision, state, _context(packet))
    assert combined.packet.source_module == SPARSE_ROUTER
    assert combined.packet.representation.shape == packet.representation.shape


@pytest.mark.parametrize("top_k", [0, 1, 3])
def test_router_rejects_non_top2_configuration(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k=2"):
        SparseRouter(top_k=top_k)
