from __future__ import annotations

import torch

from neuromorphic.core.contracts import BrainPacket, ModuleContext
from neuromorphic.core.registry import (
    EPISODIC_MEMORY,
    OPTIONAL_EXPERT_IDS,
    SENSORY_ENCODER,
    WORKING_MEMORY,
)
from neuromorphic.modules.sparse_router_v2 import SparseRouterV2


def _goal(batch: int, steps: int, task: int = 0) -> torch.Tensor:
    goal = torch.zeros(batch, steps, 104)
    goal[..., 69 + task] = 1.0
    return goal


def _packet(goal_context: torch.Tensor, valid_mask: torch.Tensor | None = None) -> BrainPacket:
    batch, steps, _ = goal_context.shape
    return BrainPacket(
        representation=torch.zeros(batch, steps, 8),
        valid_mask=torch.ones(batch, steps, dtype=torch.bool) if valid_mask is None else valid_mask,
        modality="fixture",
        step_index=torch.arange(steps).repeat(batch, 1),
        source_module=SENSORY_ENCODER,
        goal_context=goal_context,
    )


def _context(packet: BrainPacket) -> ModuleContext:
    return ModuleContext(
        task_id="associative_recall.v1",
        phase="train",
        reset_mask=torch.zeros_like(packet.valid_mask),
        eligible_modules=OPTIONAL_EXPERT_IDS,
    )


def test_sparse_router_v2_reserves_associative_recall_store_and_query_for_episodic() -> None:
    module = SparseRouterV2(feature_dim=8, task_embedding_dim=4)
    torch.nn.init.zeros_(module.scorer.weight)
    module.scorer.bias.data.copy_(torch.tensor([-1.0, 1.0]))

    goal = _goal(1, 3)
    goal[0, 0, 0] = 1.0
    goal[0, 1, 2] = 1.0
    packet = _packet(goal)

    learned = module.route(packet, mode="learned")
    no_reservation = module.route(packet, mode="no_reservation")

    assert learned.reserved_mask[0, 0].tolist() == [True, False]
    assert learned.reserved_mask[0, 1].tolist() == [True, False]
    assert learned.executed_mask[0, 0].tolist() == [True, False]
    assert learned.executed_mask[0, 1].tolist() == [True, False]
    assert learned.executed_mask[0, 2].tolist() == [False, True]
    assert no_reservation.executed_mask[0, 0].tolist() == [False, True]
    assert no_reservation.executed_mask[0, 1].tolist() == [False, True]
    assert no_reservation.executed_mask[0, 2].tolist() == [False, True]
    assert learned.capacity_drops == 0
    assert learned.capacity.sum().item() == 2 * int(packet.valid_mask.sum().item())


def test_sparse_router_v2_dense_and_stable_ties_cover_all_tokens_and_combine() -> None:
    module = SparseRouterV2(feature_dim=8, task_embedding_dim=4)
    torch.nn.init.zeros_(module.scorer.weight)
    torch.nn.init.zeros_(module.scorer.bias)
    goal = _goal(2, 2, task=1)
    packet = _packet(goal, valid_mask=torch.tensor([[True, False], [True, True]]))

    learned = module.route(packet, mode="learned")
    dense = module.route(packet, mode="dense")

    assert learned.learned_mask[0, 0].tolist() == [True, False]
    assert learned.executed_mask[1, 1].tolist() == [True, False]
    assert dense.executed_mask.sum(dim=-1).tolist() == [[2, 0], [2, 2]]
    assert dense.capacity.tolist() == [[2, 2], [1, 1]]

    state = module.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    episodic = BrainPacket(
        packet.representation + 1.0,
        packet.valid_mask,
        packet.modality,
        packet.step_index,
        EPISODIC_MEMORY,
        packet.goal_context,
    )
    working = BrainPacket(
        packet.representation + 2.0,
        packet.valid_mask,
        packet.modality,
        packet.step_index,
        WORKING_MEMORY,
        packet.goal_context,
    )
    combined = module.combine(
        packet,
        {EPISODIC_MEMORY: episodic, WORKING_MEMORY: working},
        dense,
        state,
        _context(packet),
    )
    assert combined.packet.source_module == module.module_id
    assert combined.packet.representation.shape == packet.representation.shape


def test_sparse_router_v2_legacy_capacity_reroutes_without_drops() -> None:
    module = SparseRouterV2(feature_dim=8, task_embedding_dim=4)
    torch.nn.init.zeros_(module.scorer.weight)
    module.scorer.bias.data.copy_(torch.tensor([2.0, -2.0]))
    packet = _packet(_goal(8, 1, task=1))

    decision = module.route(packet, mode="legacy_capacity")

    assert decision.capacity.tolist() == [[5, 5]]
    assert decision.raw_mask[..., 0].sum().item() == 8
    assert decision.executed_mask[..., 0].sum().item() == 5
    assert decision.executed_mask[..., 1].sum().item() == 3
    assert decision.executed_mask.sum().item() == 8
    assert decision.capacity_drops == 0
