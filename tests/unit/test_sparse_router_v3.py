from __future__ import annotations

import math

import pytest
import torch

from neuromorphic.core.contracts import BrainPacket, ModuleContext
from neuromorphic.core.registry import (
    EPISODIC_MEMORY,
    OPTIONAL_EXPERT_IDS,
    SENSORY_ENCODER,
    WORKING_MEMORY,
)
from neuromorphic.modules.sparse_router_v3 import SparseRouterV3


def _packet(task: int, batch: int = 8, steps: int = 1) -> BrainPacket:
    goal = torch.zeros(batch, steps, 104)
    goal[..., 69 + task] = 1.0
    goal[..., 4] = 1.0
    return BrainPacket(
        representation=torch.randn(batch, steps, 8),
        valid_mask=torch.ones(batch, steps, dtype=torch.bool),
        modality="fixture",
        step_index=torch.arange(steps).repeat(batch, 1),
        source_module=SENSORY_ENCODER,
        goal_context=goal,
    )


def _context(packet: BrainPacket, task_id: str) -> ModuleContext:
    return ModuleContext(
        task_id=task_id,
        phase="train",
        reset_mask=torch.zeros_like(packet.valid_mask),
        eligible_modules=OPTIONAL_EXPERT_IDS,
    )


def test_v3_drs_reserves_working_and_bounds_dual_routes() -> None:
    module = SparseRouterV3(feature_dim=8, task_embedding_dim=4, dual_route_fraction=0.25)
    packet = _packet(task=1, batch=8, steps=2)
    decision = module.route(packet, surprise=torch.zeros(8, 2, 1))

    assert decision.reserved_mask[..., 1].all()
    assert decision.executed_mask[..., 1].all()
    for step in range(2):
        dual = decision.executed_mask[:, step].sum(dim=-1).gt(1).sum().item()
        assert dual == math.ceil(8 * 0.25)
    assert decision.executed_mask.sum().item() == 20
    assert decision.capacity_drops == 0


def test_v3_ar_store_and_query_remain_episodic_only() -> None:
    module = SparseRouterV3(feature_dim=8, task_embedding_dim=4)
    packet = _packet(task=0, batch=2, steps=2)
    assert packet.goal_context is not None
    packet.goal_context[..., 0] = 0.0
    packet.goal_context[:, 0, 0] = 1.0
    packet.goal_context[:, 1, 2] = 1.0

    decision = module.route(packet)

    assert decision.reserved_mask[..., 0].all()
    assert decision.executed_mask[..., 0].all()
    assert not decision.executed_mask[..., 1].any()


def test_v3_semantic_loss_and_straight_through_fusion_train_scorer() -> None:
    torch.manual_seed(7)
    module = SparseRouterV3(feature_dim=8, task_embedding_dim=4)
    packet = _packet(task=1, batch=8)
    decision = module.route(packet, surprise=torch.zeros(8, 1, 1))
    state = module.initial_state(8, device=torch.device("cpu"), dtype=torch.float32)
    episodic = BrainPacket(
        packet.representation + 1.0,
        packet.valid_mask,
        packet.modality,
        packet.step_index,
        EPISODIC_MEMORY,
        packet.goal_context,
    )
    working = BrainPacket(
        packet.representation - 1.0,
        packet.valid_mask,
        packet.modality,
        packet.step_index,
        WORKING_MEMORY,
        packet.goal_context,
    )
    output = module.combine(
        packet,
        {EPISODIC_MEMORY: episodic, WORKING_MEMORY: working},
        decision,
        state,
        _context(packet, "delayed_rule_switch.v1"),
    )
    losses = module.routing_losses(packet, decision)
    loss = output.packet.representation.square().mean() + losses["router.semantic_alignment"]
    loss.backward()  # type: ignore[no-untyped-call]

    assert module.scorer.weight.grad is not None
    assert torch.isfinite(module.scorer.weight.grad).all()
    assert module.scorer.weight.grad.abs().sum().item() > 0.0


def test_v3_dual_route_selection_avoids_host_scalar_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = SparseRouterV3(feature_dim=8, task_embedding_dim=4, dual_route_fraction=0.25)
    packet = _packet(task=1, batch=7, steps=3)
    packet.valid_mask[:, 1] = torch.tensor([True, True, True, True, True, False, False])
    packet.valid_mask[:, 2] = torch.tensor([True, True, True, False, False, False, False])

    def forbidden_item(tensor: torch.Tensor) -> object:
        del tensor
        raise AssertionError("routing must not synchronize a device scalar through Tensor.item()")

    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)

    decision = module.route(packet, surprise=torch.zeros(7, 3, 1))

    dual_per_step = decision.executed_mask.sum(dim=-1).gt(1).sum(dim=0)
    torch.testing.assert_close(dual_per_step, torch.tensor([1, 1, 0]))
