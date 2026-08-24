from __future__ import annotations

import torch

from neuromorphic.core.contracts import BrainPacket, ModuleContext
from neuromorphic.core.registry import ACTION_SELECTOR, OPTIONAL_EXPERT_IDS
from neuromorphic.modules.predictive_adapter_v3 import PredictiveAdapterV3


def _packet(representation: torch.Tensor, step: int = 0) -> BrainPacket:
    batch = representation.shape[0]
    goal = torch.zeros(batch, 1, 104)
    goal[..., 69] = 1.0
    return BrainPacket(
        representation=representation,
        valid_mask=torch.ones(batch, 1, dtype=torch.bool),
        modality="fixture",
        step_index=torch.full((batch, 1), step, dtype=torch.long),
        source_module=ACTION_SELECTOR,
        goal_context=goal,
    )


def _context(batch: int) -> ModuleContext:
    return ModuleContext(
        task_id="associative_recall.v1",
        phase="train",
        reset_mask=torch.zeros(batch, 1, dtype=torch.bool),
        eligible_modules=OPTIONAL_EXPERT_IDS,
    )


def test_v3_forecast_starts_at_persistence_and_does_not_mutate_sensation() -> None:
    torch.manual_seed(7)
    module = PredictiveAdapterV3(feature_dim=4, action_count=4, action_dim=2)
    representation = torch.randn(3, 1, 4)
    packet = _packet(representation)
    state = module.initial_state(3, device=torch.device("cpu"), dtype=torch.float32)
    committed = module.commit(
        packet,
        state,
        _context(3),
        torch.zeros(3, 1, dtype=torch.long),
    )
    expected = module.target_norm(representation[:, 0])
    torch.testing.assert_close(committed.tensors["forecast"], expected)

    next_packet = _packet(representation.clone(), step=1)
    consumed = module.consume(next_packet, committed, _context(3))

    torch.testing.assert_close(consumed.output.packet.representation, representation)
    assert torch.count_nonzero(consumed.feedback_delta) == 0
    assert consumed.forecast_error.max().item() < 1.0e-7
    assert consumed.persistence_error.max().item() < 1.0e-7


def test_v3_surprise_is_detached_and_transition_loss_trains_residual_head() -> None:
    module = PredictiveAdapterV3(feature_dim=4, action_count=4, action_dim=2)
    first = _packet(torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]).repeat(2, 1, 1))
    initial = module.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    committed = module.commit(
        first,
        initial,
        _context(2),
        torch.zeros(2, 1, dtype=torch.long),
    )
    second = _packet(torch.tensor([[[4.0, 1.0, 3.0, 2.0]]]).repeat(2, 1, 1), step=1)
    consumed = module.consume(second, committed, _context(2))

    assert consumed.surprise.shape == (2, 1, 1)
    assert not consumed.surprise.requires_grad
    assert consumed.surprise.min().item() > 0.0
    consumed.output.auxiliary_losses["predictive_transition"].backward()  # type: ignore[no-untyped-call]
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )
