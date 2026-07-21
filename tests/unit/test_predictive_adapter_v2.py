from __future__ import annotations

import torch

from neuromorphic.core.contracts import BrainPacket, ModuleContext, ModuleState
from neuromorphic.core.registry import ACTION_SELECTOR, OPTIONAL_EXPERT_IDS
from neuromorphic.modules.predictive_adapter_v2 import PredictiveAdapterV2


def _goal(batch: int, steps: int, task: int = 0) -> torch.Tensor:
    goal = torch.zeros(batch, steps, 104)
    goal[..., 69 + task] = 1.0
    return goal


def _packet(
    representation: torch.Tensor,
    *,
    step_index: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    goal_context: torch.Tensor | None = None,
) -> BrainPacket:
    batch, steps, _ = representation.shape
    return BrainPacket(
        representation=representation,
        valid_mask=torch.ones(batch, steps, dtype=torch.bool) if valid_mask is None else valid_mask,
        modality="fixture",
        step_index=torch.arange(steps).repeat(batch, 1) if step_index is None else step_index,
        source_module=ACTION_SELECTOR,
        goal_context=_goal(batch, steps) if goal_context is None else goal_context,
    )


def _context(packet: BrainPacket, reset_mask: torch.Tensor | None = None) -> ModuleContext:
    return ModuleContext(
        task_id="associative_recall.v1",
        phase="train",
        reset_mask=torch.zeros_like(packet.valid_mask) if reset_mask is None else reset_mask,
        eligible_modules=OPTIONAL_EXPERT_IDS,
    )


def test_predictive_adapter_v2_consumes_only_consecutive_forecasts_with_bounded_feedback() -> None:
    module = PredictiveAdapterV2(feature_dim=4, action_count=4, action_dim=3)
    first, second = module.feedback[0], module.feedback[2]
    assert isinstance(first, torch.nn.Linear)
    assert isinstance(second, torch.nn.Linear)
    torch.nn.init.zeros_(first.weight)
    torch.nn.init.zeros_(first.bias)
    torch.nn.init.zeros_(second.weight)
    torch.nn.init.constant_(second.bias, 100.0)

    packet = _packet(
        torch.zeros(3, 1, 4),
        step_index=torch.tensor([[4], [5], [4]], dtype=torch.long),
    )
    state = ModuleState(
        module.module_id,
        module.state_version,
        {
            "forecast": torch.ones(3, 4),
            "forecast_valid": torch.tensor([True, True, True]),
            "source_step": torch.tensor([3, 3, 3], dtype=torch.long),
            "persistence": torch.zeros(3, 4),
        },
    )
    output = module.forward_with_action(
        packet,
        state,
        _context(packet, torch.tensor([[False], [False], [True]])),
        torch.zeros(3, 1, dtype=torch.long),
    )

    assert torch.allclose(output.packet.representation[0, 0], torch.full((4,), 0.25), atol=1.0e-4)
    assert torch.count_nonzero(output.packet.representation[1, 0]) == 0
    assert torch.count_nonzero(output.packet.representation[2, 0]) == 0
    assert output.auxiliary_losses["predictive_transition"].item() > 0.0


def test_predictive_adapter_v2_commits_next_forecast_and_ignores_padding() -> None:
    module = PredictiveAdapterV2(feature_dim=4, action_count=4, action_dim=2)
    packet_goal = _goal(2, 2)
    packet_goal[0, 0, 72 + 2] = 1.0
    packet_goal[0, 1, 72 + 1] = 1.0
    packet = _packet(
        torch.randn(2, 2, 4),
        valid_mask=torch.tensor([[True, True], [True, False]]),
        step_index=torch.tensor([[0, 1], [3, 4]], dtype=torch.long),
        goal_context=packet_goal,
    )
    state = module.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)

    output = module(packet, state, _context(packet))

    assert output.prediction_logits is not None
    assert output.prediction_logits.shape == (2, 2, 4)
    assert output.state.tensors["forecast"].shape == (2, 4)
    assert output.state.tensors["forecast_valid"].tolist() == [True, True]
    assert output.state.tensors["source_step"].tolist() == [1, 3]

    reset = module.reset_state(output.state, torch.tensor([True, False]))
    assert not reset.tensors["forecast_valid"][0].item()
    assert reset.tensors["source_step"][0].item() == -1
    assert torch.count_nonzero(reset.tensors["forecast"][0]) == 0
