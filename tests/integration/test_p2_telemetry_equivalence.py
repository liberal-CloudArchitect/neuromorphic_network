"""Numerical equivalence checks for the detached P2 telemetry path."""

from __future__ import annotations

import copy

import pytest
import torch

from neuromorphic.modules.network import ModularBrainNetwork, ModularBrainOutput
from neuromorphic.tasks.delayed_rule_switch import DelayedRuleSwitchTask
from neuromorphic.training.modular_trainer import train_one_update
from neuromorphic.training.p2_config import P2LossWeights


def _assert_outputs_equal(first: ModularBrainOutput, second: ModularBrainOutput) -> None:
    assert torch.equal(first.action_logits, second.action_logits)
    assert first.auxiliary_losses.keys() == second.auxiliary_losses.keys()
    for name in first.auxiliary_losses:
        assert torch.equal(first.auxiliary_losses[name], second.auxiliary_losses[name])
    for module_id in first.state.module_states:
        for name, tensor in first.state.get(module_id).tensors.items():
            assert torch.equal(tensor, second.state.get(module_id).tensors[name])


def test_fixed_input_telemetry_flag_preserves_packets_losses_and_state() -> None:
    torch.manual_seed(7)
    batch = DelayedRuleSwitchTask().generate("train", [0, 1])
    model = ModularBrainNetwork()
    off = model.forward_batch(batch, telemetry_enabled=False)
    on = model.forward_batch(batch, telemetry_enabled=True)
    _assert_outputs_equal(off, on)


def test_training_step_telemetry_flag_preserves_gradients_optimizer_and_parameters() -> None:
    torch.manual_seed(7)
    batch = DelayedRuleSwitchTask().generate("train", [0, 1])
    off_model = ModularBrainNetwork(feature_dim=32, working_slot_dim=8)
    on_model = copy.deepcopy(off_model)
    off_optimizer = torch.optim.AdamW(off_model.parameters(), lr=3e-4)
    on_optimizer = torch.optim.AdamW(on_model.parameters(), lr=3e-4)
    weights = P2LossWeights().by_loss_name()
    off_output, off_loss, _ = train_one_update(
        model=off_model,
        optimizer=off_optimizer,
        batch=batch,
        weights=weights,
        include_primary=True,
        gradient_clip_norm=1.0,
        forced_experts=None,
        telemetry_enabled=False,
    )
    on_output, on_loss, _ = train_one_update(
        model=on_model,
        optimizer=on_optimizer,
        batch=batch,
        weights=weights,
        include_primary=True,
        gradient_clip_norm=1.0,
        forced_experts=None,
        telemetry_enabled=True,
    )
    assert torch.equal(off_loss, on_loss)
    _assert_outputs_equal(off_output, on_output)
    for (off_name, off_parameter), (on_name, on_parameter) in zip(
        off_model.named_parameters(), on_model.named_parameters(), strict=True
    ):
        assert off_name == on_name
        assert torch.equal(off_parameter, on_parameter)
        assert (off_parameter.grad is None) == (on_parameter.grad is None)
        if off_parameter.grad is not None:
            on_gradient = on_parameter.grad
            assert on_gradient is not None
            assert torch.equal(off_parameter.grad, on_gradient)
    assert off_optimizer.state_dict().keys() == on_optimizer.state_dict().keys()


@pytest.mark.mps
def test_fixed_input_telemetry_equivalence_on_mps() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    device = torch.device("mps")
    batch = DelayedRuleSwitchTask().generate("train", [0, 1], device=device)
    model = ModularBrainNetwork(feature_dim=32, working_slot_dim=8).to(device)
    off = model.forward_batch(batch, telemetry_enabled=False)
    on = model.forward_batch(batch, telemetry_enabled=True)
    assert torch.allclose(off.action_logits, on.action_logits, rtol=1e-5, atol=1e-6)
