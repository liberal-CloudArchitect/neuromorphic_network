from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neuromorphic.modules.network_v3 import ModularBrainNetworkV3
from neuromorphic.training.checkpoint import CheckpointCompatibilityError
from neuromorphic.training.p5_checkpoint import (
    P5CheckpointState,
    load_p5_checkpoint,
    save_p5_checkpoint,
)


def _state(model: ModularBrainNetworkV3) -> P5CheckpointState:
    return P5CheckpointState(
        profile="pilot",
        candidate_id="preset-0",
        candidate_index=0,
        global_step=100,
        task_steps={
            "associative_recall.v1": 34,
            "delayed_rule_switch.v1": 33,
            "small_graph.v1": 33,
        },
        config_hash="config",
        protocol_hash="protocol",
        best_metrics={"macro": 0.5},
        validation_macro=(0.4, 0.5),
        router_gradient_seen=True,
        predictor_gradient_seen=True,
        stale_evaluations=2,
        last_loss=1.25,
        network_state=model.initial_state(2, device=torch.device("cpu"), dtype=torch.float32),
    )


def test_checkpoint_v5_roundtrip_restores_model_optimizer_state_and_rng(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = ModularBrainNetworkV3()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    first = next(model.parameters())
    first.grad = torch.ones_like(first)
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    state = _state(model)
    saved_parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}
    save_p5_checkpoint(path, model=model, optimizer=optimizer, state=state)
    expected_random = torch.rand(4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    torch.manual_seed(999)

    restored = load_p5_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expected_profile="pilot",
        expected_candidate_id="preset-0",
        expected_candidate_index=0,
        expected_config_hash="config",
        expected_protocol_hash="protocol",
        expected_network_state=state.network_state,
    )

    assert restored.profile == state.profile
    assert restored.candidate_id == state.candidate_id
    assert restored.global_step == state.global_step
    assert restored.task_steps == state.task_steps
    assert restored.network_state is not None
    assert state.network_state is not None
    torch.testing.assert_close(
        restored.network_state.valid_step_counts,
        state.network_state.valid_step_counts,
        rtol=0.0,
        atol=0.0,
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved_parameters[name], rtol=0.0, atol=0.0)
    torch.testing.assert_close(torch.rand(4), expected_random, rtol=0.0, atol=0.0)


def test_checkpoint_v5_rejects_hash_before_mutating_model(tmp_path: Path) -> None:
    model = ModularBrainNetworkV3()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    state = _state(model)
    path = tmp_path / "checkpoint.pt"
    save_p5_checkpoint(path, model=model, optimizer=optimizer, state=state)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match="config_hash"):
        load_p5_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_profile="pilot",
            expected_candidate_id="preset-0",
            expected_candidate_index=0,
            expected_config_hash="wrong",
            expected_protocol_hash="protocol",
            expected_network_state=state.network_state,
        )

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)
