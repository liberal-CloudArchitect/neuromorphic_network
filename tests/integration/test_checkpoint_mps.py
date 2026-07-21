from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neuromorphic.core.contracts import ModuleState
from neuromorphic.core.network_state import NetworkState
from neuromorphic.core.registry import PREDICTIVE_ADAPTER_V2
from neuromorphic.training.checkpoint import load_checkpoint, save_checkpoint
from neuromorphic.training.p4_checkpoint import (
    P4CheckpointState,
    load_p4_checkpoint,
    save_p4_checkpoint,
)
from neuromorphic.training.reproducibility import set_global_seed


@pytest.mark.mps
def test_mps_checkpoint_resume_matches_continuous_update(tmp_path: Path) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    device = torch.device("mps")
    set_global_seed(29)
    inputs = torch.randn(4, 3, device=device)
    model = torch.nn.Linear(3, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step(current_model: torch.nn.Module, current_optimizer: torch.optim.Optimizer) -> None:
        current_optimizer.zero_grad(set_to_none=True)
        loss = current_model(inputs).square().mean()
        loss.backward()
        current_optimizer.step()

    step(model, optimizer)
    path = tmp_path / "mps.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        training_state={"step": 1},
        sampler_state={},
        config={"seed": 29},
    )
    step(model, optimizer)
    expected = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}

    resumed = torch.nn.Linear(3, 2).to(device)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    load_checkpoint(
        path,
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=None,
        config={"seed": 29},
    )
    step(resumed, resumed_optimizer)
    for name, tensor in resumed.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), expected[name], rtol=1e-5, atol=1e-6)


@pytest.mark.mps
def test_p4_mps_checkpoint_restores_pending_forecast_and_next_update(tmp_path: Path) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    device = torch.device("mps")
    set_global_seed(43)
    inputs = torch.randn(4, 3, device=device)
    model = torch.nn.Linear(3, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step(current_model: torch.nn.Module, current_optimizer: torch.optim.Optimizer) -> None:
        current_optimizer.zero_grad(set_to_none=True)
        loss = current_model(inputs).square().mean()
        loss.backward()
        current_optimizer.step()

    step(model, optimizer)
    predictive = ModuleState(
        PREDICTIVE_ADAPTER_V2,
        "predictive-state-v2",
        {
            "forecast": torch.randn(4, 128, device=device),
            "forecast_valid": torch.tensor([True, True, False, True], device=device),
            "source_step": torch.tensor([30, 30, -1, 30], device=device),
            "persistence": torch.randn(4, 128, device=device),
        },
    )
    network_state = NetworkState(
        {PREDICTIVE_ADAPTER_V2: predictive},
        torch.tensor([31, 31, 0, 31], device=device),
    )
    state = P4CheckpointState(
        cell_id="mps-cell",
        global_step=31,
        task_steps={"task": 31},
        sampler_states={},
        best_metrics={"macro": 0.5},
        stale_evaluations=0,
        matrix_cursor=0,
        config_hash="config",
        protocol_hash="protocol",
        matrix_hash="matrix",
        pilot_lock_hash=None,
        mechanism_lock_hash=None,
        analysis_curves={"task": ((31, 0.5),)},
        last_loss=0.5,
        cumulative_wall_clock_seconds=1.0,
        transition_count=30,
        prediction_totals={"eligible": 31.0, "covered": 30.0},
        validation_prediction_totals={"eligible": 8.0, "covered": 8.0},
        network_state=network_state,
    )
    path = tmp_path / "p4-mps.pt"
    save_p4_checkpoint(path, model=model, optimizer=optimizer, state=state)
    step(model, optimizer)
    expected = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}

    resumed = torch.nn.Linear(3, 2).to(device)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    empty_predictive = ModuleState(
        PREDICTIVE_ADAPTER_V2,
        "predictive-state-v2",
        {
            "forecast": torch.zeros(4, 128, device=device),
            "forecast_valid": torch.zeros(4, dtype=torch.bool, device=device),
            "source_step": torch.full((4,), -1, dtype=torch.long, device=device),
            "persistence": torch.zeros(4, 128, device=device),
        },
    )
    restored = load_p4_checkpoint(
        path,
        model=resumed,
        optimizer=resumed_optimizer,
        expected_cell_id="mps-cell",
        expected_config_hash="config",
        expected_protocol_hash="protocol",
        expected_matrix_hash="matrix",
        expected_matrix_cursor=0,
        expected_pilot_lock_hash=None,
        expected_mechanism_lock_hash=None,
        expected_network_state=NetworkState(
            {PREDICTIVE_ADAPTER_V2: empty_predictive},
            torch.zeros(4, dtype=torch.long, device=device),
        ),
    )
    assert restored.network_state is not None
    restored_forecast = restored.network_state.get(PREDICTIVE_ADAPTER_V2)
    torch.testing.assert_close(
        restored_forecast.tensors["forecast"].cpu(),
        predictive.tensors["forecast"].cpu(),
        rtol=1e-5,
        atol=1e-6,
    )
    assert restored.network_state.valid_step_counts.tolist() == [31, 31, 0, 31]
    step(resumed, resumed_optimizer)
    for name, tensor in resumed.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), expected[name], rtol=1e-5, atol=1e-6)
