import copy
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from neuromorphic.core.contracts import ModuleState
from neuromorphic.core.network_state import NetworkState
from neuromorphic.core.registry import PREDICTIVE_ADAPTER_V2
from neuromorphic.training.checkpoint import CheckpointCompatibilityError
from neuromorphic.training.p4_checkpoint import (
    P4CheckpointState,
    load_p4_checkpoint,
    save_p4_checkpoint,
)
from neuromorphic.training.trainer import IndexSampler


def _network_state(*, count: int, forecast_value: float = 2.0) -> NetworkState:
    module_state = ModuleState(
        PREDICTIVE_ADAPTER_V2,
        "predictive-state-v2",
        {
            "forecast": torch.full((2, 3), forecast_value),
            "forecast_valid": torch.tensor([True, False]),
            "source_step": torch.tensor([30, -1], dtype=torch.long),
            "persistence": torch.ones(2, 3),
        },
    )
    return NetworkState(
        {PREDICTIVE_ADAPTER_V2: module_state},
        torch.tensor([count, 0], dtype=torch.long),
    )


def _state(*, network_state: NetworkState | None = None) -> P4CheckpointState:
    sampler = IndexSampler.create(8, 117)
    sampler.next(3)
    return P4CheckpointState(
        cell_id="cell",
        global_step=31,
        task_steps={"task": 31},
        sampler_states={"task": sampler.state_dict()},
        best_metrics={"macro": 0.5},
        stale_evaluations=0,
        matrix_cursor=2,
        config_hash="config",
        protocol_hash="protocol",
        matrix_hash="matrix",
        pilot_lock_hash="pilot",
        mechanism_lock_hash=None,
        analysis_curves={"task": ((1, 0.2), (31, 0.5))},
        last_loss=0.4,
        cumulative_wall_clock_seconds=4.0,
        transition_count=30,
        prediction_totals={"eligible": 31.0, "covered": 30.0},
        validation_prediction_totals={"eligible": 8.0, "covered": 8.0},
        network_state=network_state,
    )


def _load(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_network_state: NetworkState | None = None,
) -> P4CheckpointState:
    return load_p4_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expected_cell_id="cell",
        expected_config_hash="config",
        expected_protocol_hash="protocol",
        expected_matrix_hash="matrix",
        expected_matrix_cursor=2,
        expected_pilot_lock_hash="pilot",
        expected_mechanism_lock_hash=None,
        expected_sampler_signatures={"task": (8, 117)},
        expected_network_state=expected_network_state,
    )


def _train_once(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor([[0.2, -0.3, 0.7]])
    target = torch.tensor([[0.5, -0.5]])
    loss = torch.nn.functional.mse_loss(model(inputs), target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    return inputs, target


def test_p4_checkpoint_cpu_round_trip_and_hash_validation(tmp_path: Path) -> None:
    torch.manual_seed(9)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    inputs, _ = _train_once(model, optimizer)
    expected_output = model(inputs).detach().clone()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    path = tmp_path / "checkpoint.pt"
    save_p4_checkpoint(path, model=model, optimizer=optimizer, state=_state())

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    optimizer.state.clear()
    restored = _load(path, model, optimizer)

    assert restored.global_step == 31
    assert restored.transition_count == 30
    assert restored.prediction_totals == {"eligible": 31.0, "covered": 30.0}
    assert restored.validation_prediction_totals == {"eligible": 8.0, "covered": 8.0}
    torch.testing.assert_close(model(inputs), expected_output, rtol=0.0, atol=0.0)
    assert optimizer.state_dict()["param_groups"] == saved_optimizer["param_groups"]
    for saved, actual in zip(
        saved_optimizer["state"].values(), optimizer.state_dict()["state"].values(), strict=True
    ):
        for name in saved:
            torch.testing.assert_close(actual[name], saved[name], rtol=0.0, atol=0.0)

    with pytest.raises(CheckpointCompatibilityError, match="protocol_hash"):
        load_p4_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_cell_id="cell",
            expected_config_hash="config",
            expected_protocol_hash="wrong",
            expected_matrix_hash="matrix",
            expected_matrix_cursor=2,
            expected_pilot_lock_hash="pilot",
            expected_mechanism_lock_hash=None,
        )


def test_p4_checkpoint_restores_pending_forecast_and_31_to_32_boundary(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    pending = _network_state(count=31)
    path = tmp_path / "pending.pt"
    save_p4_checkpoint(path, model=model, optimizer=optimizer, state=_state(network_state=pending))
    expected = _network_state(count=0, forecast_value=0.0)
    restored = _load(path, model, optimizer, expected_network_state=expected)

    assert restored.network_state is not None
    predictive = restored.network_state.get(PREDICTIVE_ADAPTER_V2)
    assert predictive.tensors["forecast_valid"].tolist() == [True, False]
    assert predictive.tensors["source_step"].tolist() == [30, -1]
    torch.testing.assert_close(predictive.tensors["forecast"], torch.full((2, 3), 2.0))
    advanced, detached = restored.network_state.advance(torch.tensor([True, False]))
    assert advanced.valid_step_counts.tolist() == [32, 0]
    assert detached.tolist() == [True, False]
    # Detaching the recurrent graph does not discard the pending forecast.
    torch.testing.assert_close(
        advanced.get(PREDICTIVE_ADAPTER_V2).tensors["forecast"], torch.full((2, 3), 2.0)
    )


def test_corrupt_checkpoint_does_not_mutate_live_training_state(tmp_path: Path) -> None:
    torch.manual_seed(41)
    np.random.seed(41)
    random.seed(41)
    source = torch.nn.Linear(3, 2)
    source_optimizer = torch.optim.AdamW(source.parameters())
    _train_once(source, source_optimizer)
    path = tmp_path / "corrupt.pt"
    save_p4_checkpoint(path, model=source, optimizer=source_optimizer, state=_state())
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["model_state"]["weight"] = torch.zeros(1, 1)
    torch.save(payload, path)

    live = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(live.parameters())
    _train_once(live, optimizer)
    before_model = {name: tensor.clone() for name, tensor in live.state_dict().items()}
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_python = random.getstate()
    before_numpy: Any = np.random.get_state()
    before_torch = torch.get_rng_state().clone()

    with pytest.raises(CheckpointCompatibilityError, match="tensor is incompatible"):
        _load(path, live, optimizer)

    for name, tensor in live.state_dict().items():
        torch.testing.assert_close(tensor, before_model[name], rtol=0.0, atol=0.0)
    assert optimizer.state_dict()["param_groups"] == before_optimizer["param_groups"]
    for saved, actual in zip(
        before_optimizer["state"].values(),
        optimizer.state_dict()["state"].values(),
        strict=True,
    ):
        for name in saved:
            torch.testing.assert_close(actual[name], saved[name], rtol=0.0, atol=0.0)
    assert random.getstate() == before_python
    current_numpy = cast(tuple[object, np.ndarray, object, object, object], np.random.get_state())
    assert np.array_equal(current_numpy[1], before_numpy[1])
    torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("matrix_cursor", 3, "matrix_cursor"),
        ("pilot_lock_hash", "other", "pilot_lock_hash"),
        ("mechanism_lock_hash", "unexpected", "mechanism_lock_hash"),
    ],
)
def test_p4_checkpoint_rejects_cursor_and_lock_mismatch_without_mutation(
    tmp_path: Path, field: str, bad_value: object, message: str
) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "checkpoint.pt"
    save_p4_checkpoint(path, model=model, optimizer=optimizer, state=_state())
    arguments: dict[str, Any] = {
        "expected_cell_id": "cell",
        "expected_config_hash": "config",
        "expected_protocol_hash": "protocol",
        "expected_matrix_hash": "matrix",
        "expected_matrix_cursor": 2,
        "expected_pilot_lock_hash": "pilot",
        "expected_mechanism_lock_hash": None,
    }
    arguments[f"expected_{field}"] = bad_value
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with pytest.raises(CheckpointCompatibilityError, match=message):
        load_p4_checkpoint(path, model=model, optimizer=optimizer, **arguments)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0.0, atol=0.0)
