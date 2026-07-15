from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neuromorphic.training.checkpoint import (
    CheckpointCompatibilityError,
    load_checkpoint,
    save_checkpoint,
)
from neuromorphic.training.reproducibility import set_global_seed


def _step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, inputs: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()


def test_checkpoint_resume_is_bitwise_equal_on_cpu(tmp_path: Path) -> None:
    set_global_seed(17)
    inputs = torch.randn(4, 3)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = {"model": "linear", "seed": 17}
    _step(model, optimizer, inputs)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        training_state={"step": 1},
        sampler_state={"cursor": 4},
        config=config,
    )
    expected_random = torch.rand(5)
    _step(model, optimizer, inputs)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    resumed = torch.nn.Linear(3, 2)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    training, sampler = load_checkpoint(
        checkpoint,
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=None,
        config=config,
    )
    assert training == {"step": 1}
    assert sampler == {"cursor": 4}
    assert torch.equal(torch.rand(5), expected_random)
    _step(resumed, resumed_optimizer, inputs)
    assert all(torch.equal(resumed.state_dict()[name], value) for name, value in expected.items())


def test_checkpoint_rejects_config_mismatch(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        training_state={},
        sampler_state={},
        config={"seed": 1},
    )
    with pytest.raises(CheckpointCompatibilityError, match="configuration hash"):
        load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            config={"seed": 2},
        )
