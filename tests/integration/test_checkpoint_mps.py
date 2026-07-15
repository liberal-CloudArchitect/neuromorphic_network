from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neuromorphic.training.checkpoint import load_checkpoint, save_checkpoint
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
