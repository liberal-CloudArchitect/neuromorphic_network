from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from neuromorphic.core.contracts import ModuleState
from neuromorphic.core.registry import MODULE_IDS
from neuromorphic.training.checkpoint import CheckpointCompatibilityError
from neuromorphic.training.modular_checkpoint import (
    TASK_IDS,
    canonical_modular_config_hash,
    load_modular_checkpoint,
    save_modular_checkpoint,
)
from neuromorphic.training.reproducibility import set_global_seed


def _states(batch_size: int = 2) -> dict[str, ModuleState]:
    return {
        module_id: ModuleState(
            module_id,
            "state-v1",
            {"memory": torch.arange(batch_size * 3, dtype=torch.float32).reshape(batch_size, 3)},
        )
        for module_id in MODULE_IDS
    }


def _samplers() -> dict[str, dict[str, Any]]:
    return {task_id: {"cursor": index, "epoch": 0} for index, task_id in enumerate(TASK_IDS)}


def _step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()


def _save_fixture(path: Path) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict[str, object]]:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _step(model, optimizer)
    config: dict[str, object] = {
        "structure": {"feature_size": 128, "experts": 3},
        "curriculum": {"stages": ["sensory", "episodic", "working", "predictive"]},
        "loss_weights": {"primary": 1.0, "router": 0.01},
        "run_id": "first",
        "output_root": "artifacts/one",
        "resume": None,
        "telemetry_enabled": False,
    }
    save_modular_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        module_states=_states(),
        curriculum_stage="joint",
        stage_step=31,
        sampler_states=_samplers(),
        tbptt_counters=torch.tensor([31, 7]),
        frozen_module_ids=(MODULE_IDS[0],),
        config=config,
    )
    return model, optimizer, config


def test_modular_config_hash_ignores_only_run_local_fields() -> None:
    base = {
        "structure": {"width": 128},
        "curriculum": [100, 100, 100, 100],
        "loss_weights": {"primary": 1.0},
        "run_id": "a",
        "output_root": "one",
        "resume": None,
        "telemetry_enabled": False,
    }
    local_change = {
        **base,
        "run_id": "b",
        "output_root": "two",
        "resume": "latest.pt",
        "telemetry_enabled": True,
    }
    assert canonical_modular_config_hash(base) == canonical_modular_config_hash(local_change)
    assert canonical_modular_config_hash(base) != canonical_modular_config_hash(
        {**base, "loss_weights": {"primary": 0.5}}
    )
    assert canonical_modular_config_hash(base) != canonical_modular_config_hash(
        {**base, "curriculum": [2, 2, 2, 2]}
    )


def test_modular_checkpoint_restores_full_curriculum_and_rng(tmp_path: Path) -> None:
    set_global_seed(31)
    path = tmp_path / "modular.pt"
    model, _, config = _save_fixture(path)
    expected_parameters = {name: value.clone() for name, value in model.state_dict().items()}
    expected_random = torch.rand(4)

    resumed = torch.nn.Linear(3, 2)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    restored = load_modular_checkpoint(
        path,
        model=resumed,
        optimizer=resumed_optimizer,
        expected_module_states=_states(),
        config={
            **config,
            "run_id": "resume-run",
            "output_root": "artifacts/resumed",
            "resume": path,
            "telemetry_enabled": True,
        },
    )

    assert restored.curriculum_stage == "joint"
    assert restored.stage_step == 31
    assert set(restored.sampler_states) == set(TASK_IDS)
    assert torch.equal(restored.tbptt_counters, torch.tensor([31, 7]))
    assert restored.frozen_module_ids == (MODULE_IDS[0],)
    assert len(restored.optimizer_groups) == 1
    assert all(
        torch.equal(resumed.state_dict()[name], expected)
        for name, expected in expected_parameters.items()
    )
    assert torch.equal(torch.rand(4), expected_random)
    assert restored.module_states[MODULE_IDS[1]].version == "state-v1"


def test_modular_checkpoint_prevalidation_does_not_mutate_model(tmp_path: Path) -> None:
    path = tmp_path / "modular.pt"
    _, _, config = _save_fixture(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["module_states"][MODULE_IDS[2]]["version"] = "state-v999"
    torch.save(payload, path)

    resumed = torch.nn.Linear(3, 2)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    before = {name: value.clone() for name, value in resumed.state_dict().items()}
    with pytest.raises(CheckpointCompatibilityError, match="version"):
        load_modular_checkpoint(
            path,
            model=resumed,
            optimizer=resumed_optimizer,
            expected_module_states=_states(),
            config=config,
        )
    assert all(torch.equal(resumed.state_dict()[name], value) for name, value in before.items())
    assert resumed_optimizer.state == {}


def test_modular_checkpoint_rejects_config_and_state_contract_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "modular.pt"
    _, _, config = _save_fixture(path)
    resumed = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    with pytest.raises(CheckpointCompatibilityError, match="configuration hash"):
        load_modular_checkpoint(
            path,
            model=resumed,
            optimizer=optimizer,
            expected_module_states=_states(),
            config={**config, "structure": {"feature_size": 64, "experts": 3}},
        )

    expected = _states()
    state = expected[MODULE_IDS[0]]
    expected[MODULE_IDS[0]] = ModuleState(
        state.module_id, state.version, {"memory": torch.zeros(2, 4)}
    )
    with pytest.raises(CheckpointCompatibilityError, match="tensor is incompatible"):
        load_modular_checkpoint(
            path,
            model=resumed,
            optimizer=optimizer,
            expected_module_states=expected,
            config=config,
        )


def test_modular_checkpoint_requires_three_sampler_states_and_six_modules(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="three P1 tasks"):
        save_modular_checkpoint(
            tmp_path / "invalid.pt",
            model=model,
            optimizer=optimizer,
            module_states=_states(),
            curriculum_stage="sensory",
            stage_step=0,
            sampler_states={TASK_IDS[0]: {}},
            tbptt_counters=torch.zeros(2, dtype=torch.int64),
            frozen_module_ids=(),
            config={},
        )
    incomplete = _states()
    incomplete.pop(MODULE_IDS[-1])
    with pytest.raises(ValueError, match="six registered"):
        save_modular_checkpoint(
            tmp_path / "invalid.pt",
            model=model,
            optimizer=optimizer,
            module_states=incomplete,
            curriculum_stage="sensory",
            stage_step=0,
            sampler_states=_samplers(),
            tbptt_counters=torch.zeros(2, dtype=torch.int64),
            frozen_module_ids=(),
            config={},
        )
