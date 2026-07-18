from pathlib import Path

import pytest
import torch

from neuromorphic.inference.bundle import create_network_mvp_bundle, load_network_mvp
from neuromorphic.modules.network import ModularBrainNetwork
from neuromorphic.tasks import create_task
from neuromorphic.training.checkpoint import CheckpointCompatibilityError
from neuromorphic.training.p3_checkpoint import (
    P3CheckpointState,
    load_p3_checkpoint,
    save_p3_checkpoint,
)
from neuromorphic.training.trainer import IndexSampler


def test_checkpoint_v3_round_trip_before_mutation(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    inputs = torch.ones(2, 3)
    model(inputs).sum().backward()
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = tmp_path / "state.pt"
    sampler = IndexSampler.create(8, 7)
    sampler.next(2)
    state = P3CheckpointState(
        cell_id="cell",
        global_step=2,
        task_steps={"task": 2},
        sampler_states={"task": sampler.state_dict()},
        best_metrics={"macro": 0.5},
        stale_evaluations=0,
        matrix_cursor=1,
        config_hash="config",
        protocol_hash="protocol",
        analysis_curves={"task": ((1, 0.25), (2, 0.5))},
        validation_curve=((1, 0.2), (2, 0.4)),
        last_loss=0.75,
    )
    save_p3_checkpoint(path, model=model, optimizer=optimizer, state=state)
    with torch.no_grad():
        model.weight.zero_()
    restored = load_p3_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expected_cell_id="cell",
        expected_config_hash="config",
        expected_protocol_hash="protocol",
    )
    assert restored.global_step == 2
    assert restored.analysis_curves["task"][-1] == (2, 0.5)
    assert restored.validation_curve[-1] == (2, 0.4)
    assert restored.last_loss == 0.75
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected.items())


def test_network_mvp_qualification_fixture_round_trip(tmp_path: Path) -> None:
    model = ModularBrainNetwork(feature_dim=32, working_slots=4, working_slot_dim=8)
    bundle = create_network_mvp_bundle(
        tmp_path / "bundle",
        model=model,
        source_commit="fixture",
        gate_status="QUALIFICATION_ONLY",
        qualification_only=True,
        model_config={"feature_dim": 32, "working_slots": 4, "working_slot_dim": 8},
    )
    loaded = load_network_mvp(bundle, torch.device("cpu"))
    batch = create_task("delayed_rule_switch.v1", profile="smoke").generate("test", [0])
    output = loaded.predict_batch(batch)
    assert output.logits.shape[:2] == batch.targets.shape
    assert loaded.manifest["qualification_only"] is True


def test_checkpoint_v3_rejects_sampler_before_model_mutation(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    sampler = IndexSampler.create(8, 7)
    state = P3CheckpointState(
        cell_id="cell",
        global_step=0,
        task_steps={"task": 0},
        sampler_states={"task": sampler.state_dict()},
        best_metrics={},
        stale_evaluations=0,
        matrix_cursor=1,
        config_hash="config",
        protocol_hash="protocol",
        analysis_curves={"task": ()},
        validation_curve=(),
        last_loss=None,
    )
    path = tmp_path / "invalid.pt"
    save_p3_checkpoint(path, model=model, optimizer=optimizer, state=state)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["sampler_states"]["task"]["cursor"] = 99
    torch.save(payload, path)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    before = {name: value.clone() for name, value in model.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match="sampler counters"):
        load_p3_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_cell_id="cell",
            expected_config_hash="config",
            expected_protocol_hash="protocol",
            expected_matrix_cursor=1,
            expected_sampler_signatures={"task": (8, 7)},
        )

    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())
