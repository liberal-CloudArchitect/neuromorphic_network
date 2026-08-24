from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neuromorphic.modules.network_v3 import ModularBrainNetworkV3
from neuromorphic.training.p5_checkpoint import (
    P5CheckpointState,
    load_p5_checkpoint,
    save_p5_checkpoint,
)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_checkpoint_v5_restores_mps_model_and_network_state(tmp_path: Path) -> None:
    device = torch.device("mps")
    model = ModularBrainNetworkV3().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    network_state = model.initial_state(2, device=device, dtype=torch.float32)
    state = P5CheckpointState(
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
        stale_evaluations=0,
        last_loss=1.0,
        network_state=network_state,
    )
    path = tmp_path / "checkpoint.pt"
    save_p5_checkpoint(path, model=model, optimizer=optimizer, state=state)

    restored = load_p5_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expected_profile="pilot",
        expected_candidate_id="preset-0",
        expected_candidate_index=0,
        expected_config_hash="config",
        expected_protocol_hash="protocol",
        expected_network_state=network_state,
    )

    assert restored.network_state is not None
    assert restored.network_state.valid_step_counts.device.type == "mps"
    assert restored.network_state.get("predictive_adapter.v3").version == "predictive-state-v3"
