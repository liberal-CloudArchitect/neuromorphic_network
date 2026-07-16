"""End-to-end CPU fixture for the P2 CLI and evidence bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from neuromorphic.training.run import main

ROOT = Path(__file__).resolve().parents[2]


def test_p2_cli_writes_complete_auditable_cpu_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = {
        "schema_version": "p2-suite-v1",
        "profile": "ci",
        "seed": 7,
        "device": "cpu",
        "output_root": str(tmp_path),
        "run_id": "p2-integration",
        "model": {
            "feature_dim": 16,
            "episodic_slots": 16,
            "working_slots": 4,
            "working_slot_dim": 4,
            "action_embedding_dim": 4,
            "task_embedding_dim": 4,
            "router_top_k": 2,
            "router_capacity_factor": 1.25,
        },
        "budget": {
            "pretrain_steps_per_stage": 1,
            "joint_steps_per_task": 1,
            "batch_size": 2,
            "train_size": 4,
            "validation_size": 2,
            "test_size": 2,
            "ood_size": 2,
            "validation_interval_per_task": 1,
            "checkpoint_interval_per_task": 1,
            "tbptt_steps": 32,
        },
    }
    path = tmp_path / "p2.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["--config", str(path)]) == 0
    # Consume the JSON CLI result without depending on unstable timing values.
    result = json.loads(capsys.readouterr().out)
    assert result["run_id"] == "p2-integration"
    directory = tmp_path / "p2-integration"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "schemas" / "run-manifest-v1.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(manifest)
    assert manifest["status"] == "completed"
    assert manifest["artifacts"]
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["telemetry_equivalence"]["maximum_parameter_difference"] == 0.0
    route = summary["telemetry_off"]["routing"]
    assert route["exact_top_k"] and route["capacity_drops"] == 0
    cost = summary["telemetry_off"]["cost_profile"]
    assert cost["active_optional_macs"] < cost["dense_optional_macs"]
    assert summary["latency_ms"]["p95"] >= summary["latency_ms"]["p50"] > 0.0
