from __future__ import annotations

import json
from pathlib import Path

import jsonschema  # type: ignore[import-untyped]
import torch

from neuromorphic.tasks import create_task
from neuromorphic.training.baselines import GRUBaseline, trainable_parameter_count
from neuromorphic.training.config import RunConfig
from neuromorphic.training.manifest import build_manifest, write_manifest
from neuromorphic.training.run import execute


def test_run_manifest_validates_against_schema(tmp_path: Path) -> None:
    manifest = build_manifest(
        run_id="test-run",
        seed=7,
        device=torch.device("cpu"),
        task={"task_id": "associative_recall.v1"},
        model={"kind": "gru"},
        optimizer={"kind": "adamw"},
        data={"version": "v1"},
        budget={"steps": 1},
        parameters=10,
        estimated_macs=20,
        mac_coverage=0.98,
        unsupported_parameters=("encoder.norm.weight",),
        mac_operators=[{"name": "head", "calls": 1}],
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    schema = json.loads(Path("schemas/run-manifest-v1.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(
        json.loads(path.read_text(encoding="utf-8"))
    )
    assert manifest["status"] == "running"
    assert manifest["cost"]["mac_profiler_coverage"] == 0.98


def test_parameter_matched_run_records_effective_architecture(tmp_path: Path) -> None:
    task = create_task("associative_recall.v1")
    reference = GRUBaseline(input_dim=task.input_dim, num_classes=task.num_classes, hidden_size=16)
    target = trainable_parameter_count(reference)
    config = RunConfig.model_validate(
        {
            "matching_mode": "parameter",
            "seed": 7,
            "device": "cpu",
            "output_root": str(tmp_path),
            "run_id": "parameter-match",
            "task": {"task_id": task.task_id, "profile": "smoke"},
            "model": {"kind": "gru", "hidden_size": 128, "target_parameters": target},
            "training": {
                "batch_size": 2,
                "max_steps": 1,
                "eval_interval": 1,
                "checkpoint_interval": 1,
            },
        }
    )
    execute(config)
    manifest = json.loads(
        (tmp_path / "parameter-match" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model"]["effective_hidden_size"] == 16
    assert manifest["model"]["parameter_match"]["actual"] == target
    assert manifest["model"]["parameter_match"]["relative_error"] <= 0.05
