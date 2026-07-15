from __future__ import annotations

import json
from pathlib import Path

import jsonschema  # type: ignore[import-untyped]
import torch

from neuromorphic.training.manifest import build_manifest, write_manifest


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
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    schema = json.loads(Path("schemas/run-manifest-v1.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(
        json.loads(path.read_text(encoding="utf-8"))
    )
    assert manifest["status"] == "running"
    assert manifest["cost"]["mac_profiler_coverage"] == 0.98
