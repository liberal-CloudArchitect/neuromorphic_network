from __future__ import annotations

import json
from pathlib import Path

from neuromorphic.training.p5_config import P5QualificationConfig
from neuromorphic.training.p5_suite import execute_p5_qualification


def test_p5_cpu_qualification_exercises_trainable_mechanisms(tmp_path: Path) -> None:
    config = P5QualificationConfig(output_root=tmp_path, run_id="p5-cpu-fixture")

    result = execute_p5_qualification(config)

    assert result["status"] == "qualification_passed"
    directory = tmp_path / "p5-cpu-fixture"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert summary["router_gradient_seen"] is True
    assert summary["predictor_gradient_seen"] is True
    assert summary["routing"]["drs_reserved"] > 0
    assert manifest["status"] == "qualification_passed"
    assert set(manifest["artifacts"]) == {"config.json", "summary.json"}
