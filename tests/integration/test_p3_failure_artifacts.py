"""Failure-path coverage for the P3 matrix registry and verifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuromorphic.training.p3_config import load_p3_suite_config
from neuromorphic.training.p3_suite import _initial_registry, execute_p3_suite, verify_p3_run
from neuromorphic.training.run import main


def test_p3_failed_cells_emit_registered_failure_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_p3_suite_config(Path("configs/experiments/p3/ci.yaml")).model_copy(
        update={"output_root": tmp_path, "run_id": "failure-fixture"}
    )

    def fail_build(*args: object, **kwargs: object) -> object:
        raise FloatingPointError("injected non-finite training state")

    monkeypatch.setattr("neuromorphic.training.p3_suite._build_model", fail_build)
    summary = execute_p3_suite(config)
    directory = tmp_path / "failure-fixture"
    verification = verify_p3_run(directory)

    assert summary["status"] == "qualification_failed"
    missing = verification["missing_cells"]
    registered = verification["registered_artifacts"]
    assert isinstance(missing, list)
    assert isinstance(registered, int)
    assert len(missing) == len(config.matrix())
    assert registered >= len(config.matrix())
    assert all(
        (directory / "cells" / cell.cell_id / "failure.json").is_file() for cell in config.matrix()
    )


def test_p3_resume_does_not_reset_the_suite_wall_clock_budget(tmp_path: Path) -> None:
    config = load_p3_suite_config(Path("configs/experiments/p3/ci.yaml")).model_copy(
        update={
            "device": "cpu",
            "output_root": tmp_path,
            "run_id": "exhausted-fixture",
        }
    )
    directory = tmp_path / "exhausted-fixture"
    directory.mkdir()
    registry = _initial_registry(config, "exhausted-fixture")
    registry["wall_clock_seconds"] = config.budget.wall_clock_hours * 3600.0
    (directory / "config.json").write_text(
        config.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (directory / "registry.json").write_text(
        json.dumps(registry),
        encoding="utf-8",
    )

    summary = execute_p3_suite(config)
    updated = json.loads((directory / "registry.json").read_text(encoding="utf-8"))

    assert summary["status"] == "resource_limit"
    assert updated["cells"][0]["status"] == "RESOURCE_LIMIT"
    assert updated["completed_cells"] == 0
    assert updated["wall_clock_seconds"] >= config.budget.wall_clock_hours * 3600.0


@pytest.mark.parametrize(
    "status",
    ["qualification_failed", "pilot_failed", "completed_with_failures", "resource_limit"],
)
def test_p3_cli_returns_nonzero_for_incomplete_matrix(
    status: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "neuromorphic.training.p3_suite.execute_p3_suite",
        lambda config: {"status": status},
    )
    assert main(["--config", "configs/experiments/p3/ci.yaml"]) == 5
