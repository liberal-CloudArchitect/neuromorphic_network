from pathlib import Path

import pytest
from pydantic import ValidationError

from neuromorphic.training.p3_config import (
    InterventionSpec,
    P3SuiteConfig,
    load_p3_suite_config,
)
from neuromorphic.training.p3_suite import _early_stop_reached, _task_sequence


def test_p3_qualification_and_full_matrix_are_frozen() -> None:
    qualification = load_p3_suite_config(Path("configs/experiments/p3/ci.yaml"))
    pilot = load_p3_suite_config(Path("configs/experiments/p3/pilot.yaml"))
    formal = load_p3_suite_config(Path("configs/experiments/p3/full.yaml"))
    assert len(qualification.matrix()) == 39
    assert len(pilot.matrix()) == 12
    assert all(cell.max_steps == 1_000 for cell in pilot.matrix())
    assert len(formal.matrix()) == 81
    assert len({cell.cell_id for cell in formal.matrix()}) == 81
    control_variants = {
        cell.variant_id for cell in qualification.matrix() if cell.cell_type == "control"
    }
    assert {
        "acute-episodic-off",
        "acute-working-reset",
        "acute-predictive-off",
        "router-dense",
        "router-fixed",
        "router-random",
        "direct-head",
        "frozen-random-encoder",
        "shallow-encoder",
    } == control_variants
    assert qualification.qualification_only
    assert formal.budget.wall_clock_hours == 72


def test_intervention_rejects_incompatible_target() -> None:
    with pytest.raises(ValidationError, match="incompatible"):
        InterventionSpec(target="episodic_memory.v1", operation="dense")


def test_p3_config_hash_excludes_run_locations() -> None:
    config = load_p3_suite_config(Path("configs/experiments/p3/ci.yaml"))
    changed = config.model_copy(update={"run_id": "elsewhere", "output_root": Path("/tmp")})
    assert changed.config_hash() == config.config_hash()


def test_p3_config_hash_binds_generated_matrix() -> None:
    config = load_p3_suite_config(Path("configs/experiments/p3/ci.yaml"))
    assert len(config.matrix_hash()) == 64
    assert config.compatible_dict()["experiment_matrix"] == [
        cell.model_dump(mode="json") for cell in config.matrix()
    ]


def test_continual_cells_keep_all_three_registered_stages() -> None:
    config = load_p3_suite_config(Path("configs/experiments/p3/full.yaml"))
    cell = next(item for item in config.matrix() if item.cell_type == "continual")
    stage = config.budget.continual_steps_per_stage

    assert cell.max_steps == stage * 3
    assert _task_sequence(cell, 0) != _task_sequence(cell, stage)
    assert _task_sequence(cell, stage) != _task_sequence(cell, stage * 2)
    assert not _early_stop_reached(cell, stale=100, patience=10)
    shared = next(item for item in config.matrix() if item.cell_type == "shared")
    assert _early_stop_reached(shared, stale=10, patience=10)


def test_formal_profile_rejects_device_or_budget_drift() -> None:
    config = load_p3_suite_config(Path("configs/experiments/p3/full.yaml"))
    raw = config.model_dump(mode="json")
    with pytest.raises(ValidationError, match="MPS"):
        P3SuiteConfig.model_validate({**raw, "device": "cpu"})
    changed_budget = {**raw["budget"], "continual_steps_per_stage": 1_499}
    with pytest.raises(ValidationError, match="budgets are frozen"):
        P3SuiteConfig.model_validate({**raw, "budget": changed_budget})
