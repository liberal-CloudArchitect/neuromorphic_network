from pathlib import Path

import pytest
from pydantic import ValidationError

from neuromorphic.training.p3_config import InterventionSpec, load_p3_suite_config


def test_p3_qualification_and_full_matrix_are_frozen() -> None:
    qualification = load_p3_suite_config(Path("configs/experiments/p3/ci.yaml"))
    formal = load_p3_suite_config(Path("configs/experiments/p3/full.yaml"))
    assert len(qualification.matrix()) == 36
    assert len(formal.matrix()) == 72
    assert len({cell.cell_id for cell in formal.matrix()}) == 72
    assert qualification.qualification_only
    assert formal.budget.wall_clock_hours == 72


def test_intervention_rejects_incompatible_target() -> None:
    with pytest.raises(ValidationError, match="incompatible"):
        InterventionSpec(target="episodic_memory.v1", operation="dense")


def test_p3_config_hash_excludes_run_locations() -> None:
    config = load_p3_suite_config(Path("configs/experiments/p3/ci.yaml"))
    changed = config.model_copy(update={"run_id": "elsewhere", "output_root": Path("/tmp")})
    assert changed.config_hash() == config.config_hash()
