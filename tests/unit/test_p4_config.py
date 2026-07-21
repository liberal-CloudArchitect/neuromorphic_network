from pathlib import Path

import pytest

from neuromorphic.training.p4_config import (
    P4_PILOT_PRESETS,
    P4SuiteConfig,
    build_p4_matrix,
    load_p4_suite_config,
)


def test_p4_profile_matrix_sizes_are_frozen() -> None:
    qualification = load_p4_suite_config(Path("configs/experiments/p4/qualification.yaml"))
    pilot = load_p4_suite_config(Path("configs/experiments/p4/pilot.yaml"))
    mechanism = load_p4_suite_config(Path("configs/experiments/p4/mechanism.yaml"))
    full = load_p4_suite_config(Path("configs/experiments/p4/full.yaml"))
    assert len(build_p4_matrix(qualification)) == 8
    assert len(build_p4_matrix(pilot)) == 4
    assert len(build_p4_matrix(mechanism)) == 24
    assert len(build_p4_matrix(full)) == 81
    assert len({cell.cell_id for cell in full.matrix()}) == 81
    assert P4_PILOT_PRESETS == {
        "preset-0": (1.0e-4, 1.0e-2, 0.05),
        "preset-1": (1.0e-4, 1.0e-2, 0.10),
        "preset-2": (3.0e-4, 1.0e-2, 0.05),
        "preset-3": (3.0e-4, 1.0e-2, 0.10),
    }


def test_p4_compatible_hash_excludes_run_local_paths() -> None:
    config = load_p4_suite_config(Path("configs/experiments/p4/qualification.yaml"))
    changed = config.model_copy(
        update={"run_id": "different", "output_root": Path("elsewhere"), "resume": Path("x")}
    )
    assert changed.config_hash() == config.config_hash()
    assert changed.matrix_hash() == config.matrix_hash()


def test_p4_full_refuses_missing_mechanism_gate() -> None:
    config = load_p4_suite_config(Path("configs/experiments/p4/full.yaml"))
    with pytest.raises(ValueError, match="GATE-4-MECH"):
        P4SuiteConfig.model_validate({**config.model_dump(mode="python"), "mechanism_report": None})


def test_p4_formal_optimizer_must_match_selected_pilot() -> None:
    config = load_p4_suite_config(Path("configs/experiments/p4/mechanism.yaml"))
    with pytest.raises(ValueError, match="selected pilot preset"):
        P4SuiteConfig.model_validate(
            {
                **config.model_dump(mode="python"),
                "selected_preset": "preset-0",
            }
        )
