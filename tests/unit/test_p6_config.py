from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from neuromorphic.core.registry import WORKING_MEMORY, WORKING_MEMORY_V2
from neuromorphic.modules.network_v4 import ModularBrainNetworkV4
from neuromorphic.modules.working_memory_v2 import WorkingMemoryV2
from neuromorphic.training.p5_config import P5_PILOT_PRESETS
from neuromorphic.training.p6_config import (
    P6_BASELINE_LEARNING_RATES,
    P6_FORMAL_SEEDS,
    P6_MECHANISM_VARIANTS,
    P6_PILOT_PRESETS,
    P6FullConfig,
    P6MechanismConfig,
    P6PilotConfig,
    P6QualificationConfig,
    P6SurpriseCheckConfig,
    load_p6_full_config,
    load_p6_mechanism_config,
    load_p6_pilot_config,
    load_p6_qualification_config,
    load_p6_surprise_check_config,
)

_CONFIGS = Path("configs/experiments/p6")


def _formal_budget() -> dict[str, Any]:
    return {
        "device": "cuda",
        "seeds": list(P6_FORMAL_SEEDS),
        "train_samples": 8192,
        "validation_samples": 2048,
        "analysis_samples": 512,
        "test_samples": 2048,
        "ood_samples": 2048,
        "batch_size": 64,
        "steps_per_task": 5000,
        "validation_interval": 100,
        "checkpoint_interval": 100,
        "patience": 30,
        "bootstrap_samples": 10000,
        "control_root": "artifacts/p6-cuda/control",
    }


def _mechanism(**overrides: Any) -> dict[str, Any]:
    value = {
        **_formal_budget(),
        "profile": "mechanism",
        "wall_clock_hours": 48.0,
        "selected_preset": "preset-4",
        "qualification_report": "artifacts/p6-cuda/qualification-lock.json",
        "pilot_lock": "artifacts/p6-cuda/pilot-lock.json",
    }
    value.update(overrides)
    return value


def _full(**overrides: Any) -> dict[str, Any]:
    value = {
        **_formal_budget(),
        "profile": "full",
        "wall_clock_hours": 24.0,
        "baseline_learning_rates": {"shared-gru": 1.0e-3, "shared-transformer": 3.0e-4},
        "pilot_lock": "artifacts/p6-cuda/pilot-lock.json",
        "mechanism_lock": "artifacts/p6-cuda/mechanism-lock.json",
    }
    value.update(overrides)
    return value


def test_preset_grid_adds_only_a_faster_copy_of_preset_one() -> None:
    assert {key: P6_PILOT_PRESETS[key] for key in P5_PILOT_PRESETS} == P5_PILOT_PRESETS
    assert set(P6_PILOT_PRESETS) - set(P5_PILOT_PRESETS) == {"preset-4"}
    assert P6_PILOT_PRESETS["preset-4"] == (1.0e-3, *P5_PILOT_PRESETS["preset-1"][1:])
    assert P6_BASELINE_LEARNING_RATES == (1.0e-4, 3.0e-4, 1.0e-3)


def test_pilot_runs_five_modular_presets_and_three_rates_per_baseline() -> None:
    candidates = load_p6_pilot_config(_CONFIGS / "pilot-cuda.yaml").candidates()
    modular = [item for item in candidates if item.model == "modular-v4"]
    baselines = [item for item in candidates if item.model != "modular-v4"]

    assert [item.candidate_id for item in modular] == list(P6_PILOT_PRESETS)
    assert sorted({(item.model, item.learning_rate) for item in baselines}) == sorted(
        (model, rate)
        for model in ("shared-gru", "shared-transformer")
        for rate in P6_BASELINE_LEARNING_RATES
    )
    assert all(item.temporal_loss_weight is None for item in baselines)
    assert {item.max_steps for item in candidates} == {4000}
    assert len({item.candidate_id for item in candidates}) == 11


def test_formal_mechanism_matrix_has_four_retrained_and_two_inference_cells_per_seed() -> None:
    matrix = P6MechanismConfig.model_validate(_mechanism()).matrix()

    assert len(matrix) == 30 and len({cell.cell_id for cell in matrix}) == 30
    retrained = [cell for cell in matrix if cell.retrained]
    frozen = [cell for cell in matrix if not cell.retrained]
    assert {cell.variant for cell in retrained} == {"full", "wm-v1", "gate-off", "enrich-off"}
    assert {cell.max_steps for cell in retrained} == {20000}
    assert {cell.variant for cell in frozen} == {"shuffle-surprise", "surprise-zero"}
    assert all(
        cell.max_steps == 0 and cell.source_cell_id == f"full__s{cell.seed}" for cell in frozen
    )
    assert {cell.seed for cell in matrix} == set(P6_FORMAL_SEEDS)


@pytest.mark.parametrize(
    "overrides",
    [
        {"seeds": [17, 29, 43]},
        {"patience": 10},
        {"device": "mps"},
        {"wall_clock_hours": 24.0},
        {"steps_per_task": 4000},
        {"ood_samples": 512},
        {"selected_preset": "preset-5"},
        {"dual_route_fraction": 0.3},
        {"telemetry_enabled": True},
        {"pilot_lock": None},
        {"qualification_report": None},
    ],
)
def test_formal_mechanism_rejects_anything_the_protocol_did_not_freeze(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        P6MechanismConfig.model_validate(_mechanism(**overrides))


def test_full_stage_trains_only_the_baselines_and_reuses_mechanism_full_cells() -> None:
    matrix = P6FullConfig.model_validate(_full()).matrix()

    modular = [cell for cell in matrix if cell.model == "modular-v4"]
    baselines = [cell for cell in matrix if cell.model != "modular-v4"]
    assert len(modular) == 5 and len(baselines) == 10
    assert all(
        not cell.retrained and cell.source_cell_id == f"full__s{cell.seed}" for cell in modular
    )
    assert all(cell.retrained and cell.max_steps == 20000 for cell in baselines)
    rates = {cell.model: cell.learning_rate for cell in baselines}
    assert rates == {"shared-gru": 1.0e-3, "shared-transformer": 3.0e-4}


@pytest.mark.parametrize(
    "overrides",
    [
        {"baseline_learning_rates": {"shared-gru": 2.0e-4, "shared-transformer": 3.0e-4}},
        {"baseline_learning_rates": {"shared-gru": 3.0e-4}},
        {"mechanism_lock": None},
        {"wall_clock_hours": 48.0},
        {"patience": 10},
    ],
)
def test_full_stage_rejects_unfrozen_settings(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        P6FullConfig.model_validate(_full(**overrides))


def test_variant_specs_build_the_intended_networks() -> None:
    for variant, spec in P6_MECHANISM_VARIANTS.items():
        model = ModularBrainNetworkV4(**spec.network_kwargs())  # type: ignore[arg-type]
        with model.working_surprise(spec.working_surprise):
            pass
        if spec.working_memory == "v1":
            assert WORKING_MEMORY in model.module_ids and variant == "wm-v1"
            continue
        working = model.registry.get(WORKING_MEMORY_V2)
        assert isinstance(working, WorkingMemoryV2)
        assert working.write_source == spec.write_source
        assert working.surprise_gating is spec.surprise_gating
    assert P6_MECHANISM_VARIANTS["gate-off"].surprise_gating is False
    assert P6_MECHANISM_VARIANTS["enrich-off"].write_source == "current"
    assert P6_MECHANISM_VARIANTS["shuffle-surprise"].working_surprise == "shuffle"
    assert P6_MECHANISM_VARIANTS["surprise-zero"].working_surprise == "zero"


def test_qualification_and_surprise_check_budgets_are_frozen() -> None:
    with pytest.raises(ValidationError):
        P6QualificationConfig(learning_rate=1.0e-3)
    base = load_p6_surprise_check_config(_CONFIGS / "surprise-check-cuda.yaml").model_dump()
    with pytest.raises(ValidationError):
        P6SurpriseCheckConfig.model_validate({**base, "steps_per_task": 500})
    with pytest.raises(ValidationError):
        P6SurpriseCheckConfig.model_validate({**base, "preset": "preset-4"})
    with pytest.raises(ValidationError):
        P6SurpriseCheckConfig.model_validate({**base, "correlation_threshold": 0.3})
    with pytest.raises(ValidationError):
        P6SurpriseCheckConfig.model_validate({**base, "profile": "ci"})
    assert load_p6_surprise_check_config(_CONFIGS / "surprise-check-ci.yaml").max_steps == 16


def test_committed_configs_load_and_hash_only_experiment_fields() -> None:
    loaders = {
        "qualification-cpu": load_p6_qualification_config,
        "qualification-cuda": load_p6_qualification_config,
        "surprise-check-ci": load_p6_surprise_check_config,
        "surprise-check-cuda": load_p6_surprise_check_config,
        "pilot-cuda": load_p6_pilot_config,
        "mechanism-ci": load_p6_mechanism_config,
        "full-ci": load_p6_full_config,
    }
    assert {path.stem for path in _CONFIGS.glob("*.yaml")} == set(loaders)
    for name, loader in loaders.items():
        config = loader(_CONFIGS / f"{name}.yaml")
        moved = type(config).model_validate(
            {**config.model_dump(), "control_root": "elsewhere", "run_id": "other-run"}
        )
        assert moved.config_hash() == config.config_hash()
    pilot = load_p6_pilot_config(_CONFIGS / "pilot-cuda.yaml")
    assert (
        P6PilotConfig.model_validate({**pilot.model_dump(), "device": "mps"}).config_hash()
        != pilot.config_hash()
    )
