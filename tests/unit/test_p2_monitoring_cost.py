from __future__ import annotations

import math

import pytest
import torch

from neuromorphic.core.contracts import ModuleState
from neuromorphic.core.registry import (
    ACTION_SELECTOR,
    EPISODIC_MEMORY,
    PREDICTIVE_ADAPTER,
    SENSORY_ENCODER,
    WORKING_MEMORY,
)
from neuromorphic.training.modular_cost import (
    ModuleMacRecord,
    build_modular_mac_profile,
    linear_macs,
)
from neuromorphic.training.modular_monitoring import (
    capture_gradients,
    gradient_cosine_similarity,
    routing_statistics,
    state_dynamics,
)


def test_gradient_cosine_uses_named_finite_gradients() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    model(torch.tensor([[1.0, 2.0]])).sum().backward()
    first = capture_gradients(model)
    assert gradient_cosine_similarity(first, first) == pytest.approx(1.0)

    second = {"weight": -first["weight"]}  # type: ignore[operator]
    assert gradient_cosine_similarity(first, second) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="identical parameter names"):
        gradient_cosine_similarity(first, {})
    assert gradient_cosine_similarity(first, {"weight": None}) is None


def test_state_dynamics_reports_norm_and_change() -> None:
    before = {
        EPISODIC_MEMORY: ModuleState(EPISODIC_MEMORY, "v1", {"memory": torch.tensor([[0.0, 0.0]])})
    }
    after = {
        EPISODIC_MEMORY: ModuleState(EPISODIC_MEMORY, "v1", {"memory": torch.tensor([[3.0, 4.0]])})
    }
    dynamics = state_dynamics(before, after)[EPISODIC_MEMORY]
    assert dynamics.state_norm == pytest.approx(5.0)
    assert dynamics.change_norm == pytest.approx(5.0)
    assert dynamics.relative_change == pytest.approx(1.0)


def test_routing_statistics_discloses_raw_reroutes_and_exact_execution() -> None:
    raw = torch.tensor(
        [
            [[True, True, False], [True, False, True]],
            [[False, False, False], [False, True, True]],
        ]
    )
    executed = torch.tensor(
        [
            [[True, False, True], [True, False, True]],
            [[False, False, False], [False, True, True]],
        ]
    )
    valid = torch.tensor([[True, True], [False, True]])
    stats = routing_statistics(raw, executed, valid_mask=valid, top_k=2)
    assert stats.valid_tokens == 3
    assert stats.raw_assignments == stats.executed_assignments == 6
    assert stats.reroute_rate == pytest.approx(1 / 3)
    assert stats.capacity_drops == 0
    assert stats.exact_top_k
    assert sum(stats.raw_shares) == pytest.approx(1.0)
    assert sum(stats.executed_shares) == pytest.approx(1.0)
    assert math.isfinite(stats.executed_entropy)
    assert math.isfinite(stats.executed_coefficient_of_variation)


def test_routing_statistics_counts_drops_and_rejects_padding_routes() -> None:
    raw = torch.tensor([[[True, True, False]]])
    executed = torch.tensor([[[True, False, False]]])
    stats = routing_statistics(raw, executed, valid_mask=torch.ones(1, 1).bool(), top_k=2)
    assert stats.capacity_drops == 1
    assert not stats.exact_top_k

    padding = torch.tensor([[[True, False, False]]])
    with pytest.raises(ValueError, match=r"Padding|padding"):
        routing_statistics(padding, padding, valid_mask=torch.zeros(1, 1).bool(), top_k=1)


@pytest.mark.mps
def test_routing_statistics_runs_natively_on_mps() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    device = torch.device("mps")
    raw = torch.tensor([[[True, True, False]]], device=device)
    executed = torch.tensor([[[True, False, True]]], device=device)
    valid = torch.ones((1, 1), dtype=torch.bool, device=device)
    stats = routing_statistics(raw, executed, valid_mask=valid, top_k=2)
    assert stats.exact_top_k
    assert stats.capacity_drops == 0


@pytest.mark.mps
def test_gradient_and_state_monitoring_move_before_float64_on_mps() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    device = torch.device("mps")
    gradient = {"weight": torch.ones(2, device=device)}
    assert gradient_cosine_similarity(gradient, gradient) == pytest.approx(1.0)
    before = {
        WORKING_MEMORY: ModuleState(
            WORKING_MEMORY, "v1", {"slots": torch.zeros(1, 2, device=device)}
        )
    }
    after = {
        WORKING_MEMORY: ModuleState(
            WORKING_MEMORY, "v1", {"slots": torch.ones(1, 2, device=device)}
        )
    }
    dynamics = state_dynamics(before, after)[WORKING_MEMORY]
    assert dynamics.change_norm == pytest.approx(math.sqrt(2.0))


def test_modular_mac_profile_separates_dense_and_active_optional_cost() -> None:
    records = (
        ModuleMacRecord("boundary.associative", "boundary", "Linear", 100, 8, 8, 50, 50),
        ModuleMacRecord(SENSORY_ENCODER, "required", "Linear", 200, 8, 8, 100, 100),
        ModuleMacRecord(ACTION_SELECTOR, "required", "Linear", 50, 8, 8, 25, 25),
        ModuleMacRecord(EPISODIC_MEMORY, "optional", "attention", 300, 8, 5, 80, 80),
        ModuleMacRecord(WORKING_MEMORY, "optional", "attention", 400, 8, 6, 100, 120),
        ModuleMacRecord(PREDICTIVE_ADAPTER, "optional", "Linear", 250, 8, 5, 75, 75),
    )
    profile = build_modular_mac_profile(records)
    assert profile.boundary_macs == 800
    assert profile.required_macs == 2_000
    assert profile.dense_optional_macs == 7_600
    assert profile.active_optional_macs == 5_150
    assert profile.active_total_macs < profile.dense_total_macs
    assert profile.sparse_optional_saving == 2_450
    assert profile.parameter_coverage == pytest.approx(430 / 450)
    assert profile.to_dict()["records"]


def test_mac_records_validate_categories_and_linear_estimate() -> None:
    layer = torch.nn.Linear(4, 3)
    assert linear_macs(layer, token_count=5) == 60
    with pytest.raises(ValueError, match="required-path"):
        ModuleMacRecord(EPISODIC_MEMORY, "required", "Linear", 1, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="active calls"):
        ModuleMacRecord(PREDICTIVE_ADAPTER, "optional", "Linear", 1, 1, 2, 1, 1)
