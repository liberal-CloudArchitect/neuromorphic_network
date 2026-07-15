from __future__ import annotations

from typing import Literal, cast

import pytest
import torch

from neuromorphic.training.baselines import (
    GRUBaseline,
    TransformerBaseline,
    estimate_macs,
    profile_macs,
    select_parameter_matched_baseline,
    trainable_parameter_count,
    validate_parameter_target,
)


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_baseline_forward_backward_and_cost(kind: str) -> None:
    model: torch.nn.Module
    if kind == "gru":
        model = GRUBaseline(input_dim=12, num_classes=4, hidden_size=16)
    else:
        model = TransformerBaseline(
            input_dim=12,
            num_classes=4,
            hidden_size=16,
            layers=2,
            heads=4,
            feedforward_size=32,
        )
    inputs = torch.randn(3, 7, 12)
    mask = torch.ones(3, 7, dtype=torch.bool)
    output = model(inputs, mask)
    assert output.logits.shape == (3, 7, 4)
    output.logits.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert trainable_parameter_count(model) > 0
    assert estimate_macs(model, sequence_length=7) > 0


def test_parameter_matching_rejects_out_of_tolerance() -> None:
    model = GRUBaseline(input_dim=4, num_classes=2, hidden_size=8)
    with pytest.raises(ValueError, match="parameter target mismatch"):
        validate_parameter_target(model, target=10, tolerance=0.05)


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_parameter_search_meets_five_percent_tolerance(kind: str) -> None:
    reference: torch.nn.Module
    if kind == "gru":
        reference = GRUBaseline(input_dim=12, num_classes=4, hidden_size=32)
    else:
        reference = TransformerBaseline(
            input_dim=12,
            num_classes=4,
            hidden_size=32,
            layers=2,
            heads=4,
            feedforward_size=128,
        )
    target = trainable_parameter_count(reference)
    selected, match = select_parameter_matched_baseline(
        kind=cast(Literal["gru", "transformer"], kind),
        input_dim=12,
        num_classes=4,
        target=target,
    )
    assert trainable_parameter_count(selected) == match.actual
    assert match.relative_error <= 0.05


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_mac_profile_discloses_coverage(kind: str) -> None:
    model: torch.nn.Module
    if kind == "gru":
        model = GRUBaseline(input_dim=12, num_classes=4, hidden_size=16)
    else:
        model = TransformerBaseline(
            input_dim=12,
            num_classes=4,
            hidden_size=16,
            layers=2,
            heads=4,
            feedforward_size=32,
        )
    profile = profile_macs(model, sequence_length=7)
    assert profile.estimated_macs > 0
    assert 0.95 <= profile.coverage <= 1.0
    assert profile.supported_parameters <= profile.total_parameters
