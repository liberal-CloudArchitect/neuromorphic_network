from __future__ import annotations

import pytest

from neuromorphic.evaluation.statistics import (
    hierarchical_stratified_bootstrap_contrast,
    hierarchical_stratified_bootstrap_mean,
    holm_adjust,
    percentile_bootstrap_mean,
)


def test_bootstrap_is_deterministic_and_contains_mean() -> None:
    first = percentile_bootstrap_mean([1.0, 2.0, 3.0], samples=1_000, seed=7)
    second = percentile_bootstrap_mean([1.0, 2.0, 3.0], samples=1_000, seed=7)
    assert first == second
    assert first.lower <= first.estimate <= first.upper


def test_holm_adjustment_preserves_order_and_bounds() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_hierarchical_bootstrap_is_deterministic_and_seed_balanced() -> None:
    records = [
        {"seed": 17, "stratum": "low", "value": 1.0},
        {"seed": 17, "stratum": "high", "value": 1.0},
        {"seed": 29, "stratum": "low", "value": 0.0},
        {"seed": 29, "stratum": "high", "value": 0.0},
    ]
    first = hierarchical_stratified_bootstrap_mean(
        records, value_key="value", stratum_key="stratum", samples=500, seed=11
    )
    second = hierarchical_stratified_bootstrap_mean(
        records, value_key="value", stratum_key="stratum", samples=500, seed=11
    )
    assert first == second
    assert first.estimate == pytest.approx(0.5)


def test_hierarchical_bootstrap_contrast_preserves_direction() -> None:
    records = [
        {"seed": seed, "stratum": stratum, "value": value}
        for seed in (17, 29, 43)
        for stratum, value in (("low", 1.0), ("high", 0.25))
    ]
    interval = hierarchical_stratified_bootstrap_contrast(
        records,
        value_key="value",
        stratum_key="stratum",
        reference="low",
        comparison="high",
        samples=200,
    )
    assert interval.estimate == pytest.approx(0.75)
    assert interval.lower == pytest.approx(0.75)
    assert interval.upper == pytest.approx(0.75)


def test_hierarchical_contrast_rejects_missing_control_stratum() -> None:
    with pytest.raises(ValueError, match="every seed"):
        hierarchical_stratified_bootstrap_contrast(
            [{"seed": 17, "stratum": "high", "value": 1.0}],
            value_key="value",
            stratum_key="stratum",
            reference="low",
            comparison="high",
        )
