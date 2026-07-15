from __future__ import annotations

import pytest

from neuromorphic.evaluation.statistics import holm_adjust, percentile_bootstrap_mean


def test_bootstrap_is_deterministic_and_contains_mean() -> None:
    first = percentile_bootstrap_mean([1.0, 2.0, 3.0], samples=1_000, seed=7)
    second = percentile_bootstrap_mean([1.0, 2.0, 3.0], samples=1_000, seed=7)
    assert first == second
    assert first.lower <= first.estimate <= first.upper


def test_holm_adjustment_preserves_order_and_bounds() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
