"""Pre-registered descriptive and paired bootstrap statistics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A percentile bootstrap estimate and confidence interval."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int


def percentile_bootstrap_mean(
    values: Sequence[float],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20_260_715,
) -> BootstrapInterval:
    """Compute a deterministic percentile bootstrap interval for a mean."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite one-dimensional sequence")
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(samples, array.size))
    distribution = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(distribution, [alpha, 1.0 - alpha])
    return BootstrapInterval(
        estimate=float(array.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        samples=samples,
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in their original order."""
    array = np.asarray(p_values, dtype=np.float64)
    if array.ndim != 1 or not ((array >= 0.0) & (array <= 1.0)).all():
        raise ValueError("p-values must be one-dimensional values in [0, 1]")
    order = np.argsort(array)
    adjusted = np.empty_like(array)
    running = 0.0
    total = array.size
    for rank, index in enumerate(order):
        running = max(running, float((total - rank) * array[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()
