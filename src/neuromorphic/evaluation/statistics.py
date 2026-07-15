"""Pre-registered descriptive and paired bootstrap statistics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
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


def _validated_record_groups(
    records: Sequence[Mapping[str, object]],
    *,
    value_key: str,
    seed_key: str,
    stratum_key: str,
) -> dict[int, dict[str, np.ndarray]]:
    if not records:
        raise ValueError("records must be non-empty")
    grouped: defaultdict[int, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        try:
            seed_value = record[seed_key]
            stratum_value = record[stratum_key]
            metric_value = record[value_key]
        except KeyError as error:
            raise ValueError(f"record is missing bootstrap field: {error.args[0]}") from error
        if isinstance(seed_value, bool) or not isinstance(seed_value, int):
            raise ValueError("bootstrap seed labels must be integers")
        if not isinstance(stratum_value, str) or not stratum_value:
            raise ValueError("bootstrap strata must be non-empty strings")
        if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
            raise ValueError("bootstrap values must be numeric")
        value = float(metric_value)
        if not np.isfinite(value):
            raise ValueError("bootstrap values must be finite")
        grouped[seed_value][stratum_value].append(value)
    return {
        seed: {
            stratum: np.sort(np.asarray(strata[stratum], dtype=np.float64))
            for stratum in sorted(strata)
        }
        for seed, strata in sorted(grouped.items())
    }


def hierarchical_stratified_bootstrap_mean(
    records: Sequence[Mapping[str, object]],
    *,
    value_key: str,
    seed_key: str = "seed",
    stratum_key: str,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20_260_715,
) -> BootstrapInterval:
    """Bootstrap seeds first, then samples inside every preserved stratum.

    Training seeds receive equal weight. Within a selected seed, each stratum keeps
    its observed sample count, so the resampling does not alter the evaluation mix.
    """
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    grouped = _validated_record_groups(
        records, value_key=value_key, seed_key=seed_key, stratum_key=stratum_key
    )
    seed_labels = sorted(grouped)
    generator = np.random.default_rng(seed)
    distribution = np.empty(samples, dtype=np.float64)
    observed_seed_means = [
        np.concatenate(list(grouped[label].values())).mean() for label in seed_labels
    ]
    for bootstrap_index in range(samples):
        selected_seed_positions = generator.integers(0, len(seed_labels), size=len(seed_labels))
        selected_seed_means: list[float] = []
        for position in selected_seed_positions:
            strata = grouped[seed_labels[int(position)]]
            resampled_strata = [
                values[generator.integers(0, values.size, size=values.size)]
                for values in strata.values()
            ]
            selected_seed_means.append(float(np.concatenate(resampled_strata).mean()))
        distribution[bootstrap_index] = np.mean(selected_seed_means)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(distribution, [alpha, 1.0 - alpha])
    return BootstrapInterval(
        estimate=float(np.mean(observed_seed_means)),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        samples=samples,
    )


def hierarchical_stratified_bootstrap_contrast(
    records: Sequence[Mapping[str, object]],
    *,
    value_key: str,
    stratum_key: str,
    reference: str,
    comparison: str,
    seed_key: str = "seed",
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20_260_715,
) -> BootstrapInterval:
    """Bootstrap a within-seed ``reference - comparison`` stratum contrast."""
    if reference == comparison:
        raise ValueError("contrast strata must be distinct")
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    grouped = _validated_record_groups(
        records, value_key=value_key, seed_key=seed_key, stratum_key=stratum_key
    )
    if any(reference not in strata or comparison not in strata for strata in grouped.values()):
        raise ValueError("every seed must contain both contrast strata")
    seed_labels = sorted(grouped)
    observed = [
        grouped[label][reference].mean() - grouped[label][comparison].mean()
        for label in seed_labels
    ]
    generator = np.random.default_rng(seed)
    distribution = np.empty(samples, dtype=np.float64)
    for bootstrap_index in range(samples):
        selected_seed_positions = generator.integers(0, len(seed_labels), size=len(seed_labels))
        contrasts: list[float] = []
        for position in selected_seed_positions:
            strata = grouped[seed_labels[int(position)]]
            reference_values = strata[reference]
            comparison_values = strata[comparison]
            reference_draw = reference_values[
                generator.integers(0, reference_values.size, size=reference_values.size)
            ]
            comparison_draw = comparison_values[
                generator.integers(0, comparison_values.size, size=comparison_values.size)
            ]
            contrasts.append(float(reference_draw.mean() - comparison_draw.mean()))
        distribution[bootstrap_index] = np.mean(contrasts)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(distribution, [alpha, 1.0 - alpha])
    return BootstrapInterval(
        estimate=float(np.mean(observed)),
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
