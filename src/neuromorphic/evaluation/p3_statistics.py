"""P3 learning-curve, generalization, and strict paired statistics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from neuromorphic.evaluation.statistics import holm_adjust


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    estimate: float
    lower: float
    upper: float
    p_value: float
    samples: int
    confidence: float = 0.95


def normalized_aulc(points: Sequence[tuple[int, float]], *, maximum_step: int) -> float:
    """Integrate a finite learning curve over a fixed normalized budget."""

    if maximum_step <= 0 or not points:
        raise ValueError("AULC requires a positive budget and at least one point")
    ordered = sorted(points)
    if ordered[0][0] < 0 or ordered[-1][0] > maximum_step:
        raise ValueError("AULC steps must lie inside the fixed budget")
    if len({step for step, _ in ordered}) != len(ordered):
        raise ValueError("AULC steps must be unique")
    if not all(np.isfinite(value) for _, value in ordered):
        raise ValueError("AULC values must be finite")
    padded = ordered.copy()
    if padded[0][0] != 0:
        padded.insert(0, (0, padded[0][1]))
    if padded[-1][0] != maximum_step:
        padded.append((maximum_step, padded[-1][1]))
    x = np.asarray([step / maximum_step for step, _ in padded], dtype=np.float64)
    y = np.asarray([value for _, value in padded], dtype=np.float64)
    return float(np.trapezoid(y, x))


def chance_normalized(value: float, chance: float) -> float:
    if not np.isfinite(value) or not np.isfinite(chance) or not 0 <= chance < 1:
        raise ValueError("invalid chance-normalized score")
    return (value - chance) / (1.0 - chance)


def normalized_ood(ood: float, identifier: float, chance: float) -> float:
    if not all(np.isfinite(item) for item in (ood, identifier, chance)):
        raise ValueError("OOD inputs must be finite")
    if identifier <= chance:
        raise ValueError("ID score must exceed chance for OOD normalization")
    return (ood - chance) / (identifier - chance)


def forgetting(best_before_later_tasks: float, final_score: float) -> float:
    if not np.isfinite(best_before_later_tasks) or not np.isfinite(final_score):
        raise ValueError("forgetting inputs must be finite")
    return best_before_later_tasks - final_score


def _record_key(record: Mapping[str, object]) -> tuple[int, str, int, str]:
    try:
        seed = record["seed"]
        stratum = record["stratum"]
        sample_index = record["sample_index"]
        task = record["task_id"]
    except KeyError as error:
        raise ValueError(f"paired record is missing {error.args[0]}") from error
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("paired record seed must be an integer")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise ValueError("paired sample_index must be an integer")
    if not isinstance(stratum, str) or not stratum or not isinstance(task, str):
        raise ValueError("paired task and stratum must be non-empty strings")
    return seed, task, sample_index, stratum


def paired_hierarchical_bootstrap(
    reference: Sequence[Mapping[str, object]],
    comparison: Sequence[Mapping[str, object]],
    *,
    value_key: str = "value",
    samples: int = 10_000,
    rng_seed: int = 20_260_715,
) -> PairedBootstrapResult:
    """Strictly join A/B samples, then resample seeds and strata/sample pairs."""

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")

    def indexed(records: Sequence[Mapping[str, object]]) -> dict[tuple[int, str, int, str], float]:
        result: dict[tuple[int, str, int, str], float] = {}
        for record in records:
            key = _record_key(record)
            if key in result:
                raise ValueError(f"duplicate paired record: {key}")
            value = record.get(value_key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("paired values must be numeric")
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError("paired values must be finite")
            result[key] = numeric
        return result

    left = indexed(reference)
    right = indexed(comparison)
    if not left or left.keys() != right.keys():
        raise ValueError("paired comparisons require identical non-empty sample keys")
    seed_labels = sorted({key[0] for key in left})
    if len(seed_labels) < 1:
        raise ValueError("paired comparisons require training seeds")
    grouped: defaultdict[int, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key in sorted(left):
        grouped[key[0]][key[3]].append(left[key] - right[key])
    observed = [
        float(np.mean([value for values in grouped[seed].values() for value in values]))
        for seed in seed_labels
    ]
    generator = np.random.default_rng(rng_seed)
    distribution = np.empty(samples, dtype=np.float64)
    for iteration in range(samples):
        seed_draw = generator.integers(0, len(seed_labels), size=len(seed_labels))
        means: list[float] = []
        for position in seed_draw:
            strata = grouped[seed_labels[int(position)]]
            resampled: list[float] = []
            for values in strata.values():
                array = np.asarray(values, dtype=np.float64)
                resampled.extend(array[generator.integers(0, array.size, size=array.size)])
            means.append(float(np.mean(resampled)))
        distribution[iteration] = float(np.mean(means))
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    below = (np.count_nonzero(distribution <= 0) + 1) / (samples + 1)
    above = (np.count_nonzero(distribution >= 0) + 1) / (samples + 1)
    return PairedBootstrapResult(
        estimate=float(np.mean(observed)),
        lower=float(lower),
        upper=float(upper),
        p_value=min(1.0, 2.0 * min(below, above)),
        samples=samples,
    )


def adjust_family(results: Sequence[PairedBootstrapResult]) -> list[float]:
    return holm_adjust([result.p_value for result in results])


__all__ = [
    "PairedBootstrapResult",
    "adjust_family",
    "chance_normalized",
    "forgetting",
    "normalized_aulc",
    "normalized_ood",
    "paired_hierarchical_bootstrap",
]
