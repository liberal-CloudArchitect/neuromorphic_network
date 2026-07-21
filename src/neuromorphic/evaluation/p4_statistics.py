"""Frozen P4 evaluation helpers and exact SmallGraph chance calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

import numpy as np

from neuromorphic.evaluation.p3_statistics import (
    chance_normalized as _chance_normalized,
)
from neuromorphic.evaluation.p3_statistics import (
    forgetting as _forgetting,
)
from neuromorphic.evaluation.p3_statistics import (
    normalized_ood as _normalized_ood,
)
from neuromorphic.evaluation.p3_statistics import (
    paired_hierarchical_bootstrap as _paired_hierarchical_bootstrap,
)
from neuromorphic.tasks.base import P4_SPLIT_SEEDS as _P4_SPLIT_SEEDS

P4_SPLIT_SEEDS: Final[Mapping[str, int]] = MappingProxyType(
    {key: value for key, value in _P4_SPLIT_SEEDS.items()}
)
P4_ASSOCIATIVE_RECALL_CHANCE: Final[float] = 1.0 / 32.0
P4_DELAYED_RULE_SWITCH_CHANCE: Final[float] = 1.0 / 2.0
P4_FORGETTING_REDUCTION_THRESHOLD: Final[float] = 0.02

chance_normalized = _chance_normalized
normalized_ood = _normalized_ood
forgetting = _forgetting
paired_hierarchical_bootstrap = _paired_hierarchical_bootstrap


def p4_split_seed(split: str) -> int:
    """Return the frozen P4 split seed for one registered split."""

    if split not in P4_SPLIT_SEEDS:
        raise ValueError(f"unknown P4 dataset split: {split!r}")
    return int(P4_SPLIT_SEEDS[split])


def validate_p4_split_seed(split: str, split_seed: int) -> None:
    """Require an exact match against the frozen P4 split seed table."""

    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ValueError("P4 split_seed must be an integer")
    expected = p4_split_seed(split)
    if split_seed != expected:
        raise ValueError(f"P4 split {split!r} must use frozen split_seed {expected}")


def sample_normalized(value: float, sample_count: int) -> float:
    """Normalize a per-sample count onto ``[0, 1]`` with strict bounds."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if not np.isfinite(value) or value < 0 or value > sample_count:
        raise ValueError("sample-normalized value must be finite and lie in [0, sample_count]")
    return float(value / sample_count)


def p4_fixed_task_chance(task_id: str) -> float:
    """Return the frozen chance level for fixed-classification P4 tasks."""

    if task_id == "associative_recall.v1":
        return P4_ASSOCIATIVE_RECALL_CHANCE
    if task_id == "delayed_rule_switch.v1":
        return P4_DELAYED_RULE_SWITCH_CHANCE
    if task_id == "small_graph.v1":
        raise ValueError("SmallGraph chance must be computed per graph and rollout horizon")
    raise ValueError(f"unknown P4 task: {task_id!r}")


def equal_task_macro(values: Sequence[float]) -> float:
    """Average exactly one finite, chance-normalized score per task."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("P4 task macro requires exactly three finite task scores")
    return float(array.mean())


def validate_forgetting_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("forgetting threshold must be numeric")
    numeric = float(threshold)
    if not np.isfinite(numeric) or numeric < 0:
        raise ValueError("forgetting threshold must be finite and non-negative")
    return numeric


def forgetting_reduction(baseline_forgetting: float, model_forgetting: float) -> float:
    if not np.isfinite(baseline_forgetting) or not np.isfinite(model_forgetting):
        raise ValueError("forgetting reduction inputs must be finite")
    return float(baseline_forgetting - model_forgetting)


def passes_forgetting_threshold(
    baseline_forgetting: float,
    model_forgetting: float,
    *,
    threshold: float = P4_FORGETTING_REDUCTION_THRESHOLD,
) -> bool:
    return forgetting_reduction(
        baseline_forgetting, model_forgetting
    ) >= validate_forgetting_threshold(threshold)


def _validated_active_adjacency(
    adjacency: Sequence[Sequence[bool | int | float]] | np.ndarray,
    *,
    node_count: int,
) -> np.ndarray:
    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count <= 0:
        raise ValueError("node_count must be a positive integer")
    matrix = np.asarray(adjacency)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if matrix.shape[0] < node_count:
        raise ValueError("adjacency must cover node_count active nodes")
    if matrix.dtype == np.bool_:
        active = matrix[:node_count, :node_count].copy()
    else:
        if not np.issubdtype(matrix.dtype, np.number):
            raise ValueError("adjacency must be boolean or numeric")
        numeric = matrix.astype(np.float64, copy=False)
        if not np.isfinite(numeric).all():
            raise ValueError("adjacency must be finite")
        active = numeric[:node_count, :node_count] != 0.0
    if np.any(np.diag(active)):
        raise ValueError("adjacency must not contain self-loops")
    if not np.array_equal(active, active.T):
        raise ValueError("adjacency must be symmetric over the active node set")
    return active


def small_graph_uniform_valid_action_reach_goal_probability(
    adjacency: Sequence[Sequence[bool | int | float]] | np.ndarray,
    *,
    start: int,
    goal: int,
    node_count: int,
    horizon: int,
) -> float:
    """Exact finite-horizon reach-goal probability under a uniform valid-action policy."""

    if isinstance(start, bool) or not isinstance(start, int):
        raise ValueError("start must be an integer")
    if isinstance(goal, bool) or not isinstance(goal, int):
        raise ValueError("goal must be an integer")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer")
    active = _validated_active_adjacency(adjacency, node_count=node_count)
    if not 0 <= start < node_count or not 0 <= goal < node_count:
        raise ValueError("start and goal must lie inside the active node set")
    if start == goal:
        return 1.0

    neighbors = [np.flatnonzero(active[node]).tolist() for node in range(node_count)]
    dead_ends = [node for node, edges in enumerate(neighbors) if node != goal and not edges]
    if dead_ends:
        raise ValueError("every non-goal node must expose at least one valid action")

    probabilities = np.zeros(node_count, dtype=np.float64)
    probabilities[goal] = 1.0
    for _ in range(horizon):
        updated = np.empty(node_count, dtype=np.float64)
        updated[goal] = 1.0
        for node in range(node_count):
            if node == goal:
                continue
            updated[node] = float(np.mean(probabilities[neighbors[node]]))
        probabilities = updated
    return float(probabilities[start])


def small_graph_chance_level(
    adjacency: Sequence[Sequence[bool | int | float]] | np.ndarray,
    *,
    start: int,
    goal: int,
    node_count: int,
    horizon: int,
) -> float:
    """Alias the exact random-policy SmallGraph success probability as a chance level."""

    return small_graph_uniform_valid_action_reach_goal_probability(
        adjacency,
        start=start,
        goal=goal,
        node_count=node_count,
        horizon=horizon,
    )


def chance_normalized_small_graph_success(
    success_rate: float,
    adjacency: Sequence[Sequence[bool | int | float]] | np.ndarray,
    *,
    start: int,
    goal: int,
    node_count: int,
    horizon: int,
) -> float:
    chance = small_graph_chance_level(
        adjacency,
        start=start,
        goal=goal,
        node_count=node_count,
        horizon=horizon,
    )
    return chance_normalized(success_rate, chance)


__all__ = [
    "P4_ASSOCIATIVE_RECALL_CHANCE",
    "P4_DELAYED_RULE_SWITCH_CHANCE",
    "P4_FORGETTING_REDUCTION_THRESHOLD",
    "P4_SPLIT_SEEDS",
    "chance_normalized",
    "chance_normalized_small_graph_success",
    "equal_task_macro",
    "forgetting",
    "forgetting_reduction",
    "normalized_ood",
    "p4_fixed_task_chance",
    "p4_split_seed",
    "paired_hierarchical_bootstrap",
    "passes_forgetting_threshold",
    "sample_normalized",
    "small_graph_chance_level",
    "small_graph_uniform_valid_action_reach_goal_probability",
    "validate_forgetting_threshold",
    "validate_p4_split_seed",
]
