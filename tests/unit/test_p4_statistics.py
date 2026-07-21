from typing import Any

import pytest

from neuromorphic.evaluation.p4_statistics import (
    P4_ASSOCIATIVE_RECALL_CHANCE,
    P4_DELAYED_RULE_SWITCH_CHANCE,
    P4_FORGETTING_REDUCTION_THRESHOLD,
    P4_SPLIT_SEEDS,
    chance_normalized,
    chance_normalized_small_graph_success,
    equal_task_macro,
    forgetting,
    forgetting_reduction,
    normalized_ood,
    p4_fixed_task_chance,
    p4_split_seed,
    paired_hierarchical_bootstrap,
    passes_forgetting_threshold,
    sample_normalized,
    small_graph_chance_level,
    small_graph_uniform_valid_action_reach_goal_probability,
    validate_forgetting_threshold,
    validate_p4_split_seed,
)


def test_p4_split_seeds_are_frozen_and_strictly_validated() -> None:
    assert dict(P4_SPLIT_SEEDS) == {
        "train": 11_101,
        "validation": 12_201,
        "test": 13_301,
        "ood": 14_401,
        "analysis": 15_501,
    }
    assert p4_split_seed("analysis") == 15_501
    validate_p4_split_seed("train", 11_101)
    mutable: Any = P4_SPLIT_SEEDS
    with pytest.raises(TypeError):
        mutable["train"] = 0
    with pytest.raises(ValueError, match="frozen split_seed"):
        validate_p4_split_seed("train", 1_101)
    with pytest.raises(ValueError, match="unknown P4 dataset split"):
        p4_split_seed("dev")


def test_p4_normalization_helpers_reuse_p3_formulas() -> None:
    assert chance_normalized(0.75, 0.5) == pytest.approx(0.5)
    assert normalized_ood(0.6, 0.8, 0.5) == pytest.approx(1 / 3)
    assert sample_normalized(6.0, 8) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="positive integer"):
        sample_normalized(1.0, 0)
    with pytest.raises(ValueError, match="lie in \\[0, sample_count\\]"):
        sample_normalized(9.0, 8)


def test_fixed_chance_and_equal_task_macro_are_frozen() -> None:
    assert p4_fixed_task_chance("associative_recall.v1") == P4_ASSOCIATIVE_RECALL_CHANCE
    assert p4_fixed_task_chance("delayed_rule_switch.v1") == P4_DELAYED_RULE_SWITCH_CHANCE
    assert equal_task_macro((0.3, 0.6, 0.9)) == pytest.approx(0.6)
    with pytest.raises(ValueError, match="per graph"):
        p4_fixed_task_chance("small_graph.v1")
    with pytest.raises(ValueError, match="exactly three"):
        equal_task_macro((0.3, 0.6))


def test_small_graph_random_policy_probability_is_exact() -> None:
    line = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]
    triangle = [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]
    assert small_graph_uniform_valid_action_reach_goal_probability(
        line, start=0, goal=2, node_count=3, horizon=1
    ) == pytest.approx(0.0)
    assert small_graph_uniform_valid_action_reach_goal_probability(
        line, start=0, goal=2, node_count=3, horizon=2
    ) == pytest.approx(0.5)
    assert small_graph_chance_level(
        triangle, start=0, goal=2, node_count=3, horizon=2
    ) == pytest.approx(0.75)
    assert chance_normalized_small_graph_success(
        1.0, triangle, start=0, goal=2, node_count=3, horizon=2
    ) == pytest.approx(1.0)


def test_small_graph_probability_validation_is_strict() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        small_graph_uniform_valid_action_reach_goal_probability(
            [[0, 1]], start=0, goal=1, node_count=2, horizon=1
        )
    with pytest.raises(ValueError, match="symmetric"):
        small_graph_uniform_valid_action_reach_goal_probability(
            [[0, 1], [0, 0]], start=0, goal=1, node_count=2, horizon=1
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        small_graph_uniform_valid_action_reach_goal_probability(
            [[0, 1], [1, 0]], start=0, goal=1, node_count=2, horizon=-1
        )
    with pytest.raises(ValueError, match="at least one valid action"):
        small_graph_uniform_valid_action_reach_goal_probability(
            [[0, 0], [0, 0]], start=0, goal=1, node_count=2, horizon=1
        )


def test_forgetting_threshold_helpers_freeze_the_p4_effect_rule() -> None:
    baseline = forgetting(0.90, 0.60)
    model = forgetting(0.88, 0.73)
    assert baseline == pytest.approx(0.30)
    assert model == pytest.approx(0.15)
    assert forgetting_reduction(baseline, model) == pytest.approx(0.15)
    assert validate_forgetting_threshold(P4_FORGETTING_REDUCTION_THRESHOLD) == pytest.approx(0.02)
    assert passes_forgetting_threshold(baseline, model)
    assert not passes_forgetting_threshold(baseline, 0.29)
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_forgetting_threshold(float("nan"))


def test_qualification_bootstrap_uses_strict_paired_keys() -> None:
    common = {
        "task_id": "associative_recall.v1",
        "split": "analysis",
        "distribution": "v1",
        "stratum": "query",
    }
    full = [
        {
            **common,
            "model_id": "modular-v2",
            "variant_id": "full",
            "seed": seed,
            "sample_index": index,
            "value": 0.8,
        }
        for seed in (17, 29, 43)
        for index in range(4)
    ]
    ablated = [
        {
            **record,
            "variant_id": "predictor-off",
            "value": 0.6,
        }
        for record in full
    ]
    result = paired_hierarchical_bootstrap(full, ablated, samples=200)
    assert result.estimate == pytest.approx(0.2)
    with pytest.raises(ValueError, match="identical"):
        paired_hierarchical_bootstrap(full, ablated[:-1], samples=200)
