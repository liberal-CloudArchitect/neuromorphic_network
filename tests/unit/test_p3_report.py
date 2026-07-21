"""Focused tests for the formal P3 report metric routing."""

from __future__ import annotations

from neuromorphic.analysis.p3_report import _primary_value


def test_primary_value_requires_live_rollout_for_small_graph_test() -> None:
    supervised = {
        "task_id": "small_graph.v1",
        "schema_version": "p3-evaluation-sample-v1",
        "optimal_action_rate": 0.75,
    }
    rollout = {
        "task_id": "small_graph.v1",
        "schema_version": "p3-small-graph-rollout-v1",
        "success_rate": 1.0,
    }

    assert _primary_value(supervised) is None
    assert _primary_value(rollout) == 1.0
    assert _primary_value(supervised, ood_proxy=True) == 0.75


def test_primary_value_routes_classification_tasks() -> None:
    assert _primary_value({"task_id": "associative_recall.v1", "query_accuracy": 0.5}) == 0.5
    assert _primary_value({"task_id": "delayed_rule_switch.v1", "response_accuracy": 0.75}) == 0.75
