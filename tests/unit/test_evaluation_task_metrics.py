from __future__ import annotations

import pytest
import torch

from neuromorphic.evaluation.task_metrics import aggregate_records, evaluation_records
from neuromorphic.tasks import AssociativeRecallTask, DelayedRuleSwitchTask, SmallGraphTask
from neuromorphic.training.baselines import BaselineOutput


def _target_logits(targets: torch.Tensor, classes: int) -> torch.Tensor:
    logits = torch.zeros((*targets.shape, classes), dtype=torch.float32)
    supervised = targets.ge(0)
    logits[supervised, targets[supervised]] = 10.0
    return logits


def test_associative_records_are_per_sample_and_interference_stratified() -> None:
    batch = AssociativeRecallTask().generate("test", list(range(12)))
    output = BaselineOutput(logits=_target_logits(batch.targets, 32))
    records = evaluation_records(output, batch, run_seed=17)
    assert len(records) == 12
    assert {record["seed"] for record in records} == {17}
    assert {record["interference_stratum"] for record in records} <= {"low", "high"}
    assert all(record["query_accuracy"] == 1.0 for record in records)


def test_associative_aggregate_reports_interference_drop_or_none() -> None:
    records = [
        {
            "task_id": "associative_recall.v1",
            "split": "test",
            "query_accuracy": 1.0,
            "interference_stratum": "low",
        },
        {
            "task_id": "associative_recall.v1",
            "split": "test",
            "query_accuracy": 0.25,
            "interference_stratum": "high",
        },
    ]
    aggregate = aggregate_records(records)
    assert aggregate["query_accuracy"] == pytest.approx(0.625)
    assert aggregate["interference_drop"] == pytest.approx(0.75)
    assert aggregate_records(records[1:])["interference_drop"] is None


def test_delayed_records_and_aggregate_expose_switch_and_delay_semantics() -> None:
    batch = DelayedRuleSwitchTask().generate("test", list(range(24)))
    output = BaselineOutput(logits=_target_logits(batch.targets, 2))
    records = evaluation_records(output, batch, run_seed=29)
    aggregate = aggregate_records(records)
    assert aggregate["response_accuracy"] == pytest.approx(1.0)
    assert aggregate["switch_cost"] == pytest.approx(0.0)
    assert isinstance(aggregate["delay_accuracy"], dict)

    ood = DelayedRuleSwitchTask().generate("ood", list(range(8)))
    ood_records = evaluation_records(
        BaselineOutput(logits=_target_logits(ood.targets, 2)), ood, run_seed=29
    )
    assert aggregate_records(ood_records)["switch_cost"] is None
    assert {record["delay_stratum"] for record in ood_records} == {"ood"}


def test_switch_cost_uses_first_post_switch_response() -> None:
    records = [
        {
            "task_id": "delayed_rule_switch.v1",
            "split": "test",
            "switched": False,
            "response_accuracy": 1.0,
            "first_post_switch_accuracy": None,
            "delay_stratum": "short",
        },
        {
            "task_id": "delayed_rule_switch.v1",
            "split": "test",
            "switched": True,
            "response_accuracy": 0.75,
            "first_post_switch_accuracy": 0.0,
            "delay_stratum": "short",
        },
    ]
    aggregate = aggregate_records(records)
    assert aggregate["first_post_switch_accuracy"] == pytest.approx(0.0)
    assert aggregate["switch_cost"] == pytest.approx(1.0)


def test_small_graph_records_keep_supervised_metrics() -> None:
    batch = SmallGraphTask().generate("test", list(range(4)))
    next_state = batch.auxiliary_targets["next_state"]
    output = BaselineOutput(
        logits=_target_logits(batch.targets, 4),
        next_state_logits=_target_logits(next_state, 16),
    )
    records = evaluation_records(output, batch, run_seed=43)
    aggregate = aggregate_records(records)
    assert aggregate["optimal_action_rate"] == pytest.approx(1.0)
    assert aggregate["next_state_accuracy"] == pytest.approx(1.0)
