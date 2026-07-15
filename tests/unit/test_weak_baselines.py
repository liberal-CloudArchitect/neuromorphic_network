from __future__ import annotations

from neuromorphic.tasks import create_task
from neuromorphic.training.weak_baselines import (
    associative_key_value_predictions,
    delayed_rule_predictions,
    masked_accuracy,
    random_predictions,
)


def test_random_baseline_is_reproducible() -> None:
    task = create_task("associative_recall.v1")
    batch = task.generate("test", list(range(8)))
    first = random_predictions(batch, num_classes=task.num_classes, seed=7)
    second = random_predictions(batch, num_classes=task.num_classes, seed=7)
    assert first.equal(second)


def test_explicit_associative_memory_is_perfect() -> None:
    task = create_task("associative_recall.v1")
    batch = task.generate("ood", list(range(32)))
    predictions = associative_key_value_predictions(batch)
    assert masked_accuracy(predictions, batch) == 1.0


def test_rule_state_oracle_handles_switches_better_than_fixed_rule() -> None:
    task = create_task("delayed_rule_switch.v1")
    batch = task.generate("ood", list(range(32)))
    fixed = masked_accuracy(delayed_rule_predictions(batch, update_on_switch=False), batch)
    updated = masked_accuracy(delayed_rule_predictions(batch, update_on_switch=True), batch)
    assert updated == 1.0
    assert fixed < updated
