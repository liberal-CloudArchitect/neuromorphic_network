from __future__ import annotations

import torch

from neuromorphic.tasks import create_task
from neuromorphic.training.weak_baselines import (
    associative_key_value_predictions,
    associative_majority_predictions,
    delayed_rule_predictions,
    masked_accuracy,
    random_predictions,
    small_graph_legal_predictions,
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


def test_associative_majority_is_qualified_but_weaker_than_memory() -> None:
    task = create_task("associative_recall.v1")
    batch = task.generate("test", list(range(64)))
    majority = masked_accuracy(associative_majority_predictions(batch), batch)
    memory = masked_accuracy(associative_key_value_predictions(batch), batch)
    assert 0.0 <= majority < memory


def test_rule_state_oracle_handles_switches_better_than_fixed_rule() -> None:
    task = create_task("delayed_rule_switch.v1")
    batch = task.generate("ood", list(range(32)))
    fixed = masked_accuracy(delayed_rule_predictions(batch, update_on_switch=False), batch)
    updated = masked_accuracy(delayed_rule_predictions(batch, update_on_switch=True), batch)
    assert updated == 1.0
    assert fixed < updated


def test_small_graph_weak_policies_only_choose_legal_actions() -> None:
    task = create_task("small_graph.v1")
    batch = task.generate("test", list(range(32)))
    legal = batch.auxiliary_targets["action_nodes"] >= 0
    for mode in ("random", "node_id"):
        predictions = small_graph_legal_predictions(batch, mode=mode, seed=7)
        chosen_legal = legal.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
        assert torch.all(chosen_legal[batch.loss_mask])
