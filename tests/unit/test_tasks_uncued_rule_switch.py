from __future__ import annotations

from itertools import pairwise
from typing import cast

import pytest
import torch

from neuromorphic.tasks import TaskBatch, UncuedRuleSwitchTask, create_task
from neuromorphic.tasks.base import DatasetSplit, SequenceTask, stable_content_hash
from neuromorphic.tasks.uncued_rule_switch import BLANK, QUERY, answer
from neuromorphic.transfer.uncued_switch import T1_SPLIT_SEEDS
from neuromorphic.transfer.uncued_switch import UncuedRuleSwitchTask as T1Task

_SPLITS: tuple[DatasetSplit, ...] = ("train", "validation", "analysis", "test", "ood")


def test_mainline_task_keeps_the_t1_generation_rules_exactly() -> None:
    port = UncuedRuleSwitchTask()
    port.task_version = "uncued-rule-switch-v1"
    port.split_seeds = T1_SPLIT_SEEDS
    reference = T1Task()
    for split in _SPLITS:
        ours = port.generate(split, list(range(48)))
        theirs = reference.generate(split, list(range(48)))
        for name in ("inputs", "targets", "valid_mask", "loss_mask", "episode_ids"):
            assert torch.equal(getattr(ours, name), getattr(theirs, name)), (split, name)
        for name in ("trials_since_switch", "rule"):
            assert torch.equal(ours.auxiliary_targets[name], theirs.auxiliary_targets[name])
        assert ours.metadata["switch_trials"] == theirs.metadata["switch_trials"]
        torch.testing.assert_close(port.oracle(ours), reference.oracle(theirs), equal_nan=True)


def test_task_lives_only_in_the_p6_namespace_with_the_frozen_split_seeds() -> None:
    task = create_task("uncued_rule_switch.v1", namespace="p6")
    assert isinstance(task, SequenceTask)
    assert task.task_version == "uncued-rule-switch-p6-v1"
    seeds: dict[DatasetSplit, int] = {
        "train": 31_101,
        "validation": 32_201,
        "test": 33_301,
        "ood": 34_401,
        "analysis": 35_501,
    }
    for split, seed in seeds.items():
        batch = task.generate(split, [0])
        assert batch.metadata["split_seed"] == seed
        assert batch.metadata["namespace"] == "p6"
    with pytest.raises(ValueError, match="only in the p6 namespace"):
        create_task("uncued_rule_switch.v1")
    with pytest.raises(ValueError, match="distribution"):
        create_task("uncued_rule_switch.v1", namespace="p6", distribution="joint")


def test_generation_is_deterministic_profile_independent_and_split_disjoint() -> None:
    smoke = create_task("uncued_rule_switch.v1", namespace="p6")
    qualification = create_task("uncued_rule_switch.v1", profile="qualification", namespace="p6")
    first = smoke.generate("validation", [3, 9, 27])
    assert torch.equal(first.inputs, smoke.generate("validation", [3, 9, 27]).inputs)
    assert torch.equal(first.inputs, qualification.generate("validation", [3, 9, 27]).inputs)

    hashes: dict[str, set[str]] = {}
    episodes: dict[str, set[int]] = {}
    for split in _SPLITS:
        batch = smoke.generate(split, list(range(64)))
        hashes[split] = set(cast(tuple[str, ...], batch.metadata["content_hashes"]))
        episodes[split] = set(batch.episode_ids[batch.valid_mask].tolist())
    for index, left in enumerate(_SPLITS):
        for right in _SPLITS[index + 1 :]:
            assert hashes[left].isdisjoint(hashes[right])
            assert episodes[left].isdisjoint(episodes[right])
    assert _raw_samples(T1Task().generate("train", list(range(64)))).isdisjoint(
        _raw_samples(smoke.generate("train", list(range(64))))
    )


def _raw_samples(batch: TaskBatch) -> set[str]:
    """Version-independent fingerprints of each sample's observations."""

    return {
        stable_content_hash(batch.inputs[row, batch.valid_mask[row]], prefix="")
        for row in range(batch.batch_size)
    }


def _switches(batch_switches: object) -> list[tuple[int, ...]]:
    values = cast(tuple[str, ...], batch_switches)
    return [tuple(int(item) for item in value.split(",")) if value else () for value in values]


def test_id_and_ood_schedules_follow_the_frozen_protocol() -> None:
    task = UncuedRuleSwitchTask()
    train = task.generate("train", list(range(256)))
    ood = task.generate("ood", list(range(256)))

    assert torch.equal(train.loss_mask.sum(dim=1), torch.full((256,), 12))
    assert torch.equal(ood.loss_mask.sum(dim=1), torch.full((256,), 20))
    train_switches = _switches(train.metadata["switch_trials"])
    assert all(len(trials) <= 1 and all(4 <= t <= 8 for t in trials) for trials in train_switches)
    assert 0.6 < sum(bool(trials) for trials in train_switches) / 256 < 0.9
    for trials in _switches(ood.metadata["switch_trials"]):
        assert 2 <= len(trials) <= 3 and all(3 <= t <= 17 for t in trials)
        assert all(b - a >= 3 for a, b in pairwise(trials))
    assert int(train.auxiliary_targets["query_delay"].max()) == 2
    assert int(ood.auxiliary_targets["query_delay"].max()) == 4


def test_auxiliary_targets_describe_queries_switches_and_delays() -> None:
    task = UncuedRuleSwitchTask()
    batch = task.generate("ood", list(range(32)))
    events = batch.inputs[..., :4].argmax(dim=-1)
    delay = batch.auxiliary_targets["query_delay"]
    since = batch.auxiliary_targets["trials_since_switch"]
    rule = batch.auxiliary_targets["rule"]

    assert torch.equal(delay.ge(0), batch.loss_mask)
    assert torch.equal(events.eq(QUERY) & batch.valid_mask, batch.loss_mask)
    assert torch.all(since[~batch.loss_mask] == -1)
    for row in range(batch.batch_size):
        blanks = 0
        for step in range(batch.sequence_length):
            if not bool(batch.valid_mask[row, step]):
                break
            if int(events[row, step]) == BLANK:
                blanks += 1
            elif bool(batch.loss_mask[row, step]):
                assert int(delay[row, step]) == blanks
                first, second = (int(v) for v in batch.inputs[row, step, 8:10])
                expected = answer(int(rule[row, step]), first, second)
                assert int(batch.targets[row, step]) == expected
                blanks = 0
    switches = _switches(batch.metadata["switch_trials"])
    for row, trials in enumerate(switches):
        observed = since[row][batch.loss_mask[row]]
        assert int(observed.eq(0).sum()) == len(trials)


def test_ideal_observer_is_certain_until_the_first_switch() -> None:
    task = UncuedRuleSwitchTask()
    batch = task.generate("test", list(range(64)))
    oracle = task.oracle(batch)
    before = batch.loss_mask & batch.auxiliary_targets["trials_since_switch"].eq(-1)
    assert torch.equal(oracle[before], batch.targets[before].to(oracle.dtype))
    assert torch.isnan(oracle[~batch.loss_mask]).all()
