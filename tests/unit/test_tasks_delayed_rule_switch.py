from __future__ import annotations

from typing import cast

import torch

from neuromorphic.tasks import DelayedRuleSwitchTask


def test_delayed_rule_switch_oracle_and_masks() -> None:
    task = DelayedRuleSwitchTask()
    batch = task.generate("train", list(range(12)))
    oracle = task.oracle(batch)
    assert torch.equal(oracle[batch.loss_mask], batch.targets[batch.loss_mask])
    assert torch.all(batch.loss_mask.sum(dim=1) == task.trial_count)


def test_delayed_rule_switch_ood_uses_longer_delays_and_unseen_switches() -> None:
    task = DelayedRuleSwitchTask()
    train = task.generate("train", list(range(20)))
    ood = task.generate("ood", list(range(20)))
    train_switches = train.metadata["switch_positions"]
    ood_switches = ood.metadata["switch_positions"]
    assert isinstance(train_switches, tuple)
    assert isinstance(ood_switches, tuple)
    assert set(train_switches) <= {None, 2}
    assert set(ood_switches) == {3}
    train_lengths = cast(tuple[int, ...], train.metadata["sequence_lengths"])
    ood_lengths = cast(tuple[int, ...], ood.metadata["sequence_lengths"])
    assert min(ood_lengths) > max(train_lengths)
