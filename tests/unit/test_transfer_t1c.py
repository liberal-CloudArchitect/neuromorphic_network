from __future__ import annotations

import torch

from neuromorphic.tasks.base import DatasetSplit
from neuromorphic.transfer.multi_context import (
    CONTEXTS,
    FEEDBACK_BIT,
    MultiContextSwitchTask,
)
from neuromorphic.transfer.surprise_wm import T1GRU, T1Network, parameter_count
from neuromorphic.transfer.uncued_switch import CUE, FEEDBACK, QUERY, answer


def test_task_is_deterministic_split_disjoint_and_cues_only_open_the_episode() -> None:
    task = MultiContextSwitchTask()
    first = task.generate("train", [0, 1, 2])
    assert torch.equal(first.inputs, task.generate("train", [0, 1, 2]).inputs)
    splits: tuple[DatasetSplit, ...] = ("train", "test", "ood")
    hashes = {s: {task.content_hash(s, i) for i in range(50)} for s in splits}
    assert not (hashes["train"] & hashes["test"]) and not (hashes["test"] & hashes["ood"])
    batch = task.generate("train", list(range(64)))
    events = batch.inputs[..., :4].argmax(dim=-1)
    cues = (events == CUE) & batch.valid_mask
    assert torch.equal(cues.sum(dim=1), torch.full((64,), CONTEXTS))
    assert bool(cues[:, :CONTEXTS].all())  # one cue per context, all at the start
    cued = batch.inputs[:, :CONTEXTS, 4 : 4 + CONTEXTS].sum(dim=1)
    assert torch.equal(cued, torch.ones(64, CONTEXTS))
    query = (events == QUERY) & batch.valid_mask
    rows, cols = torch.nonzero(query, as_tuple=True)
    assert bool((events[rows, cols + 1] == FEEDBACK).all())
    assert torch.equal(batch.inputs[rows, cols + 1, FEEDBACK_BIT].long(), batch.targets[rows, cols])
    context = batch.auxiliary_targets["context"]
    assert torch.equal(context[rows, cols], context[rows, cols + 1])
    assert torch.equal(
        batch.inputs[rows, cols, 4 : 4 + CONTEXTS].argmax(dim=-1), context[rows, cols]
    )


def test_only_the_switched_context_changes_rule_and_targets_follow_it() -> None:
    task = MultiContextSwitchTask()
    batch = task.generate("ood", list(range(64)))
    rules = batch.auxiliary_targets["rules"]
    context = batch.auxiliary_targets["context"]
    rows, cols = torch.nonzero(batch.loss_mask, as_tuple=True)
    for b, t in zip(rows.tolist(), cols.tolist(), strict=True):
        k = int(context[b, t])
        x0, x1 = (
            int(batch.inputs[b, t, FEEDBACK_BIT - 2]),
            int(batch.inputs[b, t, FEEDBACK_BIT - 1]),
        )
        assert answer(int(rules[b, t, k]), x0, x1) == int(batch.targets[b, t])
    valid = batch.valid_mask
    changed = (rules[:, 1:] != rules[:, :-1]) & valid[:, 1:, None]
    assert bool((changed.sum(dim=-1) <= 1).all())  # at most one context changes per step
    switches = changed.any(dim=-1).sum(dim=1)
    schedule = batch.metadata["switch_trials"]
    assert isinstance(schedule, tuple)
    count = [len(str(s).split(",")) for s in schedule]
    assert switches.tolist() == count
    since = batch.auxiliary_targets["trials_since_switch"]
    gaps = since[batch.loss_mask]
    assert int((gaps == 0).sum()) == sum(count)


def test_oracle_is_perfect_before_switches_and_recovers_after() -> None:
    task = MultiContextSwitchTask()
    batch = task.generate("train", list(range(256)))
    oracle = task.oracle(batch)
    since = batch.auxiliary_targets["trials_since_switch"]
    target = batch.targets.float()
    accuracy = oracle * target + (1 - oracle) * (1 - target)
    query = batch.loss_mask
    assert float(accuracy[query & (since == -1)].min()) == 1.0
    after = [float(accuracy[query & (since == k)].mean()) for k in range(4)]
    assert after == sorted(after) and after[0] < 0.6 and after[3] > 0.8


def test_networks_are_causal_on_the_multi_context_layout() -> None:
    task = MultiContextSwitchTask()
    batch = task.generate("train", [7, 8])
    counts = set()
    for source in ("current", "enriched"):
        torch.manual_seed(4)
        net = T1Network(task.input_dim, mode="gate", write_source=source, feedback_bit=FEEDBACK_BIT)
        counts.add(parameter_count(net))
        logits, _ = net.run(batch)
        cut = 20
        future = batch.inputs.clone()
        future[:, cut:] = torch.rand_like(future[:, cut:])
        altered = net(
            type(batch)(
                **{**{f: getattr(batch, f) for f in batch.__dataclass_fields__}, "inputs": future}
            )
        )
        torch.testing.assert_close(altered[:, :cut], logits[:, :cut])
        plain = T1Network(task.input_dim, mode="plain", write_source=source)
        plain.load_state_dict(net.state_dict())
        assert not torch.allclose(plain(batch), logits)  # surprise read from the right bit
    assert len(counts) == 1
    gru = T1GRU(task.input_dim)
    assert abs(parameter_count(gru) / counts.pop() - 1) < 0.06
