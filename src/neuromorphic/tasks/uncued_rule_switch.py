"""Uncued rule switch with outcome feedback: the P6 mainline task (CR-015).

The generation rules are those of the T1 task in ``neuromorphic.transfer``,
rewritten for the mainline task interface.  Each episode opens with a cue
naming one of four binary rules (``x0``, ``x1``, ``xor``, ``xnor``).  Every
trial is an optional delay, a supervised query ``(x0, x1)`` and a feedback step
that reveals the correct answer.  The rule later changes without any cue, so
only the feedback can reveal the change.  ID episodes have 12 trials and at
most one switch; OOD episodes have 20 trials and two or three switches.  The
task exists only in the P6 namespace.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor

from neuromorphic.tasks.base import (
    CPU_DEVICE,
    P6_SPLIT_SEEDS,
    DatasetSplit,
    TaskBatch,
    deterministic_seed,
    make_generator,
    stable_content_hash,
)

CUE, BLANK, QUERY, FEEDBACK = 0, 1, 2, 3
RULE = slice(4, 8)
X0, X1, FEEDBACK_BIT = 8, 9, 10
NOT_AFTER_SWITCH = -1
NOT_A_QUERY = -1


def answer(rule: int, first: int, second: int) -> int:
    if rule == 0:
        return first
    if rule == 1:
        return second
    if rule == 2:
        return first ^ second
    return 1 - (first ^ second)


@dataclass(frozen=True, slots=True)
class _Sample:
    inputs: Tensor
    targets: Tensor
    loss_mask: Tensor
    trials_since_switch: Tensor  # per step; -1 except on queries after a switch
    rules: Tensor  # per step; the rule in force for that step's trial
    query_delay: Tensor  # per step; delay steps before a query, -1 elsewhere
    switch_trials: tuple[int, ...]


class UncuedRuleSwitchTask:
    """Cued initial rule, silent switches, feedback after every response."""

    task_id = "uncued_rule_switch.v1"
    task_version = "uncued-rule-switch-p6-v1"
    input_dim = 11
    num_classes = 2
    id_trials = 12
    ood_trials = 20

    def __init__(
        self, *, profile: str = "smoke", distribution: str = "v1", namespace: str = "p6"
    ) -> None:
        if distribution != "v1":
            raise ValueError(f"unknown uncued-rule-switch distribution: {distribution}")
        if namespace != "p6":
            raise ValueError("uncued-rule-switch exists only in the p6 namespace")
        self.profile = profile
        self.distribution = distribution
        self.namespace = namespace
        self.split_seeds = P6_SPLIT_SEEDS

    @staticmethod
    def _randint(generator: torch.Generator, low: int, high: int) -> int:
        return int(torch.randint(low, high, (), generator=generator).item())

    def _schedule(
        self, generator: torch.Generator, split: DatasetSplit
    ) -> tuple[int, tuple[int, ...], int]:
        """Trial count, switch trials and maximum delay for one episode."""

        if split == "ood":
            count = self._randint(generator, 2, 4)
            switches: list[int] = []
            while len(switches) < count:
                trial = self._randint(generator, 3, self.ood_trials - 2)
                if all(abs(trial - other) >= 3 for other in switches):
                    switches.append(trial)
            return self.ood_trials, tuple(sorted(switches)), 4
        has_switch = self._randint(generator, 0, 4) != 0
        trial = self._randint(generator, 4, 9)
        return self.id_trials, ((trial,) if has_switch else ()), 2

    @lru_cache(maxsize=32_768)  # noqa: B019 - bounded task instances live for one run
    def _make_sample(self, split: DatasetSplit, sample_index: int) -> _Sample:
        generator = make_generator(
            self.task_version, split, sample_index, split_seeds=self.split_seeds
        )
        trials, switch_trials, max_delay = self._schedule(generator, split)
        rule = self._randint(generator, 0, 4)
        rows: list[Tensor] = []
        labels: list[int] = []
        selected: list[bool] = []
        since: list[int] = []
        rules: list[int] = []
        delays: list[int] = []

        def append(row: Tensor, label: int, trial_since: int, delay: int) -> None:
            rows.append(row)
            labels.append(label)
            selected.append(label >= 0)
            since.append(trial_since)
            rules.append(rule)
            delays.append(delay)

        def row(event: int) -> Tensor:
            vector = torch.zeros(self.input_dim, dtype=torch.float32)
            vector[event] = 1.0
            return vector

        cue = row(CUE)
        cue[RULE.start + rule] = 1.0
        append(cue, -100, NOT_AFTER_SWITCH, NOT_A_QUERY)
        last_switch: int | None = None
        for trial in range(trials):
            if trial in switch_trials:
                rule = (rule + self._randint(generator, 1, 4)) % 4
                last_switch = trial
            delay = self._randint(generator, 0, max_delay + 1)
            for _ in range(delay):
                append(row(BLANK), -100, NOT_AFTER_SWITCH, NOT_A_QUERY)
            first = self._randint(generator, 0, 2)
            second = self._randint(generator, 0, 2)
            target = answer(rule, first, second)
            query = row(QUERY)
            query[X0] = float(first)
            query[X1] = float(second)
            append(
                query,
                target,
                NOT_AFTER_SWITCH if last_switch is None else trial - last_switch,
                delay,
            )
            feedback = row(FEEDBACK)
            feedback[FEEDBACK_BIT] = float(target)
            append(feedback, -100, NOT_AFTER_SWITCH, NOT_A_QUERY)
        return _Sample(
            inputs=torch.stack(rows),
            targets=torch.tensor(labels, dtype=torch.long),
            loss_mask=torch.tensor(selected, dtype=torch.bool),
            trials_since_switch=torch.tensor(since, dtype=torch.long),
            rules=torch.tensor(rules, dtype=torch.long),
            query_delay=torch.tensor(delays, dtype=torch.long),
            switch_trials=switch_trials,
        )

    def content_hash(self, split: DatasetSplit, sample_index: int) -> str:
        sample = self._make_sample(split, sample_index)
        return stable_content_hash(
            sample.inputs, sample.targets, sample.loss_mask, prefix=self.task_version
        )

    def generate(
        self,
        split: DatasetSplit,
        sample_indices: Sequence[int],
        *,
        device: torch.device = CPU_DEVICE,
    ) -> TaskBatch:
        if not sample_indices:
            raise ValueError("sample_indices cannot be empty")
        samples = [self._make_sample(split, index) for index in sample_indices]
        max_steps = max(sample.inputs.shape[0] for sample in samples)
        batch_size = len(samples)
        inputs = torch.zeros((batch_size, max_steps, self.input_dim), dtype=torch.float32)
        targets = torch.full((batch_size, max_steps), -100, dtype=torch.long)
        valid_mask = torch.zeros((batch_size, max_steps), dtype=torch.bool)
        loss_mask = torch.zeros((batch_size, max_steps), dtype=torch.bool)
        episode_ids = torch.full((batch_size, max_steps), -1, dtype=torch.long)
        since = torch.full((batch_size, max_steps), NOT_AFTER_SWITCH, dtype=torch.long)
        rules = torch.full((batch_size, max_steps), -1, dtype=torch.long)
        delays = torch.full((batch_size, max_steps), NOT_A_QUERY, dtype=torch.long)
        hashes: list[str] = []
        lengths: list[int] = []
        indexed_samples = zip(sample_indices, samples, strict=True)
        for batch_index, (sample_index, sample) in enumerate(indexed_samples):
            steps = sample.inputs.shape[0]
            inputs[batch_index, :steps] = sample.inputs
            targets[batch_index, :steps] = sample.targets
            valid_mask[batch_index, :steps] = True
            loss_mask[batch_index, :steps] = sample.loss_mask
            since[batch_index, :steps] = sample.trials_since_switch
            rules[batch_index, :steps] = sample.rules
            delays[batch_index, :steps] = sample.query_delay
            episode_ids[batch_index, :steps] = deterministic_seed(
                self.task_version, split, sample_index, split_seeds=self.split_seeds
            )
            hashes.append(self.content_hash(split, sample_index))
            lengths.append(steps)

        return TaskBatch(
            inputs=inputs,
            targets=targets,
            valid_mask=valid_mask,
            loss_mask=loss_mask,
            episode_ids=episode_ids,
            metadata={
                "task_id": self.task_id,
                "task_version": self.task_version,
                "namespace": self.namespace,
                "split": split,
                "split_seed": self.split_seeds[split],
                "sample_indices": tuple(sample_indices),
                "content_hashes": tuple(hashes),
                "sequence_lengths": tuple(lengths),
                "switch_trials": tuple(
                    ",".join(str(trial) for trial in sample.switch_trials) for sample in samples
                ),
                "profile": self.profile,
                "distribution": self.distribution,
            },
            auxiliary_targets={
                "trials_since_switch": since,
                "rule": rules,
                "query_delay": delays,
            },
        ).to(device)

    def oracle(self, batch: TaskBatch) -> Tensor:
        """Ideal observer's probability that the target is 1 at each query; NaN elsewhere.

        The observer keeps the rules consistent with all feedback since the
        last contradiction, starting from the cued rule.  After a contradiction
        it restarts from the rules consistent with that trial, and its
        prediction averages the answers of the rules in the set.
        """

        batch.validate()
        result = torch.full(batch.targets.shape, float("nan"))
        inputs = batch.inputs.cpu()
        for batch_index in range(batch.batch_size):
            consistent: set[int] = set()
            pending: tuple[int, int] | None = None
            for step in range(batch.sequence_length):
                if not bool(batch.valid_mask[batch_index, step]):
                    break
                event = int(inputs[batch_index, step, :4].argmax())
                if event == CUE:
                    consistent = {int(inputs[batch_index, step, RULE].argmax())}
                elif event == QUERY:
                    pending = (
                        int(inputs[batch_index, step, X0]),
                        int(inputs[batch_index, step, X1]),
                    )
                    votes = [answer(rule, *pending) for rule in consistent]
                    result[batch_index, step] = sum(votes) / len(votes)
                elif event == FEEDBACK and pending is not None:
                    observed = int(inputs[batch_index, step, FEEDBACK_BIT])
                    kept = {rule for rule in consistent if answer(rule, *pending) == observed}
                    consistent = kept or {
                        rule for rule in range(4) if answer(rule, *pending) == observed
                    }
                    pending = None
        return result


__all__ = [
    "BLANK",
    "CUE",
    "FEEDBACK",
    "FEEDBACK_BIT",
    "QUERY",
    "UncuedRuleSwitchTask",
    "answer",
]
