"""Uncued rule-switch task with outcome feedback (T1, CR-014).

Each episode opens with a cue naming one of four binary rules (``x0``, ``x1``,
``xor``, ``xnor``).  Every trial is an optional delay, a query ``(x0, x1)``
whose response is supervised, then a feedback step revealing the correct
answer.  The rule later changes **without any cue**; only the feedback can
reveal the change.  Unlike ``delayed-rule-switch-v1`` the switch is therefore
an error-driven event, which is what prediction-error-gated updating targets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor

from neuromorphic.tasks.base import (
    CPU_DEVICE,
    DatasetSplit,
    TaskBatch,
    deterministic_seed,
    make_generator,
    stable_content_hash,
)

T1_SPLIT_SEEDS: Mapping[DatasetSplit, int] = {
    "train": 41_101,
    "validation": 42_201,
    "test": 43_301,
    "ood": 44_401,
    "analysis": 45_501,
}
CUE, BLANK, QUERY, FEEDBACK = 0, 1, 2, 3
_RULE = slice(4, 8)
_X0, _X1, _FEEDBACK_BIT = 8, 9, 10
NOT_AFTER_SWITCH = -1


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
    switch_trials: tuple[int, ...]


class UncuedRuleSwitchTask:
    """Cued initial rule, silent switches, feedback after every response."""

    task_id = "uncued_rule_switch.v1"
    task_version = "uncued-rule-switch-v1"
    input_dim = 11
    num_classes = 2

    def __init__(self, *, id_trials: int = 12, ood_trials: int = 20) -> None:
        if id_trials < 9 or ood_trials < 12:
            raise ValueError("episodes are too short for the switch schedule")
        self.id_trials = id_trials
        self.ood_trials = ood_trials

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
            self.task_version, split, sample_index, split_seeds=T1_SPLIT_SEEDS
        )
        trials, switch_trials, max_delay = self._schedule(generator, split)
        rule = self._randint(generator, 0, 4)
        rows: list[Tensor] = []
        labels: list[int] = []
        selected: list[bool] = []
        since: list[int] = []

        def row(event: int) -> Tensor:
            vector = torch.zeros(self.input_dim, dtype=torch.float32)
            vector[event] = 1.0
            return vector

        cue = row(CUE)
        cue[_RULE.start + rule] = 1.0
        rows.append(cue)
        labels.append(-100)
        selected.append(False)
        since.append(NOT_AFTER_SWITCH)
        last_switch: int | None = None
        for trial in range(trials):
            if trial in switch_trials:
                rule = (rule + self._randint(generator, 1, 4)) % 4
                last_switch = trial
            for _ in range(self._randint(generator, 0, max_delay + 1)):
                rows.append(row(BLANK))
                labels.append(-100)
                selected.append(False)
                since.append(NOT_AFTER_SWITCH)
            first = self._randint(generator, 0, 2)
            second = self._randint(generator, 0, 2)
            target = answer(rule, first, second)
            query = row(QUERY)
            query[_X0] = float(first)
            query[_X1] = float(second)
            rows.append(query)
            labels.append(target)
            selected.append(True)
            since.append(NOT_AFTER_SWITCH if last_switch is None else trial - last_switch)
            feedback = row(FEEDBACK)
            feedback[_FEEDBACK_BIT] = float(target)
            rows.append(feedback)
            labels.append(-100)
            selected.append(False)
            since.append(NOT_AFTER_SWITCH)
        return _Sample(
            inputs=torch.stack(rows),
            targets=torch.tensor(labels, dtype=torch.long),
            loss_mask=torch.tensor(selected, dtype=torch.bool),
            trials_since_switch=torch.tensor(since, dtype=torch.long),
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
        steps = max(sample.inputs.shape[0] for sample in samples)
        batch = len(samples)
        inputs = torch.zeros((batch, steps, self.input_dim), dtype=torch.float32)
        targets = torch.full((batch, steps), -100, dtype=torch.long)
        valid = torch.zeros((batch, steps), dtype=torch.bool)
        loss = torch.zeros((batch, steps), dtype=torch.bool)
        episodes = torch.full((batch, steps), -1, dtype=torch.long)
        since = torch.full((batch, steps), NOT_AFTER_SWITCH, dtype=torch.long)
        for b, (index, sample) in enumerate(zip(sample_indices, samples, strict=True)):
            length = sample.inputs.shape[0]
            inputs[b, :length] = sample.inputs
            targets[b, :length] = sample.targets
            valid[b, :length] = True
            loss[b, :length] = sample.loss_mask
            since[b, :length] = sample.trials_since_switch
            episodes[b, :length] = deterministic_seed(
                self.task_version, split, index, split_seeds=T1_SPLIT_SEEDS
            )
        return TaskBatch(
            inputs=inputs,
            targets=targets,
            valid_mask=valid,
            loss_mask=loss,
            episode_ids=episodes,
            metadata={
                "task_id": self.task_id,
                "task_version": self.task_version,
                "split": split,
                "split_seed": T1_SPLIT_SEEDS[split],
                "sample_indices": tuple(sample_indices),
                "switch_trials": tuple(
                    ",".join(str(t) for t in sample.switch_trials) for sample in samples
                ),
            },
            auxiliary_targets={"trials_since_switch": since},
        ).to(device)

    def oracle(self, batch: TaskBatch) -> Tensor:
        """Ideal observer's probability that the target is 1 at each query.

        It keeps the set of rules consistent with all feedback since the last
        contradiction (starting from the cued rule) and, after a contradiction,
        restarts from the rules consistent with that trial; the prediction
        averages the answers of the rules in the set.  Non-query steps are NaN.
        """

        batch.validate()
        result = torch.full(batch.targets.shape, float("nan"))
        inputs = batch.inputs.cpu()
        for b in range(batch.batch_size):
            consistent: set[int] = set()
            pending: tuple[int, int] | None = None
            for step in range(batch.sequence_length):
                if not bool(batch.valid_mask[b, step]):
                    break
                event = int(inputs[b, step, :4].argmax())
                if event == CUE:
                    consistent = {int(inputs[b, step, _RULE].argmax())}
                elif event == QUERY:
                    pending = (int(inputs[b, step, _X0]), int(inputs[b, step, _X1]))
                    votes = [answer(r, *pending) for r in consistent]
                    result[b, step] = sum(votes) / len(votes)
                elif event == FEEDBACK and pending is not None:
                    observed = int(inputs[b, step, _FEEDBACK_BIT])
                    kept = {r for r in consistent if answer(r, *pending) == observed}
                    consistent = kept or {r for r in range(4) if answer(r, *pending) == observed}
                    pending = None
        return result


__all__ = ["T1_SPLIT_SEEDS", "UncuedRuleSwitchTask", "answer"]
