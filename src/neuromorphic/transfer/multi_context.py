"""Interleaved multi-context uncued rule switches (T1c, CR-014).

An episode opens with one cue per context giving that context's rule (one of
``x0``, ``x1``, ``xor``, ``xnor``).  Trials then draw a context at random: an
optional delay, a query ``(context, x0, x1)`` whose response is supervised,
and a feedback step ``(context, answer)``.  A context's rule later changes
**without any cue** while the other contexts keep theirs, so the network has
to hold several independent rules and update only the one the feedback is
about.  This is the regime in which addressable slots might beat one shared
recurrent state.
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
from neuromorphic.transfer.uncued_switch import BLANK, CUE, FEEDBACK, QUERY, answer

T1C_SPLIT_SEEDS: Mapping[DatasetSplit, int] = {
    "train": 51_101,
    "validation": 52_201,
    "test": 53_301,
    "ood": 54_401,
    "analysis": 55_501,
}
CONTEXTS = 3
_CONTEXT = slice(4, 4 + CONTEXTS)
_RULE = slice(_CONTEXT.stop, _CONTEXT.stop + 4)
_X0, _X1, FEEDBACK_BIT = _RULE.stop, _RULE.stop + 1, _RULE.stop + 2
NOT_AFTER_SWITCH = -1


@dataclass(frozen=True, slots=True)
class _Sample:
    inputs: Tensor
    targets: Tensor
    loss_mask: Tensor
    since: Tensor  # per step; occurrences of the step's context since its last switch
    context: Tensor  # per step; the context of a query/feedback step, else -1
    rules: Tensor  # [T, C]; every context's rule in force at each step
    context_since: Tensor  # [T, C]; every context's occurrences since its last switch
    global_since: Tensor  # per step; trials since the latest switch in any context
    switch_trials: tuple[int, ...]


class MultiContextSwitchTask:
    """Cued rules for several contexts, interleaved trials, silent per-context switches."""

    task_id = "multi_context_switch.v1"
    task_version = "multi-context-switch-v1"
    input_dim = FEEDBACK_BIT + 1
    num_classes = 2
    contexts = CONTEXTS

    def __init__(self, *, id_trials: int = 18, ood_trials: int = 30) -> None:
        if id_trials < 12 or ood_trials < 12:
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

        ood = split == "ood"
        trials = self.ood_trials if ood else self.id_trials
        count = self._randint(generator, 2, 4) if ood else 2
        low = 4 if ood else 5
        switches: list[int] = []
        while len(switches) < count:
            trial = self._randint(generator, low, trials - 2)
            if all(abs(trial - other) >= 3 for other in switches):
                switches.append(trial)
        return trials, tuple(sorted(switches)), (4 if ood else 2)

    @lru_cache(maxsize=32_768)  # noqa: B019 - bounded task instances live for one run
    def _make_sample(self, split: DatasetSplit, sample_index: int) -> _Sample:
        generator = make_generator(
            self.task_version, split, sample_index, split_seeds=T1C_SPLIT_SEEDS
        )
        trials, switch_trials, max_delay = self._schedule(generator, split)
        rule = [self._randint(generator, 0, 4) for _ in range(CONTEXTS)]
        order = torch.randperm(CONTEXTS, generator=generator).tolist()
        rows: list[Tensor] = []
        labels: list[int] = []
        selected: list[bool] = []
        since: list[int] = []
        contexts: list[int] = []
        rules: list[list[int]] = []
        context_since: list[list[int]] = []
        global_since: list[int] = []
        occurrences_since = [NOT_AFTER_SWITCH] * CONTEXTS
        last_switch: int | None = None
        trial = -1

        def emit(vector: Tensor, label: int, gap: int, context: int) -> None:
            rows.append(vector)
            labels.append(label)
            selected.append(label != -100)
            since.append(gap)
            contexts.append(context)
            rules.append(list(rule))
            context_since.append(list(occurrences_since))
            global_since.append(NOT_AFTER_SWITCH if last_switch is None else trial - last_switch)

        def row(event: int, context: int | None = None) -> Tensor:
            vector = torch.zeros(self.input_dim, dtype=torch.float32)
            vector[event] = 1.0
            if context is not None:
                vector[_CONTEXT.start + context] = 1.0
            return vector

        for context in order:
            cue = row(CUE, context)
            cue[_RULE.start + rule[context]] = 1.0
            emit(cue, -100, NOT_AFTER_SWITCH, -1)
        for trial in range(trials):
            context = self._randint(generator, 0, CONTEXTS)
            if trial in switch_trials:
                rule[context] = (rule[context] + self._randint(generator, 1, 4)) % 4
                occurrences_since[context] = 0
                last_switch = trial
            elif occurrences_since[context] != NOT_AFTER_SWITCH:
                occurrences_since[context] += 1
            for _ in range(self._randint(generator, 0, max_delay + 1)):
                emit(row(BLANK), -100, NOT_AFTER_SWITCH, -1)
            first = self._randint(generator, 0, 2)
            second = self._randint(generator, 0, 2)
            target = answer(rule[context], first, second)
            query = row(QUERY, context)
            query[_X0] = float(first)
            query[_X1] = float(second)
            emit(query, target, occurrences_since[context], context)
            feedback = row(FEEDBACK, context)
            feedback[FEEDBACK_BIT] = float(target)
            emit(feedback, -100, occurrences_since[context], context)
        return _Sample(
            inputs=torch.stack(rows),
            targets=torch.tensor(labels, dtype=torch.long),
            loss_mask=torch.tensor(selected, dtype=torch.bool),
            since=torch.tensor(since, dtype=torch.long),
            context=torch.tensor(contexts, dtype=torch.long),
            rules=torch.tensor(rules, dtype=torch.long),
            context_since=torch.tensor(context_since, dtype=torch.long),
            global_since=torch.tensor(global_since, dtype=torch.long),
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
        """Batch with auxiliary ``trials_since_switch`` (counted in the step's own
        context), ``context``, ``global_since`` (trials since any switch) and the
        per-context ``rules`` and ``context_since`` ``[B, T, C]``."""

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
        context = torch.full((batch, steps), -1, dtype=torch.long)
        rules = torch.full((batch, steps, CONTEXTS), -1, dtype=torch.long)
        context_since = torch.full((batch, steps, CONTEXTS), NOT_AFTER_SWITCH, dtype=torch.long)
        global_since = torch.full((batch, steps), NOT_AFTER_SWITCH, dtype=torch.long)
        for b, (index, sample) in enumerate(zip(sample_indices, samples, strict=True)):
            length = sample.inputs.shape[0]
            inputs[b, :length] = sample.inputs
            targets[b, :length] = sample.targets
            valid[b, :length] = True
            loss[b, :length] = sample.loss_mask
            since[b, :length] = sample.since
            context[b, :length] = sample.context
            rules[b, :length] = sample.rules
            context_since[b, :length] = sample.context_since
            global_since[b, :length] = sample.global_since
            episodes[b, :length] = deterministic_seed(
                self.task_version, split, index, split_seeds=T1C_SPLIT_SEEDS
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
                "split_seed": T1C_SPLIT_SEEDS[split],
                "sample_indices": tuple(sample_indices),
                "switch_trials": tuple(
                    ",".join(str(t) for t in sample.switch_trials) for sample in samples
                ),
            },
            auxiliary_targets={
                "trials_since_switch": since,
                "context": context,
                "global_since": global_since,
                "rules": rules,
                "context_since": context_since,
            },
        ).to(device)

    def consistent_rules(self, batch: TaskBatch) -> Tensor:
        """Ideal observer's rule set per context after each valid step, ``[B, T, C, 4]``.

        Each context keeps the rules consistent with all of its feedback since
        its last contradiction (starting from its cued rule) and, after a
        contradiction, restarts from the rules consistent with that trial,
        exactly as the T1 observer does for a single context.
        """

        batch.validate()
        result = torch.zeros((*batch.targets.shape, CONTEXTS, 4), dtype=torch.bool)
        inputs = batch.inputs.cpu()
        for b in range(batch.batch_size):
            consistent: list[set[int]] = [set(range(4)) for _ in range(CONTEXTS)]
            pending: tuple[int, int] | None = None
            for step in range(batch.sequence_length):
                if not bool(batch.valid_mask[b, step]):
                    break
                event = int(inputs[b, step, :4].argmax())
                context = int(inputs[b, step, _CONTEXT].argmax())
                if event == CUE:
                    consistent[context] = {int(inputs[b, step, _RULE].argmax())}
                elif event == QUERY:
                    pending = (int(inputs[b, step, _X0]), int(inputs[b, step, _X1]))
                elif event == FEEDBACK and pending is not None:
                    observed = int(inputs[b, step, FEEDBACK_BIT])
                    kept = {r for r in consistent[context] if answer(r, *pending) == observed}
                    consistent[context] = kept or {
                        r for r in range(4) if answer(r, *pending) == observed
                    }
                    pending = None
                for k in range(CONTEXTS):
                    result[b, step, k, sorted(consistent[k])] = True
        return result

    def oracle(self, batch: TaskBatch) -> Tensor:
        """Ideal observer's probability that the target is 1 at each query (NaN elsewhere)."""

        sets = self.consistent_rules(batch)
        result = torch.full(batch.targets.shape, float("nan"))
        inputs = batch.inputs.cpu()
        rows, cols = torch.nonzero(batch.loss_mask.cpu(), as_tuple=True)
        for b, t in zip(rows.tolist(), cols.tolist(), strict=True):
            context = int(inputs[b, t, _CONTEXT].argmax())
            first, second = int(inputs[b, t, _X0]), int(inputs[b, t, _X1])
            members = torch.nonzero(sets[b, t, context]).flatten().tolist()
            result[b, t] = sum(answer(r, first, second) for r in members) / len(members)
        return result


__all__ = ["CONTEXTS", "FEEDBACK_BIT", "T1C_SPLIT_SEEDS", "MultiContextSwitchTask"]
