"""Deterministic delayed rule-switch task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from neuromorphic.tasks.base import (
    CPU_DEVICE,
    SPLIT_SEEDS,
    DatasetSplit,
    TaskBatch,
    deterministic_seed,
    make_generator,
    stable_content_hash,
)


@dataclass(frozen=True, slots=True)
class _Sample:
    inputs: Tensor
    targets: Tensor
    loss_mask: Tensor
    switch_position: int | None


class DelayedRuleSwitchTask:
    """Apply a cued binary rule after a delay and at most one switch."""

    task_id = "delayed_rule_switch.v1"
    task_version = "delayed-rule-switch-v1"
    input_dim = 9
    num_classes = 2
    trial_count = 4

    def __init__(self, *, profile: str = "smoke") -> None:
        self.profile = profile

    @staticmethod
    def _randint(generator: torch.Generator, low: int, high: int) -> int:
        return int(torch.randint(low, high, (), generator=generator).item())

    @staticmethod
    def _answer(rule: int, first: int, second: int) -> int:
        if rule == 0:
            return first
        if rule == 1:
            return second
        if rule == 2:
            return first ^ second
        return 1 - (first ^ second)

    def _make_sample(self, split: DatasetSplit, sample_index: int) -> _Sample:
        generator = make_generator(self.task_version, split, sample_index)
        initial_rule = self._randint(generator, 0, 4)
        if split == "ood":
            switch_position: int | None = 3
            delay_low, delay_high = 9, 17
        else:
            switch_position = 2 if self._randint(generator, 0, 4) != 0 else None
            delay_low, delay_high = 2, 9
        next_rule = (initial_rule + self._randint(generator, 1, 4)) % 4

        rows: list[Tensor] = []
        labels: list[int] = []
        selected: list[bool] = []
        active_rule = initial_rule
        for trial in range(self.trial_count):
            if trial == 0 or trial == switch_position:
                if trial == switch_position:
                    active_rule = next_rule
                cue = torch.zeros(self.input_dim, dtype=torch.float32)
                cue[0] = 1.0
                cue[3 + active_rule] = 1.0
                rows.append(cue)
                labels.append(-100)
                selected.append(False)

            delay = self._randint(generator, delay_low, delay_high)
            for _ in range(delay):
                blank = torch.zeros(self.input_dim, dtype=torch.float32)
                blank[1] = 1.0
                rows.append(blank)
                labels.append(-100)
                selected.append(False)

            first = self._randint(generator, 0, 2)
            second = self._randint(generator, 0, 2)
            query = torch.zeros(self.input_dim, dtype=torch.float32)
            query[2] = 1.0
            query[7] = float(first)
            query[8] = float(second)
            rows.append(query)
            labels.append(self._answer(active_rule, first, second))
            selected.append(True)

        return _Sample(
            inputs=torch.stack(rows),
            targets=torch.tensor(labels, dtype=torch.long),
            loss_mask=torch.tensor(selected, dtype=torch.bool),
            switch_position=switch_position,
        )

    def content_hash(self, split: DatasetSplit, sample_index: int) -> str:
        sample = self._make_sample(split, sample_index)
        return stable_content_hash(
            sample.inputs,
            sample.targets,
            sample.loss_mask,
            prefix=self.task_version,
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
        hashes: list[str] = []
        lengths: list[int] = []
        switches: list[int | None] = []
        indexed_samples = zip(sample_indices, samples, strict=True)
        for batch_index, (sample_index, sample) in enumerate(indexed_samples):
            steps = sample.inputs.shape[0]
            inputs[batch_index, :steps] = sample.inputs
            targets[batch_index, :steps] = sample.targets
            valid_mask[batch_index, :steps] = True
            loss_mask[batch_index, :steps] = sample.loss_mask
            episode_ids[batch_index, :steps] = deterministic_seed(
                self.task_version, split, sample_index
            )
            hashes.append(self.content_hash(split, sample_index))
            lengths.append(steps)
            switches.append(sample.switch_position)

        return TaskBatch(
            inputs=inputs,
            targets=targets,
            valid_mask=valid_mask,
            loss_mask=loss_mask,
            episode_ids=episode_ids,
            metadata={
                "task_id": self.task_id,
                "task_version": self.task_version,
                "split": split,
                "split_seed": SPLIT_SEEDS[split],
                "sample_indices": tuple(sample_indices),
                "content_hashes": tuple(hashes),
                "sequence_lengths": tuple(lengths),
                "switch_positions": tuple(switches),
                "profile": self.profile,
            },
            auxiliary_targets={},
        ).to(device)

    def oracle(self, batch: TaskBatch) -> Tensor:
        batch.validate()
        result = torch.full_like(batch.targets, -100)
        for batch_index in range(batch.batch_size):
            active_rule: int | None = None
            for step in range(batch.sequence_length):
                if not bool(batch.valid_mask[batch_index, step].item()):
                    continue
                event = batch.inputs[batch_index, step, :3]
                event_id = int(event.argmax().item())
                if event_id == 0:
                    active_rule = int(batch.inputs[batch_index, step, 3:7].argmax().item())
                elif bool(batch.loss_mask[batch_index, step].item()):
                    if active_rule is None:
                        raise ValueError("query encountered before rule cue")
                    first = int(batch.inputs[batch_index, step, 7].item())
                    second = int(batch.inputs[batch_index, step, 8].item())
                    result[batch_index, step] = self._answer(active_rule, first, second)
        return result
