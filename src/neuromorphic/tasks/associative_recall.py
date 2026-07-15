"""Deterministic associative-recall sequence task."""

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


class AssociativeRecallTask:
    """Store key/value pairs, tolerate distractors, and answer one query."""

    task_id = "associative_recall.v1"
    task_version = "associative-recall-v1"
    input_dim = 68
    num_classes = 32

    def __init__(self, *, profile: str = "smoke") -> None:
        self.profile = profile

    @staticmethod
    def _randint(generator: torch.Generator, low: int, high: int) -> int:
        return int(torch.randint(low, high, (), generator=generator).item())

    def _make_sample(self, split: DatasetSplit, sample_index: int) -> _Sample:
        generator = make_generator(self.task_version, split, sample_index)
        if split == "ood":
            pair_count = self._randint(generator, 9, 13)
            distractor_count = self._randint(generator, 5, 9)
        else:
            pair_count = self._randint(generator, 4, 9)
            distractor_count = self._randint(generator, 0, 5)

        keys = torch.randperm(32, generator=generator)[:pair_count]
        values = torch.randperm(32, generator=generator)[:pair_count]
        query_position = self._randint(generator, 0, pair_count)
        query_key = int(keys[query_position].item())
        answer = int(values[query_position].item())

        length = pair_count + distractor_count + 1
        inputs = torch.zeros((length, self.input_dim), dtype=torch.float32)
        targets = torch.full((length,), -100, dtype=torch.long)
        loss_mask = torch.zeros((length,), dtype=torch.bool)

        order = torch.randperm(pair_count, generator=generator)
        for step, pair_position in enumerate(order.tolist()):
            key = int(keys[pair_position].item())
            value = int(values[pair_position].item())
            inputs[step, 0] = 1.0  # store event
            inputs[step, 4 + key] = 1.0
            inputs[step, 36 + value] = 1.0

        unused_keys = [key for key in range(32) if key not in set(keys.tolist())]
        for offset, step in enumerate(range(pair_count, pair_count + distractor_count)):
            inputs[step, 1] = 1.0  # distractor event
            available_index = torch.randperm(len(unused_keys), generator=generator)[offset]
            distractor_key = unused_keys[int(available_index.item())]
            distractor_value = self._randint(generator, 0, 32)
            inputs[step, 4 + distractor_key] = 1.0
            inputs[step, 36 + distractor_value] = 1.0

        inputs[-1, 2] = 1.0  # query event
        inputs[-1, 4 + query_key] = 1.0
        targets[-1] = answer
        loss_mask[-1] = True
        return _Sample(inputs, targets, loss_mask)

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
        indexed_samples = zip(sample_indices, samples, strict=True)
        for batch_index, (sample_index, sample) in enumerate(indexed_samples):
            steps = sample.inputs.shape[0]
            inputs[batch_index, :steps] = sample.inputs
            targets[batch_index, :steps] = sample.targets
            valid_mask[batch_index, :steps] = True
            loss_mask[batch_index, :steps] = sample.loss_mask
            episode_id = deterministic_seed(self.task_version, split, sample_index)
            episode_ids[batch_index, :steps] = episode_id
            hashes.append(self.content_hash(split, sample_index))
            lengths.append(steps)

        batch = TaskBatch(
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
                "profile": self.profile,
            },
            auxiliary_targets={},
        ).to(device)
        return batch

    def oracle(self, batch: TaskBatch) -> Tensor:
        batch.validate()
        result = torch.full_like(batch.targets, -100)
        for batch_index in range(batch.batch_size):
            memory: dict[int, int] = {}
            for step in range(batch.sequence_length):
                if not bool(batch.valid_mask[batch_index, step].item()):
                    continue
                event = batch.inputs[batch_index, step, :4]
                if int(event.argmax().item()) == 0:
                    key = int(batch.inputs[batch_index, step, 4:36].argmax().item())
                    value = int(batch.inputs[batch_index, step, 36:68].argmax().item())
                    memory[key] = value
                elif bool(batch.loss_mask[batch_index, step].item()):
                    key = int(batch.inputs[batch_index, step, 4:36].argmax().item())
                    result[batch_index, step] = memory[key]
        return result
