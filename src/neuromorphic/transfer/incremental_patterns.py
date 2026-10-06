"""Class-incremental multi-modal pattern recognition (T2, CR-014).

A *world* fixes ``tasks x classes_per_task`` classes; each class is the union
of ``modes`` unrelated sparse prototypes over ``dims`` input channels, so a
class mean blends unrelated patterns (F1a-v3/v4).  A sample shows one mode for
``frames`` steps with independent Gaussian noise and then a query step whose
target is the class.  Tasks are learned one after another (class-incremental:
no task label at test).  With additive Gaussian noise the frame mean is a
sufficient statistic, so the Bayes-optimal classifier is exact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

T2_SPLIT_SEEDS: Mapping[DatasetSplit, int] = {
    "train": 61_101,
    "validation": 62_201,
    "test": 63_301,
    "ood": 64_401,
    "analysis": 65_501,
}
_WORLD_SEEDS: Mapping[DatasetSplit, int] = {"train": 60_001}
SPLIT_SIZES: Mapping[DatasetSplit, int] = {
    "train": 100,  # samples per mode
    "validation": 20,
    "test": 40,
    "ood": 20,
    "analysis": 20,
}


class IncrementalPatternTask:
    """``tasks`` groups of ``classes_per_task`` multi-modal classes, learned in order."""

    task_id = "incremental_patterns.v1"
    task_version = "incremental-patterns-v1"

    def __init__(
        self,
        world: int,
        *,
        sigma: float,
        dims: int = 32,
        tasks: int = 5,
        classes_per_task: int = 4,
        modes: int = 3,
        frames: int = 3,
        active_probability: float = 0.3,
    ) -> None:
        if min(dims, tasks, classes_per_task, modes, frames) <= 0 or sigma <= 0:
            raise ValueError("invalid task size or noise")
        self.world = world
        self.sigma = sigma
        self.dims = dims
        self.tasks = tasks
        self.classes_per_task = classes_per_task
        self.classes = tasks * classes_per_task
        self.modes = modes
        self.frames = frames
        self.input_dim = dims + 1  # pattern channels plus a query flag
        self.num_classes = self.classes
        generator = torch.Generator().manual_seed(
            deterministic_seed(
                f"{self.task_version}/world", "train", world, split_seeds=_WORLD_SEEDS
            )
        )
        shape = (self.classes * modes, dims)
        active = torch.rand(shape, generator=generator) < active_probability
        values = torch.exp(0.5 * torch.randn(shape, generator=generator))
        self.prototypes = torch.where(active, values, torch.zeros(shape))  # row = class*modes+m

    def size(self, split: DatasetSplit) -> int:
        return SPLIT_SIZES[split] * self.classes * self.modes

    def label(self, sample_index: int) -> tuple[int, int]:
        """Class and mode of a sample (the layout is the same in every split)."""

        source = sample_index % (self.classes * self.modes)
        return source // self.modes, source % self.modes

    def task_of(self, label: int) -> int:
        return label // self.classes_per_task

    def indices(self, split: DatasetSplit, task: int) -> list[int]:
        """Sample indices of ``split`` whose class belongs to ``task``."""

        return [i for i in range(self.size(split)) if self.task_of(self.label(i)[0]) == task]

    @lru_cache(maxsize=65_536)  # noqa: B019 - bounded task instances live for one run
    def _frames(self, split: DatasetSplit, sample_index: int) -> Tensor:
        generator = make_generator(
            self.task_version, split, sample_index, split_seeds=T2_SPLIT_SEEDS
        )
        label, mode = self.label(sample_index)
        prototype = self.prototypes[label * self.modes + mode]
        noise = torch.randn((self.frames, self.dims), generator=generator)
        return prototype + self.sigma * noise

    def content_hash(self, split: DatasetSplit, sample_index: int) -> str:
        return stable_content_hash(
            self._frames(split, sample_index), prefix=f"{self.task_version}/{self.world}"
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
        batch = len(sample_indices)
        steps = self.frames + 1
        inputs = torch.zeros((batch, steps, self.input_dim), dtype=torch.float32)
        targets = torch.full((batch, steps), -100, dtype=torch.long)
        loss = torch.zeros((batch, steps), dtype=torch.bool)
        modes = torch.zeros(batch, dtype=torch.long)
        for b, index in enumerate(sample_indices):
            inputs[b, : self.frames, : self.dims] = self._frames(split, index)
            label, mode = self.label(index)
            targets[b, -1] = label
            modes[b] = mode
        inputs[:, -1, -1] = 1.0
        loss[:, -1] = True
        episodes = torch.tensor(
            [
                deterministic_seed(self.task_version, split, i, split_seeds=T2_SPLIT_SEEDS)
                for i in sample_indices
            ]
        )[:, None].expand(batch, steps)
        return TaskBatch(
            inputs=inputs,
            targets=targets,
            valid_mask=torch.ones((batch, steps), dtype=torch.bool),
            loss_mask=loss,
            episode_ids=episodes.clone(),
            metadata={
                "task_id": self.task_id,
                "task_version": self.task_version,
                "split": split,
                "split_seed": T2_SPLIT_SEEDS[split],
                "world": self.world,
                "sample_indices": tuple(sample_indices),
            },
            auxiliary_targets={"mode": modes[:, None].expand(batch, steps).clone()},
        ).to(device)

    def bayes_posterior(self, batch: TaskBatch, seen: int) -> Tensor:
        """Bayes-optimal class posterior over the first ``seen`` classes, ``[B, seen]``.

        The frame mean is ``N(prototype, sigma^2 / frames)``; modes are equally
        likely a priori and every seen class is equally likely.
        """

        mean = batch.inputs[:, : self.frames, : self.dims].mean(dim=1).cpu()
        prototypes = self.prototypes[: seen * self.modes]
        distance = torch.cdist(mean, prototypes).square()
        log_like = -self.frames * distance / (2 * self.sigma**2)
        per_class = torch.logsumexp(log_like.view(-1, seen, self.modes), dim=-1)
        return torch.softmax(per_class, dim=-1)


__all__ = ["SPLIT_SIZES", "T2_SPLIT_SEEDS", "IncrementalPatternTask"]
