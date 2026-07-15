"""Shared task contracts and deterministic generation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

type DatasetSplit = Literal["train", "validation", "test", "ood"]
type MetadataScalar = str | int | float | bool | None
type MetadataValue = MetadataScalar | tuple[MetadataScalar, ...]

CPU_DEVICE = torch.device("cpu")

SPLIT_SEEDS: Mapping[DatasetSplit, int] = {
    "train": 1_101,
    "validation": 2_201,
    "test": 3_301,
    "ood": 4_401,
}


def deterministic_seed(task_version: str, split: DatasetSplit, sample_index: int) -> int:
    """Return a stable torch seed independent of process RNG state."""
    if split not in SPLIT_SEEDS:
        raise ValueError(f"unknown dataset split: {split}")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    material = f"{task_version}:{SPLIT_SEEDS[split]}:{sample_index}".encode()
    return int.from_bytes(sha256(material).digest()[:8], "big") % (2**63 - 1)


def make_generator(task_version: str, split: DatasetSplit, sample_index: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(deterministic_seed(task_version, split, sample_index))
    return generator


def stable_content_hash(*tensors: Tensor, prefix: str) -> str:
    """Hash generated tensor content without depending on pickle serialization."""
    digest = sha256(prefix.encode())
    for tensor in tensors:
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TaskBatch:
    """Padded sequence batch shared by all P1 procedural tasks."""

    inputs: Tensor
    targets: Tensor
    valid_mask: Tensor
    loss_mask: Tensor
    episode_ids: Tensor
    metadata: Mapping[str, MetadataValue]
    auxiliary_targets: Mapping[str, Tensor]

    @property
    def batch_size(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.inputs.shape[1])

    @property
    def input_dim(self) -> int:
        return int(self.inputs.shape[2])

    def validate(self) -> None:
        if self.inputs.ndim != 3:
            raise ValueError("inputs must have shape [B, T, D]")
        batch, steps, features = self.inputs.shape
        if batch < 1 or steps < 1 or features < 1:
            raise ValueError("TaskBatch dimensions must be non-zero")
        expected = (batch, steps)
        for name, tensor in (
            ("targets", self.targets),
            ("valid_mask", self.valid_mask),
            ("loss_mask", self.loss_mask),
            ("episode_ids", self.episode_ids),
        ):
            if tensor.shape != expected:
                raise ValueError(f"{name} must have shape [B, T]")
            if tensor.device != self.inputs.device:
                raise ValueError(f"{name} must be on the inputs device")
        if not self.inputs.is_floating_point():
            raise TypeError("inputs must be floating point")
        if not torch.isfinite(self.inputs).all():
            raise ValueError("inputs contain non-finite values")
        if self.targets.dtype != torch.long:
            raise TypeError("targets must use torch.long")
        if self.valid_mask.dtype != torch.bool or self.loss_mask.dtype != torch.bool:
            raise TypeError("valid_mask and loss_mask must use torch.bool")
        if self.episode_ids.dtype != torch.long:
            raise TypeError("episode_ids must use torch.long")
        if torch.any(self.loss_mask & ~self.valid_mask):
            raise ValueError("loss_mask cannot select padded positions")
        if steps > 1 and torch.any(self.valid_mask[:, 1:] & ~self.valid_mask[:, :-1]):
            raise ValueError("valid_mask must describe one contiguous prefix per episode")
        if torch.any(self.episode_ids[self.valid_mask] < 0):
            raise ValueError("valid positions require non-negative episode IDs")
        if torch.any(self.episode_ids[~self.valid_mask] != -1):
            raise ValueError("padded positions must use episode ID -1")
        for batch_index in range(batch):
            valid_episode_ids = self.episode_ids[batch_index, self.valid_mask[batch_index]]
            if torch.unique(valid_episode_ids).numel() != 1:
                raise ValueError("each batch row must contain exactly one episode")
        if "task_version" not in self.metadata or "split" not in self.metadata:
            raise ValueError("metadata must include task_version and split")
        for name, tensor in self.auxiliary_targets.items():
            if tensor.device != self.inputs.device:
                raise ValueError(f"auxiliary target {name!r} must be on the inputs device")
            if tensor.ndim >= 2 and tensor.shape[:2] != expected:
                raise ValueError(f"sequence auxiliary target {name!r} must start with [B, T]")

    def to(self, device: torch.device) -> TaskBatch:
        moved = replace(
            self,
            inputs=self.inputs.to(device),
            targets=self.targets.to(device),
            valid_mask=self.valid_mask.to(device),
            loss_mask=self.loss_mask.to(device),
            episode_ids=self.episode_ids.to(device),
            auxiliary_targets={
                name: tensor.to(device) for name, tensor in self.auxiliary_targets.items()
            },
        )
        moved.validate()
        return moved


@runtime_checkable
class SequenceTask(Protocol):
    """Contract implemented by deterministic P1 sequence generators."""

    task_id: str
    task_version: str
    input_dim: int
    num_classes: int

    def generate(
        self,
        split: DatasetSplit,
        sample_indices: Sequence[int],
        *,
        device: torch.device = CPU_DEVICE,
    ) -> TaskBatch: ...

    def content_hash(self, split: DatasetSplit, sample_index: int) -> str: ...

    def oracle(self, batch: TaskBatch) -> Tensor: ...
