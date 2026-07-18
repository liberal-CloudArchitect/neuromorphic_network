"""Shared multi-task baselines used by the P3 confirmatory comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn

from neuromorphic.tasks.base import TaskBatch
from neuromorphic.training.baselines import BaselineOutput, TransformerBaseline

P3_TASK_DIMS = {
    "associative_recall.v1": (68, 32),
    "delayed_rule_switch.v1": (9, 2),
    "small_graph.v1": (356, 4),
}


def _task_id(batch: TaskBatch) -> str:
    value = batch.metadata.get("task_id")
    if not isinstance(value, str) or value not in P3_TASK_DIMS:
        raise ValueError("P3 batch has an unsupported task_id")
    return value


def _module_key(task_id: str) -> str:
    return task_id.replace(".", "__")


def sinusoidal_positions(length: int, width: int, *, device: torch.device) -> Tensor:
    """Return deterministic sinusoidal positions with shape ``[1, T, H]``."""

    if length < 1 or width < 1:
        raise ValueError("position dimensions must be positive")
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    result = torch.zeros((length, width), device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(positions * frequencies)
    if width > 1:
        result[:, 1::2] = torch.cos(positions * frequencies[: result[:, 1::2].shape[1]])
    return result.unsqueeze(0)


class SharedTaskIO(nn.Module):
    """Three task adapters and heads shared by both monolithic backbones."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict(
            {
                _module_key(task_id): nn.Linear(input_dim, hidden_size)
                for task_id, (input_dim, _) in P3_TASK_DIMS.items()
            }
        )
        self.heads = nn.ModuleDict(
            {
                _module_key(task_id): nn.Linear(hidden_size, classes)
                for task_id, (_, classes) in P3_TASK_DIMS.items()
            }
        )
        self.next_state_head = nn.Linear(hidden_size, 16)

    def encode(self, batch: TaskBatch) -> Tensor:
        return cast(Tensor, self.adapters[_module_key(_task_id(batch))](batch.inputs))

    def decode(self, encoded: Tensor, task_id: str) -> BaselineOutput:
        next_state = self.next_state_head(encoded) if task_id == "small_graph.v1" else None
        return BaselineOutput(
            logits=self.heads[_module_key(task_id)](encoded), next_state_logits=next_state
        )


class SharedGRUBaseline(nn.Module):
    """One recurrent backbone with P3 task-specific boundaries."""

    model_id = "gru-shared-v1"

    def __init__(self, hidden_size: int = 128, layers: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.io = SharedTaskIO(hidden_size)
        self.backbone = nn.GRU(hidden_size, hidden_size, num_layers=layers, batch_first=True)

    def forward(self, batch: TaskBatch) -> BaselineOutput:
        encoded = self.encode_representation(batch)
        return self.io.decode(encoded, _task_id(batch))

    def encode_representation(self, batch: TaskBatch) -> Tensor:
        """Return the shared recurrent representation for registered analysis."""

        batch.validate()
        encoded, _ = self.backbone(self.io.encode(batch))
        return cast(Tensor, encoded)


class SharedTransformerBaseline(nn.Module):
    """Causal Transformer-v2 with deterministic position information."""

    model_id = "transformer-shared-v2"

    def __init__(
        self,
        hidden_size: int = 128,
        layers: int = 2,
        heads: int = 4,
        feedforward_size: int = 512,
    ) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.hidden_size = hidden_size
        self.layers = layers
        self.heads = heads
        self.feedforward_size = feedforward_size
        self.io = SharedTaskIO(hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=feedforward_size,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)

    def forward(self, batch: TaskBatch) -> BaselineOutput:
        encoded = self.encode_representation(batch)
        return self.io.decode(encoded, _task_id(batch))

    def encode_representation(self, batch: TaskBatch) -> Tensor:
        """Return the position-aware causal representation for registered analysis."""

        batch.validate()
        length = batch.sequence_length
        causal = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=batch.inputs.device),
            diagonal=1,
        )
        encoded = self.io.encode(batch)
        encoded = encoded + sinusoidal_positions(
            length, self.hidden_size, device=batch.inputs.device
        ).to(encoded.dtype)
        encoded = self.backbone(encoded, mask=causal, src_key_padding_mask=~batch.valid_mask)
        return cast(Tensor, encoded)


class SingleTaskTransformerV2(TransformerBaseline):
    """P1-compatible single-task Transformer with P3 position encoding."""

    model_id = "transformer-monolithic-v2"

    def forward(self, inputs: Tensor, valid_mask: Tensor) -> BaselineOutput:
        encoded = self.encode_representation(inputs, valid_mask)
        next_state = None if self.next_state_head is None else self.next_state_head(encoded)
        return BaselineOutput(logits=self.output_head(encoded), next_state_logits=next_state)

    def encode_representation(self, inputs: Tensor, valid_mask: Tensor) -> Tensor:
        """Return the exact Transformer-v2 hidden sequence used by task heads."""

        if inputs.ndim != 3 or valid_mask.shape != inputs.shape[:2]:
            raise ValueError("invalid baseline input or mask shape")
        length = inputs.shape[1]
        causal = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=inputs.device), diagonal=1
        )
        encoded = self.input_projection(inputs)
        encoded = encoded + sinusoidal_positions(length, self.hidden_size, device=inputs.device).to(
            encoded.dtype
        )
        encoded = self.encoder(encoded, mask=causal, src_key_padding_mask=~valid_mask)
        return cast(Tensor, encoded)


@dataclass(frozen=True, slots=True)
class SharedParameterMatch:
    model: nn.Module
    target: int
    actual: int
    relative_error: float
    hidden_size: int
    feedforward_size: int | None


def select_shared_parameter_match(
    kind: Literal["gru", "transformer"], target: int, tolerance: float = 0.05
) -> SharedParameterMatch:
    """Deterministically choose the closest legal shared baseline."""

    if target <= 0 or not 0 < tolerance <= 0.25:
        raise ValueError("invalid parameter matching target or tolerance")
    candidates: list[tuple[int, int, int | None, nn.Module]] = []
    if kind == "gru":
        for hidden in range(16, 385):
            model: nn.Module = SharedGRUBaseline(hidden_size=hidden)
            candidates.append((sum(p.numel() for p in model.parameters()), hidden, None, model))
    else:
        for hidden in range(16, 385, 4):
            for multiplier in (2, 4, 8):
                feedforward = hidden * multiplier
                model = SharedTransformerBaseline(hidden_size=hidden, feedforward_size=feedforward)
                candidates.append(
                    (sum(p.numel() for p in model.parameters()), hidden, feedforward, model)
                )
    actual, hidden, selected_feedforward, model = min(
        candidates, key=lambda item: (abs(item[0] - target), item[0], item[1])
    )
    error = abs(actual - target) / target
    if error > tolerance:
        raise ValueError(
            f"shared parameter target mismatch: target={target}, actual={actual}, error={error:.3f}"
        )
    return SharedParameterMatch(model, target, actual, error, hidden, selected_feedforward)


__all__ = [
    "P3_TASK_DIMS",
    "SharedGRUBaseline",
    "SharedParameterMatch",
    "SharedTransformerBaseline",
    "SingleTaskTransformerV2",
    "select_shared_parameter_match",
    "sinusoidal_positions",
]
