"""Local associative readout with a frozen encoder for class-incremental learning (T2).

F1a found that strengthening only the "active unit -> correct class" synapses
and leaving every other synapse untouched almost removes catastrophic
forgetting, while SGD and perceptron readouts of the same code forget.  Here
the principle is re-implemented on main-network modules:

- ``FrameEncoder``: task adapter -> ``sensory_encoder.v1`` on every frame,
  averaged over the frames of a sample (the query step is excluded);
- ``SparseExpansion``: a fixed random projection in which each unit sums
  ``fan_in`` encoder features, followed by k-winners-take-all to a binary code;
- ``AssociativeReadout``: ``W[c] += code`` for every training sample of class
  ``c``; nothing else changes, so learning a new class cannot overwrite an old
  one.  Prediction is ``argmax_c code · W[c]`` over the classes seen so far.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from neuromorphic.core.contracts import ModuleContext
from neuromorphic.modules.sensory_encoder import SensoryEncoder
from neuromorphic.tasks.base import TaskBatch
from neuromorphic.transfer.surprise_wm import selector_head


class FrameEncoder(nn.Module):
    """Adapter -> ``sensory_encoder.v1`` per step, mean over the frame steps."""

    def __init__(self, input_dim: int, *, feature_dim: int = 64) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.adapter = nn.Linear(input_dim, feature_dim)
        self.encoder = SensoryEncoder(feature_dim=feature_dim)

    def forward(self, batch: TaskBatch) -> Tensor:
        steps = batch.inputs.shape[1]
        reset = torch.zeros_like(batch.valid_mask)
        reset[:, 0] = True
        context = ModuleContext(
            task_id="incremental_patterns.v1", phase="train", reset_mask=reset, eligible_modules=()
        )
        index = torch.arange(steps, device=batch.inputs.device).expand_as(batch.valid_mask)
        output = self.encoder.forward_inputs(
            self.adapter(batch.inputs), batch.valid_mask, index, None, context
        )
        frames = (batch.inputs[..., -1] == 0) & batch.valid_mask  # the query step is flagged
        weights = frames.float() / frames.sum(dim=1, keepdim=True).clamp(min=1)
        return torch.einsum("bt,btd->bd", weights, output.packet.representation)


class SparseExpansion(nn.Module):
    """Fixed random fan-in projection plus k-winners-take-all to a binary code."""

    weight: Tensor

    def __init__(
        self, feature_dim: int, units: int, *, fan_in: int, active: int, seed: int
    ) -> None:
        super().__init__()
        if not 0 < active <= units or not 0 < fan_in <= feature_dim:
            raise ValueError("invalid expansion size")
        generator = torch.Generator().manual_seed(seed)
        weight = torch.zeros(units, feature_dim)
        for unit in range(units):
            weight[unit, torch.randperm(feature_dim, generator=generator)[:fan_in]] = 1.0
        self.register_buffer("weight", weight)
        self.active = active

    def forward(self, features: Tensor) -> Tensor:
        drive = features @ self.weight.T
        top = torch.topk(drive, self.active, dim=-1).indices
        return torch.zeros_like(drive).scatter_(-1, top, 1.0)


class AssociativeReadout(nn.Module):
    """``W[c] += code`` for each training sample of class ``c``; nothing else changes."""

    weight: Tensor

    def __init__(self, units: int, classes: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.zeros(classes, units))

    @torch.no_grad()
    def learn(self, code: Tensor, labels: Tensor) -> None:
        self.weight.index_add_(0, labels, code)

    def scores(self, code: Tensor) -> Tensor:
        return code @ self.weight.T


class ModularClassifier(nn.Module):
    """``FrameEncoder`` -> ``action_selector.v1`` head (gradient-trained readout)."""

    def __init__(self, input_dim: int, classes: int, *, feature_dim: int = 64) -> None:
        super().__init__()
        self.frames = FrameEncoder(input_dim, feature_dim=feature_dim)
        self.head = selector_head(feature_dim, classes)

    def forward(self, batch: TaskBatch) -> Tensor:
        logits: Tensor = self.head(self.frames(batch))
        return logits


class GRUClassifier(nn.Module):
    """Shared-GRU baseline: adapter -> GRU -> selector head at the query step."""

    def __init__(self, input_dim: int, classes: int, *, hidden: int) -> None:
        super().__init__()
        self.adapter = nn.Linear(input_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = selector_head(hidden, classes)

    def forward(self, batch: TaskBatch) -> Tensor:
        output, _ = self.gru(self.adapter(batch.inputs))
        logits: Tensor = self.head(output[:, -1])
        return logits


__all__ = [
    "AssociativeReadout",
    "FrameEncoder",
    "GRUClassifier",
    "ModularClassifier",
    "SparseExpansion",
]
