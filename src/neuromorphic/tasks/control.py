"""Leakage-safe task controls for the modular P2 network.

The controls are derived exclusively from observations and masks available at
the current step.  Targets and future observations are deliberately absent
from this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

import torch
from torch import Tensor

from neuromorphic.core.contracts import internal_execution_is_trusted
from neuromorphic.tasks.base import TaskBatch

ASSOCIATIVE_RECALL: Final = "associative_recall.v1"
DELAYED_RULE_SWITCH: Final = "delayed_rule_switch.v1"
SMALL_GRAPH: Final = "small_graph.v1"

TASK_IDS: Final[tuple[str, ...]] = (
    ASSOCIATIVE_RECALL,
    DELAYED_RULE_SWITCH,
    SMALL_GRAPH,
)
TASK_INPUT_DIMS: Final[dict[str, int]] = {
    ASSOCIATIVE_RECALL: 68,
    DELAYED_RULE_SWITCH: 9,
    SMALL_GRAPH: 356,
}
TASK_CLASS_COUNTS: Final[dict[str, int]] = {
    ASSOCIATIVE_RECALL: 32,
    DELAYED_RULE_SWITCH: 2,
    SMALL_GRAPH: 4,
}

EVENT_DIM: Final = 5
KEY_DIM: Final = 32
ACTION_MASK_DIM: Final = 32
TASK_DIM: Final = 3
ACTION_COPY_DIM: Final = 32
GOAL_CONTEXT_DIM: Final = EVENT_DIM + KEY_DIM + ACTION_MASK_DIM + TASK_DIM + ACTION_COPY_DIM

_EVENT = slice(0, 5)
_KEY = slice(5, 37)
_ACTION_MASK = slice(37, 69)
_TASK = slice(69, 72)
_ACTION_COPY = slice(72, 104)


@dataclass(frozen=True, slots=True)
class TaskControl:
    """Current-step controls supplied to the modular network.

    ``action_nodes`` is decoded from the SmallGraph observation rather than
    copied from an auxiliary target.  ``-1`` denotes an unavailable slot.
    """

    task_id: str
    goal_context: Tensor
    valid_mask: Tensor
    loss_mask: Tensor
    action_nodes: Tensor

    def __post_init__(self) -> None:
        if self.task_id not in TASK_IDS:
            raise ValueError(f"unsupported task_id: {self.task_id}")
        if self.goal_context.ndim != 3 or self.goal_context.shape[-1] != GOAL_CONTEXT_DIM:
            raise ValueError("goal_context must have shape [B, T, 104]")
        expected = self.goal_context.shape[:2]
        if self.valid_mask.shape != expected or self.loss_mask.shape != expected:
            raise ValueError("valid_mask and loss_mask must have shape [B, T]")
        if self.action_nodes.shape != (*expected, ACTION_MASK_DIM):
            raise ValueError("action_nodes must have shape [B, T, 32]")
        if not self.goal_context.is_floating_point():
            raise TypeError("goal_context must use a floating-point dtype")
        if self.valid_mask.dtype is not torch.bool or self.loss_mask.dtype is not torch.bool:
            raise TypeError("valid_mask and loss_mask must use torch.bool")
        if self.action_nodes.dtype is not torch.long:
            raise TypeError("action_nodes must use torch.long")
        device = self.goal_context.device
        if self.valid_mask.device != device or self.loss_mask.device != device:
            raise ValueError("control tensors must share a device")
        if self.action_nodes.device != device:
            raise ValueError("action_nodes and goal_context must share a device")
        if (
            not internal_execution_is_trusted()
            and torch.any(self.loss_mask & ~self.valid_mask).item()
        ):
            raise ValueError("loss_mask cannot select padding")
        if (
            not internal_execution_is_trusted()
            and not torch.isfinite(self.goal_context).all().item()
        ):
            raise ValueError("goal_context must contain only finite values")

    @property
    def action_valid_mask(self) -> Tensor:
        """Return the 32-slot action availability view."""

        return self.goal_context[..., _ACTION_MASK].to(dtype=torch.bool)

    @property
    def selected_action_copy(self) -> Tensor:
        """Return the one-hot copy of the model-selected action."""

        return self.goal_context[..., _ACTION_COPY]

    def at_step(self, step: int) -> TaskControl:
        """Select one time step while preserving the sequence dimension."""

        if step < 0 or step >= self.goal_context.shape[1]:
            raise IndexError("step is outside the task control sequence")
        return TaskControl(
            task_id=self.task_id,
            goal_context=self.goal_context[:, step : step + 1],
            valid_mask=self.valid_mask[:, step : step + 1],
            loss_mask=self.loss_mask[:, step : step + 1],
            action_nodes=self.action_nodes[:, step : step + 1],
        )

    def with_selected_action(self, action: Tensor) -> TaskControl:
        """Return a control whose final 32 channels copy a selected action.

        The selected action comes from model logits. Invalid or padded rows
        retain an all-zero action copy.
        """

        if action.shape == (*self.valid_mask.shape, 1):
            action = action.squeeze(-1)
        if action.shape != self.valid_mask.shape:
            raise ValueError("action must have shape [B, T]")
        if action.dtype is not torch.long:
            raise TypeError("action must use torch.long")
        if action.device != self.goal_context.device:
            raise ValueError("action and goal_context must share a device")
        if (
            not internal_execution_is_trusted()
            and torch.any((action < 0) | (action >= ACTION_COPY_DIM)).item()
        ):
            raise ValueError("action values must be in [0, 31]")
        copied = torch.nn.functional.one_hot(action, ACTION_COPY_DIM).to(self.goal_context.dtype)
        copied = copied * self.valid_mask.unsqueeze(-1)
        goal_context = torch.cat((self.goal_context[..., :72], copied), dim=-1)
        return replace(self, goal_context=goal_context)


def _decode_small_graph_actions(inputs: Tensor) -> tuple[Tensor, Tensor]:
    """Decode the four node-valued action slots from a SmallGraph observation."""

    batch, steps, _ = inputs.shape
    slots = inputs[..., 288:352].reshape(batch, steps, 4, 16)
    valid = inputs[..., 352:356] > 0.5
    node_ids = slots.argmax(dim=-1).to(torch.long)
    node_ids = torch.where(valid, node_ids, torch.full_like(node_ids, -1))
    action_nodes = torch.full(
        (batch, steps, ACTION_MASK_DIM),
        -1,
        dtype=torch.long,
        device=inputs.device,
    )
    action_nodes[..., :4] = node_ids
    action_mask = torch.zeros(
        (batch, steps, ACTION_MASK_DIM),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    action_mask[..., :4] = valid.to(inputs.dtype)
    return action_nodes, action_mask


def task_control_from_batch(batch: TaskBatch) -> TaskControl:
    """Build current-observation controls without inspecting targets.

    ``TaskBatch.metadata['task_id']`` selects the deterministic observation
    layout. Neither ``targets`` nor any auxiliary target is read here.
    """

    batch.validate()
    task_id_value = batch.metadata.get("task_id")
    if not isinstance(task_id_value, str) or task_id_value not in TASK_IDS:
        raise ValueError("TaskBatch metadata must contain a supported task_id")
    if batch.input_dim != TASK_INPUT_DIMS[task_id_value]:
        raise ValueError(
            f"{task_id_value} requires input dimension {TASK_INPUT_DIMS[task_id_value]}"
        )

    context = torch.zeros(
        (*batch.inputs.shape[:2], GOAL_CONTEXT_DIM),
        dtype=batch.inputs.dtype,
        device=batch.inputs.device,
    )
    context[..., 3] = batch.loss_mask.to(batch.inputs.dtype)
    context[..., 4] = batch.valid_mask.to(batch.inputs.dtype)
    action_nodes = torch.full(
        (*batch.inputs.shape[:2], ACTION_MASK_DIM),
        -1,
        dtype=torch.long,
        device=batch.inputs.device,
    )

    if task_id_value == ASSOCIATIVE_RECALL:
        context[..., :3] = batch.inputs[..., :3]
        context[..., _KEY] = batch.inputs[..., 4:36]
    elif task_id_value == DELAYED_RULE_SWITCH:
        context[..., :3] = batch.inputs[..., :3]
    else:
        # Every non-terminal graph observation is a potential decision.  The
        # valid/loss channels still distinguish terminal observations.
        context[..., 2] = batch.valid_mask.to(batch.inputs.dtype)
        action_nodes, action_mask = _decode_small_graph_actions(batch.inputs)
        context[..., _ACTION_MASK] = action_mask

    task_index = TASK_IDS.index(task_id_value)
    context[..., 69 + task_index] = batch.valid_mask.to(batch.inputs.dtype)
    context = context * batch.valid_mask.unsqueeze(-1)
    action_nodes = torch.where(
        batch.valid_mask.unsqueeze(-1), action_nodes, torch.full_like(action_nodes, -1)
    )
    return TaskControl(
        task_id=task_id_value,
        goal_context=context,
        valid_mask=batch.valid_mask,
        loss_mask=batch.loss_mask,
        action_nodes=action_nodes,
    )


__all__ = [
    "ACTION_COPY_DIM",
    "ACTION_MASK_DIM",
    "ASSOCIATIVE_RECALL",
    "DELAYED_RULE_SWITCH",
    "EVENT_DIM",
    "GOAL_CONTEXT_DIM",
    "KEY_DIM",
    "SMALL_GRAPH",
    "TASK_CLASS_COUNTS",
    "TASK_DIM",
    "TASK_IDS",
    "TASK_INPUT_DIMS",
    "TaskControl",
    "task_control_from_batch",
]
