"""P6 task controls: the three mainline tasks plus uncued rule switch (CR-015, E1).

The P6 goal context has 105 channels.  Channels 0-103 keep the frozen mainline
layout and channel 104 is the uncued-rule-switch task bit, so the task one-hot
is channels 69-71 plus 104.  Uncued steps copy their cue/delay/query event bits
into channels 0-2 exactly as Delayed Rule Switch does; feedback steps leave
them at zero.  The mainline ``TASK_IDS`` and ``TaskControl`` are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final

import torch
from torch import Tensor

from neuromorphic.tasks.base import TaskBatch
from neuromorphic.tasks.control import (
    ACTION_MASK_DIM,
    GOAL_CONTEXT_DIM,
    TASK_CLASS_COUNTS,
    TASK_IDS,
    TASK_INPUT_DIMS,
    TaskControl,
    task_control_from_batch,
)

UNCUED_RULE_SWITCH: Final = "uncued_rule_switch.v1"
P6_TASK_IDS: Final[tuple[str, ...]] = (*TASK_IDS, UNCUED_RULE_SWITCH)
P6_TASK_INPUT_DIMS: Final[dict[str, int]] = {**TASK_INPUT_DIMS, UNCUED_RULE_SWITCH: 11}
P6_TASK_CLASS_COUNTS: Final[dict[str, int]] = {**TASK_CLASS_COUNTS, UNCUED_RULE_SWITCH: 2}
UNCUED_TASK_CHANNEL: Final = GOAL_CONTEXT_DIM
P6_GOAL_CONTEXT_DIM: Final = GOAL_CONTEXT_DIM + 1


def p6_task_one_hot(goal_context: Tensor) -> Tensor:
    """Return the four task bits (AR, DRS, SmallGraph, uncued) of a P6 goal context."""

    if goal_context.shape[-1] < P6_GOAL_CONTEXT_DIM:
        raise ValueError(f"P6 goal_context needs {P6_GOAL_CONTEXT_DIM} channels")
    uncued = goal_context[..., UNCUED_TASK_CHANNEL : UNCUED_TASK_CHANNEL + 1]
    return torch.cat((goal_context[..., 69:72], uncued), dim=-1)


@dataclass(frozen=True, slots=True)
class P6TaskControl(TaskControl):
    """``TaskControl`` over the four P6 tasks and the 105-channel goal context."""

    supported_task_ids: ClassVar[tuple[str, ...]] = P6_TASK_IDS
    goal_context_dim: ClassVar[int] = P6_GOAL_CONTEXT_DIM


def p6_task_control_from_batch(batch: TaskBatch) -> P6TaskControl:
    """Build P6 controls from current observations only, never from targets."""

    batch.validate()
    task_id = batch.metadata.get("task_id")
    if not isinstance(task_id, str) or task_id not in P6_TASK_IDS:
        raise ValueError("TaskBatch metadata must contain a supported P6 task_id")
    if task_id != UNCUED_RULE_SWITCH:
        base = task_control_from_batch(batch)
        padding = base.goal_context.new_zeros((*base.goal_context.shape[:2], 1))
        return P6TaskControl(
            task_id=base.task_id,
            goal_context=torch.cat((base.goal_context, padding), dim=-1),
            valid_mask=base.valid_mask,
            loss_mask=base.loss_mask,
            action_nodes=base.action_nodes,
        )
    if batch.input_dim != P6_TASK_INPUT_DIMS[task_id]:
        raise ValueError(f"{task_id} requires input dimension {P6_TASK_INPUT_DIMS[task_id]}")

    valid = batch.valid_mask.to(batch.inputs.dtype)
    context = batch.inputs.new_zeros((*batch.inputs.shape[:2], P6_GOAL_CONTEXT_DIM))
    context[..., :3] = batch.inputs[..., :3]
    context[..., 3] = batch.loss_mask.to(batch.inputs.dtype)
    context[..., 4] = valid
    context[..., UNCUED_TASK_CHANNEL] = valid
    context = context * batch.valid_mask.unsqueeze(-1)
    action_nodes = torch.full(
        (*batch.inputs.shape[:2], ACTION_MASK_DIM),
        -1,
        dtype=torch.long,
        device=batch.inputs.device,
    )
    return P6TaskControl(
        task_id=task_id,
        goal_context=context,
        valid_mask=batch.valid_mask,
        loss_mask=batch.loss_mask,
        action_nodes=action_nodes,
    )


__all__ = [
    "P6_GOAL_CONTEXT_DIM",
    "P6_TASK_CLASS_COUNTS",
    "P6_TASK_IDS",
    "P6_TASK_INPUT_DIMS",
    "UNCUED_RULE_SWITCH",
    "UNCUED_TASK_CHANNEL",
    "P6TaskControl",
    "p6_task_control_from_batch",
    "p6_task_one_hot",
]
