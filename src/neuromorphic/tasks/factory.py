"""Task construction with named smoke and qualification profiles."""

from __future__ import annotations

from typing import Literal

from neuromorphic.tasks.associative_recall import AssociativeRecallTask
from neuromorphic.tasks.base import SequenceTask
from neuromorphic.tasks.delayed_rule_switch import DelayedRuleSwitchTask
from neuromorphic.tasks.small_graph import SmallGraphTask

type TaskProfile = Literal["smoke", "qualification"]

_TASKS: dict[str, type[AssociativeRecallTask | DelayedRuleSwitchTask | SmallGraphTask]] = {
    AssociativeRecallTask.task_id: AssociativeRecallTask,
    DelayedRuleSwitchTask.task_id: DelayedRuleSwitchTask,
    SmallGraphTask.task_id: SmallGraphTask,
}


def create_task(task_id: str, *, profile: TaskProfile = "smoke") -> SequenceTask:
    """Create a task; profiles select run budgets, never sample content."""
    if profile not in ("smoke", "qualification"):
        raise ValueError(f"unknown task profile: {profile}")
    try:
        task_type = _TASKS[task_id]
    except KeyError as error:
        known = ", ".join(sorted(_TASKS))
        raise ValueError(f"unknown task_id {task_id!r}; expected one of: {known}") from error
    return task_type(profile=profile)
