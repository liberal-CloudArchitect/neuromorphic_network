"""Deterministic sequence tasks used by the P1 baselines."""

from neuromorphic.tasks.associative_recall import AssociativeRecallTask
from neuromorphic.tasks.base import DatasetSplit, SequenceTask, TaskBatch
from neuromorphic.tasks.control import TaskControl, task_control_from_batch
from neuromorphic.tasks.delayed_rule_switch import DelayedRuleSwitchTask
from neuromorphic.tasks.factory import TaskProfile, create_task
from neuromorphic.tasks.small_graph import SmallGraphTask

__all__ = [
    "AssociativeRecallTask",
    "DatasetSplit",
    "DelayedRuleSwitchTask",
    "SequenceTask",
    "SmallGraphTask",
    "TaskBatch",
    "TaskControl",
    "TaskProfile",
    "create_task",
    "task_control_from_batch",
]
