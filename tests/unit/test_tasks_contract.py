from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from neuromorphic.tasks import TaskBatch, create_task
from neuromorphic.tasks.base import SequenceTask


@pytest.mark.parametrize(
    "task_id",
    ["associative_recall.v1", "delayed_rule_switch.v1", "small_graph.v1"],
)
def test_task_factory_and_batch_contract(task_id: str) -> None:
    task = create_task(task_id)
    assert isinstance(task, SequenceTask)
    batch = task.generate("train", [0, 1, 2])
    batch.validate()
    assert batch.inputs.shape[:2] == batch.targets.shape
    assert batch.inputs.shape[-1] == task.input_dim
    assert torch.all(batch.loss_mask <= batch.valid_mask)
    assert torch.all(batch.targets[batch.loss_mask] >= 0)
    assert torch.all(batch.episode_ids[~batch.valid_mask] == -1)


def test_batch_validation_rejects_loss_on_padding() -> None:
    task = create_task("associative_recall.v1")
    batch = task.generate("train", [0, 1])
    invalid_loss_mask = batch.loss_mask.clone()
    padded = torch.nonzero(~batch.valid_mask, as_tuple=False)
    assert padded.numel() > 0
    invalid_loss_mask[tuple(padded[0])] = True
    invalid = replace(batch, loss_mask=invalid_loss_mask)
    with pytest.raises(ValueError, match="padded"):
        invalid.validate()


def test_batch_validation_rejects_noncontiguous_sequence_boundary() -> None:
    task = create_task("associative_recall.v1")
    batch = task.generate("train", [0, 1])
    valid_mask = batch.valid_mask.clone()
    valid_mask[0, 1] = False
    invalid = replace(batch, valid_mask=valid_mask)
    with pytest.raises(ValueError, match="contiguous prefix"):
        invalid.validate()


def test_batch_validation_rejects_cross_episode_state_boundary() -> None:
    task = create_task("delayed_rule_switch.v1")
    batch = task.generate("train", [0])
    episode_ids = batch.episode_ids.clone()
    episode_ids[0, 1] += 1
    invalid = replace(batch, episode_ids=episode_ids)
    with pytest.raises(ValueError, match="exactly one episode"):
        invalid.validate()


def test_batch_device_copy_preserves_content() -> None:
    batch = create_task("delayed_rule_switch.v1").generate("test", [5, 8])
    copied = batch.to(torch.device("cpu"))
    assert isinstance(copied, TaskBatch)
    assert torch.equal(copied.inputs, batch.inputs)
    assert copied.metadata == batch.metadata


def test_factory_rejects_unknown_task_and_profile() -> None:
    with pytest.raises(ValueError, match="unknown task_id"):
        create_task("not-a-task")
    with pytest.raises(ValueError, match="unknown task profile"):
        create_task("associative_recall.v1", profile="large")  # type: ignore[arg-type]
