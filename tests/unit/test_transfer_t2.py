from __future__ import annotations

import torch

from neuromorphic.tasks.base import DatasetSplit
from neuromorphic.transfer.associative_readout import (
    AssociativeReadout,
    FrameEncoder,
    GRUClassifier,
    ModularClassifier,
    SparseExpansion,
)
from neuromorphic.transfer.incremental_patterns import IncrementalPatternTask
from neuromorphic.transfer.surprise_wm import parameter_count


def test_task_layout_splits_and_tasks() -> None:
    task = IncrementalPatternTask(5, sigma=0.5)
    assert task.size("train") == 100 * 20 * 3
    assert torch.equal(task.prototypes, IncrementalPatternTask(5, sigma=0.5).prototypes)
    assert not torch.equal(task.prototypes, IncrementalPatternTask(6, sigma=0.5).prototypes)
    splits: tuple[DatasetSplit, ...] = ("train", "validation", "test", "analysis")
    hashes = {s: {task.content_hash(s, i) for i in range(60)} for s in splits}
    for a in splits:
        for b in splits:
            assert a == b or not (hashes[a] & hashes[b])
    pool = task.indices("train", 2)
    labels = {task.label(i)[0] for i in pool}
    assert labels == {8, 9, 10, 11} and len(pool) == 4 * 3 * 100
    batch = task.generate("test", pool[:7])
    assert batch.inputs.shape == (7, 4, 33)
    assert bool((batch.inputs[:, -1, -1] == 1).all()) and bool((batch.inputs[:, :3, -1] == 0).all())
    assert bool(batch.loss_mask[:, -1].all()) and not bool(batch.loss_mask[:, :3].any())


def test_bayes_posterior_is_near_perfect_at_low_noise_and_class_means_fail() -> None:
    task = IncrementalPatternTask(7, sigma=0.5)
    batch = task.generate("validation", list(range(task.size("validation"))))
    labels = batch.targets[:, -1]
    posterior = task.bayes_posterior(batch, task.classes)
    assert float(posterior.argmax(dim=-1).eq(labels).float().mean()) > 0.99
    torch.testing.assert_close(posterior.sum(dim=-1), torch.ones(len(labels)))
    train = task.generate("train", list(range(task.size("train"))))
    x = train.inputs[:, :3, :32].mean(dim=1)
    means = torch.stack([x[train.targets[:, -1] == c].mean(dim=0) for c in range(20)])
    mean_x = batch.inputs[:, :3, :32].mean(dim=1)
    assert float(torch.cdist(mean_x, means).argmin(dim=1).eq(labels).float().mean()) < 0.9


def test_associative_readout_only_changes_the_trained_class() -> None:
    expansion = SparseExpansion(16, 200, fan_in=4, active=10, seed=0)
    code = expansion(torch.randn(5, 16))
    assert torch.equal(code.sum(dim=-1), torch.full((5,), 10.0))
    assert set(code.unique().tolist()) <= {0.0, 1.0}
    readout = AssociativeReadout(200, 6)
    readout.learn(code[:3], torch.tensor([1, 1, 4]))
    before = readout.weight.clone()
    readout.learn(code[3:], torch.tensor([2, 2]))
    changed = (readout.weight != before).any(dim=-1)
    assert changed.tolist() == [False, False, True, False, False, False]
    torch.testing.assert_close(readout.weight[2], code[3:].sum(dim=0))


def test_frame_encoder_ignores_the_query_step_and_models_are_parameter_matched() -> None:
    task = IncrementalPatternTask(9, sigma=0.5)
    batch = task.generate("train", [0, 1, 2])
    torch.manual_seed(0)
    encoder = FrameEncoder(task.input_dim)
    pooled = encoder(batch)
    altered = batch.inputs.clone()
    altered[:, -1, :32] = torch.randn(3, 32)  # the query step carries no pattern
    other = encoder(
        type(batch)(
            **{**{f: getattr(batch, f) for f in batch.__dataclass_fields__}, "inputs": altered}
        )
    )
    torch.testing.assert_close(pooled, other)
    modular = parameter_count(ModularClassifier(task.input_dim, 20))
    gru = parameter_count(GRUClassifier(task.input_dim, 20, hidden=56))
    assert abs(gru / modular - 1) < 0.06
    assert abs(20 * 1280 / gru - 1) < 0.06
