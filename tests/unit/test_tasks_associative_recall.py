from __future__ import annotations

import torch

from neuromorphic.tasks import AssociativeRecallTask


def test_associative_recall_oracle_answers_every_query() -> None:
    task = AssociativeRecallTask()
    batch = task.generate("train", list(range(16)))
    oracle = task.oracle(batch)
    assert torch.equal(oracle[batch.loss_mask], batch.targets[batch.loss_mask])
    assert torch.all(batch.loss_mask.sum(dim=1) == 1)


def test_associative_recall_id_and_ood_ranges() -> None:
    task = AssociativeRecallTask()
    train = task.generate("train", list(range(32)))
    ood = task.generate("ood", list(range(32)))
    train_store = train.inputs[:, :, 0].sum(dim=1)
    train_distractors = train.inputs[:, :, 1].sum(dim=1)
    ood_store = ood.inputs[:, :, 0].sum(dim=1)
    ood_distractors = ood.inputs[:, :, 1].sum(dim=1)
    assert torch.all((train_store >= 4) & (train_store <= 8))
    assert torch.all((train_distractors >= 0) & (train_distractors <= 4))
    assert torch.all((ood_store >= 9) & (ood_store <= 12))
    assert torch.all((ood_distractors >= 5) & (ood_distractors <= 8))
    for batch in (train, ood):
        for sample in batch.inputs:
            store_rows = sample[:, 0].bool()
            distractor_rows = sample[:, 1].bool()
            query_rows = sample[:, 2].bool()
            store_keys = set(sample[store_rows, 4:36].argmax(dim=-1).tolist())
            distractor_keys = set(sample[distractor_rows, 4:36].argmax(dim=-1).tolist())
            query_key = int(sample[query_rows, 4:36].argmax(dim=-1).item())
            assert distractor_keys.isdisjoint(store_keys)
            assert query_key in store_keys
