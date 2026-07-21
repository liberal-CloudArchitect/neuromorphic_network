from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import yaml

from neuromorphic.tasks import create_task

TASK_IDS = (
    "associative_recall.v1",
    "delayed_rule_switch.v1",
    "small_graph.v1",
)


def test_generation_is_deterministic_and_profile_independent() -> None:
    for task_id in TASK_IDS:
        smoke = create_task(task_id, profile="smoke")
        qualification = create_task(task_id, profile="qualification")
        first = smoke.generate("validation", [3, 9, 27])
        second = smoke.generate("validation", [3, 9, 27])
        other_profile = qualification.generate("validation", [3, 9, 27])
        assert torch.equal(first.inputs, second.inputs)
        assert torch.equal(first.targets, second.targets)
        assert torch.equal(first.inputs, other_profile.inputs)
        assert first.metadata["content_hashes"] == second.metadata["content_hashes"]


def test_splits_have_distinct_episode_ids_and_content() -> None:
    indices = list(range(64))
    for task_id in TASK_IDS:
        task = create_task(task_id)
        hashes: dict[str, set[object]] = {}
        episode_ids: dict[str, set[int]] = {}
        for split in ("train", "validation", "test", "ood"):
            batch = task.generate(split, indices)
            content_hashes = cast(tuple[str, ...], batch.metadata["content_hashes"])
            hashes[split] = set(content_hashes)
            episode_ids[split] = set(batch.episode_ids[batch.valid_mask].tolist())
        for left_index, left in enumerate(hashes):
            for right in list(hashes)[left_index + 1 :]:
                assert hashes[left].isdisjoint(hashes[right])
                assert episode_ids[left].isdisjoint(episode_ids[right])


def test_p4_namespace_has_fresh_versions_and_split_seeds() -> None:
    for task_id in TASK_IDS:
        legacy = create_task(task_id)
        p4 = create_task(task_id, namespace="p4")
        legacy_batch = legacy.generate("test", [0])
        p4_batch = p4.generate("test", [0])
        assert "-p4-" in p4.task_version
        assert p4_batch.metadata["split_seed"] == 13_301
        assert legacy_batch.metadata["content_hashes"] != p4_batch.metadata["content_hashes"]


def test_delayed_switch_profile_matches_the_frozen_protocol() -> None:
    profile = yaml.safe_load(Path("configs/tasks/profiles.yaml").read_text(encoding="utf-8"))
    specification = profile["tasks"]["delayed_rule_switch.v1"]
    assert specification["id"]["switch_positions"] == [2]
    assert specification["ood"]["switch_positions"] == [3]
    task = create_task("delayed_rule_switch.v1")
    train = task.generate("train", list(range(32)))
    ood = task.generate("ood", list(range(32)))
    train_switches = cast(tuple[int | None, ...], train.metadata["switch_positions"])
    ood_switches = cast(tuple[int | None, ...], ood.metadata["switch_positions"])
    assert set(train_switches) <= {None, 2}
    assert set(ood_switches) == {3}
