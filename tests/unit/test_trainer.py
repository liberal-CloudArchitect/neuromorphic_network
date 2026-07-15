from __future__ import annotations

from pathlib import Path

import torch

from neuromorphic.tasks import create_task
from neuromorphic.training.baselines import GRUBaseline
from neuromorphic.training.config import RunConfig
from neuromorphic.training.reproducibility import set_global_seed
from neuromorphic.training.trainer import IndexSampler, train_baseline


def test_sampler_restore_produces_identical_indices() -> None:
    sampler = IndexSampler.create(10, 7)
    sampler.next(7)
    state = sampler.state_dict()
    expected = sampler.next(15)
    restored = IndexSampler.create(10, 7)
    restored.load_state_dict(state)
    assert restored.next(15) == expected


def test_short_training_run_writes_reproducible_artifacts(tmp_path: Path) -> None:
    config = RunConfig.model_validate(
        {
            "seed": 7,
            "device": "cpu",
            "output_root": str(tmp_path),
            "task": {"task_id": "associative_recall.v1", "profile": "smoke"},
            "model": {"kind": "gru", "hidden_size": 8},
            "optimizer": {"learning_rate": 0.001, "weight_decay": 0.0},
            "training": {
                "batch_size": 4,
                "max_steps": 2,
                "eval_interval": 1,
                "patience": 2,
                "checkpoint_interval": 1,
            },
        }
    )
    set_global_seed(config.seed)
    task = create_task(config.task.task_id, profile=config.task.profile)
    model = GRUBaseline(
        input_dim=task.input_dim,
        num_classes=task.num_classes,
        hidden_size=config.model.hidden_size,
    )
    result = train_baseline(
        config=config,
        task=task,
        model=model,
        device=torch.device("cpu"),
        run_directory=tmp_path / "run",
    )
    assert result["steps"] == 2
    assert (tmp_path / "run" / "latest.pt").is_file()
    assert (tmp_path / "run" / "metrics.jsonl").is_file()
