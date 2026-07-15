from __future__ import annotations

from pathlib import Path

import torch

from neuromorphic.tasks import create_task
from neuromorphic.training.baselines import BaselineOutput, GRUBaseline
from neuromorphic.training.config import RunConfig
from neuromorphic.training.metrics import masked_task_loss
from neuromorphic.training.reproducibility import set_global_seed
from neuromorphic.training.trainer import (
    IndexSampler,
    forward_training_batch,
    train_baseline,
)


def test_sampler_restore_produces_identical_indices() -> None:
    sampler = IndexSampler.create(10, 7)
    sampler.next(7)
    state = sampler.state_dict()
    expected = sampler.next(15)
    restored = IndexSampler.create(10, 7)
    restored.load_state_dict(state)
    assert restored.next(15) == expected


def test_gru_training_detaches_state_every_tbptt_window() -> None:
    task = create_task("delayed_rule_switch.v1")
    batch = task.generate("ood", [0, 1])

    class CountingGRU(GRUBaseline):
        chunk_calls = 0

        def forward_chunk(
            self,
            inputs: torch.Tensor,
            valid_mask: torch.Tensor,
            state: torch.Tensor | None = None,
        ) -> tuple[BaselineOutput, torch.Tensor]:
            self.chunk_calls += 1
            return super().forward_chunk(inputs, valid_mask, state)

    model = CountingGRU(input_dim=task.input_dim, num_classes=task.num_classes, hidden_size=8)
    output = forward_training_batch(model, batch, tbptt_steps=32)
    assert model.chunk_calls == (batch.sequence_length + 31) // 32
    loss, _ = masked_task_loss(output, batch, auxiliary_weight=0.1)
    loss.backward()  # type: ignore[no-untyped-call]


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
    assert result["selected_checkpoint"] == "best.pt"
    assert (tmp_path / "run" / "latest.pt").is_file()
    best = torch.load(tmp_path / "run" / "best.pt", map_location="cpu", weights_only=False)
    assert all(
        torch.equal(model.state_dict()[name], value) for name, value in best["model_state"].items()
    )
    assert (tmp_path / "run" / "metrics.jsonl").is_file()
    assert (tmp_path / "run" / "evaluation_samples.jsonl").is_file()
