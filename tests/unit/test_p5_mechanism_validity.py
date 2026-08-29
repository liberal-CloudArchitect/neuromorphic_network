from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

import neuromorphic.training.p5_mechanism as mechanism
from neuromorphic.training.p5_config import P5MechanismCell, load_p5_mechanism_config


class _Task:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id

    def generate(self, split: str, indices: list[int], *, device: torch.device) -> object:
        del split, indices, device
        return object()


class _ScalarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward_batch(self, batch: object, **kwargs: object) -> torch.Tensor:
        del batch, kwargs
        return self.weight


def test_analysis_aulc_uses_fixed_budget_and_step_axis() -> None:
    stopped = ((100, 0.2), (200, 0.6))
    explicit_plateau = ((100, 0.2), (200, 0.6), (300, 0.6), (400, 0.6))

    assert mechanism._analysis_macro_aulc(stopped, maximum_step=400) == pytest.approx(0.45)
    assert mechanism._analysis_macro_aulc(explicit_plateau, maximum_step=400) == pytest.approx(0.45)


def test_train_cell_uses_analysis_curve_and_restores_best_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_ids = (
        "associative_recall.v1",
        "delayed_rule_switch.v1",
        "small_graph.v1",
    )
    monkeypatch.setattr(mechanism, "_TASKS", tuple(_Task(task_id) for task_id in task_ids))
    monkeypatch.setattr(mechanism, "_check_routing", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mechanism,
        "_weighted_loss",
        lambda output, *args, **kwargs: (output, {"total": float(output.detach())}),
    )
    values = {
        "validation": iter((0.9, 0.8)),
        "analysis": iter((0.2, 0.6)),
    }

    def evaluation(
        model: nn.Module,
        config: object,
        device: torch.device,
        *,
        split: str = "validation",
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        del model, config, device
        score = next(values[split])
        return {task_id: score for task_id in task_ids}, {}

    monkeypatch.setattr(mechanism, "_validation_summary", evaluation)

    config = load_p5_mechanism_config(Path("configs/experiments/p5/mechanism-ci.yaml")).model_copy(
        update={
            "profile": "mechanism",
            "validation_interval": 1,
            "checkpoint_interval": 1,
            "patience": 1,
        }
    )
    cell = P5MechanismCell(
        cell_id="full__s7",
        seed=7,
        variant="full",
        retrained=True,
        max_steps=4,
    )
    model = _ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    directory = tmp_path / "run"
    directory.mkdir()
    registry: dict[str, Any] = {"wall_clock_seconds": 0.0}
    entry: dict[str, object] = {"step": 0, "validation_macro": []}
    started = time.monotonic()

    result = mechanism._train_cell(
        cast(Any, model),
        optimizer,
        cell,
        0,
        config,
        torch.device("cpu"),
        directory,
        registry,
        entry,
        started + 60.0,
        0.0,
        started,
    )

    assert result["steps"] == 2
    assert result["analysis_macro_aulc"] == pytest.approx(0.45)
    analysis_curve = result["analysis_curve"]
    assert isinstance(analysis_curve, tuple)
    assert tuple(step for step, _ in analysis_curve) == (1, 2)
    assert tuple(value for _, value in analysis_curve) == pytest.approx((0.2, 0.6))
    assert result["analysis_budget_steps"] == 4
    assert result["selected_checkpoint"] == "best.pt"
    assert model.weight.item() == pytest.approx(0.9)
    assert (directory / "cells/full__s7/best.pt").is_file()
