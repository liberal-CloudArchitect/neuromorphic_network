from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from neuromorphic.evaluation.p3_statistics import normalized_aulc
from neuromorphic.tasks.associative_recall import AssociativeRecallTask
from neuromorphic.training import p5_mechanism, p5_suite
from neuromorphic.training.p5_config import P5QualificationConfig, load_p5_mechanism_config


def test_validation_macro_aulc_uses_fixed_budget_padding() -> None:
    curve = [0.2, 0.8]

    score = p5_suite._validation_macro_aulc(curve, validation_interval=100, max_steps=500)

    assert score == pytest.approx(normalized_aulc([(100, 0.2), (200, 0.8)], maximum_step=500))
    assert score != pytest.approx(sum(curve) / len(curve))


def test_p5_diagnostic_terms_cannot_change_optimizer_objective() -> None:
    batch = AssociativeRecallTask(namespace="p5").generate("train", [0, 1])
    logits = torch.zeros((*batch.targets.shape, 32), requires_grad=True)

    def output(*, separation: float, communication: float, semantic: float) -> object:
        return SimpleNamespace(
            logits=logits,
            auxiliary_losses={
                "episodic.separation": torch.tensor(separation),
                "router.communication_cost": torch.tensor(communication),
                "router.semantic_alignment": torch.tensor(semantic),
            },
        )

    reference, _ = p5_suite._weighted_loss(
        output(separation=0.0, communication=0.0, semantic=0.0),
        batch,
        P5QualificationConfig(),
    )
    diagnostics_changed, values = p5_suite._weighted_loss(
        output(separation=100.0, communication=100.0, semantic=0.0),
        batch,
        P5QualificationConfig(),
    )
    trainable_changed, _ = p5_suite._weighted_loss(
        output(separation=0.0, communication=0.0, semantic=1.0),
        batch,
        P5QualificationConfig(),
    )

    torch.testing.assert_close(diagnostics_changed, reference)
    assert trainable_changed.item() > reference.item()
    assert values["episodic.separation"] == 100.0
    assert values["router.communication_cost"] == 100.0


def test_formal_evidence_rejects_missing_prediction_fields(tmp_path: Path) -> None:
    config = load_p5_mechanism_config(
        Path("configs/experiments/p5/mechanism-cuda.yaml")
    ).model_copy(update={"bootstrap_samples": 200})
    summaries = _mechanism_summaries(config.seeds)
    summaries["full__s17"]["prediction"] = {"associative_recall.v1": {"forecast_error": 0.18}}
    _write_formal_sample_records(tmp_path, config.seeds)

    with pytest.raises(ValueError, match="prediction"):
        p5_mechanism._formal_evidence(config, tmp_path, summaries)


def test_execute_p5_mechanism_scores_retrained_cell_from_best_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_p5_mechanism_config(Path("configs/experiments/p5/mechanism-ci.yaml")).model_copy(
        update={
            "output_root": tmp_path,
            "run_id": "mechanism-best-checkpoint",
            "steps_per_task": 1,
            "validation_interval": 1,
            "checkpoint_interval": 1,
            "patience": 1,
        }
    )

    class FakeTask:
        task_id = "associative_recall.v1"

        def generate(self, split: str, indices: list[int], device: torch.device) -> object:
            del split, device
            desired = 1.0 if indices and indices[0] == 0 else 0.0
            return SimpleNamespace(desired=desired)

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quality = torch.nn.Parameter(torch.tensor(0.0))

        def forward_batch(self, batch: object, **_: object) -> object:
            return SimpleNamespace(owner=self, batch=batch)

    def fake_validation_summary(
        model: FakeModel,
        config: object,
        device: torch.device,
        *,
        split: str = "validation",
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        del config, device, split
        value = 0.9 if float(model.quality.detach().cpu()) > 0.5 else 0.1
        return (
            {"associative_recall.v1": value},
            {
                "associative_recall.v1": {
                    "forecast_error": 0.1,
                    "persistence_error": 0.2,
                    "covered": 1.0,
                }
            },
        )

    def fake_weighted_loss(
        output: object, batch: object, config: object, **_: object
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del config
        output_value = cast(Any, output)
        batch_value = cast(Any, batch)
        model = output_value.owner
        desired = torch.tensor(batch_value.desired, dtype=model.quality.dtype)
        loss = (model.quality - desired).square()
        return loss, {"total": float(loss.detach().cpu())}

    def fake_score_records(
        model: FakeModel,
        cell: object,
        config: object,
        device: torch.device,
        deadline: float,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        del cell, config, device, deadline
        value = 0.9 if float(model.quality.detach().cpu()) > 0.5 else 0.1
        records = [
            {
                "seed": 7,
                "task_id": task_id,
                "split": "test",
                "distribution": "v1",
                "sample_index": 0,
                "stratum": task_id,
                "model_id": "modular-v3",
                "variant_id": "full",
                "value": value,
            }
            for task_id in (
                "associative_recall.v1",
                "delayed_rule_switch.v1",
                "small_graph.v1",
            )
        ]
        return records, {
            "prediction": {
                task_id: {"forecast_error": 0.1, "persistence_error": 0.2, "covered": 1.0}
                for task_id in (
                    "associative_recall.v1",
                    "delayed_rule_switch.v1",
                    "small_graph.v1",
                )
            },
            "routing": {
                "active_macs": 1.0,
                "dense_macs": 2.0,
                "semantic_required": 1.0,
                "semantic_executed": 1.0,
                "dual_tokens": 0.0,
                "valid_tokens": 1.0,
                "capacity_drops": 0,
            },
        }

    monkeypatch.setattr(p5_mechanism, "_TASKS", (FakeTask(),))
    monkeypatch.setattr(p5_mechanism, "_check_routing", lambda *args, **kwargs: {})
    monkeypatch.setattr(p5_mechanism, "_git", lambda *args: "")
    monkeypatch.setattr(p5_mechanism, "_model", lambda *args: FakeModel())
    monkeypatch.setattr(p5_mechanism, "_settings", lambda config: (1.0, 0.1, 0.01))
    monkeypatch.setattr(p5_mechanism, "_validation_summary", fake_validation_summary)
    monkeypatch.setattr(p5_mechanism, "_weighted_loss", fake_weighted_loss)
    monkeypatch.setattr(p5_mechanism, "_score_records", fake_score_records)
    monkeypatch.setattr(torch.optim, "AdamW", torch.optim.SGD)

    result = p5_mechanism.execute_p5_mechanism(config)

    assert result["status"] == "qualification_passed"
    summary = json.loads(
        (tmp_path / "mechanism-best-checkpoint" / "cells" / "full__s7" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["test_scores"]["associative_recall.v1"] == pytest.approx(0.9)


def _mechanism_summaries(seeds: tuple[int, ...]) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for seed in seeds:
        for variant in (
            "full",
            "predictor-off",
            "surprise-off",
            "no-dual-route",
            "dense-memory",
        ):
            full = variant == "full"
            aulc = 0.8 if full else 0.7
            summaries[f"{variant}__s{seed}"] = {
                "analysis_macro_aulc": aulc,
                "analysis_curve": [[0, aulc], [15000, aulc]],
                "analysis_budget_steps": 15000,
                "selected_checkpoint": "best.pt",
                "drs_score": 0.8 if full else 0.7,
                "test_scores": {
                    "associative_recall.v1": 0.9 if full else 0.89,
                    "delayed_rule_switch.v1": 0.9 if full else 0.89,
                    "small_graph.v1": 0.9 if full else 0.89,
                },
                "prediction": {
                    task: {"forecast_error": 0.18, "persistence_error": 0.20}
                    for task in (
                        "associative_recall.v1",
                        "delayed_rule_switch.v1",
                        "small_graph.v1",
                    )
                },
                "routing": {
                    "active_macs": 50.0,
                    "dense_macs": 100.0,
                    "semantic_required": 10.0,
                    "semantic_executed": 10.0,
                    "dual_tokens": 2.0,
                    "dual_eligible_tokens": 10.0,
                    "valid_tokens": 10.0,
                    "capacity_drops": 0.0,
                },
            }
    return summaries


def _write_formal_sample_records(root: Path, seeds: tuple[int, ...]) -> None:
    for seed in seeds:
        for variant, value in (("full", 0.9), ("dense-memory", 0.89)):
            directory = root / "cells" / f"{variant}__s{seed}"
            directory.mkdir(parents=True)
            records = [
                {
                    "seed": seed,
                    "task_id": task,
                    "split": "test",
                    "distribution": "v1",
                    "sample_index": 0,
                    "stratum": task,
                    "model_id": "modular-v3",
                    "variant_id": variant,
                    "value": value,
                }
                for task in (
                    "associative_recall.v1",
                    "delayed_rule_switch.v1",
                    "small_graph.v1",
                )
            ]
            (directory / "sample-records.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
