from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from neuromorphic.modules.network_v4 import ModularBrainNetworkV4
from neuromorphic.tasks.delayed_rule_switch import DelayedRuleSwitchTask
from neuromorphic.tasks.uncued_rule_switch import FEEDBACK, FEEDBACK_BIT, UncuedRuleSwitchTask
from neuromorphic.training import p6_suite
from neuromorphic.training.p5_config import P5QualificationConfig
from neuromorphic.training.p5_suite import _sha256, _weighted_loss
from neuromorphic.training.p6_config import P6QualificationConfig, P6SurpriseCheckConfig
from neuromorphic.training.p6_suite import (
    check_p6_routing,
    combine_p6_qualification_lock,
    execute_p6_qualification,
    execute_p6_surprise_check,
    feedback_surprise_pairs,
    p6_weighted_loss,
    spearman_correlation,
)


def test_spearman_handles_order_ties_and_constants() -> None:
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert spearman_correlation(values, values * 10) == pytest.approx(1.0)
    assert spearman_correlation(values, -values) == pytest.approx(-1.0)
    # Average ranks for ties: x ranks (1, 2.5, 2.5, 4), y ranks (1, 2, 3, 4).
    tied = spearman_correlation(torch.tensor([1.0, 2.0, 2.0, 3.0]), values)
    assert tied == pytest.approx(4.5 / math.sqrt(4.5 * 5.0))
    assert math.isnan(spearman_correlation(torch.ones(4), values))
    with pytest.raises(ValueError, match="equal-length"):
        spearman_correlation(values, values[:3])


def _uncued_output() -> tuple[Any, Any]:
    torch.manual_seed(3)
    batch = UncuedRuleSwitchTask().generate("validation", list(range(4)))
    model = ModularBrainNetworkV4()
    model.eval()
    with torch.no_grad():
        return model.forward_batch(batch, phase="evaluate"), batch


def test_feedback_pairs_compare_forecast_surprise_with_the_previous_answer() -> None:
    output, batch = _uncued_output()
    forecast, outcome, other, mismatches = feedback_surprise_pairs(output, batch)

    events = batch.inputs[..., :4].argmax(dim=-1)
    feedback = events.eq(FEEDBACK) & batch.valid_mask
    rows, steps = torch.nonzero(feedback, as_tuple=True)
    probability = torch.softmax(output.logits, dim=-1)[..., 1]
    expected = (batch.inputs[rows, steps, FEEDBACK_BIT] - probability[rows, steps - 1]).abs()
    assert int(mismatches) == 0 and forecast.numel() == int(feedback.sum())
    torch.testing.assert_close(outcome, expected)
    torch.testing.assert_close(forecast, torch.log1p(output.forecast_error[rows, steps]))
    assert other.numel() == int((output.forecast_transition_mask & ~feedback).sum())
    assert float(torch.log1p(output.forecast_error).sum()) == pytest.approx(
        float(output.module_metrics["predictive.surprise_sum"]), rel=1.0e-5
    )
    drs = DelayedRuleSwitchTask(namespace="p6").generate("validation", [0])
    with pytest.raises(ValueError, match="uncued"):
        feedback_surprise_pairs(output, drs)


def test_p6_loss_uses_the_p5_auxiliary_weights() -> None:
    torch.manual_seed(2)
    batch = DelayedRuleSwitchTask(namespace="p6").generate("train", [0, 1, 2])
    output = ModularBrainNetworkV4().forward_batch(batch)
    p5_total, p5_parts = _weighted_loss(output, batch, P5QualificationConfig())
    p6_total, p6_parts = p6_weighted_loss(
        output, batch, temporal_weight=0.1, semantic_weight=0.01, dual_budget_weight=0.001
    )
    assert torch.equal(p6_total, p5_total) and p6_parts == p5_parts


def test_routing_check_flags_missing_working_memory_on_uncued_steps() -> None:
    output, batch = _uncued_output()
    counts, failures = check_p6_routing(output, batch, 0.25)
    assert not failures and counts["working_reserved"] == counts["valid"] > 0

    first = output.routing_trace[0]
    executed = first.executed_mask.clone()
    executed[..., 1] = False
    broken = dataclasses.replace(
        output,
        routing_trace=(
            dataclasses.replace(first, executed_mask=executed),
            *output.routing_trace[1:],
        ),
    )
    assert check_p6_routing(broken, batch, 0.25)[1] == [
        "uncued_rule_switch.v1 working-memory coverage is incomplete"
    ]


def _clean_repository(monkeypatch: pytest.MonkeyPatch, commit: str = "c0ffee") -> None:
    monkeypatch.setattr(
        p6_suite, "_repository_state", lambda: {"git_commit": commit, "git_dirty": False}
    )
    monkeypatch.setattr(p6_suite, "_repository_relative", lambda path: str(path))


def test_engineering_qualification_passes_and_writes_its_part_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_repository(monkeypatch)
    control = tmp_path / "p6-cpu" / "control"
    result = execute_p6_qualification(
        P6QualificationConfig(device="cpu", output_root=tmp_path / "runs", control_root=control)
    )
    summary = json.loads((Path(result["artifact_dir"]) / "summary.json").read_text())

    assert result["status"] == "qualification_passed" and summary["failures"] == []
    assert all(summary["gradients"].values())
    assert summary["routing"]["drops"] == 0 and summary["updates"] == 16
    assert set(summary["validation_scores"]) == {task.task_id for task in p6_suite.P6_TASKS}
    lock = json.loads((control.parent / "engineering-lock.json").read_text())
    assert lock["status"] == "PASSED" and lock["device"] == "cpu"
    assert not (control.parent / "qualification-lock.json").exists()


def test_ci_surprise_check_reports_but_never_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_repository(monkeypatch)
    control = tmp_path / "p6-cpu" / "control"
    config = P6SurpriseCheckConfig(
        profile="ci",
        device="cpu",
        train_samples=64,
        validation_samples=32,
        batch_size=8,
        steps_per_task=4,
        output_root=tmp_path / "runs",
        control_root=control,
    )
    result = execute_p6_surprise_check(config)
    summary = json.loads((Path(result["artifact_dir"]) / "summary.json").read_text())

    assert summary["feedback_steps"] == 32 * 12
    assert summary["surprise_reconstruction_gap"] < 1.0e-4
    assert {"mean", "q05", "q50", "q95"} <= set(summary["forecast_surprise_feedback"])
    passed = summary["spearman"] >= 0.5
    assert result["status"] == ("qualification_passed" if passed else "qualification_failed")
    assert not (control.parent / "surprise-check-lock.json").exists()


def _part_lock(root: Path, part: str, *, commit: str = "c0ffee", device: str = "cuda") -> None:
    report = root / f"{part}-report.json"
    report.write_text(json.dumps({"part": part}), encoding="utf-8")
    lock = {
        "status": "PASSED",
        "git_commit": commit,
        "device": device,
        "qualification_report": str(report),
        "qualification_report_sha256": _sha256(report),
    }
    (root / f"{part}-lock.json").write_text(json.dumps(lock), encoding="utf-8")


def test_qualification_lock_needs_both_parts_on_one_commit_and_device(tmp_path: Path) -> None:
    _part_lock(tmp_path, "engineering")
    assert combine_p6_qualification_lock(tmp_path) is None
    _part_lock(tmp_path, "surprise-check", commit="other")
    assert combine_p6_qualification_lock(tmp_path) is None
    _part_lock(tmp_path, "surprise-check", device="cpu")
    assert combine_p6_qualification_lock(tmp_path) is None
    _part_lock(tmp_path, "surprise-check")
    (tmp_path / "surprise-check-report.json").write_text("tampered", encoding="utf-8")
    assert combine_p6_qualification_lock(tmp_path) is None

    _part_lock(tmp_path, "surprise-check")
    lock_path = combine_p6_qualification_lock(tmp_path)
    assert lock_path == tmp_path / "qualification-lock.json"
    lock = json.loads(lock_path.read_text())
    assert (
        lock["status"] == "PASSED" and lock["git_commit"] == "c0ffee" and lock["device"] == "cuda"
    )
    assert lock["surprise_report"].endswith("surprise-check-report.json")
