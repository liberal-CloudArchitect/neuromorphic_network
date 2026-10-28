"""P6 qualification: the engineering check and the surprise-validity check (CR-015, E2).

Both parts use seed 7 and only train/validation data.  Each writes a report;
when the repository is clean, each also writes a part lock, and the
qualification lock that the pilot requires appears only once both parts have
passed on the same commit and device.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from neuromorphic.core.registry import PREDICTIVE_ADAPTER_V3, SPARSE_ROUTER_V3
from neuromorphic.modules.network_v2 import ModularBrainOutputV2
from neuromorphic.modules.network_v4 import ModularBrainNetworkV4
from neuromorphic.tasks.associative_recall import AssociativeRecallTask
from neuromorphic.tasks.base import TaskBatch
from neuromorphic.tasks.delayed_rule_switch import DelayedRuleSwitchTask
from neuromorphic.tasks.small_graph import SmallGraphTask
from neuromorphic.tasks.uncued_rule_switch import (
    FEEDBACK,
    FEEDBACK_BIT,
    QUERY,
    UncuedRuleSwitchTask,
)
from neuromorphic.training.config import resolve_device
from neuromorphic.training.p5_suite import (
    _git,
    _primary_loss,
    _repository_relative,
    _score,
    _sha256,
    _write_json,
)
from neuromorphic.training.p6_config import (
    P6_PILOT_PRESETS,
    P6QualificationConfig,
    P6SurpriseCheckConfig,
)
from neuromorphic.training.reproducibility import set_global_seed

P6_TASKS = (
    AssociativeRecallTask(namespace="p6"),
    DelayedRuleSwitchTask(namespace="p6"),
    SmallGraphTask(namespace="p6"),
    UncuedRuleSwitchTask(namespace="p6"),
)
_RULE_SWITCH_TASKS = {"delayed_rule_switch.v1", "uncued_rule_switch.v1"}
_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

# The P5 auxiliary weights; P6 changes only working memory (CR-015).
_FIXED_AUXILIARY_WEIGHTS = {
    "episodic.retrieval": 0.1,
    "episodic.separation": 0.0,
    "working.state_consistency": 0.05,
    "working.gate_regularization": 0.001,
    "router.load_balance": 0.01,
    "router.communication_cost": 0.0,
}


def p6_weighted_loss(
    output: ModularBrainOutputV2,
    batch: TaskBatch,
    *,
    temporal_weight: float,
    semantic_weight: float,
    dual_budget_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    """Primary task loss plus the P5-weighted auxiliary losses."""

    weights = {
        **_FIXED_AUXILIARY_WEIGHTS,
        "predictive.temporal": temporal_weight,
        "router.semantic_alignment": semantic_weight,
        "router.dual_budget": dual_budget_weight,
    }
    primary = _primary_loss(output.logits, batch)
    total = primary
    values = {"primary": float(primary.detach().cpu())}
    for name, value in output.auxiliary_losses.items():
        if value.ndim != 0 or not torch.isfinite(value).item():
            raise FloatingPointError(f"invalid P6 auxiliary loss: {name}")
        total = total + weights.get(name, 0.0) * value
        values[name] = float(value.detach().cpu())
    values["total"] = float(total.detach().cpu())
    return total, values


def check_p6_routing(
    output: ModularBrainOutputV2, batch: TaskBatch, dual_fraction: float
) -> tuple[dict[str, int], list[str]]:
    """Count semantic coverage and dual routes; return any violated frozen condition."""

    task_id = str(batch.metadata["task_id"])
    counts = {"valid": 0, "ar_reserved": 0, "working_reserved": 0, "dual": 0, "drops": 0}
    failures: set[str] = set()
    for step, decision in enumerate(output.routing_trace):
        valid = batch.valid_mask[:, step]
        valid_count = int(valid.sum().item())
        counts["valid"] += valid_count
        counts["drops"] += int(decision.capacity_drops)
        if decision.capacity_drops != 0:
            failures.add("capacity drop")
        if task_id == "associative_recall.v1":
            events = batch.inputs[:, step, :3]
            semantic = valid & (events[:, 0].gt(0.5) | events[:, 2].gt(0.5))
            if semantic.any() and not decision.executed_mask[:, 0, 0][semantic].all():
                failures.add("AR episodic coverage is incomplete")
            counts["ar_reserved"] += int(semantic.sum().item())
        elif task_id in _RULE_SWITCH_TASKS:
            if valid.any() and not decision.executed_mask[:, 0, 1][valid].all():
                failures.add(f"{task_id} working-memory coverage is incomplete")
            dual = decision.executed_mask[:, 0].sum(dim=-1).gt(1) & valid
            if int(dual.sum().item()) > math.floor(valid_count * dual_fraction):
                failures.add(f"{task_id} dual-route budget was exceeded")
            counts["working_reserved"] += valid_count
            counts["dual"] += int(dual.sum().item())
    return counts, sorted(failures)


def _ranks(values: Tensor) -> Tensor:
    """Ranks starting at 1, with tied values sharing their average rank."""

    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float64)
    start = 0
    count = values.numel()
    while start < count:
        end = start
        while end + 1 < count and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return ranks


def spearman_correlation(first: Tensor, second: Tensor) -> float:
    """Spearman's rho with average ranks for ties; NaN when either side is constant."""

    if first.shape != second.shape or first.ndim != 1 or first.numel() < 2:
        raise ValueError("Spearman inputs must be two equal-length vectors of length >= 2")
    left = _ranks(first.detach().cpu().to(torch.float64))
    right = _ranks(second.detach().cpu().to(torch.float64))
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt(left.square().sum() * right.square().sum())
    if denominator.item() == 0.0:
        return float("nan")
    return float((left * right).sum() / denominator)


def feedback_surprise_pairs(
    output: ModularBrainOutputV2, batch: TaskBatch
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Forecast surprise and outcome surprise on uncued feedback steps.

    Returns ``(forecast, outcome, other_forecast, mismatches)``.  The forecast
    surprise is ``log1p`` of the per-step forecast error, which is exactly the
    signal ``predictive_adapter.v3`` gives the router and working memory.  The
    outcome surprise is ``|feedback - p|`` with ``p`` the network's probability
    of answering 1 at the immediately preceding query.  ``other_forecast``
    holds the surprise on every other step that had a forecast.
    """

    if batch.metadata["task_id"] != UncuedRuleSwitchTask.task_id:
        raise ValueError("feedback surprise is defined on uncued rule switch only")
    events = batch.inputs[..., :4].argmax(dim=-1)
    valid = batch.valid_mask
    feedback = events.eq(FEEDBACK) & valid
    previous_is_query = torch.zeros_like(feedback)
    previous_is_query[:, 1:] = events[:, :-1].eq(QUERY) & valid[:, :-1]
    mismatches = (feedback & ~previous_is_query).sum()
    forecast = torch.log1p(output.forecast_error.detach())
    probability = torch.softmax(output.logits.detach(), dim=-1)[..., 1]
    previous_probability = torch.zeros_like(probability)
    previous_probability[:, 1:] = probability[:, :-1]
    outcome = (batch.inputs[..., FEEDBACK_BIT] - previous_probability).abs()
    transitions = output.forecast_transition_mask & valid
    other = transitions & ~feedback
    return forecast[feedback], outcome[feedback], forecast[other], mismatches


def _distribution(values: Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"count": 0.0}
    values = values.detach().cpu().to(torch.float64)
    quantiles = torch.quantile(values, torch.tensor(_QUANTILES, dtype=torch.float64))
    summary = {"count": float(values.numel()), "mean": float(values.mean())}
    for level, value in zip(_QUANTILES, quantiles, strict=True):
        summary[f"q{round(level * 100):02d}"] = float(value)
    return summary


def _nonzero_gradient(parameter: nn.Parameter) -> bool:
    gradient = parameter.grad
    return gradient is not None and gradient.abs().sum().item() > 0.0


def _has_gradient(module: nn.Module, names: tuple[str, ...]) -> bool:
    parameters = dict(module.named_parameters())
    return all(_nonzero_gradient(parameters[name]) for name in names)


def _train_round_robin(
    model: ModularBrainNetworkV4,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    train_samples: int,
    batch_size: int,
    steps_per_task: int,
    temporal_weight: float,
    semantic_weight: float,
    dual_budget_weight: float,
    dual_fraction: float,
    gradient_clip_norm: float,
    gradient_flags: dict[str, bool] | None = None,
) -> tuple[dict[str, list[float]], dict[str, int], list[str]]:
    """Interleave the four tasks one batch at a time, as the P5 suites do."""

    histories: dict[str, list[float]] = {task.task_id: [] for task in P6_TASKS}
    routing = {"valid": 0, "ar_reserved": 0, "working_reserved": 0, "dual": 0, "drops": 0}
    failures: set[str] = set()
    counters = dict.fromkeys(histories, 0)
    working = cast(nn.Module, model.registry.get(model.working_module_id))
    router = cast(nn.Module, model.registry.get(SPARSE_ROUTER_V3))
    predictor = cast(nn.Module, model.registry.get(PREDICTIVE_ADAPTER_V3))
    model.train()
    for global_step in range(steps_per_task * len(P6_TASKS)):
        task = P6_TASKS[global_step % len(P6_TASKS)]
        start = counters[task.task_id] * batch_size
        counters[task.task_id] += 1
        indices = [(start + offset) % train_samples for offset in range(batch_size)]
        batch = task.generate("train", indices, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_batch(batch)
        counts, step_failures = check_p6_routing(output, batch, dual_fraction)
        failures.update(step_failures)
        for name, value in counts.items():
            routing[name] += value
        loss, parts = p6_weighted_loss(
            output,
            batch,
            temporal_weight=temporal_weight,
            semantic_weight=semantic_weight,
            dual_budget_weight=dual_budget_weight,
        )
        if not torch.isfinite(loss).item():
            raise FloatingPointError("P6 qualification produced a non-finite loss")
        loss.backward()  # type: ignore[no-untyped-call]
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError("P6 qualification produced a non-finite gradient")
        if gradient_flags is not None:
            gradient_flags["working_gate"] |= _has_gradient(
                working, ("update_gate.weight", "surprise_gate")
            )
            gradient_flags["working_candidate"] |= _has_gradient(working, ("candidate.weight",))
            gradient_flags["router"] |= any(map(_nonzero_gradient, router.parameters()))
            gradient_flags["predictor"] |= any(map(_nonzero_gradient, predictor.parameters()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        histories[task.task_id].append(parts["total"])
    return histories, routing, sorted(failures)


class _SurprisePairs:
    """Accumulate uncued feedback-step surprise pairs across validation batches."""

    def __init__(self) -> None:
        self.forecast: list[Tensor] = []
        self.outcome: list[Tensor] = []
        self.other: list[Tensor] = []
        self.mismatches = 0
        self.reconstruction_gap = 0.0

    def add(self, output: ModularBrainOutputV2, batch: TaskBatch) -> None:
        forecast, outcome, other, mismatches = feedback_surprise_pairs(output, batch)
        self.forecast.append(forecast.cpu())
        self.outcome.append(outcome.cpu())
        self.other.append(other.cpu())
        self.mismatches += int(mismatches)
        recorded = float(output.module_metrics["predictive.surprise_sum"])
        rebuilt = float(torch.log1p(output.forecast_error).sum())
        self.reconstruction_gap = max(
            self.reconstruction_gap, abs(recorded - rebuilt) / max(1.0, abs(recorded))
        )


def _validate(
    model: ModularBrainNetworkV4,
    *,
    device: torch.device,
    validation_samples: int,
    batch_size: int,
    dual_fraction: float,
) -> tuple[dict[str, float], list[str], _SurprisePairs]:
    """Score every task on validation and collect the uncued surprise pairs."""

    scores: dict[str, float] = {}
    failures: set[str] = set()
    uncued = _SurprisePairs()
    model.eval()
    with torch.no_grad():
        for task in P6_TASKS:
            correct = 0.0
            total = 0
            for start in range(0, validation_samples, batch_size):
                indices = list(range(start, min(start + batch_size, validation_samples)))
                batch = task.generate("validation", indices, device=device)
                output = model.forward_batch(batch, phase="evaluate")
                failures.update(check_p6_routing(output, batch, dual_fraction)[1])
                queries = int(batch.loss_mask.sum().item())
                correct += _score(output.logits, batch) * queries
                total += queries
                if task.task_id == UncuedRuleSwitchTask.task_id:
                    uncued.add(output, batch)
            scores[task.task_id] = correct / max(total, 1)
    return scores, sorted(failures), uncued


def _run_directory(
    output_root: Path, prefix: str, run_id: str | None, device: torch.device
) -> tuple[str, Path]:
    identifier = run_id or (
        f"{prefix}-{str(device).replace(':', '-')}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    directory = output_root / identifier
    directory.mkdir(parents=True, exist_ok=False)
    return identifier, directory


def _finish(
    *,
    directory: Path,
    run_id: str,
    part: str,
    passed: bool,
    summary: dict[str, Any],
    config_hash: str,
    device: torch.device,
    control_root: Path,
    lock_allowed: bool,
) -> dict[str, Any]:
    """Write summary, report and manifest; write the part lock on a clean repository."""

    _write_json(directory / "summary.json", summary)
    report_path = directory / "qualification-report.json"
    _write_json(
        report_path,
        {
            "schema_version": "p6-qualification-report-v1",
            "part": part,
            "status": "PASSED" if passed else "FAILED",
            "qualification_only": True,
            "run_id": run_id,
            "git_commit": summary["git_commit"],
            "git_dirty": summary["git_dirty"],
            "device": str(device),
            "config_hash": config_hash,
            "summary_sha256": _sha256(directory / "summary.json"),
        },
    )
    _write_json(
        directory / "manifest.json",
        {
            "schema_version": "p6-qualification-manifest-v1",
            "run_id": run_id,
            "status": summary["status"],
            "artifacts": {
                path.name: _sha256(path)
                for path in sorted(directory.iterdir())
                if path.is_file() and path.name != "manifest.json"
            },
        },
    )
    relative_report = _repository_relative(report_path)
    if passed and lock_allowed and not summary["git_dirty"] and relative_report is not None:
        _write_json(
            control_root.parent / f"{part}-lock.json",
            {
                "schema_version": "p6-qualification-part-lock-v1",
                "part": part,
                "status": "PASSED",
                "git_commit": summary["git_commit"],
                "device": device.type,
                "config_hash": config_hash,
                "qualification_report": relative_report,
                "qualification_report_sha256": _sha256(report_path),
            },
        )
        combine_p6_qualification_lock(control_root.parent)
    return {
        "run_id": run_id,
        "part": part,
        "status": summary["status"],
        "device": str(device),
        "artifact_dir": str(directory),
    }


def combine_p6_qualification_lock(lock_root: Path) -> Path | None:
    """Write ``qualification-lock.json`` once both parts passed on one commit and device."""

    parts: dict[str, dict[str, Any]] = {}
    for part in ("engineering", "surprise-check"):
        path = lock_root / f"{part}-lock.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        report = Path(str(value.get("qualification_report")))
        if (
            value.get("status") != "PASSED"
            or not report.is_file()
            or _sha256(report) != value.get("qualification_report_sha256")
        ):
            return None
        parts[part] = {**value, "lock_sha256": _sha256(path)}
    engineering, surprise = parts["engineering"], parts["surprise-check"]
    if (engineering["git_commit"], engineering["device"]) != (
        surprise["git_commit"],
        surprise["device"],
    ):
        return None
    lock = lock_root / "qualification-lock.json"
    _write_json(
        lock,
        {
            "schema_version": "p6-qualification-lock-v1",
            "status": "PASSED",
            "git_commit": engineering["git_commit"],
            "device": engineering["device"],
            "qualification_report": engineering["qualification_report"],
            "qualification_report_sha256": engineering["qualification_report_sha256"],
            "surprise_report": surprise["qualification_report"],
            "surprise_report_sha256": surprise["qualification_report_sha256"],
            "engineering_lock_sha256": engineering["lock_sha256"],
            "surprise_lock_sha256": surprise["lock_sha256"],
        },
    )
    return lock


def _repository_state() -> dict[str, Any]:
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def _build_model(
    device: torch.device, dual_fraction: float, learning_rate: float, weight_decay: float
) -> tuple[ModularBrainNetworkV4, torch.optim.Optimizer]:
    model = ModularBrainNetworkV4(dual_route_fraction=dual_fraction).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return model, optimizer


def execute_p6_qualification(config: P6QualificationConfig) -> dict[str, Any]:
    """Engineering qualification: finite training, gradients and frozen routing (E2)."""

    set_global_seed(config.seed)
    device = resolve_device(config.device)
    model, optimizer = _build_model(
        device, config.dual_route_fraction, config.learning_rate, config.weight_decay
    )
    run_id, directory = _run_directory(
        config.output_root, "p6-qualification", config.run_id, device
    )
    _write_json(directory / "config.json", config.model_dump(mode="json"))
    gradients = dict.fromkeys(("working_gate", "working_candidate", "router", "predictor"), False)
    histories, routing, failures = _train_round_robin(
        model,
        optimizer,
        device=device,
        train_samples=config.train_samples,
        batch_size=config.batch_size,
        steps_per_task=config.steps_per_task,
        temporal_weight=config.temporal_loss_weight,
        semantic_weight=config.semantic_loss_weight,
        dual_budget_weight=config.dual_budget_weight,
        dual_fraction=config.dual_route_fraction,
        gradient_clip_norm=config.gradient_clip_norm,
        gradient_flags=gradients,
    )
    scores, validation_failures, _ = _validate(
        model,
        device=device,
        validation_samples=config.validation_samples,
        batch_size=config.batch_size,
        dual_fraction=config.dual_route_fraction,
    )
    failures = sorted({*failures, *validation_failures})
    if not gradients["working_gate"]:
        failures.append("working_memory.v2 gate received no gradient")
    if not gradients["working_candidate"]:
        failures.append("working_memory.v2 candidate received no gradient")
    passed = not failures
    summary = {
        "schema_version": "p6-qualification-summary-v1",
        "part": "engineering",
        "status": "qualification_passed" if passed else "qualification_failed",
        "qualification_only": True,
        "run_id": run_id,
        **_repository_state(),
        "device": str(device),
        "config_hash": config.config_hash(),
        "updates": config.steps_per_task * len(P6_TASKS),
        "loss_history": histories,
        "validation_scores": scores,
        "routing": routing,
        "gradients": gradients,
        "failures": failures,
    }
    return _finish(
        directory=directory,
        run_id=run_id,
        part="engineering",
        passed=passed,
        summary=summary,
        config_hash=config.config_hash(),
        device=device,
        control_root=config.control_root,
        lock_allowed=True,
    )


def execute_p6_surprise_check(config: P6SurpriseCheckConfig) -> dict[str, Any]:
    """Train V4 ``full`` at ``preset-1`` and correlate forecast with outcome surprise (E2)."""

    set_global_seed(config.seed)
    device = resolve_device(config.device)
    learning_rate, temporal_weight, semantic_weight = P6_PILOT_PRESETS[config.preset]
    model, optimizer = _build_model(
        device, config.dual_route_fraction, learning_rate, config.weight_decay
    )
    run_id, directory = _run_directory(
        config.output_root, f"p6-surprise-check-{config.profile}", config.run_id, device
    )
    _write_json(directory / "config.json", config.model_dump(mode="json"))
    histories, routing, failures = _train_round_robin(
        model,
        optimizer,
        device=device,
        train_samples=config.train_samples,
        batch_size=config.batch_size,
        steps_per_task=config.steps_per_task,
        temporal_weight=temporal_weight,
        semantic_weight=semantic_weight,
        dual_budget_weight=config.dual_budget_weight,
        dual_fraction=config.dual_route_fraction,
        gradient_clip_norm=config.gradient_clip_norm,
    )
    scores, validation_failures, uncued = _validate(
        model,
        device=device,
        validation_samples=config.validation_samples,
        batch_size=config.batch_size,
        dual_fraction=config.dual_route_fraction,
    )
    forecast_values = torch.cat(uncued.forecast)
    outcome_values = torch.cat(uncued.outcome)
    correlation = spearman_correlation(forecast_values, outcome_values)
    failures = sorted({*failures, *validation_failures})
    if uncued.mismatches:
        failures.append("a feedback step did not follow its query")
    if uncued.reconstruction_gap > 1.0e-4:
        failures.append("log1p(forecast_error) does not reproduce the routed surprise")
    if not correlation >= config.correlation_threshold:
        failures.append(
            f"Spearman correlation {correlation:.4f} is below {config.correlation_threshold}"
        )
    passed = not failures
    summary = {
        "schema_version": "p6-surprise-check-summary-v1",
        "part": "surprise-check",
        "profile": config.profile,
        "status": "qualification_passed" if passed else "qualification_failed",
        "qualification_only": True,
        "run_id": run_id,
        **_repository_state(),
        "device": str(device),
        "config_hash": config.config_hash(),
        "preset": config.preset,
        "updates": config.max_steps,
        "loss_history": histories,
        "validation_scores": scores,
        "routing": routing,
        "spearman": correlation,
        "threshold": config.correlation_threshold,
        "feedback_steps": int(forecast_values.numel()),
        "forecast_surprise_feedback": _distribution(forecast_values),
        "forecast_surprise_other": _distribution(torch.cat(uncued.other)),
        "outcome_surprise_feedback": _distribution(outcome_values),
        "surprise_reconstruction_gap": uncued.reconstruction_gap,
        "failures": failures,
    }
    return _finish(
        directory=directory,
        run_id=run_id,
        part="surprise-check",
        passed=passed,
        summary=summary,
        config_hash=config.config_hash(),
        device=device,
        control_root=config.control_root,
        lock_allowed=config.profile == "formal",
    )


__all__ = [
    "P6_TASKS",
    "check_p6_routing",
    "combine_p6_qualification_lock",
    "execute_p6_qualification",
    "execute_p6_surprise_check",
    "feedback_surprise_pairs",
    "p6_weighted_loss",
    "spearman_correlation",
]
