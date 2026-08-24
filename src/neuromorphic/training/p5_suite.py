"""Small deterministic qualification for the P5 predictive-routing mechanisms."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import functional as F

from neuromorphic.core.registry import PREDICTIVE_ADAPTER_V3, SPARSE_ROUTER_V3
from neuromorphic.modules.network_v3 import ModularBrainNetworkV3
from neuromorphic.tasks.associative_recall import AssociativeRecallTask
from neuromorphic.tasks.base import TaskBatch
from neuromorphic.tasks.delayed_rule_switch import DelayedRuleSwitchTask
from neuromorphic.tasks.small_graph import SmallGraphTask
from neuromorphic.training.checkpoint import CheckpointCompatibilityError
from neuromorphic.training.config import resolve_device
from neuromorphic.training.p5_checkpoint import (
    P5CheckpointState,
    load_p5_checkpoint,
    save_p5_checkpoint,
)
from neuromorphic.training.p5_config import (
    P5_PILOT_PRESETS,
    P5PilotConfig,
    P5QualificationConfig,
)
from neuromorphic.training.reproducibility import set_global_seed

_TASKS = (
    AssociativeRecallTask(namespace="p5"),
    DelayedRuleSwitchTask(namespace="p5"),
    SmallGraphTask(namespace="p5"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], text=True, check=True, capture_output=True
    ).stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _repository_relative(path: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return None


def _primary_loss(logits: Tensor, batch: TaskBatch) -> Tensor:
    optimal = batch.auxiliary_targets.get("optimal_action_mask")
    if optimal is None:
        return F.cross_entropy(logits[batch.loss_mask], batch.targets[batch.loss_mask])
    allowed = F.log_softmax(logits, dim=-1).masked_fill(~optimal, -torch.inf)
    return -torch.logsumexp(allowed[batch.loss_mask], dim=-1).mean()


def _score(logits: Tensor, batch: TaskBatch) -> float:
    selected = logits.argmax(dim=-1)
    optimal = batch.auxiliary_targets.get("optimal_action_mask")
    if optimal is None:
        return float((selected[batch.loss_mask] == batch.targets[batch.loss_mask]).float().mean())
    success = optimal.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
    return float(success[batch.loss_mask].float().mean())


def _check_routing(output: Any, batch: TaskBatch, dual_fraction: float) -> dict[str, int]:
    ar_reserved = 0
    drs_reserved = 0
    drs_dual = 0
    valid_total = 0
    for step, decision in enumerate(output.routing_trace):
        valid = batch.valid_mask[:, step]
        valid_count = int(valid.sum().item())
        valid_total += valid_count
        if decision.capacity_drops != 0:
            raise RuntimeError("P5 qualification observed a capacity drop")
        if batch.metadata["task_id"] == "associative_recall.v1":
            events = batch.inputs[:, step, :3]
            semantic = valid & (events[:, 0].gt(0.5) | events[:, 2].gt(0.5))
            if semantic.any() and not decision.executed_mask[:, 0, 0][semantic].all():
                raise RuntimeError("AR semantic episodic coverage is incomplete")
            ar_reserved += int(semantic.sum().item())
        elif batch.metadata["task_id"] == "delayed_rule_switch.v1":
            if valid.any() and not decision.executed_mask[:, 0, 1][valid].all():
                raise RuntimeError("DRS working-memory coverage is incomplete")
            dual = decision.executed_mask[:, 0].sum(dim=-1).gt(1) & valid
            if int(dual.sum().item()) > math.floor(valid_count * dual_fraction):
                raise RuntimeError("DRS dual-route budget was exceeded")
            drs_reserved += valid_count
            drs_dual += int(dual.sum().item())
    if (
        output.cost_statistics["optional.active_calls"]
        >= output.cost_statistics["optional.dense_calls"]
    ):
        raise RuntimeError("P5 sparse execution did not reduce optional calls")
    return {
        "ar_reserved": ar_reserved,
        "drs_reserved": drs_reserved,
        "drs_dual": drs_dual,
        "valid": valid_total,
    }


def _weighted_loss(
    output: Any,
    batch: TaskBatch,
    config: P5QualificationConfig,
    *,
    temporal_weight: float | None = None,
    semantic_weight: float | None = None,
    dual_budget_weight: float | None = None,
) -> tuple[Tensor, dict[str, float]]:
    primary = _primary_loss(output.logits, batch)
    weights = {
        "episodic.retrieval": 0.1,
        "episodic.separation": 0.01,
        "working.state_consistency": 0.05,
        "working.gate_regularization": 0.001,
        "predictive.temporal": (
            config.temporal_loss_weight if temporal_weight is None else temporal_weight
        ),
        "router.semantic_alignment": (
            config.semantic_loss_weight if semantic_weight is None else semantic_weight
        ),
        "router.dual_budget": (
            config.dual_budget_weight if dual_budget_weight is None else dual_budget_weight
        ),
        "router.load_balance": 0.01,
        "router.communication_cost": 0.001,
    }
    total = primary
    values = {"primary": float(primary.detach().cpu())}
    for name, value in output.auxiliary_losses.items():
        if value.ndim != 0 or not torch.isfinite(value).item():
            raise FloatingPointError(f"invalid P5 auxiliary loss: {name}")
        total = total + weights.get(name, 0.0) * value
        values[name] = float(value.detach().cpu())
    values["total"] = float(total.detach().cpu())
    return total, values


def execute_p5_qualification(config: P5QualificationConfig) -> dict[str, Any]:
    set_global_seed(config.seed)
    device = resolve_device(config.device)
    model = ModularBrainNetworkV3(dual_route_fraction=config.dual_route_fraction).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    run_id = config.run_id or (
        f"p5-qualification-{str(device).replace(':', '-')}-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    directory = config.output_root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(directory / "config.json", config.model_dump(mode="json"))

    histories: dict[str, list[float]] = {task.task_id: [] for task in _TASKS}
    routing_totals = {"ar_reserved": 0, "drs_reserved": 0, "drs_dual": 0, "valid": 0}
    router_gradient_seen = False
    predictor_gradient_seen = False
    telemetry_events = 0
    counters = {task.task_id: 0 for task in _TASKS}
    total_updates = config.steps_per_task * len(_TASKS)
    for global_step in range(total_updates):
        task = _TASKS[global_step % len(_TASKS)]
        task_step = counters[task.task_id]
        counters[task.task_id] += 1
        start = task_step * config.batch_size
        indices = [(start + offset) % config.train_samples for offset in range(config.batch_size)]
        batch = task.generate("train", indices, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_batch(batch)
        for name, value in _check_routing(output, batch, config.dual_route_fraction).items():
            routing_totals[name] += value
        loss, parts = _weighted_loss(output, batch, config)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("P5 qualification produced a non-finite loss")
        loss.backward()  # type: ignore[no-untyped-call]
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError("P5 qualification produced a non-finite gradient")
        router = cast(torch.nn.Module, model.registry.get(SPARSE_ROUTER_V3))
        predictor = cast(torch.nn.Module, model.registry.get(PREDICTIVE_ADAPTER_V3))
        router_gradient_seen |= any(
            parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
            for parameter in router.parameters()
        )
        predictor_gradient_seen |= any(
            parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
            for parameter in predictor.parameters()
        )
        if config.telemetry_enabled:
            from neuromorphic.training.p5_telemetry import build_p5_telemetry_event

            scorer = model.registry.get(SPARSE_ROUTER_V3)
            scorer_gradients = [
                parameter.grad.detach().square().sum()
                for parameter in cast(torch.nn.Module, scorer).parameters()
                if parameter.grad is not None
            ]
            scorer_grad_norm = (
                torch.stack(scorer_gradients).sum().sqrt() if scorer_gradients else None
            )
            event = build_p5_telemetry_event(
                run_id=run_id,
                global_step=global_step,
                batch=batch,
                output=output,
                scorer_grad_norm=scorer_grad_norm,
            )
            with (directory / "telemetry-v3.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
            telemetry_events += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        histories[task.task_id].append(parts["total"])

    scores: dict[str, float] = {}
    forecast: dict[str, dict[str, float]] = {}
    model.eval()
    with torch.no_grad():
        for task in _TASKS:
            batch = task.generate(
                "validation", list(range(config.validation_samples)), device=device
            )
            output = model.forward_batch(batch, phase="evaluate")
            _check_routing(output, batch, config.dual_route_fraction)
            scores[task.task_id] = _score(output.logits, batch)
            covered = output.forecast_transition_mask.sum().item()
            forecast[task.task_id] = {
                "coverage": float(covered / max(batch.valid_mask.sum().item(), 1)),
                "forecast_error": float(
                    output.forecast_error.sum().cpu() / max(float(covered), 1.0)
                ),
                "persistence_error": float(
                    output.persistence_error.sum().cpu() / max(float(covered), 1.0)
                ),
            }
    if not router_gradient_seen or not predictor_gradient_seen:
        raise RuntimeError("P5 qualification did not exercise both mechanism gradients")
    summary = {
        "schema_version": "p5-qualification-summary-v1",
        "status": "qualification_passed",
        "qualification_only": True,
        "run_id": run_id,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "device": str(device),
        "config_hash": config.config_hash(),
        "updates": total_updates,
        "loss_history": histories,
        "validation_scores": scores,
        "forecast": forecast,
        "routing": routing_totals,
        "router_gradient_seen": router_gradient_seen,
        "predictor_gradient_seen": predictor_gradient_seen,
        "telemetry_events": telemetry_events,
    }
    _write_json(directory / "summary.json", summary)
    report_path = directory / "qualification-report.json"
    _write_json(
        report_path,
        {
            "schema_version": "p5-qualification-report-v1",
            "status": "PASSED",
            "qualification_only": True,
            "run_id": run_id,
            "git_commit": summary["git_commit"],
            "git_dirty": summary["git_dirty"],
            "device": str(device),
            "config_hash": config.config_hash(),
            "summary_sha256": _sha256(directory / "summary.json"),
        },
    )
    manifest = {
        "schema_version": "p5-qualification-manifest-v1",
        "run_id": run_id,
        "status": summary["status"],
        "artifacts": {
            path.name: _sha256(path)
            for path in sorted(directory.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    _write_json(directory / "manifest.json", manifest)
    relative_report = _repository_relative(report_path)
    if not summary["git_dirty"] and relative_report is not None:
        _write_json(
            config.control_root.parent / "qualification-lock.json",
            {
                "schema_version": "p5-qualification-lock-v1",
                "status": "PASSED",
                "git_commit": summary["git_commit"],
                "device": str(device),
                "config_hash": config.config_hash(),
                "qualification_report": relative_report,
                "qualification_report_sha256": _sha256(report_path),
            },
        )
    return {
        "run_id": run_id,
        "status": summary["status"],
        "device": str(device),
        "artifact_dir": str(directory),
        "updates": total_updates,
    }


def _validate_qualification_lock(path: Path, *, commit: str) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "PASSED":
        raise ValueError("P5 qualification lock is missing or failed")
    if value.get("git_commit") != commit:
        raise ValueError("P5 qualification lock belongs to another commit")
    report = value.get("qualification_report")
    expected = value.get("qualification_report_sha256")
    if not isinstance(report, str) or not isinstance(expected, str):
        raise ValueError("P5 qualification lock evidence is incomplete")
    report_path = Path(report)
    if not report_path.is_file() or _sha256(report_path) != expected:
        raise ValueError("P5 qualification report checksum does not match")
    return _sha256(path)


def _validation_summary(
    model: ModularBrainNetworkV3,
    config: P5PilotConfig,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    scores: dict[str, float] = {}
    forecast: dict[str, dict[str, float]] = {}
    model.eval()
    with torch.no_grad():
        for task in _TASKS:
            correct = 0.0
            selected_count = 0
            covered = 0.0
            error_sum = 0.0
            persistence_sum = 0.0
            for start in range(0, config.validation_samples, config.batch_size):
                indices = list(
                    range(start, min(start + config.batch_size, config.validation_samples))
                )
                batch = task.generate("validation", indices, device=device)
                output = model.forward_batch(batch, phase="evaluate")
                _check_routing(output, batch, config.dual_route_fraction)
                optimal = batch.auxiliary_targets.get("optimal_action_mask")
                chosen = output.logits.argmax(dim=-1)
                if optimal is None:
                    correct += float(
                        (chosen[batch.loss_mask] == batch.targets[batch.loss_mask]).sum().cpu()
                    )
                else:
                    correct += float(
                        optimal.gather(-1, chosen.unsqueeze(-1))
                        .squeeze(-1)[batch.loss_mask]
                        .sum()
                        .cpu()
                    )
                selected_count += int(batch.loss_mask.sum().item())
                transitions = output.forecast_transition_mask
                covered += float(transitions.sum().cpu())
                error_sum += float(output.forecast_error.sum().cpu())
                persistence_sum += float(output.persistence_error.sum().cpu())
            scores[task.task_id] = correct / max(selected_count, 1)
            forecast[task.task_id] = {
                "forecast_error": error_sum / max(covered, 1.0),
                "persistence_error": persistence_sum / max(covered, 1.0),
                "covered": covered,
            }
    model.train()
    return scores, forecast


def _pilot_registry(config: P5PilotConfig, run_id: str, commit: str) -> dict[str, object]:
    return {
        "schema_version": "p5-pilot-registry-v1",
        "status": "running",
        "run_id": run_id,
        "git_commit": commit,
        "config_hash": config.config_hash(),
        "cells": [
            {
                "candidate_id": preset,
                "status": "PENDING",
                "step": 0,
                "validation_macro": [],
                "router_gradient_seen": False,
                "predictor_gradient_seen": False,
            }
            for preset in P5_PILOT_PRESETS
        ],
    }


def _truncate_telemetry(path: Path, *, completed_steps: int) -> None:
    if not path.is_file():
        return
    retained: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if isinstance(value, dict) and int(value.get("global_step", -1)) < completed_steps:
            retained.append(json.dumps(value, sort_keys=True))
    path.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")


def execute_p5_pilot(config: P5PilotConfig) -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    if config.expected_git_commit is not None and config.expected_git_commit != commit:
        raise ValueError("P5 pilot expected_git_commit does not match HEAD")
    qualification_hash = _validate_qualification_lock(config.qualification_report, commit=commit)
    device = resolve_device(config.device)
    if config.resume is None:
        run_id = config.run_id or (
            f"p5-pilot-{str(device).replace(':', '-')}-"
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
        directory = config.output_root / run_id
        directory.mkdir(parents=True, exist_ok=False)
        registry = _pilot_registry(config, run_id, commit)
        _write_json(directory / "config.json", config.model_dump(mode="json"))
        _write_json(directory / "registry.json", registry)
    else:
        directory = config.resume
        run_id = directory.name
        registry = json.loads((directory / "registry.json").read_text(encoding="utf-8"))
        if (
            registry.get("git_commit") != commit
            or registry.get("config_hash") != config.config_hash()
        ):
            raise CheckpointCompatibilityError("P5 pilot registry is incompatible")
        registry["status"] = "running"

    stop_file = directory / "STOP"
    stop_file.unlink(missing_ok=True)
    cells = registry["cells"]
    if not isinstance(cells, list):
        raise ValueError("P5 pilot registry cells are invalid")
    for candidate_index, candidate in enumerate(cells):
        if not isinstance(candidate, dict):
            raise ValueError("P5 pilot candidate entry is invalid")
        if candidate.get("status") == "COMPLETED":
            continue
        candidate_id = str(candidate["candidate_id"])
        learning_rate, temporal_weight, semantic_weight = P5_PILOT_PRESETS[candidate_id]
        set_global_seed(config.seed)
        model = ModularBrainNetworkV3(dual_route_fraction=config.dual_route_fraction).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
        )
        cell_directory = directory / "cells" / candidate_id
        cell_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = cell_directory / "checkpoint.pt"
        task_steps = {task.task_id: 0 for task in _TASKS}
        start_step = 0
        if checkpoint.is_file():
            restored = load_p5_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                expected_profile="pilot",
                expected_candidate_id=candidate_id,
                expected_candidate_index=candidate_index,
                expected_config_hash=config.config_hash(),
                expected_protocol_hash=config.protocol_version,
            )
            start_step = restored.global_step
            task_steps = dict(restored.task_steps)
            candidate["validation_macro"] = list(restored.validation_macro)
            candidate["router_gradient_seen"] = restored.router_gradient_seen
            candidate["predictor_gradient_seen"] = restored.predictor_gradient_seen
            if config.telemetry_enabled:
                _truncate_telemetry(
                    cell_directory / "telemetry-v3.jsonl", completed_steps=start_step
                )
        candidate["status"] = "RUNNING"
        losses: list[float] = []
        router_gradient_seen = bool(candidate.get("router_gradient_seen", False))
        predictor_gradient_seen = bool(candidate.get("predictor_gradient_seen", False))
        for step in range(start_step, config.steps_per_preset):
            if stop_file.is_file():
                registry["status"] = "stopped"
                candidate["step"] = step
                _write_json(directory / "registry.json", registry)
                return {
                    "run_id": run_id,
                    "status": "stopped",
                    "artifact_dir": str(directory),
                }
            task = _TASKS[step % len(_TASKS)]
            task_step = task_steps[task.task_id]
            start = task_step * config.batch_size
            indices = [
                (start + offset) % config.train_samples for offset in range(config.batch_size)
            ]
            task_steps[task.task_id] += 1
            batch = task.generate("train", indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_batch(batch)
            _check_routing(output, batch, config.dual_route_fraction)
            loss, _ = _weighted_loss(
                output,
                batch,
                P5QualificationConfig(),
                temporal_weight=temporal_weight,
                semantic_weight=semantic_weight,
                dual_budget_weight=config.dual_budget_weight,
            )
            loss.backward()  # type: ignore[no-untyped-call]
            router = cast(torch.nn.Module, model.registry.get(SPARSE_ROUTER_V3))
            predictor = cast(torch.nn.Module, model.registry.get(PREDICTIVE_ADAPTER_V3))
            router_gradient_seen |= any(
                parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
                for parameter in router.parameters()
            )
            predictor_gradient_seen |= any(
                parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0
                for parameter in predictor.parameters()
            )
            if config.telemetry_enabled:
                from neuromorphic.training.p5_telemetry import build_p5_telemetry_event

                gradient_terms = [
                    parameter.grad.detach().square().sum()
                    for parameter in router.parameters()
                    if parameter.grad is not None
                ]
                event = build_p5_telemetry_event(
                    run_id=run_id,
                    global_step=step,
                    batch=batch,
                    output=output,
                    scorer_grad_norm=(
                        torch.stack(gradient_terms).sum().sqrt() if gradient_terms else None
                    ),
                )
                with (cell_directory / "telemetry-v3.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            completed = step + 1
            candidate["step"] = completed
            candidate["router_gradient_seen"] = router_gradient_seen
            candidate["predictor_gradient_seen"] = predictor_gradient_seen
            if completed % config.validation_interval == 0:
                scores, _ = _validation_summary(model, config, device)
                macro = sum(scores.values()) / len(scores)
                candidate.setdefault("validation_macro", []).append(macro)
            if completed % config.checkpoint_interval == 0:
                curve = candidate.get("validation_macro", [])
                best = max(curve) if isinstance(curve, list) and curve else 0.0
                save_p5_checkpoint(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    state=P5CheckpointState(
                        profile="pilot",
                        candidate_id=candidate_id,
                        candidate_index=candidate_index,
                        global_step=completed,
                        task_steps=task_steps,
                        config_hash=config.config_hash(),
                        protocol_hash=config.protocol_version,
                        best_metrics={"macro": float(best)},
                        validation_macro=tuple(float(value) for value in curve),
                        router_gradient_seen=router_gradient_seen,
                        predictor_gradient_seen=predictor_gradient_seen,
                        stale_evaluations=0,
                        last_loss=losses[-1],
                    ),
                )
                _write_json(
                    directory / "heartbeat.json",
                    {
                        "candidate_id": candidate_id,
                        "candidate_index": candidate_index,
                        "step": completed,
                        "max_steps": config.steps_per_preset,
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                )
                _write_json(directory / "registry.json", registry)
        scores, forecast = _validation_summary(model, config, device)
        curve_value = candidate.get("validation_macro", [])
        curve = [float(value) for value in curve_value] if isinstance(curve_value, list) else []
        eligible = (
            router_gradient_seen
            and predictor_gradient_seen
            and all(
                value["forecast_error"] < value["persistence_error"] for value in forecast.values()
            )
        )
        summary = {
            "schema_version": "p5-pilot-candidate-v1",
            "candidate_id": candidate_id,
            "settings": {
                "learning_rate": learning_rate,
                "temporal_loss_weight": temporal_weight,
                "semantic_loss_weight": semantic_weight,
                "weight_decay": config.weight_decay,
            },
            "eligible": eligible,
            "validation_scores": scores,
            "validation_macro_aulc": sum(curve) / max(len(curve), 1),
            "final_loss": sum(losses[-10:]) / max(len(losses[-10:]), 1),
            "forecast": forecast,
            "router_gradient_seen": router_gradient_seen,
            "predictor_gradient_seen": predictor_gradient_seen,
        }
        _write_json(cell_directory / "summary.json", summary)
        candidate.update(
            {
                "status": "COMPLETED",
                "eligible": eligible,
                "validation_macro_aulc": summary["validation_macro_aulc"],
                "final_loss": summary["final_loss"],
                "summary_sha256": _sha256(cell_directory / "summary.json"),
            }
        )
        _write_json(directory / "registry.json", registry)

    candidates = [
        json.loads(
            (directory / "cells" / str(candidate["candidate_id"]) / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        for candidate in cells
        if isinstance(candidate, dict)
    ]
    eligible_candidates = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible_candidates:
        registry["status"] = "pilot_failed"
        _write_json(directory / "registry.json", registry)
        return {"run_id": run_id, "status": "pilot_failed", "artifact_dir": str(directory)}
    selected = sorted(
        eligible_candidates,
        key=lambda item: (
            -float(item["validation_macro_aulc"]),
            float(item["final_loss"]),
            str(item["candidate_id"]),
        ),
    )[0]
    selection_path = directory / "pilot-selection.json"
    _write_json(
        selection_path,
        {
            "schema_version": "p5-pilot-selection-v1",
            "status": "PASSED",
            "run_id": run_id,
            "git_commit": commit,
            "git_dirty": bool(_git("status", "--porcelain")),
            "qualification_lock_sha256": qualification_hash,
            "config_hash": config.config_hash(),
            "selection_rule": "eligible,validation_macro_aulc_desc,final_loss_asc,preset_id_asc",
            "selected_preset": selected["candidate_id"],
            "settings": selected["settings"],
            "candidates": candidates,
        },
    )
    registry["status"] = "pilot_passed"
    registry["selected_preset"] = selected["candidate_id"]
    _write_json(directory / "registry.json", registry)
    relative = _repository_relative(selection_path)
    if not _git("status", "--porcelain") and relative is not None:
        _write_json(
            config.control_root.parent / "pilot-lock.json",
            {
                "schema_version": "p5-pilot-lock-v1",
                "status": "PASSED",
                "git_commit": commit,
                "device": str(device),
                "config_hash": config.config_hash(),
                "selected_preset": selected["candidate_id"],
                "settings": selected["settings"],
                "pilot_selection": relative,
                "pilot_selection_sha256": _sha256(selection_path),
            },
        )
    return {
        "run_id": run_id,
        "status": "pilot_passed",
        "artifact_dir": str(directory),
        "selected_preset": selected["candidate_id"],
    }


def verify_p5_run(directory: Path) -> dict[str, object]:
    if (directory / "manifest.json").is_file():
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        artifacts = manifest.get("artifacts", {})
        if not isinstance(artifacts, dict):
            raise ValueError("P5 manifest artifacts are invalid")
        checksums_ok = all(
            (directory / name).is_file() and _sha256(directory / name) == digest
            for name, digest in artifacts.items()
        )
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        return {
            "run_id": summary["run_id"],
            "status": summary["status"],
            "checksums_ok": checksums_ok,
            "artifacts": len(artifacts),
        }
    registry = json.loads((directory / "registry.json").read_text(encoding="utf-8"))
    cells = registry.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError("P5 pilot registry cells are invalid")
    missing = [
        str(cell.get("candidate_id"))
        for cell in cells
        if not isinstance(cell, dict)
        or cell.get("status") != "COMPLETED"
        or not (directory / "cells" / str(cell.get("candidate_id")) / "summary.json").is_file()
    ]
    return {
        "run_id": registry["run_id"],
        "status": registry["status"],
        "cells": len(cells),
        "missing_cells": missing,
        "checksums_ok": not missing,
    }


__all__ = ["execute_p5_pilot", "execute_p5_qualification", "verify_p5_run"]
