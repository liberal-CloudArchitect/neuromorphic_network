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
from neuromorphic.training.config import resolve_device
from neuromorphic.training.p5_config import P5QualificationConfig
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    output: Any, batch: TaskBatch, config: P5QualificationConfig
) -> tuple[Tensor, dict[str, float]]:
    primary = _primary_loss(output.logits, batch)
    weights = {
        "episodic.retrieval": 0.1,
        "episodic.separation": 0.01,
        "working.state_consistency": 0.05,
        "working.gate_regularization": 0.001,
        "predictive.temporal": config.temporal_loss_weight,
        "router.semantic_alignment": config.semantic_loss_weight,
        "router.dual_budget": config.dual_budget_weight,
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
    }
    _write_json(directory / "summary.json", summary)
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
    return {
        "run_id": run_id,
        "status": summary["status"],
        "device": str(device),
        "artifact_dir": str(directory),
        "updates": total_updates,
    }


__all__ = ["execute_p5_qualification"]
