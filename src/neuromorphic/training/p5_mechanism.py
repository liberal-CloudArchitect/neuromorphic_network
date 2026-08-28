"""P5 three-seed mechanism matrix with strict recovery and paired statistics."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch

from neuromorphic.evaluation.p3_records import p3_sample_records
from neuromorphic.evaluation.p3_statistics import (
    adjust_family,
    paired_hierarchical_bootstrap,
)
from neuromorphic.modules.network_v3 import ModularBrainNetworkV3
from neuromorphic.tasks.base import DatasetSplit
from neuromorphic.training.checkpoint import CheckpointCompatibilityError
from neuromorphic.training.config import resolve_device
from neuromorphic.training.p5_checkpoint import (
    P5CheckpointState,
    load_p5_checkpoint,
    save_p5_checkpoint,
)
from neuromorphic.training.p5_config import (
    P5_PILOT_PRESETS,
    P5MechanismCell,
    P5MechanismConfig,
    P5QualificationConfig,
)
from neuromorphic.training.p5_suite import (
    _TASKS,
    _check_routing,
    _git,
    _repository_relative,
    _sha256,
    _validation_summary,
    _weighted_loss,
    _write_json,
)
from neuromorphic.training.reproducibility import set_global_seed

_METRIC_KEYS = {
    "associative_recall.v1": "query_accuracy",
    "delayed_rule_switch.v1": "response_accuracy",
    "small_graph.v1": "optimal_action_rate",
}


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_lock(path: Path | None, *, commit: str, label: str) -> str | None:
    if path is None:
        return None
    value = _json(path)
    if value.get("status") != "PASSED" or value.get("git_commit") != commit:
        raise ValueError(f"P5 {label} lock is failed or belongs to another commit")
    evidence_key = "qualification_report" if label == "qualification" else "pilot_selection"
    checksum_key = f"{evidence_key}_sha256"
    evidence = value.get(evidence_key)
    checksum = value.get(checksum_key)
    if not isinstance(evidence, str) or not isinstance(checksum, str):
        raise ValueError(f"P5 {label} lock evidence is incomplete")
    evidence_path = Path(evidence)
    if not evidence_path.is_file() or _sha256(evidence_path) != checksum:
        raise ValueError(f"P5 {label} evidence checksum does not match")
    return _sha256(path)


def _settings(config: P5MechanismConfig) -> tuple[float, float, float]:
    return P5_PILOT_PRESETS[config.selected_preset]


def _validate_mechanism_qualification_lock(path: Path | None, *, commit: str) -> str | None:
    if path is None:
        return None
    value = _json(path)
    if value.get("status") != "PASSED" or value.get("git_commit") != commit:
        raise ValueError("P5 mechanism qualification lock is invalid")
    report = value.get("mechanism_qualification_report")
    checksum = value.get("mechanism_qualification_report_sha256")
    if not isinstance(report, str) or not isinstance(checksum, str):
        raise ValueError("P5 mechanism qualification evidence is incomplete")
    report_path = Path(report)
    if not report_path.is_file() or _sha256(report_path) != checksum:
        raise ValueError("P5 mechanism qualification checksum does not match")
    return _sha256(path)


def _model(
    cell: P5MechanismCell, config: P5MechanismConfig, device: torch.device
) -> ModularBrainNetworkV3:
    dual = 0.0 if cell.variant == "no-dual-route" else config.dual_route_fraction
    return ModularBrainNetworkV3(dual_route_fraction=dual).to(device)


def _forward_kwargs(cell: P5MechanismCell) -> dict[str, object]:
    if cell.variant == "predictor-off":
        return {"predictor_mode": "off"}
    if cell.variant in {"surprise-off", "acute-surprise-off"}:
        return {"predictor_mode": "feedback_zero"}
    if cell.variant == "shuffle-surprise":
        return {"predictor_mode": "shuffle_forecast"}
    if cell.variant == "dense-memory":
        return {"routing_mode": "dense"}
    if cell.variant == "no-semantic-reservation":
        return {"routing_mode": "no_reservation"}
    return {}


def _registry(config: P5MechanismConfig, run_id: str, commit: str) -> dict[str, object]:
    return {
        "schema_version": "p5-mechanism-registry-v1",
        "status": "running",
        "run_id": run_id,
        "profile": config.profile,
        "git_commit": commit,
        "config_hash": config.config_hash(),
        "cells": [
            {
                "cell_id": cell.cell_id,
                "status": "PENDING",
                "step": 0,
                "validation_macro": [],
            }
            for cell in config.matrix()
        ],
    }


def _score_records(
    model: ModularBrainNetworkV3,
    cell: P5MechanismCell,
    config: P5MechanismConfig,
    device: torch.device,
    deadline: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    prediction: dict[str, dict[str, float]] = {}
    active_calls = 0.0
    dense_calls = 0.0
    active_macs = 0.0
    dense_macs = 0.0
    semantic_required = 0.0
    semantic_executed = 0.0
    dual_tokens = 0.0
    valid_tokens = 0.0
    model.eval()
    with torch.no_grad():
        for task in _TASKS:
            covered = 0.0
            error_sum = 0.0
            persistence_sum = 0.0
            split_sizes: tuple[tuple[DatasetSplit, int], ...] = (
                ("analysis", config.analysis_samples),
                ("test", config.test_samples),
            )
            for split, size in split_sizes:
                for start in range(0, size, config.batch_size):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("P5 mechanism wall-clock budget exhausted")
                    indices = list(range(start, min(start + config.batch_size, size)))
                    batch = task.generate(split, indices, device=device)
                    output = model.forward_batch(
                        batch,
                        phase="evaluate",
                        **_forward_kwargs(cell),  # type: ignore[arg-type]
                    )
                    if cell.variant not in {"dense-memory", "no-semantic-reservation"}:
                        _check_routing(
                            output,
                            batch,
                            0.0 if cell.variant == "no-dual-route" else config.dual_route_fraction,
                        )
                    enriched = p3_sample_records(
                        output,
                        batch,
                        run_seed=cell.seed,
                        model_id="modular-v3",
                        variant_id=cell.variant,
                    )
                    key = _METRIC_KEYS[task.task_id]
                    for record in enriched:
                        record["value"] = _number(record[key], label=key)
                        record["cell_id"] = cell.cell_id
                    records.extend(enriched)
                    if split == "analysis":
                        transitions = float(output.forecast_transition_mask.sum().cpu())
                        covered += transitions
                        error_sum += float(output.forecast_error.sum().cpu())
                        persistence_sum += float(output.persistence_error.sum().cpu())
                    active_calls += float(output.cost_statistics["optional.active_calls"].cpu())
                    dense_calls += float(output.cost_statistics["optional.dense_calls"].cpu())
                    episodic_calls = float(
                        output.module_metrics["selected.episodic_memory.v1"].cpu()
                    )
                    working_calls = float(output.module_metrics["selected.working_memory.v1"].cpu())
                    active_macs += episodic_calls * 32768.0 + working_calls * 12800.0
                    dense_macs += float(batch.valid_mask.sum().cpu()) * (32768.0 + 12800.0)
                    semantic_required += float(
                        output.module_metrics["routing.reserved_tokens"].cpu()
                    )
                    semantic_executed += float(
                        output.module_metrics["routing.reserved_executed"].cpu()
                    )
                    dual_tokens += float(output.module_metrics["routing.dual_tokens"].cpu())
                    valid_tokens += float(batch.valid_mask.sum().cpu())
            prediction[task.task_id] = {
                "covered": covered,
                "forecast_error": error_sum / max(covered, 1.0),
                "persistence_error": persistence_sum / max(covered, 1.0),
            }
    model.train()
    return records, {
        "prediction": prediction,
        "routing": {
            "active_calls": active_calls,
            "dense_calls": dense_calls,
            "active_macs": active_macs,
            "dense_macs": dense_macs,
            "semantic_required": semantic_required,
            "semantic_executed": semantic_executed,
            "dual_tokens": dual_tokens,
            "valid_tokens": valid_tokens,
            "capacity_drops": 0,
        },
    }


def _parent_checkpoint(directory: Path, seed: int) -> Path:
    return directory / "cells" / f"full__s{seed}" / "checkpoint.pt"


def _load_parent(
    directory: Path,
    cell: P5MechanismCell,
    config: P5MechanismConfig,
    model: ModularBrainNetworkV3,
    optimizer: torch.optim.Optimizer,
) -> None:
    parent = _parent_checkpoint(directory, cell.seed)
    if not parent.is_file():
        raise CheckpointCompatibilityError(f"missing full parent checkpoint for seed {cell.seed}")
    load_p5_checkpoint(
        parent,
        model=model,
        optimizer=optimizer,
        expected_profile="mechanism",
        expected_candidate_id=f"full__s{cell.seed}",
        expected_candidate_index=(
            list(config.matrix()).index(
                next(item for item in config.matrix() if item.cell_id == f"full__s{cell.seed}")
            )
        ),
        expected_config_hash=config.config_hash(),
        expected_protocol_hash=config.protocol_version,
        restore_rng=False,
    )


def _train_cell(
    model: ModularBrainNetworkV3,
    optimizer: torch.optim.Optimizer,
    cell: P5MechanismCell,
    cell_index: int,
    config: P5MechanismConfig,
    device: torch.device,
    directory: Path,
    registry: dict[str, object],
    entry: dict[str, object],
    deadline: float,
    prior_wall_clock: float,
    suite_started: float,
) -> dict[str, object]:
    learning_rate, temporal_weight, semantic_weight = _settings(config)
    del learning_rate
    checkpoint = directory / "cells" / cell.cell_id / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    task_steps = {task.task_id: 0 for task in _TASKS}
    start_step = 0
    curve_value = entry.get("validation_macro", [])
    curve = [float(value) for value in curve_value] if isinstance(curve_value, list) else []
    best = max(curve) if curve else -math.inf
    stale = 0
    last_loss: float | None = None
    if checkpoint.is_file() and _integer(entry.get("step", 0), label="cell step") > 0:
        restored = load_p5_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            expected_profile="mechanism",
            expected_candidate_id=cell.cell_id,
            expected_candidate_index=cell_index,
            expected_config_hash=config.config_hash(),
            expected_protocol_hash=config.protocol_version,
        )
        start_step = restored.global_step
        task_steps = dict(restored.task_steps)
        curve = list(restored.validation_macro)
        best = restored.best_metrics.get("macro", -math.inf)
        stale = restored.stale_evaluations
        last_loss = restored.last_loss
    for step in range(start_step, cell.max_steps):
        if time.monotonic() >= deadline:
            raise TimeoutError("P5 mechanism wall-clock budget exhausted")
        if (directory / "STOP").is_file():
            return {"stopped": True, "steps": step, "validation_macro": curve}
        task = _TASKS[step % len(_TASKS)]
        task_step = task_steps[task.task_id]
        start = task_step * config.batch_size
        indices = [(start + offset) % config.train_samples for offset in range(config.batch_size)]
        task_steps[task.task_id] += 1
        batch = task.generate("train", indices, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_batch(batch, **_forward_kwargs(cell))  # type: ignore[arg-type]
        _check_routing(output, batch, 0.0 if cell.variant == "no-dual-route" else 0.25)
        loss, _ = _weighted_loss(
            output,
            batch,
            P5QualificationConfig(),
            temporal_weight=temporal_weight,
            semantic_weight=semantic_weight,
            dual_budget_weight=config.dual_budget_weight,
        )
        loss.backward()  # type: ignore[no-untyped-call]
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError("P5 mechanism produced a non-finite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        completed = step + 1
        entry["step"] = completed
        if completed % config.validation_interval == 0:
            scores, _ = _validation_summary(model, cast(Any, config), device)
            macro = sum(scores.values()) / len(scores)
            curve.append(macro)
            entry["validation_macro"] = curve
            if macro >= best + 0.001:
                best = macro
                stale = 0
            else:
                stale += 1
        if completed % config.checkpoint_interval == 0:
            save_p5_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                state=P5CheckpointState(
                    profile="mechanism",
                    candidate_id=cell.cell_id,
                    candidate_index=cell_index,
                    global_step=completed,
                    task_steps=task_steps,
                    config_hash=config.config_hash(),
                    protocol_hash=config.protocol_version,
                    best_metrics={"macro": best if math.isfinite(best) else 0.0},
                    validation_macro=tuple(curve),
                    router_gradient_seen=True,
                    predictor_gradient_seen=True,
                    stale_evaluations=stale,
                    last_loss=last_loss,
                ),
            )
            _write_json(
                directory / "heartbeat.json",
                {
                    "cell_id": cell.cell_id,
                    "cell_index": cell_index,
                    "step": completed,
                    "max_steps": cell.max_steps,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
            registry["wall_clock_seconds"] = prior_wall_clock + time.monotonic() - suite_started
            _write_json(directory / "registry.json", registry)
        if config.profile == "mechanism" and stale >= config.patience:
            break
    completed = _integer(entry.get("step", 0), label="cell step")
    if not checkpoint.is_file() or completed % config.checkpoint_interval != 0:
        save_p5_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            state=P5CheckpointState(
                profile="mechanism",
                candidate_id=cell.cell_id,
                candidate_index=cell_index,
                global_step=completed,
                task_steps=task_steps,
                config_hash=config.config_hash(),
                protocol_hash=config.protocol_version,
                best_metrics={"macro": best if math.isfinite(best) else 0.0},
                validation_macro=tuple(curve),
                router_gradient_seen=True,
                predictor_gradient_seen=True,
                stale_evaluations=stale,
                last_loss=last_loss,
            ),
        )
    return {
        "steps": completed,
        "validation_macro": curve,
        "macro_aulc": sum(curve) / max(len(curve), 1),
        "last_loss": last_loss,
    }


def _paired_records(
    summaries: Mapping[str, Mapping[str, object]],
    *,
    left: str,
    right: str,
    field: str,
    seeds: tuple[int, ...],
    task: str = "macro",
    relative: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    left_records: list[dict[str, object]] = []
    right_records: list[dict[str, object]] = []
    for seed in seeds:
        common = {
            "seed": seed,
            "task_id": task,
            "split": "analysis",
            "distribution": "v1",
            "sample_index": 0,
            "stratum": f"{field}:{task}",
            "model_id": "modular-v3",
        }
        left_value = _number(summaries[f"{left}__s{seed}"][field], label=field)
        right_value = _number(summaries[f"{right}__s{seed}"][field], label=field)
        if relative:
            left_value = (left_value - right_value) / max(abs(right_value), 1.0e-12)
            right_value = 0.0
        left_records.append({**common, "variant_id": left, "value": left_value})
        right_records.append({**common, "variant_id": right, "value": right_value})
    return left_records, right_records


def _formal_evidence(
    config: P5MechanismConfig,
    directory: Path,
    summaries: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    surprise_left, surprise_right = _paired_records(
        summaries,
        left="full",
        right="surprise-off",
        field="macro_aulc",
        seeds=config.seeds,
        relative=True,
    )
    predictor_left, predictor_right = _paired_records(
        summaries,
        left="full",
        right="predictor-off",
        field="macro_aulc",
        seeds=config.seeds,
        relative=True,
    )
    dual_left, dual_right = _paired_records(
        summaries,
        left="full",
        right="no-dual-route",
        field="drs_score",
        seeds=config.seeds,
        task="delayed_rule_switch.v1",
        relative=True,
    )
    surprise = paired_hierarchical_bootstrap(
        surprise_left, surprise_right, samples=config.bootstrap_samples
    )
    predictor = paired_hierarchical_bootstrap(
        predictor_left, predictor_right, samples=config.bootstrap_samples
    )
    dual = paired_hierarchical_bootstrap(dual_left, dual_right, samples=config.bootstrap_samples)

    full_records: list[dict[str, object]] = []
    dense_records: list[dict[str, object]] = []
    for seed in config.seeds:
        for variant, target in (("full", full_records), ("dense-memory", dense_records)):
            path = directory / "cells" / f"{variant}__s{seed}" / "sample-records.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record["split"] == "test":
                    value = dict(record)
                    if variant == "full":
                        value["value"] = _number(value["value"], label="sample value") + 0.02
                    target.append(value)
    sparse = paired_hierarchical_bootstrap(
        full_records, dense_records, samples=config.bootstrap_samples
    )
    forecast_left: list[dict[str, object]] = []
    forecast_right: list[dict[str, object]] = []

    final_deltas: dict[str, float] = {}
    predictor_deltas: dict[str, float] = {}
    dense_deltas: dict[str, float] = {}
    prediction_improvements: dict[str, float] = {}
    active = dense = required = executed = dual_tokens = valid = 0.0
    for seed in config.seeds:
        full = summaries[f"full__s{seed}"]
        for comparator, delta_target in (
            ("surprise-off", final_deltas),
            ("dense-memory", dense_deltas),
        ):
            other = summaries[f"{comparator}__s{seed}"]
            for task_id, score in cast(Mapping[str, float], full["test_scores"]).items():
                delta_target[f"s{seed}:{task_id}"] = (
                    score - cast(Mapping[str, float], other["test_scores"])[task_id]
                )
        predictor_other = summaries[f"predictor-off__s{seed}"]
        for task_id, score in cast(Mapping[str, float], full["test_scores"]).items():
            predictor_deltas[f"s{seed}:{task_id}"] = (
                score - cast(Mapping[str, float], predictor_other["test_scores"])[task_id]
            )
        for task_id, values in cast(Mapping[str, Mapping[str, float]], full["prediction"]).items():
            persistence = values["persistence_error"]
            improvement = (persistence - values["forecast_error"]) / max(persistence, 1.0e-12)
            prediction_improvements[f"s{seed}:{task_id}"] = improvement
            common = {
                "seed": seed,
                "task_id": task_id,
                "split": "analysis",
                "distribution": "v1",
                "sample_index": 0,
                "stratum": f"forecast:{task_id}",
                "model_id": "modular-v3",
            }
            forecast_left.append({**common, "variant_id": "forecast", "value": improvement})
            forecast_right.append({**common, "variant_id": "persistence", "value": 0.0})
        routing = cast(Mapping[str, float], full["routing"])
        active += routing["active_macs"]
        dense += routing["dense_macs"]
        required += routing["semantic_required"]
        executed += routing["semantic_executed"]
        dual_tokens += routing["dual_tokens"]
        valid += routing["valid_tokens"]
    mac_reduction = 1.0 - active / max(dense, 1.0)
    forecast_bootstrap = paired_hierarchical_bootstrap(
        forecast_left, forecast_right, samples=config.bootstrap_samples
    )
    adjusted = adjust_family([surprise, predictor, dual, sparse, forecast_bootstrap])
    passed = (
        surprise.estimate >= 0.05
        and surprise.lower > 0.0
        and adjusted[0] <= 0.05
        and predictor.estimate >= 0.05
        and predictor.lower > 0.0
        and adjusted[1] <= 0.05
        and dual.estimate >= 0.03
        and dual.lower > 0.0
        and adjusted[2] <= 0.05
        and sparse.lower > -0.02
        and adjusted[3] <= 0.05
        and min(final_deltas.values()) >= -0.02
        and min(predictor_deltas.values()) >= -0.02
        and min(dense_deltas.values()) >= -0.02
        and min(prediction_improvements.values()) >= 0.05
        and forecast_bootstrap.lower > 0.0
        and adjusted[4] <= 0.05
        and mac_reduction >= 0.20
        and required > 0
        and executed == required
        and dual_tokens / max(valid, 1.0) <= 0.25
    )
    return {
        "status": "PASSED" if passed else "FAILED",
        "surprise_causality": asdict(surprise) | {"holm_adjusted_p": adjusted[0]},
        "predictor_causality": asdict(predictor) | {"holm_adjusted_p": adjusted[1]},
        "dual_route_causality": asdict(dual) | {"holm_adjusted_p": adjusted[2]},
        "sparse_noninferiority": asdict(sparse) | {"holm_adjusted_p": adjusted[3]},
        "forecast_quality": asdict(forecast_bootstrap) | {"holm_adjusted_p": adjusted[4]},
        "minimum_surprise_final_delta": min(final_deltas.values()),
        "minimum_predictor_final_delta": min(predictor_deltas.values()),
        "minimum_dense_delta": min(dense_deltas.values()),
        "minimum_prediction_improvement": min(prediction_improvements.values()),
        "mac_reduction": mac_reduction,
        "semantic_coverage": executed / max(required, 1.0),
        "dual_fraction": dual_tokens / max(valid, 1.0),
    }


def execute_p5_mechanism(config: P5MechanismConfig) -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    if config.expected_git_commit is not None and config.expected_git_commit != commit:
        raise ValueError("P5 mechanism expected_git_commit does not match HEAD")
    qualification_hash = _validate_lock(
        config.qualification_report, commit=commit, label="qualification"
    )
    pilot_hash = _validate_lock(config.pilot_lock, commit=commit, label="pilot")
    if config.pilot_lock is not None:
        pilot_value = _json(config.pilot_lock)
        if pilot_value.get("selected_preset") != config.selected_preset:
            raise ValueError("P5 mechanism preset does not match the pilot lock")
    mechanism_qualification_hash = _validate_mechanism_qualification_lock(
        config.mechanism_qualification_report, commit=commit
    )
    device = resolve_device(config.device)
    if config.resume is None:
        run_id = config.run_id or (
            f"p5-{config.profile}-{str(device).replace(':', '-')}-"
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
        directory = config.output_root / run_id
        directory.mkdir(parents=True, exist_ok=False)
        registry = _registry(config, run_id, commit)
        _write_json(directory / "config.json", config.model_dump(mode="json"))
        _write_json(directory / "registry.json", registry)
    else:
        directory = config.resume
        run_id = directory.name
        registry = _json(directory / "registry.json")
        if (
            registry.get("git_commit") != commit
            or registry.get("config_hash") != config.config_hash()
        ):
            raise CheckpointCompatibilityError("P5 mechanism registry is incompatible")
        registry["status"] = "running"
    stop_file = directory / "STOP"
    stop_file.unlink(missing_ok=True)
    prior_wall_clock = _number(registry.get("wall_clock_seconds", 0.0), label="wall_clock_seconds")
    suite_started = time.monotonic()
    remaining_budget = config.wall_clock_hours * 3600.0 - prior_wall_clock
    if remaining_budget <= 0.0:
        registry["status"] = "resource_limit"
        _write_json(directory / "registry.json", registry)
        return {"run_id": run_id, "status": "resource_limit", "artifact_dir": str(directory)}
    deadline = suite_started + remaining_budget
    cells = config.matrix()
    entries = registry.get("cells")
    if not isinstance(entries, list) or len(entries) != len(cells):
        raise ValueError("P5 mechanism registry cells are invalid")
    summaries: dict[str, Mapping[str, object]] = {}
    for index, (cell, entry_value) in enumerate(zip(cells, entries, strict=True)):
        if not isinstance(entry_value, dict):
            raise ValueError("P5 mechanism registry entry is invalid")
        entry = entry_value
        cell_directory = directory / "cells" / cell.cell_id
        summary_path = cell_directory / "summary.json"
        if entry.get("status") == "COMPLETED" and summary_path.is_file():
            summaries[cell.cell_id] = _json(summary_path)
            continue
        set_global_seed(cell.seed)
        model = _model(cell, config, device)
        learning_rate, _, _ = _settings(config)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
        )
        entry["status"] = "RUNNING"
        _write_json(directory / "registry.json", registry)
        if cell.retrained:
            try:
                training = _train_cell(
                    model,
                    optimizer,
                    cell,
                    index,
                    config,
                    device,
                    directory,
                    registry,
                    entry,
                    deadline,
                    prior_wall_clock,
                    suite_started,
                )
            except TimeoutError:
                entry["status"] = "RESOURCE_LIMIT"
                registry["status"] = "resource_limit"
                registry["wall_clock_seconds"] = prior_wall_clock + time.monotonic() - suite_started
                _write_json(directory / "registry.json", registry)
                return {
                    "run_id": run_id,
                    "status": "resource_limit",
                    "artifact_dir": str(directory),
                }
            if training.get("stopped"):
                registry["status"] = "stopped"
                entry["status"] = "PENDING"
                _write_json(directory / "registry.json", registry)
                return {"run_id": run_id, "status": "stopped", "artifact_dir": str(directory)}
        else:
            _load_parent(directory, cell, config, model, optimizer)
            training = {"steps": 0, "validation_macro": [], "macro_aulc": 0.0}
        try:
            records, evaluation = _score_records(model, cell, config, device, deadline)
        except TimeoutError:
            entry["status"] = "RESOURCE_LIMIT"
            registry["status"] = "resource_limit"
            registry["wall_clock_seconds"] = prior_wall_clock + time.monotonic() - suite_started
            _write_json(directory / "registry.json", registry)
            return {"run_id": run_id, "status": "resource_limit", "artifact_dir": str(directory)}
        cell_directory.mkdir(parents=True, exist_ok=True)
        with (cell_directory / "sample-records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        test_scores: dict[str, float] = {}
        for task_id in _METRIC_KEYS:
            values = [
                _number(record["value"], label="sample value")
                for record in records
                if record["split"] == "test" and record["task_id"] == task_id
            ]
            test_scores[task_id] = sum(values) / max(len(values), 1)
        summary = {
            "schema_version": "p5-mechanism-cell-v1",
            "cell_id": cell.cell_id,
            "variant": cell.variant,
            "seed": cell.seed,
            "steps": training.get("steps", 0),
            "macro_aulc": training.get("macro_aulc", 0.0),
            "drs_score": test_scores["delayed_rule_switch.v1"],
            "test_scores": test_scores,
            **evaluation,
        }
        _write_json(summary_path, summary)
        entry["status"] = "COMPLETED"
        entry["summary_sha256"] = _sha256(summary_path)
        entry["sample_records_sha256"] = _sha256(cell_directory / "sample-records.jsonl")
        summaries[cell.cell_id] = summary
        registry["wall_clock_seconds"] = prior_wall_clock + time.monotonic() - suite_started
        _write_json(directory / "registry.json", registry)
    registry["status"] = (
        "qualification_passed" if config.profile == "qualification" else "completed"
    )
    if config.profile == "qualification":
        report = {
            "schema_version": "p5-mechanism-qualification-report-v1",
            "status": "PASSED",
            "run_id": run_id,
            "git_commit": commit,
            "git_dirty": bool(_git("status", "--porcelain")),
            "config_hash": config.config_hash(),
            "qualification_lock_sha256": qualification_hash,
            "pilot_lock_sha256": pilot_hash,
            "mechanism_qualification_lock_sha256": mechanism_qualification_hash,
            "cells": len(cells),
        }
        report_path = directory / "mechanism-qualification-report.json"
        _write_json(report_path, report)
        relative = _repository_relative(report_path)
        if not report["git_dirty"] and relative is not None:
            _write_json(
                config.control_root.parent / "mechanism-qualification-lock.json",
                {
                    "schema_version": "p5-mechanism-qualification-lock-v1",
                    "status": "PASSED",
                    "git_commit": commit,
                    "config_hash": config.config_hash(),
                    "mechanism_qualification_report": relative,
                    "mechanism_qualification_report_sha256": _sha256(report_path),
                },
            )
    else:
        evidence = _formal_evidence(config, directory, summaries)
        registry["status"] = (
            "mechanism_passed" if evidence["status"] == "PASSED" else "mechanism_failed"
        )
        report_path = directory / "mechanism-report.json"
        _write_json(
            report_path,
            {
                "schema_version": "p5-mechanism-report-v1",
                "run_id": run_id,
                "git_commit": commit,
                "git_dirty": bool(_git("status", "--porcelain")),
                "config_hash": config.config_hash(),
                "mechanism_qualification_lock_sha256": mechanism_qualification_hash,
                "status": evidence["status"],
                "evidence": evidence,
            },
        )
        relative = _repository_relative(report_path)
        if evidence["status"] == "PASSED" and not _git("status", "--porcelain") and relative:
            _write_json(
                config.control_root.parent / "mechanism-lock.json",
                {
                    "schema_version": "p5-mechanism-lock-v1",
                    "status": "PASSED",
                    "git_commit": commit,
                    "config_hash": config.config_hash(),
                    "mechanism_report": relative,
                    "mechanism_report_sha256": _sha256(report_path),
                },
            )
    _write_json(directory / "registry.json", registry)
    registry["wall_clock_seconds"] = prior_wall_clock + time.monotonic() - suite_started
    _write_json(directory / "registry.json", registry)
    return {
        "run_id": run_id,
        "status": registry["status"],
        "artifact_dir": str(directory),
        "completed_cells": len(cells),
    }


__all__ = ["execute_p5_mechanism"]
