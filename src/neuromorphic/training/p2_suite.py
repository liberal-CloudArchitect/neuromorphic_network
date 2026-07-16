"""Execute the frozen P2 pretraining and paired joint-smoke suite."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch

from neuromorphic.tasks import create_task
from neuromorphic.training.config import resolve_device
from neuromorphic.training.manifest import (
    build_manifest,
    file_sha256,
    write_manifest,
)
from neuromorphic.training.modular_trainer import (
    build_modular_network,
    clone_model_state,
    compare_branch_models,
    run_joint_branch,
    run_pretraining,
)
from neuromorphic.training.p2_config import P2SuiteConfig
from neuromorphic.training.reproducibility import set_global_seed


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _paired_metric_difference(first: Mapping[str, object], second: Mapping[str, object]) -> float:
    maximum = 0.0

    def visit(left: object, right: object) -> None:
        nonlocal maximum
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if set(left) != set(right):
                raise ValueError("paired telemetry metric keys differ")
            for key in left:
                visit(left[key], right[key])
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            maximum = max(maximum, abs(float(left) - float(right)))

    visit(first, second)
    return maximum


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile_latency(
    config: P2SuiteConfig,
    model_state: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    model = build_modular_network(config).to(device)
    model.load_state_dict(model_state)
    model.eval()
    task = create_task("delayed_rule_switch.v1", profile="smoke")
    batch = task.generate(
        "validation",
        list(range(min(config.budget.validation_size, 8))),
        device=device,
    )
    durations: list[float] = []
    with torch.no_grad():
        for index in range(12):
            _synchronize(device)
            started = time.perf_counter()
            model.forward_batch(batch, phase="evaluate")
            _synchronize(device)
            if index >= 2:
                durations.append((time.perf_counter() - started) * 1_000.0)
    durations.sort()
    return {
        "p50": durations[len(durations) // 2],
        "p95": durations[math.ceil(0.95 * len(durations)) - 1],
        "samples": float(len(durations)),
    }


def execute_p2_suite(config: P2SuiteConfig) -> dict[str, Any]:
    """Run pretraining once and two semantically paired joint branches."""

    set_global_seed(config.seed)
    device = resolve_device(config.device)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = config.run_id or f"p2-suite-s{config.seed}-{timestamp}"
    directory = config.output_root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(directory / "config.json", config.model_dump(mode="json"))
    model = build_modular_network(config).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    manifest = build_manifest(
        run_id=run_id,
        seed=config.seed,
        device=device,
        task={"task_order": list(config.task_order), "profile": config.profile},
        model={"kind": "modular", **config.model.model_dump(mode="json")},
        optimizer=config.optimizer.model_dump(mode="json"),
        data={
            "train": config.budget.train_size,
            "validation": config.budget.validation_size,
            "test": config.budget.test_size,
            "ood": config.budget.ood_size,
        },
        budget=config.budget.model_dump(mode="json"),
        parameters=parameters,
        estimated_macs=0,
        mac_coverage=0.0,
        unsupported_parameters=(),
        mac_operators=[],
    )
    write_manifest(directory / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        pretrain = run_pretraining(
            model=model,
            config=config,
            device=device,
            directory=directory / "pretrain",
        )
        shared = clone_model_state(model)
        set_global_seed(config.seed)
        off, off_model = run_joint_branch(
            name=f"{run_id}-telemetry-off",
            initial_model_state=shared,
            config=config,
            device=device,
            directory=directory / "telemetry-off",
            telemetry_enabled=False,
        )
        set_global_seed(config.seed)
        on, on_model = run_joint_branch(
            name=f"{run_id}-telemetry-on",
            initial_model_state=shared,
            config=config,
            device=device,
            directory=directory / "telemetry-on",
            telemetry_enabled=True,
        )
        maximum_parameter_difference = compare_branch_models(off_model, on_model, device=device)
        off_evaluations = {key: dict(value) for key, value in off.evaluations.items()}
        on_evaluations = {key: dict(value) for key, value in on.evaluations.items()}
        maximum_metric_difference = _paired_metric_difference(off_evaluations, on_evaluations)
        if maximum_metric_difference > 1e-4:
            raise ValueError("telemetry paired training metric difference exceeds 1e-4")
        if config.profile == "gate":
            for branch in (off, on):
                if not branch.routing.get("exact_top_k"):
                    raise ValueError("GATE-2 requires exact executed top-2 routing")
                if branch.routing.get("capacity_drops") != 0:
                    raise ValueError("GATE-2 forbids routing capacity drops")
                if not branch.routing.get("raw_health_guard"):
                    raise ValueError("GATE-2 raw routing health guard failed")
                active = cast(int, branch.cost_profile["active_optional_macs"])
                dense = cast(int, branch.cost_profile["dense_optional_macs"])
                if active >= dense:
                    raise ValueError("GATE-2 optional active MAC must be below dense MAC")
                if not all(branch.gradient_coverage.values()):
                    raise ValueError("GATE-2 requires gradient coverage for all six modules")
            expected_events = config.budget.joint_steps_per_task * len(config.task_order) * 6
            if on.telemetry_event_count != expected_events or off.telemetry_event_count != 0:
                raise ValueError("GATE-2 telemetry event cardinality is invalid")
        latency = _profile_latency(config, off_model, device)
        telemetry_overhead_seconds = on.wall_clock_seconds - off.wall_clock_seconds
        summary = {
            "schema_version": "p2-suite-summary-v1",
            "run_id": run_id,
            "device": str(device),
            "profile": config.profile,
            "pretrain": pretrain.to_dict(),
            "telemetry_off": off.to_dict(),
            "telemetry_on": on.to_dict(),
            "telemetry_equivalence": {
                "maximum_parameter_difference": maximum_parameter_difference,
                "maximum_metric_difference": maximum_metric_difference,
                "metric_tolerance": 1e-4,
                "event_identity_compared": False,
                "wall_clock_overhead_seconds": telemetry_overhead_seconds,
                "wall_clock_overhead_ratio": telemetry_overhead_seconds
                / max(off.wall_clock_seconds, 1e-12),
            },
            "latency_ms": latency,
            "wall_clock_seconds": time.perf_counter() - started,
        }
        _write_json(directory / "summary.json", summary)
        manifest["status"] = "completed"
        cost_profile = off.cost_profile
        manifest["cost"]["estimated_macs_per_sequence"] = cost_profile["active_total_macs"]
        manifest["cost"]["mac_profiler_coverage"] = cost_profile["parameter_coverage"]
        manifest["cost"]["mac_operators"] = cost_profile["records"]
        manifest["cost"]["latency_ms"] = latency
        manifest["cost"]["wall_clock_seconds"] = summary["wall_clock_seconds"]
        manifest["cost"]["peak_memory_bytes"] = (
            int(torch.mps.current_allocated_memory()) if device.type == "mps" else None
        )
        manifest["cost"]["peak_memory_method"] = (
            "torch.mps.current_allocated_memory" if device.type == "mps" else "unavailable"
        )
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        manifest["cost"]["wall_clock_seconds"] = time.perf_counter() - started
        manifest["artifacts"] = {
            path.relative_to(directory).as_posix(): {"sha256": file_sha256(path)}
            for path in directory.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        write_manifest(directory / "manifest.json", manifest)
        raise
    manifest["artifacts"] = {
        path.relative_to(directory).as_posix(): {"sha256": file_sha256(path)}
        for path in directory.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    write_manifest(directory / "manifest.json", manifest)
    return {"run_id": run_id, "artifact_dir": str(directory), **summary}


__all__ = ["execute_p2_suite"]
