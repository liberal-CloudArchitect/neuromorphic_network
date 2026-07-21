"""Command-line entry point for P1 baselines and P2--P4 experiment suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import ValidationError

from neuromorphic.tasks import create_task
from neuromorphic.training.baselines import (
    GRUBaseline,
    TransformerBaseline,
    profile_macs,
    select_parameter_matched_baseline,
    trainable_parameter_count,
    validate_parameter_target,
)
from neuromorphic.training.checkpoint import CheckpointCompatibilityError
from neuromorphic.training.config import RunConfig, load_run_config, resolve_device
from neuromorphic.training.manifest import build_manifest, file_sha256, write_manifest
from neuromorphic.training.reproducibility import set_global_seed
from neuromorphic.training.trainer import profile_sizes, train_baseline


def _split_hash(task: Any, split: str, size: int) -> str:
    digest = hashlib.sha256()
    for index in range(size):
        digest.update(task.content_hash(split, index).encode("ascii"))
    return digest.hexdigest()


def _validate_match(label: str, actual: float, target: float, tolerance: float) -> None:
    relative_error = abs(actual - target) / target
    if relative_error > tolerance:
        raise ValueError(
            f"{label} target mismatch: target={target}, actual={actual}, "
            f"relative_error={relative_error:.3f}"
        )


def build_baseline(config: RunConfig, task: Any) -> torch.nn.Module:
    """Construct the selected monolithic baseline."""
    auxiliary_classes = 16 if config.task.task_id == "small_graph.v1" else None
    common = {
        "input_dim": task.input_dim,
        "num_classes": task.num_classes,
        "hidden_size": config.model.hidden_size,
        "dropout": config.model.dropout,
        "auxiliary_classes": auxiliary_classes,
    }
    if config.model.target_parameters is not None:
        matched_model, _ = select_parameter_matched_baseline(
            kind=config.model.kind,
            input_dim=task.input_dim,
            num_classes=task.num_classes,
            target=config.model.target_parameters,
            tolerance=config.model.parameter_tolerance,
            layers=config.model.layers,
            heads=config.model.heads,
            dropout=config.model.dropout,
            auxiliary_classes=auxiliary_classes,
        )
        return matched_model
    model: torch.nn.Module
    if config.model.kind == "gru":
        model = GRUBaseline(**common, layers=config.model.layers)
    else:
        model = TransformerBaseline(
            **common,
            layers=config.model.layers,
            heads=config.model.heads,
            feedforward_size=config.model.feedforward_size,
        )
    validate_parameter_target(
        model, config.model.target_parameters, config.model.parameter_tolerance
    )
    return model


def execute(config: RunConfig) -> dict[str, Any]:
    """Execute one validated baseline run and persist its evidence."""
    set_global_seed(config.seed)
    device = resolve_device(config.device)
    task = create_task(config.task.task_id, profile=config.task.profile)
    model = build_baseline(config, task)
    sizes = profile_sizes(config.task.profile)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = config.run_id or (
        f"{config.task.task_id.replace('.', '-')}-{config.model.kind}-s{config.seed}-{timestamp}"
    )
    run_directory = config.output_root / run_id
    frozen_config = config.model_dump(mode="json")
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / "config.json").write_text(
        json.dumps(frozen_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sample = task.generate("train", [0])
    parameters = trainable_parameter_count(model)
    mac_profile = profile_macs(model, sample.sequence_length)
    effective_model = config.model.model_dump(mode="json")
    effective_model["effective_hidden_size"] = getattr(model, "hidden_size", None)
    effective_model["effective_layers"] = getattr(model, "layers", None)
    effective_model["effective_feedforward_size"] = getattr(model, "feedforward_size", None)
    if config.model.target_parameters is not None:
        effective_model["parameter_match"] = {
            "target": config.model.target_parameters,
            "actual": parameters,
            "relative_error": abs(parameters - config.model.target_parameters)
            / config.model.target_parameters,
        }
    if config.model.target_macs is not None:
        _validate_match(
            "MAC",
            mac_profile.estimated_macs,
            config.model.target_macs,
            config.model.cost_tolerance,
        )
    data = {
        "task_version": task.task_version,
        "split_sizes": sizes,
        "split_hashes": {split: _split_hash(task, split, size) for split, size in sizes.items()},
    }
    manifest = build_manifest(
        run_id=run_id,
        seed=config.seed,
        device=device,
        task={"task_id": config.task.task_id, "profile": config.task.profile},
        model=effective_model,
        optimizer=config.optimizer.model_dump(mode="json"),
        data=data,
        budget=config.training.model_dump(mode="json"),
        parameters=parameters,
        estimated_macs=mac_profile.estimated_macs,
        mac_coverage=mac_profile.coverage,
        unsupported_parameters=mac_profile.unsupported_parameters,
        mac_operators=[asdict(operator) for operator in mac_profile.operators],
    )
    write_manifest(run_directory / "manifest.json", manifest)
    try:
        summary = train_baseline(
            config=config,
            task=task,
            model=model,
            device=device,
            run_directory=run_directory,
        )
        if config.training.target_optimizer_steps is not None:
            _validate_match(
                "optimizer steps",
                summary["steps"],
                config.training.target_optimizer_steps,
                config.training.compute_tolerance,
            )
        if config.training.target_training_tokens is not None:
            _validate_match(
                "training tokens",
                summary["training_tokens"],
                config.training.target_training_tokens,
                config.training.compute_tolerance,
            )
        if config.model.target_latency_ms is not None:
            _validate_match(
                "P50 latency",
                summary["latency_ms"]["p50"],
                config.model.target_latency_ms,
                config.model.cost_tolerance,
            )
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        manifest["artifacts"] = {
            path.name: {"sha256": file_sha256(path)}
            for path in run_directory.iterdir()
            if path.is_file() and path.name != "manifest.json"
        }
        write_manifest(run_directory / "manifest.json", manifest)
        raise
    (run_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["status"] = "completed"
    manifest["cost"]["latency_ms"] = summary["latency_ms"]
    manifest["cost"]["wall_clock_seconds"] = summary["wall_clock_seconds"]
    manifest["cost"]["peak_memory_bytes"] = summary["peak_memory_bytes"]
    manifest["cost"]["peak_memory_method"] = summary["peak_memory_method"]
    manifest["budget"]["actual_optimizer_steps"] = summary["steps"]
    manifest["budget"]["actual_training_examples"] = summary["training_examples"]
    manifest["budget"]["actual_training_tokens"] = summary["training_tokens"]
    manifest["budget"]["actual_validation_evaluations"] = summary["validation_evaluations"]
    manifest["artifacts"] = {
        path.name: {"sha256": file_sha256(path)}
        for path in run_directory.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    write_manifest(run_directory / "manifest.json", manifest)
    return {"run_id": run_id, "device": str(device), "artifact_dir": str(run_directory), **summary}


def main(arguments: list[str] | None = None) -> int:
    """Parse arguments and return the documented CLI exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parsed = parser.parse_args(arguments)
    try:
        raw_config = yaml.safe_load(parsed.config.read_text(encoding="utf-8"))
        if isinstance(raw_config, dict) and raw_config.get("schema_version") == "p2-suite-v1":
            from neuromorphic.training.p2_config import load_p2_suite_config
            from neuromorphic.training.p2_suite import execute_p2_suite

            result = execute_p2_suite(load_p2_suite_config(parsed.config))
        elif isinstance(raw_config, dict) and raw_config.get("schema_version") == "p3-suite-v1":
            from neuromorphic.training.p3_config import load_p3_suite_config
            from neuromorphic.training.p3_suite import execute_p3_suite

            result = execute_p3_suite(load_p3_suite_config(parsed.config))
        elif isinstance(raw_config, dict) and raw_config.get("schema_version") == "p4-suite-v1":
            from neuromorphic.training.p4_config import load_p4_suite_config
            from neuromorphic.training.p4_suite import execute_p4_suite

            result = execute_p4_suite(load_p4_suite_config(parsed.config))
        else:
            result = execute(load_run_config(parsed.config))
    except FloatingPointError as error:
        print(json.dumps({"error": str(error), "exit_code": 3}), file=sys.stderr)
        return 3
    except CheckpointCompatibilityError as error:
        print(json.dumps({"error": str(error), "exit_code": 4}), file=sys.stderr)
        return 4
    except (ValidationError, ValueError, OSError) as error:
        print(json.dumps({"error": str(error), "exit_code": 2}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    if result.get("status") in {
        "qualification_failed",
        "pilot_failed",
        "completed_with_failures",
        "mechanism_failed",
        "full_failed",
        "resource_limit",
        "stopped",
    }:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
