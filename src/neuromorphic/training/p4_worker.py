"""Short-lived P4 worker processes used by the durable suite supervisor."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Literal

import torch

from neuromorphic.training.config import resolve_device
from neuromorphic.training.p4_config import P4ExperimentCell, P4SuiteConfig
from neuromorphic.training.p4_suite import (
    PROTOCOL_HASH,
    P4ResourceLimit,
    _build_model,
    _evaluate_cell,
    _evaluate_live_rollout_view,
    _load_parent,
    _sha256,
    _train_cell,
    _write_json,
)


def _load_config(path: Path) -> P4SuiteConfig:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("P4 worker config must be a JSON object")
    return P4SuiteConfig.model_validate(value)


def _find_cell(config: P4SuiteConfig, cell_id: str) -> tuple[int, P4ExperimentCell]:
    for cursor, cell in enumerate(config.matrix()):
        if cell.cell_id == cell_id:
            return cursor, cell
    raise ValueError(f"P4 worker cell is not in the frozen matrix: {cell_id}")


def _load_evaluation_model(
    config: P4SuiteConfig,
    cell: P4ExperimentCell,
    suite_directory: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, str, Path]:
    model, matching = _build_model(config, cell, device)
    if cell.max_steps == 0:
        _load_parent(model, cell, suite_directory, device)
        parent = suite_directory / "cells" / f"shared__shared__modular-v2__full__s{cell.seed}__all"
        checkpoint = parent / ("best.pt" if (parent / "best.pt").is_file() else "checkpoint.pt")
    else:
        cell_directory = suite_directory / "cells" / cell.cell_id
        checkpoint = cell_directory / (
            "best.pt" if (cell_directory / "best.pt").is_file() else "checkpoint.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"P4 evaluation checkpoint is missing: {checkpoint}")
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
    return model, matching, checkpoint


def _base_result(
    config: P4SuiteConfig,
    cell: P4ExperimentCell,
    *,
    stage: str,
) -> dict[str, object]:
    return {
        "schema_version": "p4-worker-result-v1",
        "status": "COMPLETED",
        "stage": stage,
        "cell_id": cell.cell_id,
        "config_hash": config.config_hash(),
        "matrix_hash": config.matrix_hash(),
        "protocol_hash": PROTOCOL_HASH,
    }


def run_training_stage(
    config: P4SuiteConfig,
    cell: P4ExperimentCell,
    *,
    cursor: int,
    suite_directory: Path,
    output: Path,
    remaining_seconds: float,
    pilot_hash: str | None,
    mechanism_hash: str | None,
) -> dict[str, object]:
    device = resolve_device(config.device)
    started = time.perf_counter()
    deadline = started + max(remaining_seconds, 0.0)
    model, matching = _build_model(config, cell, device)
    cell_directory = suite_directory / "cells" / cell.cell_id
    cell_directory.mkdir(parents=True, exist_ok=True)
    if cell.max_steps == 0:
        _load_parent(model, cell, suite_directory, device)
        training: dict[str, object] = {"steps": 0, "wall_clock_seconds": 0.0}
    else:
        training = _train_cell(
            model,
            cell,
            config,
            device,
            cell_directory,
            suite_directory,
            cursor,
            deadline,
            pilot_hash,
            mechanism_hash,
        )
    status = "COMPLETED"
    if training.get("stopped"):
        status = "STOPPED"
    elif training.get("resource_limited"):
        status = "RESOURCE_LIMIT"
    result = {
        **_base_result(config, cell, stage="train"),
        "status": status,
        "matching": matching,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "training": training,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    checkpoint = cell_directory / (
        "best.pt" if (cell_directory / "best.pt").is_file() else "checkpoint.pt"
    )
    if checkpoint.is_file():
        result["checkpoint"] = checkpoint.relative_to(suite_directory).as_posix()
        result["checkpoint_sha256"] = _sha256(checkpoint)
    _write_json(output, result)
    return result


def run_evaluation_stage(
    config: P4SuiteConfig,
    cell: P4ExperimentCell,
    *,
    suite_directory: Path,
    output: Path,
    artifact_directory: Path,
    remaining_seconds: float,
    kind: Literal["standard", "rollout"],
    task_id: str,
    split: Literal["test", "ood"] | None,
    distribution: str | None,
) -> dict[str, object]:
    device = resolve_device(config.device)
    started = time.perf_counter()
    deadline = started + max(remaining_seconds, 0.0)
    model, matching, checkpoint = _load_evaluation_model(config, cell, suite_directory, device)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    evaluation_cell = cell.model_copy(update={"task_id": task_id})
    if kind == "standard":
        evaluation = _evaluate_cell(
            model,
            evaluation_cell,
            config,
            device,
            artifact_directory,
            suite_directory,
            deadline,
            include_live_rollouts=False,
        )
    else:
        if task_id != "small_graph.v1" or split is None or distribution is None:
            raise ValueError("rollout workers require SmallGraph split and distribution")
        evaluation = _evaluate_live_rollout_view(
            model,
            evaluation_cell,
            config,
            device,
            artifact_directory,
            suite_directory,
            deadline,
            split=split,
            distribution=distribution,
        )
    artifacts = {
        path.name: _sha256(path) for path in artifact_directory.iterdir() if path.is_file()
    }
    result = {
        **_base_result(config, cell, stage=f"evaluate-{kind}"),
        "matching": matching,
        "checkpoint": checkpoint.relative_to(suite_directory).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "task_id": task_id,
        "split": split,
        "distribution": distribution,
        "evaluation": evaluation,
        "artifacts": artifacts,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    _write_json(output, result)
    return result


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--suite-directory", required=True, type=Path)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--stage", required=True, choices=("train", "standard", "rollout"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--remaining-seconds", required=True, type=float)
    parser.add_argument("--pilot-hash")
    parser.add_argument("--mechanism-hash")
    parser.add_argument("--task-id")
    parser.add_argument("--split", choices=("test", "ood"))
    parser.add_argument("--distribution")
    parsed = parser.parse_args(arguments)
    try:
        config = _load_config(parsed.config)
        cursor, cell = _find_cell(config, parsed.cell_id)
        if parsed.stage == "train":
            result = run_training_stage(
                config,
                cell,
                cursor=cursor,
                suite_directory=parsed.suite_directory,
                output=parsed.output,
                remaining_seconds=parsed.remaining_seconds,
                pilot_hash=parsed.pilot_hash,
                mechanism_hash=parsed.mechanism_hash,
            )
        else:
            if parsed.artifact_directory is None or parsed.task_id is None:
                raise ValueError("evaluation workers require artifact-directory and task-id")
            result = run_evaluation_stage(
                config,
                cell,
                suite_directory=parsed.suite_directory,
                output=parsed.output,
                artifact_directory=parsed.artifact_directory,
                remaining_seconds=parsed.remaining_seconds,
                kind=parsed.stage,
                task_id=parsed.task_id,
                split=parsed.split,
                distribution=parsed.distribution,
            )
    except P4ResourceLimit as error:
        _write_json(
            parsed.output,
            {
                "schema_version": "p4-worker-result-v1",
                "status": "RESOURCE_LIMIT",
                "stage": parsed.stage,
                "cell_id": parsed.cell_id,
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        return 75
    except Exception as error:
        _write_json(
            parsed.output,
            {
                "schema_version": "p4-worker-result-v1",
                "status": "FAILED",
                "stage": parsed.stage,
                "cell_id": parsed.cell_id,
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        print(json.dumps({"error": str(error), "type": type(error).__name__}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "COMPLETED" else 75


if __name__ == "__main__":
    raise SystemExit(main())
