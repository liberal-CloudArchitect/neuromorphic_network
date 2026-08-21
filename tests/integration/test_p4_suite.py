from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import jsonschema  # type: ignore[import-untyped]
import pytest
import torch
import yaml
from torch import nn

from neuromorphic.core.registry import PREDICTIVE_ADAPTER_V2, SPARSE_ROUTER_V2
from neuromorphic.tasks import SmallGraphTask
from neuromorphic.telemetry.events_v2 import TelemetryV2Event
from neuromorphic.training import p4_suite
from neuromorphic.training.p4_config import P4_TASK_ORDER, P4SuiteConfig, load_p4_suite_config


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


def _config(root: Path, *, run_id: str = "p4-test") -> P4SuiteConfig:
    return P4SuiteConfig.model_validate(
        {
            "schema_version": "p4-suite-v1",
            "protocol_version": "p4-protocol-v2",
            "profile": "qualification",
            "qualification_only": True,
            "device": "cpu",
            "output_root": str(root),
            "control_root": str(root / "control"),
            "run_id": run_id,
            "seeds": [7],
            "data": {"train": 64, "validation": 32, "analysis": 32, "test": 32, "ood": 32},
            "budget": {
                "batch_size": 8,
                "shared_steps_per_task": 4,
                "per_task_steps": 4,
                "continual_steps_per_stage": 2,
                "validation_interval": 2,
                "checkpoint_interval": 2,
                "patience": 10,
                "min_delta": 0.001,
                "wall_clock_hours": 2,
                "bootstrap_samples": 200,
            },
        }
    )


def _patch_fast_execution(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    def build(config: P4SuiteConfig, cell: Any, device: torch.device) -> tuple[nn.Module, str]:
        del config, device
        calls.append(cell.cell_id)
        return _TinyModel(), "fixture"

    def train(*args: Any, **kwargs: Any) -> dict[str, object]:
        del kwargs
        directory = cast(Path, args[4])
        cell = args[1]
        (directory / "checkpoint.pt").write_bytes(b"fixture")
        aulc = 0.6 if cell.variant_id == "predictor-off" else 0.7
        return {
            "steps": 12,
            "analysis_aulc": {
                "associative_recall.v1": aulc,
                "delayed_rule_switch.v1": aulc,
                "small_graph.v1": aulc,
            },
            "validation_macro_aulc": 0.6,
            "last_loss": 0.4,
            "settings": {
                "learning_rate": 0.0001,
                "weight_decay": 0.01,
                "temporal_loss_weight": 0.05,
            },
            "prediction": {
                "eligible": 8.0,
                "covered": 8.0,
                "coverage": 1.0,
                "relative_improvement": 0.1,
                "feedback_nonzero": 1.0,
            },
        }

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, object]:
        del kwargs
        directory = cast(Path, args[4])
        cell = args[1]
        events: list[dict[str, object]] = []
        for task_id in (
            "associative_recall.v1",
            "delayed_rule_switch.v1",
            "small_graph.v1",
        ):
            events.extend(
                (
                    TelemetryV2Event(
                        event_id=f"{cell.cell_id}:{task_id}:route",
                        run_id=cell.cell_id,
                        global_step=12,
                        task=task_id,
                        event_type="summary",
                        phase="evaluate",
                        module_id=SPARSE_ROUTER_V2,
                        compute_gate=True,
                    ).to_dict(),
                    TelemetryV2Event(
                        event_id=f"{cell.cell_id}:{task_id}:predict",
                        run_id=cell.cell_id,
                        global_step=12,
                        task=task_id,
                        event_type="summary",
                        phase="evaluate",
                        module_id=PREDICTIVE_ADAPTER_V2,
                        compute_gate=True,
                    ).to_dict(),
                )
            )
        p4_suite._write_jsonl(directory / "telemetry-v2.jsonl", events)
        return {
            "record_count": 0,
            "views": {},
            "scores": {task_id: 0.8 for task_id in P4_TASK_ORDER},
            "routing": {
                task_id: {
                    "active_optional_macs": 75.0,
                    "dense_optional_macs": 100.0,
                    "capacity_drops": 0.0,
                    "reserved_total": 10.0 if task_id == "associative_recall.v1" else 0.0,
                    "reserved_executed": (10.0 if task_id == "associative_recall.v1" else 0.0),
                }
                for task_id in P4_TASK_ORDER
            },
            "prediction": {
                task_id: {
                    "eligible": 100.0,
                    "covered": 100.0,
                    "error_sum": 90.0,
                    "persistence_sum": 100.0,
                }
                for task_id in P4_TASK_ORDER
            },
        }

    monkeypatch.setattr(p4_suite, "_build_model", build)
    monkeypatch.setattr(p4_suite, "_train_cell", train)
    monkeypatch.setattr(p4_suite, "_evaluate_cell", evaluate)
    monkeypatch.setattr(p4_suite, "_load_parent", lambda *args, **kwargs: None)


def test_registry_is_atomic_complete_verifiable_and_resume_skips_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _patch_fast_execution(monkeypatch, calls)
    monkeypatch.setattr(p4_suite, "_repository_state", lambda: ("clean-sha", False))
    config = _config(tmp_path)

    result = p4_suite.run_p4_suite(config)

    assert result["status"] == "qualification_passed"
    assert len(calls) == 8
    directory = tmp_path / "p4-test"
    qualification_report = json.loads(
        (directory / "qualification-report.json").read_text(encoding="utf-8")
    )
    assert qualification_report["status"] == "PASSED"
    assert qualification_report["all_cells_completed"] is True
    assert qualification_report["cpu_micro_required"] is True
    assert qualification_report["forecast_path_seen"] is True
    assert qualification_report["feedback_nonzero"] is True
    assert qualification_report["sparse_mac_less_than_dense"] is True
    assert qualification_report["dense_control_matches_dense_macs"] is True
    assert isinstance(qualification_report["registry_checksum"], str)
    verification = p4_suite.verify_p4_run(directory)
    assert verification["missing_cells"] == []
    assert verification["checksums_ok"] is True
    assert not (tmp_path / "control/qualification-lock.json").exists()
    before = list(calls)
    resumed = p4_suite.run_p4_suite(config)
    assert resumed["completed_cells"] == 8
    assert calls == before

    schema = json.loads(
        (Path(__file__).parents[2] / "schemas" / "p4-suite-registry-v1.json").read_text(
            encoding="utf-8"
        )
    )
    registry = json.loads((directory / "registry.json").read_text(encoding="utf-8"))
    jsonschema.validate(registry, schema)


def test_registry_records_terminal_timestamp_after_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _patch_fast_execution(monkeypatch, calls)
    monkeypatch.setattr(p4_suite, "_repository_state", lambda: ("clean-sha", False))
    config = _config(tmp_path, run_id="ended-at")

    p4_suite.run_p4_suite(config)

    registry = json.loads((tmp_path / "ended-at" / "registry.json").read_text(encoding="utf-8"))
    assert isinstance(registry["ended_at"], str)


def test_cell_execution_mode_is_isolated_for_mps(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="isolated").model_copy(update={"device": "mps"})

    assert p4_suite._cell_execution_mode(config) == "isolated"


def test_cell_execution_mode_is_direct_for_cpu(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="direct")

    assert p4_suite._cell_execution_mode(config) == "direct"


def test_isolated_cell_runner_returns_worker_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, run_id="worker")
    cell = config.matrix()[0]
    cell_directory = tmp_path / "worker" / "cells" / cell.cell_id
    suite_directory = tmp_path / "worker"

    monkeypatch.setattr(
        p4_suite,
        "_run_worker",
        lambda *args, **kwargs: {
            "matching": "fixture",
            "training": {"steps": 12},
            "parameters": 1,
            "trainable_parameters": 1,
        },
    )
    monkeypatch.setattr(
        p4_suite,
        "_run_isolated_evaluation",
        lambda *args, **kwargs: {"record_count": 0, "views": {}},
    )

    summary = p4_suite._run_cell_isolated(
        config,
        cell,
        torch.device("cpu"),
        cell_directory,
        suite_directory,
        cursor=0,
        deadline=time.perf_counter() + 60.0,
        pilot_hash=None,
        mechanism_hash=None,
    )

    assert summary["matching"] == "fixture"


def test_failed_cell_is_preserved_and_retried_without_repeating_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _patch_fast_execution(monkeypatch, calls)
    original = p4_suite._build_model
    failed_once = False

    def flaky(config: P4SuiteConfig, cell: Any, device: torch.device) -> tuple[nn.Module, str]:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            calls.append(cell.cell_id)
            raise RuntimeError("injected")
        return original(config, cell, device)

    monkeypatch.setattr(p4_suite, "_build_model", flaky)
    config = _config(tmp_path, run_id="retry")
    first = p4_suite.run_p4_suite(config)
    assert first["status"] == "qualification_failed"
    registry_path = tmp_path / "retry" / "registry.json"
    first_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert first_registry["cells"][0]["status"] == "FAILED"
    failure = tmp_path / "retry" / "cells" / first_registry["cells"][0]["cell_id"] / "failure.json"
    assert failure.is_file()

    completed_before = {
        item["cell_id"] for item in first_registry["cells"] if item["status"] == "COMPLETED"
    }
    calls.clear()
    second = p4_suite.run_p4_suite(config)
    assert second["status"] == "qualification_passed"
    assert len(calls) == 1
    assert calls[0] not in completed_before


def test_p4_task_factory_uses_fresh_namespace() -> None:
    config = _config(Path("artifacts/runs"), run_id="unused")
    task = p4_suite._task(config, "associative_recall.v1")
    batch = task.generate("train", [0])
    assert batch.metadata["namespace"] == "p4"
    assert batch.metadata["task_version"] == "associative-recall-p4-v1"
    assert batch.metadata["split_seed"] == 11101


def test_modular_small_graph_rollout_uses_incremental_causal_steps(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="rollout")
    cell = config.matrix()[0]
    model, _ = p4_suite._build_model(config, cell, torch.device("cpu"))
    task = cast(SmallGraphTask, p4_suite._task(config, "small_graph.v1"))

    records = task.rollout_records(
        p4_suite._rollout_policy(model, cell, task, torch.device("cpu")),
        "test",
        [0, 1],
        max_steps=3,
    )

    assert len(records) == 2
    assert {record["sample_index"] for record in records} == {0, 1}


@pytest.mark.parametrize("model_id", ["modular-v2", "gru"])
def test_batched_small_graph_rollout_matches_scalar_episode_semantics(
    tmp_path: Path, model_id: str
) -> None:
    config = _config(tmp_path, run_id=f"rollout-equivalence-{model_id}")
    cell = config.matrix()[0].model_copy(update={"model_id": model_id})
    model, _ = p4_suite._build_model(config, cell, torch.device("cpu"))
    task = cast(SmallGraphTask, p4_suite._task(config, "small_graph.v1"))
    sample_indices = list(range(6))

    scalar = task.rollout_records(
        p4_suite._rollout_policy(model, cell, task, torch.device("cpu")),
        "test",
        sample_indices,
    )
    batched = p4_suite._rollout_records_with_chance(
        model,
        cell,
        task,
        "test",
        len(sample_indices),
        torch.device("cpu"),
        batch_size=3,
        deadline=time.perf_counter() + 120.0,
    )

    comparable_fields = {
        "task_id",
        "task_version",
        "distribution",
        "split",
        "sample_index",
        "success_rate",
        "path_excess",
        "optimal_action_rate",
        "invalid_actions",
        "node_count",
        "shortest_distance",
        "bootstrap_stratum",
    }
    assert [{field: record[field] for field in comparable_fields} for record in batched] == [
        {field: record[field] for field in comparable_fields} for record in scalar
    ]


def test_batched_small_graph_rollout_honors_deadline(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="rollout-deadline")
    cell = config.matrix()[0]
    model, _ = p4_suite._build_model(config, cell, torch.device("cpu"))
    task = cast(SmallGraphTask, p4_suite._task(config, "small_graph.v1"))

    with pytest.raises(p4_suite.P4ResourceLimit, match="wall-clock budget"):
        p4_suite._rollout_records_with_chance(
            model,
            cell,
            task,
            "test",
            2,
            torch.device("cpu"),
            batch_size=2,
            deadline=time.perf_counter() - 1.0,
        )


def test_live_rollout_batch_is_separate_from_training_batch(tmp_path: Path) -> None:
    qualification = _config(tmp_path, run_id="rollout-batch")
    mechanism = load_p4_suite_config(Path("configs/experiments/p4/mechanism.yaml"))

    assert qualification.budget.batch_size == 8
    assert p4_suite._live_rollout_batch_size(qualification, qualification.data.test) == 32
    assert mechanism.budget.batch_size == 64
    assert mechanism.budget.wall_clock_hours == 48
    assert p4_suite._live_rollout_batch_size(mechanism, mechanism.data.test) == 1024


def test_batched_rollout_reports_intra_view_progress(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="rollout-progress")
    cell = config.matrix()[0]
    model, _ = p4_suite._build_model(config, cell, torch.device("cpu"))
    task = cast(SmallGraphTask, p4_suite._task(config, "small_graph.v1"))
    updates: list[tuple[int, int, int]] = []

    records = p4_suite._rollout_records_with_chance(
        model,
        cell,
        task,
        "test",
        4,
        torch.device("cpu"),
        batch_size=2,
        deadline=time.perf_counter() + 120.0,
        progress=lambda completed, total, turn: updates.append((completed, total, turn)),
    )

    assert len(records) == 4
    assert updates
    assert {total for _, total, _ in updates} == {4}
    assert {completed for completed, _, _ in updates} == {0, 2}
    assert all(turn >= 1 for _, _, turn in updates)


def test_suite_emits_durable_structured_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []
    _patch_fast_execution(monkeypatch, calls)
    monkeypatch.setattr(p4_suite, "_repository_state", lambda: ("clean-sha", False))

    result = p4_suite.run_p4_suite(_config(tmp_path, run_id="progress"))

    events = [
        json.loads(line)["event"] for line in capsys.readouterr().out.splitlines() if line.strip()
    ]
    assert result["status"] == "qualification_passed"
    assert events[0] == "suite_started"
    assert events.count("cell_started") == 8
    assert events.count("cell_completed") == 8
    assert events[-1] == "suite_finished"


def test_suite_releases_device_cache_after_every_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    released: list[str] = []
    _patch_fast_execution(monkeypatch, calls)
    monkeypatch.setattr(
        p4_suite,
        "_release_device_cache",
        lambda device: released.append(device.type),
    )

    result = p4_suite.run_p4_suite(_config(tmp_path, run_id="cache-release"))

    assert result["status"] == "qualification_passed"
    assert released == ["cpu"] * 8


def test_phase_heartbeat_replaces_stale_training_cursor(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="phase-heartbeat")
    cell = config.matrix()[0]

    p4_suite._write_phase_heartbeat(
        tmp_path,
        cell,
        phase="live_rollout",
        task_id="small_graph.v1",
        split="ood",
        distribution="topology",
        completed=1024,
        total=2048,
        deadline=time.perf_counter() + 120.0,
        turn=3,
    )

    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["phase"] == "live_rollout"
    assert heartbeat["distribution"] == "topology"
    assert heartbeat["completed"] == 1024
    assert heartbeat["total"] == 2048
    assert heartbeat["turn"] == 3


def test_short_lived_worker_stage_is_atomic_and_reused(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="isolated-worker")
    directory = tmp_path / "isolated-worker"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(config.model_dump(mode="json")), encoding="utf-8"
    )
    cell = config.matrix()[0]
    cell_directory = directory / "cells" / cell.cell_id
    result_path = cell_directory / "training-stage.json"
    deadline = time.perf_counter() + 180.0

    first = p4_suite._run_worker(
        config,
        cell,
        suite_directory=directory,
        result_path=result_path,
        deadline=deadline,
        arguments=["--stage", "train"],
    )

    assert first["status"] == "COMPLETED"
    assert first["training"]["steps"] > 0
    assert p4_suite._worker_result_valid(
        result_path,
        config=config,
        cell=cell,
        suite_directory=directory,
    )
    worker_schema = json.loads(
        (Path(__file__).parents[2] / "schemas" / "p4-worker-result-v1.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(first, worker_schema)
    before = result_path.stat().st_mtime_ns
    second = p4_suite._run_worker(
        config,
        cell,
        suite_directory=directory,
        result_path=result_path,
        deadline=deadline,
        arguments=["--stage", "train"],
    )
    assert second == first
    assert result_path.stat().st_mtime_ns == before


def test_isolated_standard_evaluation_loads_frozen_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path, run_id="isolated-evaluation")
    directory = tmp_path / "isolated-evaluation"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps(config.model_dump(mode="json")), encoding="utf-8"
    )
    cell = config.matrix()[0]
    deadline = time.perf_counter() + 180.0
    cell_directory = directory / "cells" / cell.cell_id
    p4_suite._run_worker(
        config,
        cell,
        suite_directory=directory,
        result_path=cell_directory / "training-stage.json",
        deadline=deadline,
        arguments=["--stage", "train"],
    )
    part_directory = cell_directory / "evaluation" / "standard__associative_recall.v1"

    result = p4_suite._run_worker(
        config,
        cell,
        suite_directory=directory,
        result_path=part_directory / "result.json",
        deadline=deadline,
        arguments=[
            "--stage",
            "standard",
            "--task-id",
            "associative_recall.v1",
            "--artifact-directory",
            str(part_directory),
        ],
    )

    assert result["status"] == "COMPLETED"
    assert result["evaluation"]["record_count"] == 160
    assert result["evaluation"]["views"]["associative_recall.v1"] == {
        "analysis:v1": 32,
        "test:v1": 32,
        "ood:capacity": 32,
        "ood:interference": 32,
        "ood:joint": 32,
    }


def test_isolated_evaluation_preregisters_every_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, run_id="evaluation-registry")
    cell = config.matrix()[0]
    suite_directory = tmp_path / "evaluation-registry"
    cell_directory = suite_directory / "cells" / cell.cell_id

    def fake_worker(
        config: P4SuiteConfig,
        cell: Any,
        *,
        suite_directory: Path,
        result_path: Path,
        deadline: float,
        arguments: list[str],
    ) -> dict[str, Any]:
        del config, suite_directory, deadline
        artifact_directory = Path(arguments[arguments.index("--artifact-directory") + 1])
        artifact_directory.mkdir(parents=True, exist_ok=True)
        (artifact_directory / "sample_records.jsonl").write_text("", encoding="utf-8")
        (artifact_directory / "telemetry-v2.jsonl").write_text("", encoding="utf-8")
        task_id = arguments[arguments.index("--task-id") + 1]
        result = {
            "schema_version": "p4-worker-result-v1",
            "status": "COMPLETED",
            "stage": arguments[arguments.index("--stage") + 1],
            "cell_id": cell.cell_id,
            "evaluation": {
                "record_count": 0,
                "views": {task_id: {}},
                "routing": {},
                "prediction": {},
                "scores": {task_id: 0.0},
                "telemetry_v2_events": 0,
            },
        }
        p4_suite._write_json(result_path, result)
        return result

    monkeypatch.setattr(p4_suite, "_run_worker", fake_worker)

    result = p4_suite._run_isolated_evaluation(
        config,
        cell,
        suite_directory=suite_directory,
        cell_directory=cell_directory,
        deadline=time.perf_counter() + 30.0,
    )

    assert result["record_count"] == 0
    registry = json.loads((cell_directory / "evaluation-registry.json").read_text(encoding="utf-8"))
    assert len(registry["partitions"]) == 7
    assert all(partition["status"] == "COMPLETED" for partition in registry["partitions"])
    schema = json.loads(
        (Path(__file__).parents[2] / "schemas" / "p4-evaluation-registry-v1.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.validate(registry, schema)


def test_pilot_never_enters_analysis_test_or_ood_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _patch_fast_execution(monkeypatch, calls)
    raw = yaml.safe_load(Path("configs/experiments/p4/pilot.yaml").read_text(encoding="utf-8"))
    raw.update(
        {
            "output_root": str(tmp_path),
            "control_root": str(tmp_path / "control"),
            "run_id": "pilot-validation-only",
            "device": "cpu",
            "qualification_report": str(tmp_path / "qualification-lock.json"),
        }
    )
    config = P4SuiteConfig.model_validate(raw)
    monkeypatch.setattr(
        p4_suite,
        "_evaluate_cell",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("pilot accessed post-selection evaluation")
        ),
    )
    monkeypatch.setattr(p4_suite, "_lock_hash", lambda *args, **kwargs: "qualification")

    result = p4_suite.run_p4_suite(config)

    assert result["status"] == "pilot_passed"
    assert result["completed_cells"] == 4
    selection = json.loads(
        (tmp_path / "pilot-validation-only/pilot-selection.json").read_text(encoding="utf-8")
    )
    assert selection["status"] == "PASSED"


def test_formal_training_consumes_frozen_early_stop_patience(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path, run_id="early-stop")
    config = config.model_copy(update={"budget": config.budget.model_copy(update={"patience": 1})})
    cell = config.matrix()[0]
    model, _ = p4_suite._build_model(config, cell, torch.device("cpu"))
    directory = tmp_path / "cell"
    suite_directory = tmp_path / "suite"
    directory.mkdir(parents=True)
    suite_directory.mkdir(parents=True)

    monkeypatch.setattr(
        p4_suite,
        "_score_split",
        lambda *args, **kwargs: ({task_id: 0.5 for task_id in P4_TASK_ORDER}, {}),
    )

    result = p4_suite._train_cell(
        model,
        cell,
        config,
        torch.device("cpu"),
        directory,
        suite_directory,
        0,
        time.perf_counter() + 120.0,
        None,
        None,
    )

    assert result["steps"] == 4
    assert (directory / "checkpoint.pt").is_file()
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert [event["event"] for event in progress] == ["cell_progress", "cell_progress"]
    assert progress[-1]["step"] == 4
    assert progress[-1]["last_loss"] is not None


def test_direct_suite_rejects_incomplete_prerequisite_evidence(tmp_path: Path) -> None:
    lock = tmp_path / "qualification-lock.json"
    lock.write_text(
        json.dumps(
            {
                "status": "PASSED",
                "git_commit": "abc123",
                "qualification_report": "artifacts/runs/missing/report.json",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incomplete evidence"):
        p4_suite._lock_hash(lock, required_status="PASSED", expected_commit="abc123")


def test_mechanism_run_emits_a_scientific_gate_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _patch_fast_execution(monkeypatch, calls)
    monkeypatch.setattr(p4_suite, "_lock_hash", lambda *args, **kwargs: "frozen-lock")
    raw = yaml.safe_load(Path("configs/experiments/p4/mechanism.yaml").read_text(encoding="utf-8"))
    raw.update(
        {
            "output_root": str(tmp_path),
            "control_root": str(tmp_path / "control"),
            "run_id": "mechanism-fixture",
            "device": "cpu",
        }
    )
    config = P4SuiteConfig.model_validate(raw)

    result = p4_suite.run_p4_suite(config)

    assert result["status"] == "mechanism_passed"
    assert result["completed_cells"] == 24
    report = json.loads(
        (tmp_path / "mechanism-fixture/mechanism-report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "PASSED"
    assert report["evidence"]["predictive_causality"]["bootstrap_samples"] == 10_000
    assert report["evidence"]["prediction_quality"]["passed"] is True
    assert report["evidence"]["sparse_routing"]["passed"] is True
