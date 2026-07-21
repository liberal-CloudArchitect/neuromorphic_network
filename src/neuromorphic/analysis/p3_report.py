"""Recompute the frozen P3 formal statistics from immutable run artifacts."""

# ruff: noqa: RUF001

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any, cast

from neuromorphic.evaluation.p3_statistics import (
    PairedBootstrapResult,
    adjust_family,
    paired_hierarchical_bootstrap,
)
from neuromorphic.training.p3_suite import verify_p3_run

TASKS = (
    "associative_recall.v1",
    "delayed_rule_switch.v1",
    "small_graph.v1",
)
SEEDS = (17, 29, 43)
BASELINES = ("gru", "transformer")
PRIMARY_METRICS = {
    "associative_recall.v1": "query_accuracy",
    "delayed_rule_switch.v1": "response_accuracy",
    "small_graph.v1": "success_rate",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield cast(dict[str, Any], value)


def _summary(cells: Path, cell_id: str) -> dict[str, Any]:
    return _read_json(cells / cell_id / "summary.json")


def _task_slug(task_id: str) -> str:
    return task_id.removesuffix(".v1")


def _shared_cell(model_id: str, seed: int) -> str:
    return f"shared__shared__{model_id}__full__s{seed}__all"


def _continual_cell(model_id: str, seed: int) -> str:
    return f"continual__continual__{model_id}__latin__s{seed}__all"


def _per_task_cell(seed: int, task_id: str) -> str:
    return f"per_task__per_task__modular__full__s{seed}__{_task_slug(task_id)}"


def _causal_cell(variant_id: str, seed: int, task_id: str) -> str:
    return f"causal__per_task__modular__{variant_id}__s{seed}__{_task_slug(task_id)}"


def _primary_value(record: Mapping[str, object], *, ood_proxy: bool = False) -> float | None:
    task_id = record.get("task_id")
    if task_id not in TASKS:
        return None
    if task_id == "small_graph.v1":
        if ood_proxy:
            metric = "optimal_action_rate"
        elif record.get("schema_version") == "p3-small-graph-rollout-v1":
            metric = "success_rate"
        else:
            return None
    else:
        metric = PRIMARY_METRICS[task_id]
    value = record.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"non-finite primary metric {metric}")
    return numeric


def _evaluation_records(
    cells: Path,
    cell_ids: Sequence[str],
    *,
    view: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for cell_id in cell_ids:
        for source in _read_jsonl(cells / cell_id / "sample_records.jsonl"):
            if view == "test":
                selected = source.get("split") == "test" and source.get("distribution") == "v1"
                value = _primary_value(source)
            elif view == "ood_proxy":
                selected = source.get("split") == "ood"
                value = _primary_value(source, ood_proxy=True)
            else:
                raise ValueError(f"unsupported evaluation view: {view}")
            if selected and value is not None:
                record = dict(source)
                record["value"] = value
                records.append(record)
    if not records:
        raise ValueError(f"no records for {view}: {cell_ids}")
    return records


def _scalar_records(
    cells: Path,
    *,
    model_id: str,
    cell_ids: Sequence[str],
    training_key: str,
    variant_id: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for seed, cell_id in zip(SEEDS, cell_ids, strict=True):
        values = _summary(cells, cell_id)["training"][training_key]
        if not isinstance(values, dict):
            raise ValueError(f"invalid {training_key}: {cell_id}")
        for task_id, value in sorted(values.items()):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"non-finite {training_key}: {cell_id}")
            records.append(
                {
                    "model_id": model_id,
                    "variant_id": variant_id,
                    "seed": seed,
                    "task_id": task_id,
                    "split": "analysis",
                    "distribution": training_key,
                    "sample_index": 0,
                    "stratum": task_id,
                    "value": numeric,
                }
            )
    return records


def _causal_aulc_records(
    cells: Path, *, variant_id: str, task_id: str, ablated: bool
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for seed in SEEDS:
        cell_id = (
            _causal_cell(variant_id, seed, task_id) if ablated else _per_task_cell(seed, task_id)
        )
        value = float(_summary(cells, cell_id)["training"]["analysis_aulc"][task_id])
        records.append(
            {
                "model_id": "modular",
                "variant_id": variant_id if ablated else "full",
                "seed": seed,
                "task_id": task_id,
                "split": "analysis",
                "distribution": "analysis_aulc",
                "sample_index": 0,
                "stratum": task_id,
                "value": value,
            }
        )
    return records


def _record_mean(records: Sequence[Mapping[str, object]]) -> float:
    return mean(float(cast(int | float, record["value"])) for record in records)


def _result_dict(result: PairedBootstrapResult) -> dict[str, float | int]:
    return {
        "estimate": float(result.estimate),
        "ci95_lower": float(result.lower),
        "ci95_upper": float(result.upper),
        "p_value": float(result.p_value),
        "bootstrap_samples": result.samples,
        "confidence": result.confidence,
    }


def _comparison(
    reference: Sequence[Mapping[str, object]],
    comparison: Sequence[Mapping[str, object]],
    *,
    samples: int,
) -> tuple[PairedBootstrapResult, float, float]:
    result = paired_hierarchical_bootstrap(reference, comparison, samples=samples)
    return result, _record_mean(reference), _record_mean(comparison)


def _audit_records(cells: Path, registry: Mapping[str, Any]) -> dict[str, object]:
    splits: Counter[str] = Counter()
    schemas: Counter[str] = Counter()
    records = 0
    identity_errors: list[str] = []
    for cell in registry["cells"]:
        cell_id = str(cell["cell_id"])
        path = cells / cell_id / "sample_records.jsonl"
        for record in _read_jsonl(path):
            records += 1
            splits[str(record.get("split"))] += 1
            schemas[str(record.get("schema_version"))] += 1
            if (
                record.get("seed") != cell["seed"]
                or record.get("model_id") != cell["model_id"]
                or record.get("variant_id") != cell["variant_id"]
            ):
                identity_errors.append(cell_id)
                break
    return {
        "record_count": records,
        "splits": dict(sorted(splits.items())),
        "schemas": dict(sorted(schemas.items())),
        "identity_errors": sorted(set(identity_errors)),
        "selection_splits_absent": not ({"train", "validation"} & splits.keys()),
    }


def _network_comparisons(
    cells: Path, *, samples: int
) -> tuple[list[dict[str, object]], dict[str, bool]]:
    shared_records = {
        model: {
            view: _evaluation_records(
                cells, [_shared_cell(model, seed) for seed in SEEDS], view=view
            )
            for view in ("test", "ood_proxy")
        }
        for model in ("modular", *BASELINES)
    }
    aulc = {
        model: _scalar_records(
            cells,
            model_id=model,
            cell_ids=[_shared_cell(model, seed) for seed in SEEDS],
            training_key="analysis_aulc",
            variant_id="full",
        )
        for model in ("modular", *BASELINES)
    }
    forgetting = {
        model: _scalar_records(
            cells,
            model_id=model,
            cell_ids=[_continual_cell(model, seed) for seed in SEEDS],
            training_key="analysis_forgetting",
            variant_id="latin",
        )
        for model in ("modular", *BASELINES)
    }
    rows: list[dict[str, object]] = []
    statistics: list[PairedBootstrapResult] = []
    for category in ("task_score", "ood", "aulc", "forgetting"):
        for baseline in BASELINES:
            if category == "task_score":
                result, modular_mean, baseline_mean = _comparison(
                    shared_records["modular"]["test"],
                    shared_records[baseline]["test"],
                    samples=samples,
                )
                valid = False
                reason = "SmallGraph chance level was not frozen for chance normalization"
                threshold = 0.05
            elif category == "ood":
                result, modular_mean, baseline_mean = _comparison(
                    shared_records["modular"]["ood_proxy"],
                    shared_records[baseline]["ood_proxy"],
                    samples=samples,
                )
                valid = False
                reason = (
                    "SmallGraph OOD records lack rollout success_rate and task chance levels "
                    "were not frozen; optimal_action_rate is descriptive only"
                )
                threshold = None
            elif category == "aulc":
                result, modular_mean, baseline_mean = _comparison(
                    aulc["modular"], aulc[baseline], samples=samples
                )
                valid = True
                reason = None
                threshold = 0.15
            else:
                result, baseline_mean, modular_mean = _comparison(
                    forgetting[baseline], forgetting["modular"], samples=samples
                )
                valid = False
                reason = "protocol-v2 did not freeze a minimum forgetting effect threshold"
                threshold = None
            relative = result.estimate / abs(baseline_mean) if baseline_mean != 0 else None
            row: dict[str, object] = {
                "category": category,
                "baseline": baseline,
                "effect_direction": (
                    "modular_minus_baseline"
                    if category != "forgetting"
                    else "baseline_minus_modular"
                ),
                "modular_mean": modular_mean,
                "baseline_mean": baseline_mean,
                "relative_effect": relative,
                "minimum_relative_effect": threshold,
                "primary_metric_valid": valid,
                "invalid_reason": reason,
                **_result_dict(result),
            }
            rows.append(row)
            statistics.append(result)
    for row, adjusted in zip(rows, adjust_family(statistics), strict=True):
        row["holm_adjusted_p"] = adjusted
        row_threshold = row.get("minimum_relative_effect")
        relative_effect = row.get("relative_effect")
        lower = row.get("ci95_lower")
        row["passed"] = bool(
            row["primary_metric_valid"]
            and isinstance(row_threshold, float)
            and isinstance(relative_effect, float)
            and isinstance(lower, float)
            and relative_effect >= row_threshold
            and lower > 0
            and adjusted <= 0.05
        )
    categories = {
        category: all(bool(row["passed"]) for row in rows if row["category"] == category)
        for category in ("task_score", "ood", "aulc", "forgetting")
    }
    return rows, categories


def _causal_comparisons(cells: Path, *, samples: int) -> list[dict[str, object]]:
    specifications = (
        ("episodic", "episodic-no-read-write", "associative_recall.v1", 0.15, "test"),
        ("working", "working-reset", "delayed_rule_switch.v1", 0.10, "test"),
        ("predictive", "predictive-loss-zero", "small_graph.v1", 0.10, "aulc"),
    )
    rows: list[dict[str, object]] = []
    statistics: list[PairedBootstrapResult] = []
    for name, variant, task_id, threshold, metric in specifications:
        if metric == "test":
            full = _evaluation_records(
                cells, [_per_task_cell(seed, task_id) for seed in SEEDS], view="test"
            )
            ablated = _evaluation_records(
                cells,
                [_causal_cell(variant, seed, task_id) for seed in SEEDS],
                view="test",
            )
        else:
            full = _causal_aulc_records(cells, variant_id=variant, task_id=task_id, ablated=False)
            ablated = _causal_aulc_records(cells, variant_id=variant, task_id=task_id, ablated=True)
        result, full_mean, ablated_mean = _comparison(full, ablated, samples=samples)
        relative = result.estimate / abs(full_mean) if metric == "aulc" and full_mean else None
        rows.append(
            {
                "module": name,
                "variant": variant,
                "task_id": task_id,
                "metric": "analysis_aulc" if metric == "aulc" else PRIMARY_METRICS[task_id],
                "effect_direction": "full_minus_ablated",
                "full_mean": full_mean,
                "ablated_mean": ablated_mean,
                "absolute_effect_threshold": threshold if metric == "test" else None,
                "relative_effect_threshold": threshold if metric == "aulc" else None,
                "relative_effect": relative,
                **_result_dict(result),
            }
        )
        statistics.append(result)
    for row, adjusted in zip(rows, adjust_family(statistics), strict=True):
        row["holm_adjusted_p"] = adjusted
        effect = float(cast(int | float, row["estimate"]))
        absolute_threshold = row.get("absolute_effect_threshold")
        if absolute_threshold is None:
            relative_effect = row.get("relative_effect")
            relative_threshold = row.get("relative_effect_threshold")
            if not isinstance(relative_effect, float) or not isinstance(relative_threshold, float):
                raise ValueError("causal relative threshold fields are invalid")
            threshold_passed = relative_effect >= relative_threshold
        else:
            if not isinstance(absolute_threshold, float):
                raise ValueError("causal absolute threshold field is invalid")
            threshold_passed = effect >= absolute_threshold
        lower = row.get("ci95_lower")
        row["passed"] = bool(
            threshold_passed and isinstance(lower, float) and lower > 0 and adjusted <= 0.05
        )
    return rows


def _task_means(records: Sequence[Mapping[str, object]]) -> dict[str, float]:
    values: defaultdict[str, list[float]] = defaultdict(list)
    for record in records:
        values[cast(str, record["task_id"])].append(float(cast(int | float, record["value"])))
    return {task: mean(task_values) for task, task_values in sorted(values.items())}


def _sparse_routing(cells: Path) -> dict[str, object]:
    savings: list[float] = []
    coverages: list[float] = []
    sparse_latency: list[float] = []
    dense_latency: list[float] = []
    expert_calls: Counter[str] = Counter()
    sparse_records = _evaluation_records(
        cells, [_shared_cell("modular", seed) for seed in SEEDS], view="test"
    )
    dense_records = _evaluation_records(
        cells,
        [f"control__shared__modular__router-dense__s{seed}__all" for seed in SEEDS],
        view="test",
    )
    for seed in SEEDS:
        sparse = _summary(cells, _shared_cell("modular", seed))
        dense = _summary(cells, f"control__shared__modular__router-dense__s{seed}__all")
        sparse_profile = sparse["cost"]["profile"]
        dense_profile = dense["cost"]["profile"]
        savings.append(
            1.0
            - float(sparse_profile["active_optional_macs"])
            / float(dense_profile["active_optional_macs"])
        )
        coverages.append(float(sparse_profile["parameter_coverage"]))
        sparse_latency.append(float(sparse["cost"]["latency_ms"]["p50"]))
        dense_latency.append(float(dense["cost"]["latency_ms"]["p50"]))
        for record in sparse_profile["records"]:
            if record["category"] == "optional":
                expert_calls[str(record["module_id"])] += int(record["active_calls"])
    sparse_means = _task_means(sparse_records)
    dense_means = _task_means(dense_records)
    drops = {task: dense_means[task] - sparse_means[task] for task in TASKS}
    total_calls = sum(expert_calls.values())
    shares = {name: count / total_calls for name, count in sorted(expert_calls.items())}
    return {
        "mean_optional_mac_saving": mean(savings),
        "per_seed_optional_mac_saving": dict(zip(map(str, SEEDS), savings, strict=True)),
        "minimum_profiler_coverage": min(coverages),
        "sparse_p50_latency_ms_median": median(sparse_latency),
        "dense_p50_latency_ms_median": median(dense_latency),
        "task_score_drop": drops,
        "maximum_task_score_drop": max(drops.values()),
        "expert_call_shares": shares,
        "all_experts_active": len(shares) == 3 and all(value > 0 for value in shares.values()),
        "capacity_drops": None,
        "capacity_evidence_available": False,
        "passed": mean(savings) >= 0.20 and max(drops.values()) <= 0.02,
    }


def _negative_control(cells: Path) -> dict[str, object]:
    differences: dict[str, list[float]] = {task: [] for task in TASKS}
    for seed in SEEDS:
        full = _evaluation_records(cells, [_shared_cell("modular", seed)], view="test")
        acute = _evaluation_records(
            cells,
            [f"control__shared__modular__acute-predictive-off__s{seed}__all"],
            view="test",
        )
        full_means = _task_means(full)
        acute_means = _task_means(acute)
        for task in TASKS:
            differences[task].append(full_means[task] - acute_means[task])
    return {
        "per_task_per_seed_difference": differences,
        "maximum_absolute_difference": max(
            abs(value) for values in differences.values() for value in values
        ),
        "passed": all(value == 0.0 for values in differences.values() for value in values),
    }


def _model_costs(cells: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for model in ("modular", *BASELINES):
        summaries = [_summary(cells, _shared_cell(model, seed)) for seed in SEEDS]
        result[model] = {
            "parameters": [int(summary["parameters"]) for summary in summaries],
            "parameter_matching": [str(summary["matching"]) for summary in summaries],
            "profiler_coverage_min": min(
                float(summary["cost"]["profile"]["parameter_coverage"]) for summary in summaries
            ),
            "p50_latency_ms_median": median(
                float(summary["cost"]["latency_ms"]["p50"]) for summary in summaries
            ),
            "p95_latency_ms_median": median(
                float(summary["cost"]["latency_ms"]["p95"]) for summary in summaries
            ),
            "wall_clock_seconds": [float(summary["wall_clock_seconds"]) for summary in summaries],
            "peak_memory_bytes_max": max(
                int(summary["cost"]["peak_memory_bytes"]) for summary in summaries
            ),
        }
    return result


def _representation_summary(cells: Path) -> dict[str, object]:
    values: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    for model in ("modular", *BASELINES):
        for seed in SEEDS:
            tasks = _summary(cells, _shared_cell(model, seed))["evaluation"][
                "representation_analysis"
            ]["tasks"]
            for task_id, metrics in tasks.items():
                for name in ("linear_cka", "rsa_spearman", "linear_probe_accuracy"):
                    values[(model, task_id, name)].append(float(metrics[name]))
    result: defaultdict[str, defaultdict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (model, task, name), measurements in sorted(values.items()):
        result[model][task][name] = mean(measurements)
    return {model: dict(tasks) for model, tasks in result.items()}


def summarize_p3_run(run_directory: Path, *, bootstrap_samples: int = 10_000) -> dict[str, object]:
    """Validate and summarize one complete formal P3 run."""

    run_directory = run_directory.resolve()
    registry = _read_json(run_directory / "registry.json")
    config = _read_json(run_directory / "config.json")
    verification = verify_p3_run(run_directory)
    cells = run_directory / "cells"
    if registry.get("status") != "completed" or config.get("profile") != "full":
        raise ValueError("P3 report requires a completed full-profile run")
    statuses = Counter(str(cell["status"]) for cell in registry["cells"])
    seeds = sorted({int(cell["seed"]) for cell in registry["cells"]})
    if statuses != {"COMPLETED": 81} or seeds != list(SEEDS):
        raise ValueError("formal matrix status or seed set is incomplete")
    audit = _audit_records(cells, registry)
    if audit["identity_errors"] or not audit["selection_splits_absent"]:
        raise ValueError("formal sample record identity or split discipline failed")

    network_rows, network_categories = _network_comparisons(cells, samples=bootstrap_samples)
    causal_rows = _causal_comparisons(cells, samples=bootstrap_samples)
    sparse = _sparse_routing(cells)
    negative = _negative_control(cells)
    scientific_categories = {**network_categories, "sparse_routing": bool(sparse["passed"])}
    causal_passed = all(bool(row["passed"]) for row in causal_rows)
    benefit_count = sum(scientific_categories.values())
    gate_nn_passed = bool(
        causal_passed
        and benefit_count >= 2
        and sparse["all_experts_active"]
        and sparse["capacity_evidence_available"]
        and negative["passed"]
        and float(cast(int | float, sparse["minimum_profiler_coverage"])) >= 0.95
    )
    integrity_passed = bool(
        verification["status"] == "completed"
        and verification["checksums_ok"]
        and not verification["missing_cells"]
        and verification["cells"] == 81
        and audit["record_count"] == verification["sample_records"]
        and statuses == {"COMPLETED": 81}
        and seeds == list(SEEDS)
        and bootstrap_samples == 10_000
        and int(config["budget"]["bootstrap_samples"]) == 10_000
        and float(registry["wall_clock_seconds"])
        <= float(config["budget"]["wall_clock_hours"]) * 3600
        and audit["selection_splits_absent"]
        and not audit["identity_errors"]
    )
    return {
        "schema_version": "p3-formal-report-v1",
        "run": {
            "run_id": registry["run_id"],
            "artifact_directory": str(run_directory),
            "source_git_commit": config["expected_git_commit"],
            "protocol_version": config["protocol_version"],
            "device": config["device"],
            "seeds": seeds,
            "selected_presets": config["selected_presets"],
            "wall_clock_seconds": float(registry["wall_clock_seconds"]),
            "wall_clock_limit_seconds": float(config["budget"]["wall_clock_hours"]) * 3600,
        },
        "integrity": {
            "passed": integrity_passed,
            "verification": verification,
            "cell_statuses": dict(statuses),
            "sample_audit": audit,
            "failed_cells": [],
            "resource_limited_cells": [],
        },
        "statistics": {
            "method": "strict paired seed-to-stratum/sample percentile bootstrap",
            "bootstrap_samples": bootstrap_samples,
            "rng_seed": 20_260_715,
            "confidence": 0.95,
            "multiple_comparisons": "Holm step-down within frozen network and causal families",
        },
        "network_benefit": {
            "comparisons": network_rows,
            "categories": scientific_categories,
            "categories_passed": benefit_count,
            "minimum_categories": 2,
        },
        "causal": {
            "comparisons": causal_rows,
            "all_three_passed": causal_passed,
        },
        "sparse_routing": sparse,
        "predictive_acute_negative_control": negative,
        "model_costs": _model_costs(cells),
        "representation_analysis": _representation_summary(cells),
        "protocol_deviations": [
            {
                "id": "DR-001",
                "scope": "task_score_and_ood",
                "effect": "confirmatory categories cannot pass",
                "reason": (
                    "protocol-v2 did not freeze SmallGraph chance normalization and formal "
                    "SmallGraph OOD artifacts contain optimal_action_rate but no "
                    "rollout success_rate"
                ),
            },
            {
                "id": "DR-002",
                "scope": "forgetting",
                "effect": "confirmatory category cannot pass",
                "reason": "protocol-v2 did not freeze a minimum forgetting effect threshold",
            },
            {
                "id": "DR-003",
                "scope": "routing_capacity",
                "effect": "GATE-NN-MVP routing health cannot be established from formal artifacts",
                "reason": (
                    "formal summaries record expert calls and MACs but not capacity-drop counts"
                ),
            },
        ],
        "gates": {
            "GATE-3": "PASSED" if integrity_passed else "FAILED",
            "GATE-NN-MVP": "PASSED" if gate_nn_passed else "FAILED",
            "network_mvp_bundle_allowed": gate_nn_passed,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the concise tracked P3 formal report."""

    run = report["run"]
    network = report["network_benefit"]
    causal = report["causal"]
    sparse = report["sparse_routing"]
    lines = [
        "# P3 正式实验与科学裁决",
        "",
        f"- Run：`{run['run_id']}`",
        f"- 训练提交：`{run['source_git_commit']}`",
        f"- 设备：`{run['device']}`；seeds：`{run['seeds']}`",
        (
            f"- 实际 suite 墙钟：{run['wall_clock_seconds'] / 3600:.2f} 小时"
            f"（上限 {run['wall_clock_limit_seconds'] / 3600:.0f} 小时）"
        ),
        f"- `GATE-3`：**{report['gates']['GATE-3']}**",
        f"- `GATE-NN-MVP`：**{report['gates']['GATE-NN-MVP']}**",
        "",
        "## 完整性",
        "",
        (
            "81/81 cells 完成；"
            f"{report['integrity']['verification']['registered_artifacts']} "
            "个登记产物 checksum 通过；"
        ),
        (
            f"共复核 {report['integrity']['verification']['sample_records']:,} "
            "条逐样本记录。没有失败、缺 seed 或资源截断。"
        ),
        "",
        "## 网络收益 family",
        "",
        "| 类别 | 基线 | 效应 | 95% CI | Holm p | 结论 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in network["comparisons"]:
        conclusion = "PASS" if row["passed"] else "FAIL"
        if not row["primary_metric_valid"]:
            conclusion = "INVALID/FAIL"
        lines.append(
            f"| {row['category']} | {row['baseline']} | {row['estimate']:.4f} | "
            f"[{row['ci95_lower']:.4f}, {row['ci95_upper']:.4f}] | "
            f"{row['holm_adjusted_p']:.4g} | {conclusion} |"
        )
    lines.extend(
        [
            "",
            f"通过类别数：**{network['categories_passed']}/{network['minimum_categories']}**。",
            "",
            "## 模块因果 family",
            "",
            "| 模块 | 指标 | full−ablated | 95% CI | Holm p | 结论 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in causal["comparisons"]:
        lines.append(
            f"| {row['module']} | {row['metric']} | {row['estimate']:.4f} | "
            f"[{row['ci95_lower']:.4f}, {row['ci95_upper']:.4f}] | "
            f"{row['holm_adjusted_p']:.4g} | {'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            (
                "情景记忆与工作记忆获得支持；预测适配的重新训练消融未达到 "
                "10% AULC 门槛，因此三模块联合因果条件失败。"
            ),
            "",
            "## 稀疏执行",
            "",
            f"optional MAC 平均减少 **{100 * sparse['mean_optional_mac_saving']:.1f}%**，",
            (
                "但最大任务分数下降 "
                f"**{100 * sparse['maximum_task_score_drop']:.1f}pp**，"
                "超过 2pp 非劣界；结论为 **FAIL**。"
            ),
            (
                "profiler coverage 最低为 "
                f"{100 * sparse['minimum_profiler_coverage']:.2f}%。"
                "所有三个专家均有真实调用，但 formal summary 未记录 capacity-drop 数。"
            ),
            "",
            "## 结论与边界",
            "",
            "`GATE-3` 只说明预注册矩阵、恢复、产物和诚实统计完成。`GATE-NN-MVP` 失败，",
            "因此不生成 `network-mvp-v1` 正式 bundle，也不得使用“网络 MVP qualified”表述。",
            "这些结果只适用于人工计算模型，不构成脑区等价或生物学结论。",
            "",
            (
                "协议缺口按 `DR-001`～`DR-003` 原样保留；不得用事后定义的 "
                "chance、阈值或替代指标反转本次 Gate。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


__all__ = ["render_markdown", "summarize_p3_run", "write_report"]
