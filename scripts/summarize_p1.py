"""Summarize three frozen Associative Recall qualification runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from neuromorphic.evaluation.statistics import (
    hierarchical_stratified_bootstrap_contrast,
    hierarchical_stratified_bootstrap_mean,
)
from neuromorphic.evaluation.task_metrics import aggregate_records
from neuromorphic.training.manifest import file_sha256

EXPECTED_SEEDS = {17, 29, 43}


def load_runs(paths: list[Path]) -> list[dict[str, Any]]:
    """Load and validate the frozen P1 run evidence."""
    runs: list[dict[str, Any]] = []
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        samples_path = path / "evaluation_samples.jsonl"
        if not samples_path.is_file():
            raise ValueError(f"formal run lacks per-sample evaluation evidence: {path}")
        if manifest["git"]["dirty"]:
            raise ValueError(f"formal run has a dirty worktree: {path}")
        if manifest.get("status") != "completed" or manifest.get("failure") is not None:
            raise ValueError(f"formal run did not complete successfully: {path}")
        for artifact_name, evidence in manifest.get("artifacts", {}).items():
            artifact_path = path / artifact_name
            if not artifact_path.is_file() or file_sha256(artifact_path) != evidence.get("sha256"):
                raise ValueError(f"artifact checksum mismatch: {artifact_path}")
        if manifest["task"] != {
            "profile": "qualification",
            "task_id": "associative_recall.v1",
        }:
            raise ValueError(f"unexpected formal task: {path}")
        if manifest["model"]["kind"] != "gru":
            raise ValueError(f"unexpected formal model: {path}")
        sample_records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            samples_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("task_id") != "associative_recall.v1":
                raise ValueError(f"unexpected task record at {samples_path}:{line_number}")
            if record.get("split") not in {"test", "ood"}:
                raise ValueError(f"unexpected split record at {samples_path}:{line_number}")
            recorded_seed = record.get("seed")
            if recorded_seed not in {None, manifest["seed"]}:
                raise ValueError(f"sample seed mismatch at {samples_path}:{line_number}")
            record["seed"] = int(manifest["seed"])
            sample_records.append(record)
        if not sample_records:
            raise ValueError(f"formal run has no evaluation sample records: {path}")
        runs.append(
            {
                "path": str(path),
                "manifest": manifest,
                "summary": summary,
                "samples": sample_records,
            }
        )
    seeds = {int(run["manifest"]["seed"]) for run in runs}
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"expected seeds {sorted(EXPECTED_SEEDS)}, received {sorted(seeds)}")
    commits = {run["manifest"]["git"]["commit"] for run in runs}
    if len(commits) != 1:
        raise ValueError("formal runs must use the same commit")
    return sorted(runs, key=lambda run: int(run["manifest"]["seed"]))


def build_report(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Build descriptive multi-seed statistics without claiming a model comparison."""
    records = [record for run in runs for record in run["samples"]]
    split_reports: dict[str, dict[str, Any]] = {}
    for split in ("test", "ood"):
        split_records = [record for record in records if record["split"] == split]
        if not split_records:
            raise ValueError(f"formal runs lack {split} sample records")
        strata = {str(record["interference_stratum"]) for record in split_records}
        stratum_intervals = {
            stratum: asdict(
                hierarchical_stratified_bootstrap_mean(
                    [
                        record
                        for record in split_records
                        if record["interference_stratum"] == stratum
                    ],
                    value_key="query_accuracy",
                    stratum_key="interference_stratum",
                )
            )
            for stratum in sorted(strata)
        }
        interference_drop: dict[str, Any] | None = None
        unavailable_reason: str | None = None
        if {"low", "high"}.issubset(strata):
            interference_drop = asdict(
                hierarchical_stratified_bootstrap_contrast(
                    split_records,
                    value_key="query_accuracy",
                    stratum_key="interference_stratum",
                    reference="low",
                    comparison="high",
                )
            )
        else:
            unavailable_reason = "low and high interference strata are not both present"
        split_reports[split] = {
            "query_accuracy": asdict(
                hierarchical_stratified_bootstrap_mean(
                    split_records,
                    value_key="query_accuracy",
                    stratum_key="bootstrap_stratum",
                )
            ),
            "interference_accuracy": stratum_intervals,
            "interference_drop": interference_drop,
            "interference_drop_unavailable_reason": unavailable_reason,
            "sample_count": len(split_records),
            "seed_median": sorted(
                _required_float(
                    aggregate_records(
                        [record for record in run["samples"] if record["split"] == split]
                    ),
                    "query_accuracy",
                )
                for run in runs
            )[1],
        }
    return {
        "schema_version": "p1-baseline-report-v1",
        "commit": runs[0]["manifest"]["git"]["commit"],
        "seeds": sorted(EXPECTED_SEEDS),
        "runs": [
            {
                "seed": run["manifest"]["seed"],
                "run_id": run["manifest"]["run_id"],
                "test": aggregate_records(
                    [record for record in run["samples"] if record["split"] == "test"]
                ),
                "ood": aggregate_records(
                    [record for record in run["samples"] if record["split"] == "ood"]
                ),
                "steps": run["summary"]["steps"],
                "wall_clock_seconds": run["summary"]["wall_clock_seconds"],
                "training_examples": run["summary"]["training_examples"],
                "training_tokens": run["summary"]["training_tokens"],
                "parameters": run["manifest"]["cost"]["trainable_parameters"],
                "macs_per_sequence": run["manifest"]["cost"]["estimated_macs_per_sequence"],
                "mac_coverage": run["manifest"]["cost"]["mac_profiler_coverage"],
                "latency_ms": run["summary"]["latency_ms"],
                "peak_memory_bytes": run["summary"]["peak_memory_bytes"],
            }
            for run in runs
        ],
        "splits": split_reports,
        "statistics": {
            "method": "hierarchical seed-to-stratum/sample percentile bootstrap",
            "samples": 10_000,
            "seed": 20_260_715,
            "confidence": 0.95,
            "multiple_comparisons": "Holm N/A: one descriptive baseline, no model comparisons",
        },
        "failed_runs": [],
        "interpretation": "Descriptive P1 baseline only; no brain-inspired model comparison.",
    }


def _required_float(values: dict[str, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"report metric {key!r} must be numeric")
    return float(value)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a concise tracked evidence report."""
    lines = [
        "# P1 Associative Recall GRU 基线报告",
        "",
        f"- Commit: `{report['commit']}`",
        f"- Seeds: `{report['seeds']}`",
        "- 性质：P1 描述性单体基线，不构成类脑模型收益结论。",  # noqa: RUF001
        "",
        "| Seed | Run ID | Test | OOD | Steps | Params | MAC/seq | Coverage | "
        "P50/P95 ms | Wall s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        lines.append(
            f"| {run['seed']} | `{run['run_id']}` | {run['test']['query_accuracy']:.6f} | "
            f"{run['ood']['query_accuracy']:.6f} | {run['steps']} | {run['parameters']} | "
            f"{run['macs_per_sequence']} | {run['mac_coverage']:.3f} | "
            f"{run['latency_ms']['p50']:.3f}/{run['latency_ms']['p95']:.3f} | "
            f"{run['wall_clock_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Bootstrap 95% CI",
            "",
            "- Test query accuracy: "
            f"{_format_interval(report['splits']['test']['query_accuracy'])}",
            f"- Test interference drop (low - high): "
            f"{_format_optional_interval(report['splits']['test']['interference_drop'])}",
            f"- OOD query accuracy: {_format_interval(report['splits']['ood']['query_accuracy'])}",
            f"- OOD interference drop: "
            f"{_format_optional_interval(report['splits']['ood']['interference_drop'])}",
            f"- Seed median (test/OOD): {report['splits']['test']['seed_median']:.6f} / "
            f"{report['splits']['ood']['seed_median']:.6f}",
            "- 方法：10,000 次 seed→stratum/sample 两级 percentile bootstrap；seed=20260715。",  # noqa: RUF001
            "- Holm 校正：N/A；当前只有一个描述性基线，没有模型间多重比较。",  # noqa: RUF001
            "- 失败/缺失 run：0；所有冻结 seed 均纳入。",  # noqa: RUF001
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_interval(interval: dict[str, Any]) -> str:
    return f"{interval['estimate']:.6f} [{interval['lower']:.6f}, {interval['upper']:.6f}]"


def _format_optional_interval(interval: dict[str, Any] | None) -> str:
    return "N/A（缺少对照分层）" if interval is None else _format_interval(interval)  # noqa: RUF001


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs=3, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    arguments = parser.parse_args()
    report = build_report(load_runs(arguments.runs))
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(arguments.markdown, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
