"""Summarize three frozen Associative Recall qualification runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from neuromorphic.evaluation.statistics import percentile_bootstrap_mean

EXPECTED_SEEDS = {17, 29, 43}


def load_runs(paths: list[Path]) -> list[dict[str, Any]]:
    """Load and validate the frozen P1 run evidence."""
    runs: list[dict[str, Any]] = []
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if manifest["git"]["dirty"]:
            raise ValueError(f"formal run has a dirty worktree: {path}")
        if manifest["task"] != {
            "profile": "qualification",
            "task_id": "associative_recall.v1",
        }:
            raise ValueError(f"unexpected formal task: {path}")
        if manifest["model"]["kind"] != "gru":
            raise ValueError(f"unexpected formal model: {path}")
        runs.append({"path": str(path), "manifest": manifest, "summary": summary})
    seeds = {int(run["manifest"]["seed"]) for run in runs}
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"expected seeds {sorted(EXPECTED_SEEDS)}, received {sorted(seeds)}")
    commits = {run["manifest"]["git"]["commit"] for run in runs}
    if len(commits) != 1:
        raise ValueError("formal runs must use the same commit")
    return sorted(runs, key=lambda run: int(run["manifest"]["seed"]))


def build_report(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Build descriptive multi-seed statistics without claiming a model comparison."""
    test_values = [float(run["summary"]["test"]["accuracy"]) for run in runs]
    ood_values = [float(run["summary"]["ood"]["accuracy"]) for run in runs]
    test_interval = percentile_bootstrap_mean(test_values)
    ood_interval = percentile_bootstrap_mean(ood_values)
    return {
        "schema_version": "p1-baseline-report-v1",
        "commit": runs[0]["manifest"]["git"]["commit"],
        "seeds": sorted(EXPECTED_SEEDS),
        "runs": [
            {
                "seed": run["manifest"]["seed"],
                "run_id": run["manifest"]["run_id"],
                "test_accuracy": run["summary"]["test"]["accuracy"],
                "ood_accuracy": run["summary"]["ood"]["accuracy"],
                "steps": run["summary"]["steps"],
                "wall_clock_seconds": run["summary"]["wall_clock_seconds"],
            }
            for run in runs
        ],
        "test_accuracy": asdict(test_interval),
        "ood_accuracy": asdict(ood_interval),
        "interpretation": "Descriptive P1 baseline only; no brain-inspired model comparison.",
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a concise tracked evidence report."""
    lines = [
        "# P1 Associative Recall GRU 基线报告",
        "",
        f"- Commit: `{report['commit']}`",
        f"- Seeds: `{report['seeds']}`",
        "- 性质：P1 描述性单体基线，不构成类脑模型收益结论。",  # noqa: RUF001
        "",
        "| Seed | Run ID | Test accuracy | OOD accuracy | Steps | Wall-clock (s) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        lines.append(
            f"| {run['seed']} | `{run['run_id']}` | {run['test_accuracy']:.6f} | "
            f"{run['ood_accuracy']:.6f} | {run['steps']} | {run['wall_clock_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Bootstrap 95% CI",
            "",
            f"- Test mean: {report['test_accuracy']['estimate']:.6f} "
            f"[{report['test_accuracy']['lower']:.6f}, {report['test_accuracy']['upper']:.6f}]",
            f"- OOD mean: {report['ood_accuracy']['estimate']:.6f} "
            f"[{report['ood_accuracy']['lower']:.6f}, {report['ood_accuracy']['upper']:.6f}]",
            "- 方法：10,000 次 percentile bootstrap；仅描述三个训练 seed 的变异。",  # noqa: RUF001
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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
