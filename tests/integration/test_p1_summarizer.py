from __future__ import annotations

import json
from importlib import util
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest


def _summarizer_module() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "summarize_p1.py"
    spec = util.spec_from_file_location("p1_summarizer_test_target", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load P1 summarizer")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(path: Path, seed: int) -> None:
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "git": {"dirty": False, "commit": "abc123"},
                "task": {"profile": "qualification", "task_id": "associative_recall.v1"},
                "model": {"kind": "gru"},
                "seed": seed,
                "run_id": f"run-{seed}",
                "status": "completed",
                "failure": None,
                "artifacts": {},
                "cost": {
                    "trainable_parameters": 100,
                    "estimated_macs_per_sequence": 200,
                    "mac_profiler_coverage": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "summary.json").write_text(
        json.dumps(
            {
                "steps": 100,
                "wall_clock_seconds": 1.5,
                "training_examples": 6400,
                "training_tokens": 32000,
                "latency_ms": {"p50": 1.0, "p95": 1.5},
                "peak_memory_bytes": 1024,
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "schema_version": "evaluation-sample-v1",
            "task_id": "associative_recall.v1",
            "split": split,
            "sample_index": index,
            "seed": seed,
            "query_accuracy": value,
            "distractor_count": distractors,
            "interference_stratum": stratum,
            "bootstrap_stratum": f"pairs-4/interference-{distractors}",
        }
        for split in ("test", "ood")
        for index, value, distractors, stratum in (
            ((0, 1.0, 1, "low"), (1, 0.0, 4, "high")) if split == "test" else ((0, 0.5, 6, "high"),)
        )
    ]
    (path / "evaluation_samples.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_summarizer_uses_sample_evidence_and_marks_unavailable_contrast(
    tmp_path: Path,
) -> None:
    module = _summarizer_module()
    paths = [tmp_path / f"run-{seed}" for seed in (17, 29, 43)]
    for path, seed in zip(paths, (17, 29, 43), strict=True):
        _write_run(path, seed)
    load_runs = cast(Any, module.load_runs)
    build_report = cast(Any, module.build_report)
    write_markdown = cast(Any, module.write_markdown)
    report = build_report(load_runs(paths))
    assert report["splits"]["test"]["interference_drop"]["estimate"] == pytest.approx(1.0)
    assert report["splits"]["ood"]["interference_drop"] is None
    assert report["statistics"]["multiple_comparisons"].startswith("Holm N/A")

    markdown = tmp_path / "report.md"
    write_markdown(markdown, report)
    contents = markdown.read_text(encoding="utf-8")
    assert "seed→stratum/sample" in contents
    assert "Holm" in contents
