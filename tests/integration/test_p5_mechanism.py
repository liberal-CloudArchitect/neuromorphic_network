from __future__ import annotations

import json
from pathlib import Path

from neuromorphic.training.p5_config import load_p5_mechanism_config
from neuromorphic.training.p5_mechanism import _formal_evidence, execute_p5_mechanism
from neuromorphic.training.p5_suite import verify_p5_run


def test_p5_mechanism_cpu_micro_covers_all_interventions(tmp_path: Path) -> None:
    config = load_p5_mechanism_config(Path("configs/experiments/p5/mechanism-ci.yaml")).model_copy(
        update={"output_root": tmp_path, "run_id": "mechanism-micro"}
    )

    result = execute_p5_mechanism(config)

    assert result["status"] == "qualification_passed"
    assert result["completed_cells"] == 8
    verified = verify_p5_run(tmp_path / "mechanism-micro")
    assert verified["cells"] == 8
    assert verified["missing_cells"] == []
    assert verified["checksums_ok"] is True


def test_p5_formal_gate_fixture_covers_bootstrap_families(tmp_path: Path) -> None:
    config = load_p5_mechanism_config(
        Path("configs/experiments/p5/mechanism-cuda.yaml")
    ).model_copy(update={"bootstrap_samples": 200})
    summaries: dict[str, dict[str, object]] = {}
    for seed in config.seeds:
        for variant in (
            "full",
            "predictor-off",
            "surprise-off",
            "no-dual-route",
            "dense-memory",
        ):
            full = variant == "full"
            summaries[f"{variant}__s{seed}"] = {
                "macro_aulc": 0.80 if full else 0.70,
                "drs_score": 0.80 if full else 0.70,
                "test_scores": {
                    "associative_recall.v1": 0.90 if full else 0.89,
                    "delayed_rule_switch.v1": 0.90 if full else 0.89,
                    "small_graph.v1": 0.90 if full else 0.89,
                },
                "prediction": {
                    task: {"forecast_error": 0.18, "persistence_error": 0.20}
                    for task in (
                        "associative_recall.v1",
                        "delayed_rule_switch.v1",
                        "small_graph.v1",
                    )
                },
                "routing": {
                    "active_macs": 50.0,
                    "dense_macs": 100.0,
                    "semantic_required": 10.0,
                    "semantic_executed": 10.0,
                    "dual_tokens": 2.0,
                    "valid_tokens": 10.0,
                },
            }
        for variant, value in (("full", 0.90), ("dense-memory", 0.89)):
            directory = tmp_path / "cells" / f"{variant}__s{seed}"
            directory.mkdir(parents=True)
            records = [
                {
                    "seed": seed,
                    "task_id": task,
                    "split": "test",
                    "distribution": "v1",
                    "sample_index": 0,
                    "stratum": task,
                    "model_id": "modular-v3",
                    "variant_id": variant,
                    "value": value,
                }
                for task in (
                    "associative_recall.v1",
                    "delayed_rule_switch.v1",
                    "small_graph.v1",
                )
            ]
            (directory / "sample-records.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

    evidence = _formal_evidence(config, tmp_path, summaries)

    assert evidence["status"] == "PASSED"
    assert evidence["mac_reduction"] == 0.5
