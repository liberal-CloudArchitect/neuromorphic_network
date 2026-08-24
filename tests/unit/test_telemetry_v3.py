from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import ValidationError  # type: ignore[import-untyped]

from neuromorphic.telemetry.events_v3 import TelemetryV3Event

ROOT = Path(__file__).resolve().parents[2]


def _schema() -> dict[str, object]:
    value = json.loads((ROOT / "schemas/telemetry-v3.json").read_text(encoding="utf-8"))
    return cast(dict[str, object], value)


def _event() -> TelemetryV3Event:
    return TelemetryV3Event(
        event_id="p5:1:drs",
        run_id="p5-run",
        global_step=1,
        task="delayed_rule_switch.v1",
        phase="train",
        module_id="sparse_router.v3",
        compute_gate=True,
        surprise_count=8,
        surprise_sum=1.5,
        semantic_required=10,
        semantic_executed=10,
        semantic_coverage=1.0,
        dual_count=2,
        dual_fraction=0.2,
        forecast_coverage=0.9,
        forecast_error=0.1,
        persistence_error=0.2,
        scorer_grad_norm=0.5,
        active_calls=12.0,
        dense_calls=20.0,
        metadata={"model_id": "modular-v3"},
    )


def test_telemetry_v3_round_trips_schema() -> None:
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads(json.dumps(_event().to_dict())))


def test_telemetry_v3_rejects_invalid_coverage_and_visual_fields() -> None:
    with pytest.raises(ValueError, match="dual_fraction"):
        replace(_event(), dual_fraction=1.1)
    serialized = _event().to_dict()
    serialized["metadata"] = {"atlas_region": "x"}
    with pytest.raises(ValidationError):
        Draft202012Validator(_schema()).validate(serialized)
