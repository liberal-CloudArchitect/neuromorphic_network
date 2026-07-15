from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import ValidationError  # type: ignore[import-untyped]

from neuromorphic.telemetry import SCHEMA_VERSION, TelemetryEdge, TelemetryEvent

ROOT = Path(__file__).resolve().parents[2]


def _schema() -> dict[str, object]:
    loaded = json.loads((ROOT / "schemas" / "telemetry-v1.json").read_text(encoding="utf-8"))
    return cast(dict[str, object], loaded)


def _event() -> TelemetryEvent:
    return TelemetryEvent(
        event_id="event-1",
        parent_event_id=None,
        run_id="run-1",
        episode_id=3,
        step=4,
        token_id=2,
        monotonic_time_ns=100,
        module_clock=4,
        phase="train",
        module_id="episodic_memory.v1",
        source="forward",
        reducer_version="mean-rms-v1",
        baseline_version="ema-v1",
        compute_gate=True,
        activity_raw=0.5,
        activity_z=1.2,
        routing_mass=0.6,
        state_change=0.2,
        surprise=0.3,
        confidence=0.8,
        event_tags=("retrieve", "write"),
        edges=(TelemetryEdge("working_memory.v1", 0.4, routed=True),),
        metadata={"task": "associative-recall-v1", "loss": 0.25},
    )


def test_telemetry_event_round_trips_through_accepted_schema() -> None:
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    serialized = json.loads(json.dumps(_event().to_dict()))
    Draft202012Validator(schema).validate(serialized)
    assert schema["x-status"] == "ACCEPTED"
    assert serialized["schema_version"] == SCHEMA_VERSION == "telemetry-v1"


def test_schema_rejects_unregistered_module_and_nested_metadata() -> None:
    validator = Draft202012Validator(_schema())
    serialized = json.loads(json.dumps(_event().to_dict()))
    serialized["module_id"] = "visual_cortex"
    with pytest.raises(ValidationError):
        validator.validate(serialized)
    serialized["module_id"] = "episodic_memory.v1"
    serialized["metadata"] = {"raw_tensor": [1, 2, 3]}
    with pytest.raises(ValidationError):
        validator.validate(serialized)


def test_event_validator_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="confidence"):
        values = _event().to_dict()
        values.pop("schema_version")
        values["confidence"] = 1.5
        TelemetryEvent(**values)  # type: ignore[arg-type]
