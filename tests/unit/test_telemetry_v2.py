from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import ValidationError  # type: ignore[import-untyped]

from neuromorphic.core.contracts import BrainPacket, ModuleState
from neuromorphic.telemetry import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    TelemetryV2Event,
    detached_scalar,
)

ROOT = Path(__file__).resolve().parents[2]


def _schema() -> dict[str, object]:
    loaded = json.loads((ROOT / "schemas" / "telemetry-v2.json").read_text(encoding="utf-8"))
    return cast(dict[str, object], loaded)


def _event() -> TelemetryV2Event:
    return TelemetryV2Event(
        event_id="route-12",
        run_id="p4-qualification",
        global_step=12,
        task="associative_recall.v1",
        event_type="route",
        phase="train",
        module_id="sparse_router.v2",
        compute_gate=True,
        reserved_count=8,
        learned_count=6,
        raw_count=14,
        executed_count=14,
        capacity=16,
        drop_count=0,
        reserved_coverage=1.0,
        forecast_coverage=0.96,
        forecast_error=0.2,
        persistence_error=0.25,
        active_mac=120.0,
        dense_mac=240.0,
        metadata={"event_class": "query", "reroute_rate": 0.0},
    )


def test_v2_event_round_trips_strict_schema() -> None:
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    serialized = json.loads(json.dumps(_event().to_dict()))
    Draft202012Validator(schema).validate(serialized)
    assert serialized["schema_version"] == SCHEMA_VERSION_V2 == "telemetry-v2"
    assert "Artificial computational abstraction" in serialized["scientific_disclaimer"]


def test_v2_rejects_nested_nonfinite_and_visualization_metadata() -> None:
    with pytest.raises(TypeError, match="detached scalar"):
        replace(_event(), metadata={"raw": [1, 2]})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="finite"):
        replace(_event(), forecast_error=float("nan"))
    with pytest.raises(ValueError, match="atlas, web, and viewer"):
        replace(_event(), metadata={"atlas_region": "x"})

    serialized = json.loads(json.dumps(_event().to_dict()))
    serialized["metadata"] = {"viewer_state": 1}
    with pytest.raises(ValidationError):
        Draft202012Validator(_schema()).validate(serialized)


def test_detached_scalar_accepts_only_scalar_tensor() -> None:
    value = torch.tensor(0.5, requires_grad=True)
    assert detached_scalar(value) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="exactly one"):
        detached_scalar(torch.ones(2))


def test_v2_ids_are_valid_contract_owners_without_changing_v1_alias() -> None:
    representation = torch.zeros((1, 1, 4))
    packet = BrainPacket(
        representation=representation,
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
        modality="latent",
        step_index=torch.zeros((1, 1), dtype=torch.long),
        source_module="predictive_adapter.v2",
    )
    state = ModuleState(
        module_id="sparse_router.v2",
        version="router-state-v2",
        tensors={"count": torch.zeros(1)},
    )
    assert packet.source_module == "predictive_adapter.v2"
    assert state.module_id == "sparse_router.v2"
    assert SCHEMA_VERSION == "telemetry-v1"
