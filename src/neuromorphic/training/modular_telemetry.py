"""Detached, bounded telemetry reduction for P2 modular-network runs."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping

from neuromorphic.core.registry import MODULE_IDS
from neuromorphic.telemetry.events import TelemetryEdge, TelemetryEvent


def _event_id(run_id: str, step: int, module_id: str) -> str:
    material = f"{run_id}:{step}:{module_id}".encode()
    return hashlib.sha256(material).hexdigest()[:24]


def build_step_events(
    *,
    run_id: str,
    step: int,
    phase: str,
    split: str,
    module_metrics: Mapping[str, Mapping[str, float | bool | None]],
    reducer_version: str,
    baseline_version: str,
) -> tuple[TelemetryEvent, ...]:
    """Create at most one aggregate event per registered module and step.

    Inputs must already be detached Python scalars. This function does not use
    model tensors or random-number generators.
    """

    events: list[TelemetryEvent] = []
    timestamp = time.monotonic_ns()
    for clock, module_id in enumerate(MODULE_IDS):
        metrics = module_metrics.get(module_id)
        if metrics is None:
            continue
        compute_gate = bool(metrics.get("compute_gate", True))
        activity = metrics.get("activity_raw")
        routing_mass = metrics.get("routing_mass")
        state_change = metrics.get("state_change")
        surprise = metrics.get("surprise")
        confidence = metrics.get("confidence")
        grad_rms = metrics.get("grad_rms")
        update_rms = metrics.get("update_rms")
        tag = {
            "episodic_memory.v1": "retrieve",
            "predictive_adapter.v1": "predict",
            "action_selector.v1": "select",
            "sparse_router.v1": "route",
        }.get(module_id)
        edge_target = {
            "sensory_encoder.v1": "sparse_router.v1",
            "episodic_memory.v1": "sparse_router.v1",
            "working_memory.v1": "sparse_router.v1",
            "predictive_adapter.v1": "action_selector.v1",
            "sparse_router.v1": "action_selector.v1",
        }.get(module_id)
        edges: tuple[TelemetryEdge, ...] = ()
        if edge_target is not None:
            edges = (
                TelemetryEdge(
                    target_module_id=edge_target,
                    flow=max(float(routing_mass or 0.0), 0.0),
                    routed=module_id
                    in {
                        "episodic_memory.v1",
                        "working_memory.v1",
                        "predictive_adapter.v1",
                        "sparse_router.v1",
                    },
                ),
            )
        events.append(
            TelemetryEvent(
                event_id=_event_id(run_id, step, module_id),
                parent_event_id=None,
                run_id=run_id,
                episode_id="batch-aggregate",
                step=step,
                token_id=None,
                monotonic_time_ns=timestamp,
                module_clock=clock,
                phase=phase,  # type: ignore[arg-type]
                module_id=module_id,
                source="p2-modular-network",
                reducer_version=reducer_version,
                baseline_version=baseline_version,
                compute_gate=compute_gate,
                activity_raw=None if activity is None else float(activity),
                activity_z=None,
                routing_mass=None if routing_mass is None else max(float(routing_mass), 0.0),
                state_change=None if state_change is None else max(float(state_change), 0.0),
                surprise=None if surprise is None else max(float(surprise), 0.0),
                confidence=(None if confidence is None else min(max(float(confidence), 0.0), 1.0)),
                event_tags=() if tag is None else (tag,),  # type: ignore[arg-type]
                edges=edges,
                metadata={"split": split, "aggregation": "batch-mean"},
                grad_rms=None if grad_rms is None else max(float(grad_rms), 0.0),
                update_rms=None if update_rms is None else max(float(update_rms), 0.0),
            )
        )
    return tuple(events)


__all__ = ["build_step_events"]
