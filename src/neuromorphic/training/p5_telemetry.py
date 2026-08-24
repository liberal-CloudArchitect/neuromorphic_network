"""Build telemetry-v3 only after P5 outputs, loss, and gradients exist."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from neuromorphic.core.registry import SPARSE_ROUTER_V3
from neuromorphic.tasks.base import TaskBatch
from neuromorphic.telemetry.events_v3 import TelemetryV3Event


def _scalar(value: Tensor) -> float:
    if value.numel() != 1:
        raise ValueError("P5 telemetry metrics must be scalar tensors")
    return float(value.detach().cpu())


def build_p5_telemetry_event(
    *,
    run_id: str,
    global_step: int,
    batch: TaskBatch,
    output: Any,
    scorer_grad_norm: Tensor | None,
    phase: str = "train",
) -> TelemetryV3Event:
    """Synchronize detached scalars; callers must skip this function when disabled."""

    transition_count = int(_scalar(output.module_metrics["predictive.transition_count"]))
    eligible = int(_scalar(output.module_metrics["predictive.eligible_transition_count"]))
    surprise_sum = _scalar(output.module_metrics["predictive.surprise_sum"])
    required = int(_scalar(output.module_metrics["routing.reserved_tokens"]))
    executed = int(_scalar(output.module_metrics["routing.reserved_executed"]))
    dual = int(_scalar(output.module_metrics["routing.dual_tokens"]))
    valid = int(batch.valid_mask.sum().item())
    forecast_error = float(output.forecast_error.detach().sum().cpu() / max(transition_count, 1))
    persistence_error = float(
        output.persistence_error.detach().sum().cpu() / max(transition_count, 1)
    )
    return TelemetryV3Event(
        event_id=f"{run_id}:{global_step}:{batch.metadata['task_id']}",
        run_id=run_id,
        global_step=global_step,
        task=str(batch.metadata["task_id"]),
        phase=phase,  # type: ignore[arg-type]
        module_id=SPARSE_ROUTER_V3,
        compute_gate=True,
        surprise_count=transition_count,
        surprise_sum=surprise_sum,
        semantic_required=required,
        semantic_executed=executed,
        semantic_coverage=executed / max(required, 1) if required else 1.0,
        dual_count=dual,
        dual_fraction=dual / max(valid, 1),
        forecast_coverage=transition_count / max(eligible, 1),
        forecast_error=forecast_error,
        persistence_error=persistence_error,
        scorer_grad_norm=None if scorer_grad_norm is None else _scalar(scorer_grad_norm),
        active_calls=_scalar(output.cost_statistics["optional.active_calls"]),
        dense_calls=_scalar(output.cost_statistics["optional.dense_calls"]),
        metadata={"model_id": "modular-v3", "qualification_only": True},
    )


__all__ = ["build_p5_telemetry_event"]
