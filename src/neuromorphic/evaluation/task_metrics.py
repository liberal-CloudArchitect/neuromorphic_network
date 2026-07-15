"""Task-specific per-sample evaluation records and descriptive aggregates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from neuromorphic.tasks.base import TaskBatch
from neuromorphic.training.baselines import BaselineOutput

type EvaluationRecord = dict[str, str | int | float | bool | None]


def _metadata_sequence(batch: TaskBatch, name: str) -> tuple[Any, ...]:
    value = batch.metadata.get(name)
    if not isinstance(value, tuple) or len(value) != batch.batch_size:
        raise ValueError(f"metadata {name!r} must contain one value per sample")
    return value


def evaluation_records(
    output: BaselineOutput,
    batch: TaskBatch,
    *,
    run_seed: int | None = None,
) -> list[EvaluationRecord]:
    """Convert model output and a task batch into JSON-safe per-sample records."""
    batch.validate()
    if output.logits.shape[:2] != batch.targets.shape:
        raise ValueError("output logits must start with [B, T]")
    task_id = batch.metadata.get("task_id")
    split = batch.metadata.get("split")
    if not isinstance(task_id, str) or not isinstance(split, str):
        raise ValueError("task metadata must contain string task_id and split")
    sample_indices = _metadata_sequence(batch, "sample_indices")
    predictions = output.logits.argmax(dim=-1)
    records: list[EvaluationRecord] = []

    if task_id == "associative_recall.v1":
        distractor_counts = _metadata_sequence(batch, "distractor_counts")
        pair_counts = _metadata_sequence(batch, "pair_counts")
        for row in range(batch.batch_size):
            selected = batch.loss_mask[row]
            if int(selected.sum().item()) != 1:
                raise ValueError("associative recall requires one query per sample")
            correct = predictions[row, selected].eq(batch.targets[row, selected])
            distractors = int(distractor_counts[row])
            pairs = int(pair_counts[row])
            records.append(
                {
                    "schema_version": "evaluation-sample-v1",
                    "task_id": task_id,
                    "split": split,
                    "sample_index": int(sample_indices[row]),
                    "seed": run_seed,
                    "query_accuracy": float(correct.float().mean().cpu()),
                    "distractor_count": distractors,
                    "pair_count": pairs,
                    "interference_stratum": "low" if distractors <= 2 else "high",
                    "bootstrap_stratum": f"pairs-{pairs}/interference-{distractors}",
                }
            )
        return records

    if task_id == "delayed_rule_switch.v1":
        switch_positions = _metadata_sequence(batch, "switch_positions")
        mean_delays = _metadata_sequence(batch, "mean_delays")
        for row in range(batch.batch_size):
            selected = batch.loss_mask[row]
            if not selected.any():
                raise ValueError("delayed rule switch requires supervised responses")
            correct = predictions[row, selected].eq(batch.targets[row, selected])
            mean_delay = float(mean_delays[row])
            switch_position = switch_positions[row]
            first_post_switch_accuracy = (
                None
                if switch_position is None
                else float(correct[int(switch_position)].float().cpu())
            )
            delay_stratum = (
                "ood" if mean_delay >= 9.0 else ("short" if mean_delay <= 5.0 else "long")
            )
            records.append(
                {
                    "schema_version": "evaluation-sample-v1",
                    "task_id": task_id,
                    "split": split,
                    "sample_index": int(sample_indices[row]),
                    "seed": run_seed,
                    "response_accuracy": float(correct.float().mean().cpu()),
                    "switched": switch_position is not None,
                    "first_post_switch_accuracy": first_post_switch_accuracy,
                    "mean_delay": mean_delay,
                    "delay_stratum": delay_stratum,
                    "bootstrap_stratum": (
                        f"delay-{delay_stratum}/"
                        f"{'switch' if switch_position is not None else 'no-switch'}"
                    ),
                }
            )
        return records

    if task_id == "small_graph.v1":
        optimal = batch.auxiliary_targets.get("optimal_action_mask")
        if optimal is None:
            raise ValueError("SmallGraph requires optimal_action_mask")
        next_state = batch.auxiliary_targets.get("next_state")
        next_predictions = (
            None if output.next_state_logits is None else output.next_state_logits.argmax(dim=-1)
        )
        for row in range(batch.batch_size):
            selected = batch.loss_mask[row]
            selected_predictions = predictions[row, selected]
            selected_optimal = optimal[row, selected]
            correct = selected_optimal.gather(-1, selected_predictions.unsqueeze(-1)).squeeze(-1)
            record: EvaluationRecord = {
                "schema_version": "evaluation-sample-v1",
                "task_id": task_id,
                "split": split,
                "sample_index": int(sample_indices[row]),
                "seed": run_seed,
                "optimal_action_rate": float(correct.float().mean().cpu()),
                "bootstrap_stratum": (
                    f"nodes-{int(batch.auxiliary_targets['node_count'][row, 0].item())}/"
                    f"distance-{int(batch.auxiliary_targets['optimal_distance'][row, 0].item())}"
                ),
            }
            if next_state is not None and next_predictions is not None:
                next_correct = next_predictions[row, selected].eq(next_state[row, selected])
                record["next_state_accuracy"] = float(next_correct.float().mean().cpu())
            records.append(record)
        return records

    raise ValueError(f"unsupported task_id: {task_id}")


def _mean(records: Sequence[Mapping[str, object]], key: str) -> float | None:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"metric {key!r} must be numeric")
        values.append(float(value))
    return None if not values else sum(values) / len(values)


def aggregate_records(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate one task/split collection without hiding unavailable contrasts."""
    if not records:
        raise ValueError("records cannot be empty")
    task_ids = {record.get("task_id") for record in records}
    splits = {record.get("split") for record in records}
    if len(task_ids) != 1 or len(splits) != 1:
        raise ValueError("records must belong to one task and split")
    task_id = next(iter(task_ids))

    if task_id == "associative_recall.v1":
        low = [record for record in records if record.get("interference_stratum") == "low"]
        high = [record for record in records if record.get("interference_stratum") == "high"]
        low_accuracy = _mean(low, "query_accuracy")
        high_accuracy = _mean(high, "query_accuracy")
        return {
            "query_accuracy": _mean(records, "query_accuracy"),
            "interference_low_accuracy": low_accuracy,
            "interference_high_accuracy": high_accuracy,
            "interference_drop": (
                None
                if low_accuracy is None or high_accuracy is None
                else low_accuracy - high_accuracy
            ),
            "sample_count": len(records),
        }

    if task_id == "delayed_rule_switch.v1":
        switched = [record for record in records if record.get("switched") is True]
        controls = [record for record in records if record.get("switched") is False]
        first_post_switch_accuracy = _mean(switched, "first_post_switch_accuracy")
        control_accuracy = _mean(controls, "response_accuracy")
        delay_strata = sorted({str(record["delay_stratum"]) for record in records})
        return {
            "response_accuracy": _mean(records, "response_accuracy"),
            "first_post_switch_accuracy": first_post_switch_accuracy,
            "no_switch_accuracy": control_accuracy,
            "switch_cost": (
                None
                if first_post_switch_accuracy is None or control_accuracy is None
                else control_accuracy - first_post_switch_accuracy
            ),
            "delay_accuracy": {
                stratum: _mean(
                    [record for record in records if record.get("delay_stratum") == stratum],
                    "response_accuracy",
                )
                for stratum in delay_strata
            },
            "sample_count": len(records),
        }

    if task_id == "small_graph.v1":
        return {
            "optimal_action_rate": _mean(records, "optimal_action_rate"),
            "next_state_accuracy": _mean(records, "next_state_accuracy"),
            "sample_count": len(records),
        }
    raise ValueError(f"unsupported task_id: {task_id}")
