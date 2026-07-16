"""Model-agnostic P3 sample records and bounded representation analyses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from neuromorphic.evaluation.task_metrics import evaluation_records
from neuromorphic.tasks.base import TaskBatch
from neuromorphic.training.baselines import BaselineOutput


class TensorOutput(Protocol):
    logits: Tensor
    next_state_logits: Tensor | None


def as_baseline_output(output: object) -> BaselineOutput:
    """Normalize P1, shared baseline, or modular outputs without wrapping tensors in Pydantic."""

    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise TypeError("model output does not expose tensor logits")
    next_state = getattr(output, "next_state_logits", None)
    if next_state is None:
        next_state = getattr(output, "prediction_logits", None)
    if next_state is not None and not isinstance(next_state, Tensor):
        raise TypeError("model next-state output must be a tensor or None")
    return BaselineOutput(logits=logits, next_state_logits=next_state)


def p3_sample_records(
    output: object,
    batch: TaskBatch,
    *,
    run_seed: int,
    model_id: str,
    variant_id: str,
) -> list[dict[str, object]]:
    """Add P3 comparison identity to the existing per-sample task metrics."""

    distribution = batch.metadata.get("distribution", "v1")
    if not isinstance(distribution, str):
        raise ValueError("distribution metadata must be a string")
    records = evaluation_records(as_baseline_output(output), batch, run_seed=run_seed)
    enriched: list[dict[str, object]] = []
    for record in records:
        value: dict[str, object] = dict(record)
        value.update(
            {
                "schema_version": "p3-evaluation-sample-v1",
                "model_id": model_id,
                "variant_id": variant_id,
                "distribution": distribution,
                "stratum": str(record["bootstrap_stratum"]),
            }
        )
        enriched.append(value)
    return enriched


def linear_cka(first: Tensor, second: Tensor) -> float:
    """Compute bounded linear centered-kernel alignment without extra dependencies."""

    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
        raise ValueError("CKA inputs must be [N, F] with equal N")
    x = first.detach().to(dtype=torch.float64, device="cpu")
    y = second.detach().to(dtype=torch.float64, device="cpu")
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(x.T @ y).square()
    denominator = torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y)
    if float(denominator) == 0.0:
        raise ValueError("CKA is undefined for constant representations")
    return float((cross / denominator).clamp(0.0, 1.0))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0
        index = end
    return ranks


def rsa_spearman(first: Tensor, second: Tensor) -> float:
    """Compare pairwise cosine dissimilarities using Spearman correlation."""

    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
        raise ValueError("RSA inputs must be [N, F] with equal N")
    if first.shape[0] < 3:
        raise ValueError("RSA requires at least three observations")

    def distances(value: Tensor) -> np.ndarray:
        normalized = torch.nn.functional.normalize(value.detach().to(torch.float64), dim=-1)
        matrix = 1.0 - normalized @ normalized.T
        indices = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1)
        return matrix[indices[0], indices[1]].cpu().numpy()

    left = _rank(distances(first))
    right = _rank(distances(second))
    if np.std(left) == 0 or np.std(right) == 0:
        raise ValueError("RSA is undefined for constant dissimilarities")
    return float(np.corrcoef(left, right)[0, 1])


def linear_probe_accuracy(
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
    *,
    ridge: float = 1e-3,
) -> float:
    """Fit a deterministic closed-form ridge probe and report held-out accuracy."""

    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("probe features must be two-dimensional")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("probe feature widths must match")
    classes = int(torch.cat((train_labels, test_labels)).max().item()) + 1
    x = train_features.detach().to(device="cpu", dtype=torch.float64)
    y = torch.nn.functional.one_hot(train_labels.to("cpu"), classes).to(torch.float64)
    identity = torch.eye(x.shape[1], dtype=torch.float64)
    weights = torch.linalg.solve(x.T @ x + ridge * identity, x.T @ y)
    prediction = (test_features.detach().to(device="cpu", dtype=torch.float64) @ weights).argmax(-1)
    return float(prediction.eq(test_labels.to("cpu")).to(torch.float64).mean())


def primary_metric_name(task_id: str) -> str:
    mapping = {
        "associative_recall.v1": "query_accuracy",
        "delayed_rule_switch.v1": "response_accuracy",
        "small_graph.v1": "success_rate",
    }
    try:
        return mapping[task_id]
    except KeyError as error:
        raise ValueError(f"unsupported P3 task: {task_id}") from error


def metric_values(records: Sequence[Mapping[str, object]], name: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"record is missing numeric metric {name}")
        values.append(float(value))
    return values


__all__ = [
    "as_baseline_output",
    "linear_cka",
    "linear_probe_accuracy",
    "metric_values",
    "p3_sample_records",
    "primary_metric_name",
    "rsa_spearman",
]
