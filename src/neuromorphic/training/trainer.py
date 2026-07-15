"""Deterministic baseline training and evaluation."""

from __future__ import annotations

import json
import resource
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from neuromorphic.evaluation.task_metrics import (
    EvaluationRecord,
    aggregate_records,
    evaluation_records,
)
from neuromorphic.tasks.base import DatasetSplit, SequenceTask, TaskBatch
from neuromorphic.tasks.small_graph import SmallGraphTask
from neuromorphic.training.baselines import BaselineOutput, GRUBaseline
from neuromorphic.training.checkpoint import load_checkpoint, save_checkpoint
from neuromorphic.training.config import RunConfig
from neuromorphic.training.metrics import (
    ensure_finite_training_state,
    masked_task_loss,
)


@dataclass(slots=True)
class IndexSampler:
    """Serializable deterministic shuffled-index sampler."""

    size: int
    seed: int
    permutation: np.ndarray
    cursor: int
    epoch: int
    generator: np.random.Generator

    @classmethod
    def create(cls, size: int, seed: int) -> IndexSampler:
        if size <= 0:
            raise ValueError("sampler size must be positive")
        generator = np.random.default_rng(seed)
        return cls(
            size=size,
            seed=seed,
            permutation=generator.permutation(size),
            cursor=0,
            epoch=0,
            generator=generator,
        )

    def next(self, batch_size: int) -> list[int]:
        values: list[int] = []
        while len(values) < batch_size:
            available = min(batch_size - len(values), self.size - self.cursor)
            values.extend(self.permutation[self.cursor : self.cursor + available].tolist())
            self.cursor += available
            if self.cursor == self.size:
                self.epoch += 1
                self.permutation = self.generator.permutation(self.size)
                self.cursor = 0
        return values

    def state_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "seed": self.seed,
            "permutation": self.permutation.tolist(),
            "cursor": self.cursor,
            "epoch": self.epoch,
            "generator_state": self.generator.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("size") != self.size or state.get("seed") != self.seed:
            raise ValueError("sampler state is incompatible")
        permutation = np.asarray(state["permutation"], dtype=np.int64)
        if permutation.shape != (self.size,) or set(permutation.tolist()) != set(range(self.size)):
            raise ValueError("sampler permutation is invalid")
        self.permutation = permutation
        self.cursor = int(state["cursor"])
        self.epoch = int(state["epoch"])
        self.generator.bit_generator.state = state["generator_state"]


def profile_sizes(profile: str) -> dict[DatasetSplit, int]:
    """Return the frozen P1 split budgets."""
    if profile == "smoke":
        return {"train": 64, "validation": 32, "test": 32, "ood": 32}
    if profile == "qualification":
        return {"train": 8_192, "validation": 2_048, "test": 2_048, "ood": 2_048}
    raise ValueError(f"unknown profile: {profile}")


def evaluate_model(
    model: nn.Module,
    task: SequenceTask,
    *,
    split: DatasetSplit,
    size: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate a baseline over a deterministic split."""
    metrics, _ = evaluate_model_with_records(
        model,
        task,
        split=split,
        size=size,
        batch_size=batch_size,
        device=device,
    )
    return metrics


def evaluate_model_with_records(
    model: nn.Module,
    task: SequenceTask,
    *,
    split: DatasetSplit,
    size: int,
    batch_size: int,
    device: torch.device,
    run_seed: int | None = None,
) -> tuple[dict[str, object], list[EvaluationRecord]]:
    """Evaluate a split and retain task-specific per-sample evidence."""
    model.eval()
    records: list[EvaluationRecord] = []
    with torch.no_grad():
        for start in range(0, size, batch_size):
            indices = list(range(start, min(start + batch_size, size)))
            batch = task.generate(split, indices, device=device)
            output = model(batch.inputs, batch.valid_mask)
            if not isinstance(output, BaselineOutput):
                raise TypeError("baseline model must return BaselineOutput")
            records.extend(evaluation_records(output, batch, run_seed=run_seed))
    return aggregate_records(records), records


def forward_training_batch(
    model: nn.Module, batch: TaskBatch, *, tbptt_steps: int
) -> BaselineOutput:
    """Run a training forward pass with recurrent state detached at TBPTT boundaries."""
    if not isinstance(model, GRUBaseline) or batch.sequence_length <= tbptt_steps:
        output = model(batch.inputs, batch.valid_mask)
        if not isinstance(output, BaselineOutput):
            raise TypeError("baseline model must return BaselineOutput")
        return output
    hidden: torch.Tensor | None = None
    logits: list[torch.Tensor] = []
    next_state_logits: list[torch.Tensor] = []
    for start in range(0, batch.sequence_length, tbptt_steps):
        stop = min(start + tbptt_steps, batch.sequence_length)
        output, hidden = model.forward_chunk(
            batch.inputs[:, start:stop], batch.valid_mask[:, start:stop], hidden
        )
        hidden = hidden.detach()
        logits.append(output.logits)
        if output.next_state_logits is not None:
            next_state_logits.append(output.next_state_logits)
    return BaselineOutput(
        logits=torch.cat(logits, dim=1),
        next_state_logits=(torch.cat(next_state_logits, dim=1) if next_state_logits else None),
    )


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True) + "\n")


def measure_latency_ms(
    model: nn.Module, batch: TaskBatch, *, warmup: int = 10, samples: int = 50
) -> dict[str, object]:
    """Measure end-to-end baseline forward P50/P95 latency for one batch."""
    model.eval()
    measurements: list[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            model(batch.inputs, batch.valid_mask)
        for _ in range(samples):
            if batch.inputs.device.type == "mps":
                torch.mps.synchronize()
            if batch.inputs.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            model(batch.inputs, batch.valid_mask)
            if batch.inputs.device.type == "mps":
                torch.mps.synchronize()
            if batch.inputs.device.type == "cuda":
                torch.cuda.synchronize()
            measurements.append((time.perf_counter() - started) * 1_000.0)
    return {
        "p50": float(np.percentile(measurements, 50)),
        "p95": float(np.percentile(measurements, 95)),
        "samples": samples,
        "warmup": warmup,
        "batch_size": batch.batch_size,
        "sequence_length": batch.sequence_length,
        "dtype": str(batch.inputs.dtype),
        "device": str(batch.inputs.device),
        "sequences_per_second": float(batch.batch_size * 1_000.0 / np.percentile(measurements, 50)),
    }


def _device_memory_bytes(device: torch.device) -> tuple[int, str]:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device)), "torch.cuda.max_memory_allocated"
    if device.type == "mps":
        return int(
            torch.mps.current_allocated_memory()
        ), "sampled torch.mps.current_allocated_memory"
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        maximum *= 1024
    return maximum, "process ru_maxrss"


def train_baseline(
    *,
    config: RunConfig,
    task: SequenceTask,
    model: nn.Module,
    device: torch.device,
    run_directory: Path,
) -> dict[str, Any]:
    """Train one P1 baseline with checkpointing and early stopping."""
    sizes = profile_sizes(config.task.profile)
    run_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = run_directory / "metrics.jsonl"
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    sampler = IndexSampler.create(sizes["train"], config.seed)
    state: dict[str, Any] = {"step": 0, "best_metric": -float("inf"), "stale_evals": 0}
    compatible_config = config.checkpoint_compatible_dict()
    if config.resume is not None:
        loaded_state, sampler_state = load_checkpoint(
            config.resume,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            config=compatible_config,
        )
        state.update(loaded_state)
        sampler.load_state_dict(sampler_state)

    started = time.perf_counter()
    final_loss = float("nan")
    training_examples = 0
    training_tokens = 0
    validation_evaluations = 0
    loss_history: list[float] = []
    peak_memory_bytes, peak_memory_method = _device_memory_bytes(device)
    while int(state["step"]) < config.training.max_steps:
        model.train()
        indices = sampler.next(config.training.batch_size)
        batch = task.generate("train", indices, device=device)
        training_examples += len(indices)
        training_tokens += int(batch.valid_mask.sum().item())
        optimizer.zero_grad(set_to_none=True)
        output = forward_training_batch(model, batch, tbptt_steps=config.training.tbptt_steps)
        loss, loss_parts = masked_task_loss(
            output, batch, auxiliary_weight=config.training.auxiliary_loss_weight
        )
        loss.backward()  # type: ignore[no-untyped-call]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.optimizer.gradient_clip_norm
        )
        numerical = {**loss_parts, "gradient_norm": float(gradient_norm.detach().cpu())}
        ensure_finite_training_state(loss=loss, model=model, metrics=numerical)
        optimizer.step()
        ensure_finite_training_state(loss=loss, model=model, metrics=numerical)
        sampled_memory, peak_memory_method = _device_memory_bytes(device)
        peak_memory_bytes = max(peak_memory_bytes, sampled_memory)
        state["step"] = int(state["step"]) + 1
        final_loss = float(loss.detach().cpu())
        loss_history.append(final_loss)
        _append_jsonl(metrics_path, {"step": state["step"], "split": "train", **numerical})

        should_evaluate = int(state["step"]) % config.training.eval_interval == 0
        final_step = int(state["step"]) == config.training.max_steps
        if should_evaluate or final_step:
            validation_evaluations += 1
            validation = evaluate_model(
                model,
                task,
                split="validation",
                size=sizes["validation"],
                batch_size=config.training.batch_size,
                device=device,
            )
            if isinstance(task, SmallGraphTask):

                def validation_policy(observation: torch.Tensor) -> torch.Tensor:
                    with torch.no_grad():
                        inputs = observation.reshape(1, 1, -1)
                        mask = torch.ones((1, 1), dtype=torch.bool, device=observation.device)
                        policy_output = model(inputs, mask)
                        if not isinstance(policy_output, BaselineOutput):
                            raise TypeError("baseline model must return BaselineOutput")
                        return policy_output.logits[0, 0]

                rollout_metrics = task.rollout(
                    validation_policy,
                    "validation",
                    list(range(sizes["validation"])),
                    device=device,
                )
                validation = {**rollout_metrics, **validation}
            _append_jsonl(
                metrics_path, {"step": state["step"], "split": "validation", **validation}
            )
            primary_name = {
                "associative_recall.v1": "query_accuracy",
                "delayed_rule_switch.v1": "response_accuracy",
                "small_graph.v1": "success_rate",
            }[task.task_id]
            primary_value = validation.get(primary_name, next(iter(validation.values())))
            if isinstance(primary_value, bool) or not isinstance(primary_value, (int, float)):
                raise TypeError(f"primary metric {primary_name!r} must be numeric")
            primary = float(primary_value)
            if primary > float(state["best_metric"]) + config.training.min_delta:
                state["best_metric"] = primary
                state["stale_evals"] = 0
                save_checkpoint(
                    run_directory / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    training_state=state,
                    sampler_state=sampler.state_dict(),
                    config=compatible_config,
                )
            else:
                state["stale_evals"] = int(state["stale_evals"]) + 1

        if int(state["step"]) % config.training.checkpoint_interval == 0 or final_step:
            save_checkpoint(
                run_directory / "latest.pt",
                model=model,
                optimizer=optimizer,
                scheduler=None,
                training_state=state,
                sampler_state=sampler.state_dict(),
                config=compatible_config,
            )
        if int(state["stale_evals"]) >= config.training.patience:
            break

    training_wall_clock_seconds = time.perf_counter() - started
    comparison_window = min(10, len(loss_history))
    initial_loss_mean = float(np.mean(loss_history[:comparison_window]))
    final_loss_mean = float(np.mean(loss_history[-comparison_window:]))
    loss_decreased = final_loss_mean < initial_loss_mean
    if config.training.require_loss_decrease and not loss_decreased:
        raise ValueError(
            f"training loss did not decrease: initial={initial_loss_mean}, final={final_loss_mean}"
        )
    completed_steps = int(state["step"])
    selected_state, _ = load_checkpoint(
        run_directory / "best.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        config=compatible_config,
        restore_rng=False,
    )
    test_metrics, test_records = evaluate_model_with_records(
        model,
        task,
        split="test",
        size=sizes["test"],
        batch_size=config.training.batch_size,
        device=device,
        run_seed=config.seed,
    )
    ood_metrics, ood_records = evaluate_model_with_records(
        model,
        task,
        split="ood",
        size=sizes["ood"],
        batch_size=config.training.batch_size,
        device=device,
        run_seed=config.seed,
    )
    sample_metrics_path = run_directory / "evaluation_samples.jsonl"
    for record in [*test_records, *ood_records]:
        _append_jsonl(sample_metrics_path, record)
    latency_batch = task.generate("test", [0], device=device)
    latency = measure_latency_ms(model, latency_batch)
    summary: dict[str, Any] = {
        "steps": completed_steps,
        "selected_checkpoint": "best.pt",
        "selected_checkpoint_step": int(selected_state["step"]),
        "best_validation_metric": state["best_metric"],
        "final_loss": final_loss,
        "test": test_metrics,
        "ood": ood_metrics,
        "wall_clock_seconds": time.perf_counter() - started,
        "training_wall_clock_seconds": training_wall_clock_seconds,
        "latency_ms": latency,
        "training_examples": training_examples,
        "training_tokens": training_tokens,
        "validation_evaluations": validation_evaluations,
        "initial_loss_mean": initial_loss_mean,
        "final_loss_mean": final_loss_mean,
        "loss_decreased": loss_decreased,
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_method": peak_memory_method,
    }
    if isinstance(task, SmallGraphTask):

        def policy(observation: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                inputs = observation.reshape(1, 1, -1)
                mask = torch.ones((1, 1), dtype=torch.bool, device=observation.device)
                output = model(inputs, mask)
                if not isinstance(output, BaselineOutput):
                    raise TypeError("baseline model must return BaselineOutput")
                return output.logits[0, 0]

        summary["graph_rollout"] = {
            "test": task.rollout(policy, "test", list(range(sizes["test"])), device=device),
            "ood": task.rollout(policy, "ood", list(range(sizes["ood"])), device=device),
        }
    final_memory, peak_memory_method = _device_memory_bytes(device)
    summary["peak_memory_bytes"] = max(peak_memory_bytes, final_memory)
    summary["peak_memory_method"] = peak_memory_method
    summary["wall_clock_seconds"] = time.perf_counter() - started
    return summary
