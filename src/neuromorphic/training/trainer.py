"""Deterministic baseline training and evaluation."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from neuromorphic.tasks.base import DatasetSplit, SequenceTask, TaskBatch
from neuromorphic.tasks.small_graph import SmallGraphTask
from neuromorphic.training.baselines import BaselineOutput
from neuromorphic.training.checkpoint import load_checkpoint, save_checkpoint
from neuromorphic.training.config import RunConfig
from neuromorphic.training.metrics import (
    ensure_finite_training_state,
    masked_task_loss,
    task_metrics,
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
) -> dict[str, float]:
    """Evaluate a baseline over a deterministic split."""
    model.eval()
    totals: dict[str, float] = {}
    weight = 0
    with torch.no_grad():
        for start in range(0, size, batch_size):
            indices = list(range(start, min(start + batch_size, size)))
            batch = task.generate(split, indices, device=device)
            output = model(batch.inputs, batch.valid_mask)
            if not isinstance(output, BaselineOutput):
                raise TypeError("baseline model must return BaselineOutput")
            values = task_metrics(output, batch)
            batch_weight = int(batch.loss_mask.sum().item())
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value * batch_weight
            weight += batch_weight
    if weight == 0:
        raise ValueError("evaluation split contains no supervised positions")
    return {name: value / weight for name, value in totals.items()}


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True) + "\n")


def measure_latency_ms(
    model: nn.Module, batch: TaskBatch, *, samples: int = 20
) -> dict[str, float | int]:
    """Measure end-to-end baseline forward P50/P95 latency for one batch."""
    model.eval()
    measurements: list[float] = []
    with torch.no_grad():
        for _ in range(3):
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
    }


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
    while int(state["step"]) < config.training.max_steps:
        model.train()
        indices = sampler.next(config.training.batch_size)
        batch = task.generate("train", indices, device=device)
        training_examples += len(indices)
        training_tokens += int(batch.valid_mask.sum().item())
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.inputs, batch.valid_mask)
        if not isinstance(output, BaselineOutput):
            raise TypeError("baseline model must return BaselineOutput")
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
        state["step"] = int(state["step"]) + 1
        final_loss = float(loss.detach().cpu())
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
            _append_jsonl(
                metrics_path, {"step": state["step"], "split": "validation", **validation}
            )
            primary = next(iter(validation.values()))
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

    elapsed = time.perf_counter() - started
    test_metrics = evaluate_model(
        model,
        task,
        split="test",
        size=sizes["test"],
        batch_size=config.training.batch_size,
        device=device,
    )
    ood_metrics = evaluate_model(
        model,
        task,
        split="ood",
        size=sizes["ood"],
        batch_size=config.training.batch_size,
        device=device,
    )
    latency_batch = task.generate("test", [0], device=device)
    summary: dict[str, Any] = {
        "steps": state["step"],
        "best_validation_metric": state["best_metric"],
        "final_loss": final_loss,
        "test": test_metrics,
        "ood": ood_metrics,
        "wall_clock_seconds": elapsed,
        "latency_ms": measure_latency_ms(model, latency_batch),
        "training_examples": training_examples,
        "training_tokens": training_tokens,
        "validation_evaluations": validation_evaluations,
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
    return summary
