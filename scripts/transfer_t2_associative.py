"""T2: does a local associative readout on a frozen encoder stop class-incremental forgetting?

Protocol docs/transfer/T2-associative-readout-protocol.md (CR-014, exploratory).
Five tasks of four multi-modal classes are learned one after another.  The
transfer model keeps the main-network encoder fixed at its random
initialization (as F1a's PN->KC map) and grows a local associative readout on
a sparse expansion; baselines are the shared GRU trained by SGD, with replay
and with EWC, and the modular network with an SGD head; ablations swap the
learning rule, let the encoder learn, or drop the expansion.
``--pilot`` runs world/seed 999 on train/validation only (design check).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neuromorphic.tasks.base import DatasetSplit, TaskBatch
from neuromorphic.transfer.associative_readout import (
    AssociativeReadout,
    FrameEncoder,
    GRUClassifier,
    ModularClassifier,
    SparseExpansion,
)
from neuromorphic.transfer.incremental_patterns import IncrementalPatternTask
from neuromorphic.transfer.statistics import paired_contrasts
from neuromorphic.transfer.surprise_wm import parameter_count, selector_head

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/transfer"
SIGMA = 0.5
TASKS, CLASSES = 5, 20
SEQUENTIAL = (
    "gru_sgd",
    "gru_replay",
    "gru_ewc",
    "mod_sgd",
    "mod_assoc",
    "abl_sgdread",
    "abl_plastic",
    "abl_ncm",
    "abl_pretrained",
)
MODELS = (*SEQUENTIAL, "gru_joint")
SEEDS = tuple(range(801, 811))
PILOT_SEED = 999
LEARNING_RATES = (1e-3, 3e-4)
EWC_LAMBDAS = (1e2, 1e3, 1e4, 1e5, 1e6)
UPDATES_PER_TASK, BATCH = 1000, 64
GRU_HIDDEN, UNITS, FAN_IN, ACTIVE = 56, 1280, 6, 64
REPLAY_PER_CLASS, FISHER_SAMPLES = 10, 200
SPECIFICATION = {
    "H1": ("test_final/mod_assoc", "test_final/gru_sgd", 0.10, "superiority"),
    "H2": ("forgetting/gru_sgd", "forgetting/mod_assoc", 0.02, "superiority"),
    "H3": ("test_final/mod_assoc", "test_final/abl_sgdread", 0.05, "superiority"),
    "H4": ("test_final/mod_assoc", "test_final/abl_plastic", 0.05, "superiority"),
    "H5": ("test_final/mod_assoc", "test_final/abl_ncm", 0.03, "superiority"),
    "H6": ("test_final/mod_assoc", "test_final/gru_replay", 0.02, "superiority"),
    "H7": ("test_final/mod_assoc", "test_final/gru_ewc", 0.02, "superiority"),
    "H8": ("test_final/mod_assoc", "test_final/gru_replay", 0.02, "noninferiority"),
}
Predict = Callable[[TaskBatch, int], Tensor]  # (batch, seen classes) -> class scores


def evaluate(
    predict: Predict, task: IncrementalPatternTask, split: DatasetSplit, tasks_seen: int
) -> list[float]:
    """Accuracy on each seen task, choosing among all classes seen so far."""

    seen = tasks_seen * task.classes_per_task
    accuracies = []
    with torch.no_grad():
        for j in range(tasks_seen):
            batch = task.generate(split, task.indices(split, j))
            scores = predict(batch, seen)[:, :seen]
            accuracies.append(float(scores.argmax(dim=-1).eq(batch.targets[:, -1]).float().mean()))
    return accuracies


def masked_loss(logits: Tensor, labels: Tensor, seen: int) -> Tensor:
    return F.cross_entropy(logits[:, :seen], labels)


def sample(rng: np.random.Generator, pool: list[int], count: int) -> list[int]:
    return [pool[i] for i in rng.integers(0, len(pool), count)]


class Run:
    """Per-task accuracy after every task (analysis split; validation in a pilot)."""

    def __init__(self, task: IncrementalPatternTask, split: DatasetSplit) -> None:
        self.task = task
        self.split: DatasetSplit = split
        self.matrix: list[list[float]] = []

    def record(self, predict: Predict, tasks_seen: int) -> None:
        self.matrix.append(evaluate(predict, self.task, self.split, tasks_seen))


def train_gradient(
    name: str, task: IncrementalPatternTask, seed: int, lr: float, lam: float | None, run: Run
) -> tuple[Predict, dict[str, int]]:
    """GRU / modular networks trained by SGD (sequential, replay, EWC or joint)."""

    input_dim = task.input_dim
    model: nn.Module = (
        ModularClassifier(input_dim, CLASSES)
        if name == "mod_sgd"
        else GRUClassifier(input_dim, CLASSES, hidden=GRU_HIDDEN)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    rng = np.random.default_rng(seed)
    pools = [task.indices("train", t) for t in range(TASKS)]
    buffer: list[int] = []
    fisher = [torch.zeros_like(p) for p in model.parameters()]
    anchor = [p.detach().clone() for p in model.parameters()]

    def predict(batch: TaskBatch, seen: int) -> Tensor:
        model.eval()
        out: Tensor = model(batch)
        model.train()
        return out

    def step(indices: list[int], seen: int, penalty: bool) -> None:
        batch = task.generate("train", indices)
        loss = masked_loss(model(batch), batch.targets[:, -1], seen)
        if penalty and lam is not None:
            loss = loss + 0.5 * lam * sum(
                (f * (p - a).square()).sum()
                for f, p, a in zip(fisher, model.parameters(), anchor, strict=True)
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    if name == "gru_joint":
        everything = [i for pool in pools for i in pool]
        for _ in range(UPDATES_PER_TASK * TASKS):
            step(sample(rng, everything, BATCH), CLASSES, False)
        run.record(predict, TASKS)
        return predict, {"learned": parameter_count(model), "fixed": 0}
    for t in range(TASKS):
        seen = (t + 1) * task.classes_per_task
        for _ in range(UPDATES_PER_TASK):
            if name == "gru_replay" and buffer:
                indices = sample(rng, pools[t], BATCH // 2) + sample(rng, buffer, BATCH // 2)
            else:
                indices = sample(rng, pools[t], BATCH)
            step(indices, seen, t > 0)
        run.record(predict, t + 1)
        if name == "gru_replay":
            for c in range(t * task.classes_per_task, seen):
                members = [i for i in pools[t] if task.label(i)[0] == c]
                buffer += [members[i] for i in rng.permutation(len(members))[:REPLAY_PER_CLASS]]
        if name == "gru_ewc":
            for i in sample(rng, pools[t], FISHER_SAMPLES):
                batch = task.generate("train", [i])
                model.zero_grad(set_to_none=True)
                masked_loss(model(batch), batch.targets[:, -1], seen).backward()  # type: ignore[no-untyped-call]
                for f, p in zip(fisher, model.parameters(), strict=True):
                    if p.grad is not None:
                        f += p.grad.detach().square() / FISHER_SAMPLES
            anchor = [p.detach().clone() for p in model.parameters()]
    return predict, {"learned": parameter_count(model), "fixed": 0}


def train_encoder(
    encoder: FrameEncoder,
    head: nn.Module,
    task: IncrementalPatternTask,
    t: int,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer,
) -> None:
    """SGD on task ``t`` through an ``action_selector.v1`` head (classes seen so far)."""

    seen = (t + 1) * task.classes_per_task
    pool = task.indices("train", t)
    params = [p for group in optimizer.param_groups for p in group["params"]]
    for _ in range(UPDATES_PER_TASK):
        batch = task.generate("train", sample(rng, pool, BATCH))
        loss = masked_loss(head(encoder(batch)), batch.targets[:, -1], seen)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()


def train_associative(
    name: str, task: IncrementalPatternTask, seed: int, lr: float, run: Run
) -> tuple[Predict, dict[str, int]]:
    """Fixed random (or pretrained, or plastic) encoder with an associative, SGD or NCM readout.

    ``mod_assoc``, ``abl_sgdread`` and ``abl_ncm`` never train the encoder;
    ``abl_pretrained`` trains it on task 1 and then freezes it;
    ``abl_plastic`` keeps training it on every task through an SGD head.
    """

    rng = np.random.default_rng(seed)
    encoder = FrameEncoder(task.input_dim)
    head = selector_head(encoder.feature_dim, CLASSES)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *head.parameters()], lr=lr, weight_decay=1e-2
    )
    expansion = SparseExpansion(encoder.feature_dim, UNITS, fan_in=FAN_IN, active=ACTIVE, seed=seed)
    associative = AssociativeReadout(UNITS, CLASSES)
    linear = nn.Linear(UNITS, CLASSES)
    linear_optimizer = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=1e-2)
    sums = torch.zeros(CLASSES, encoder.feature_dim)
    counts = torch.zeros(CLASSES)

    def features(batch: TaskBatch) -> Tensor:
        encoder.eval()
        with torch.no_grad():
            out: Tensor = encoder(batch)
        encoder.train()
        return out

    def predict(batch: TaskBatch, seen: int) -> Tensor:
        h = features(batch)
        if name == "abl_ncm":
            means = sums / counts.clamp(min=1)[:, None]
            return -torch.cdist(h, means)
        code = expansion(h)
        return linear(code) if name == "abl_sgdread" else associative.scores(code)

    if name in {"abl_pretrained", "abl_plastic"}:
        train_encoder(encoder, head, task, 0, rng, optimizer)
    for t in range(TASKS):
        seen = (t + 1) * task.classes_per_task
        if name == "abl_plastic" and t > 0:
            train_encoder(encoder, head, task, t, rng, optimizer)
        pool = task.indices("train", t)
        if name == "abl_sgdread":
            for _ in range(UPDATES_PER_TASK):
                batch = task.generate("train", sample(rng, pool, BATCH))
                loss = masked_loss(linear(expansion(features(batch))), batch.targets[:, -1], seen)
                linear_optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                nn.utils.clip_grad_norm_(linear.parameters(), 1.0)
                linear_optimizer.step()
        else:
            for start in range(0, len(pool), 256):
                batch = task.generate("train", pool[start : start + 256])
                h, labels = features(batch), batch.targets[:, -1]
                associative.learn(expansion(h), labels)
                sums.index_add_(0, labels, h)
                counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
        run.record(predict, t + 1)
    readout = {
        "abl_ncm": CLASSES * encoder.feature_dim,
        "abl_sgdread": parameter_count(linear),
    }.get(name, CLASSES * UNITS)
    trained = name in {"abl_pretrained", "abl_plastic"}
    encoder_size = parameter_count(encoder)
    return predict, {
        "learned": readout + (encoder_size + parameter_count(head) if trained else 0),
        "fixed": (0 if trained else encoder_size) + FAN_IN * UNITS,
    }


def forgetting(matrix: list[list[float]]) -> float:
    """Main-line definition: mean over earlier tasks of (best analysis score - final)."""

    final = matrix[-1]
    return float(
        np.mean([max(row[j] for row in matrix[j:]) - final[j] for j in range(len(final) - 1)])
    )


def train(job: tuple[str, int, float, float | None, bool]) -> dict[str, Any]:
    name, seed, lr, lam, pilot = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    started = time.perf_counter()
    task = IncrementalPatternTask(seed, sigma=SIGMA)
    run = Run(task, "validation" if pilot else "analysis")
    if name.startswith(("mod_assoc", "abl_")):
        predict, parameters = train_associative(name, task, seed, lr, run)
    else:
        predict, parameters = train_gradient(name, task, seed, lr, lam, run)
    result: dict[str, Any] = {
        "model": name,
        "seed": seed,
        "lr": lr,
        "lambda": lam,
        "parameters": parameters,
        "validation": evaluate(predict, task, "validation", TASKS),
        "train_seconds": round(time.perf_counter() - started, 1),
    }
    result["validation_final"] = float(np.mean(result["validation"]))
    result[f"{run.split}_matrix"] = run.matrix
    result["forgetting"] = forgetting(run.matrix) if name != "gru_joint" else None
    if not pilot:
        result["test"] = evaluate(predict, task, "test", TASKS)
        result["test_final"] = float(np.mean(result["test"]))
    return result


def references(seed: int, split: DatasetSplit) -> dict[str, float]:
    """Bayes oracle and raw-input nearest neighbour / class mean on the frame mean."""

    task = IncrementalPatternTask(seed, sigma=SIGMA)
    train_batch = task.generate("train", list(range(task.size("train"))))
    batch = task.generate(split, list(range(task.size(split))))
    x_train = train_batch.inputs[:, : task.frames, : task.dims].mean(dim=1)
    y_train = train_batch.targets[:, -1]
    x = batch.inputs[:, : task.frames, : task.dims].mean(dim=1)
    y = batch.targets[:, -1]
    means = torch.stack([x_train[y_train == c].mean(dim=0) for c in range(CLASSES)])
    return {
        "bayes": float(task.bayes_posterior(batch, CLASSES).argmax(dim=-1).eq(y).float().mean()),
        "raw_1nn": float(y_train[torch.cdist(x, x_train).argmin(dim=1)].eq(y).float().mean()),
        "raw_ncm": float(torch.cdist(x, means).argmin(dim=1).eq(y).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="seed 999, train/validation only")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = (PILOT_SEED,) if args.pilot else SEEDS
    jobs: list[tuple[str, int, float, float | None, bool]] = [
        (m, s, lr, lam, args.pilot)
        for s in seeds
        for m in MODELS
        for lr in LEARNING_RATES
        for lam in (EWC_LAMBDAS if m == "gru_ewc" else (None,))
    ]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        runs = pool.map(train, jobs, chunksize=1)
    chosen: dict[int, dict[str, dict[str, Any]]] = {}
    for run in runs:
        current = chosen.setdefault(run["seed"], {}).get(run["model"])
        if current is None or run["validation_final"] > current["validation_final"]:
            chosen[run["seed"]][run["model"]] = run
    report: dict[str, Any] = {
        "protocol": "docs/transfer/T2-associative-readout-protocol.md",
        "pilot": args.pilot,
        "references_validation": {str(s): references(s, "validation") for s in seeds},
        "runs": runs,
    }
    if not args.pilot:
        report["references_test"] = {str(s): references(s, "test") for s in seeds}
        per_seed = {
            seed: {
                **{f"test_final/{m}": float(r["test_final"]) for m, r in by_model.items()},
                **{
                    f"forgetting/{m}": float(r["forgetting"])
                    for m, r in by_model.items()
                    if r["forgetting"] is not None
                },
            }
            for seed, by_model in chosen.items()
        }
        report["per_seed"] = {str(s): v for s, v in per_seed.items()}
        report["summary"] = {
            key: float(np.mean([v[key] for v in per_seed.values()]))
            for key in next(iter(per_seed.values()))
        }
        report["contrasts"] = paired_contrasts(per_seed, SPECIFICATION)  # type: ignore[arg-type]
        report["selected"] = {
            str(s): {m: {"lr": r["lr"], "lambda": r["lambda"]} for m, r in by.items()}
            for s, by in chosen.items()
        }
    for seed, by_model in sorted(chosen.items()):
        print(
            f"seed {seed}: "
            + "  ".join(
                f"{m} val {r['validation_final']:.3f} fgt {r['forgetting']}"
                + (f" test {r['test_final']:.3f}" if "test" in r else "")
                for m, r in by_model.items()
            ),
            flush=True,
        )
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    name = "t2_pilot.json" if args.pilot else "t2_report.json"
    (OUT / name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
