"""T1: does surprise-gated working memory help with uncued rule switches?

Protocol docs/transfer/T1-surprise-gate-protocol.md (CR-014, exploratory).
Five models share one budget and one learning-rate grid; each (model, seed)
keeps the learning rate with the best validation accuracy.  ``--pilot`` runs
seed 999 on train/validation only (design check; no test or OOD access).
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neuromorphic.tasks.base import DatasetSplit, TaskBatch
from neuromorphic.transfer.statistics import paired_contrasts
from neuromorphic.transfer.surprise_wm import MODES, T1GRU, T1Network, parameter_count
from neuromorphic.transfer.uncued_switch import UncuedRuleSwitchTask

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/transfer"
MODELS = (*MODES, "gru")
SEEDS = tuple(range(501, 511))
PILOT_SEED = 999
LEARNING_RATES = (1e-3, 3e-4)
UPDATES, BATCH, EVAL_EVERY = 5000, 64, 250
TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE, OOD_SIZE = 8192, 1024, 2048, 2048
RECOVERY = (1, 2, 3, 4)  # queries this many trials after a switch
SPECIFICATION = {
    "H1": ("test_recovery/gate", "test_recovery/plain", 0.05, "superiority"),
    "H2": ("test_recovery/gate", "test_recovery/input", 0.03, "superiority"),
    "H3": ("test_recovery/gate", "test_recovery/shuffled", 0.05, "superiority"),
    "H4": ("ood_accuracy/gate", "ood_accuracy/gru", 0.03, "superiority"),
}


def build(name: str, input_dim: int) -> nn.Module:
    return T1GRU(input_dim) if name == "gru" else T1Network(input_dim, mode=name)  # type: ignore[arg-type]


def loss_of(model: nn.Module, batch: TaskBatch) -> Tensor:
    logits = model(batch)
    return F.cross_entropy(logits[batch.loss_mask], batch.targets[batch.loss_mask])


@torch.no_grad()
def measure(
    model: nn.Module, task: UncuedRuleSwitchTask, split: DatasetSplit, size: int
) -> dict[str, float]:
    """Accuracy overall, before any switch, on the switch trial and during recovery."""

    model.eval()
    correct: list[Tensor] = []
    since: list[Tensor] = []
    for start in range(0, size, 256):
        batch = task.generate(split, list(range(start, min(start + 256, size))))
        hit = model(batch).argmax(dim=-1).eq(batch.targets)
        correct.append(hit[batch.loss_mask])
        since.append(batch.auxiliary_targets["trials_since_switch"][batch.loss_mask])
    model.train()
    hits, gaps = torch.cat(correct).float(), torch.cat(since)
    recovery = torch.isin(gaps, torch.tensor(RECOVERY))
    return {
        "accuracy": float(hits.mean()),
        "pre_switch": float(hits[gaps == -1].mean()),
        "switch_trial": float(hits[gaps == 0].mean()),
        "recovery": float(hits[recovery].mean()),
    }


def train(job: tuple[str, int, float, bool]) -> dict[str, Any]:
    name, seed, lr, pilot = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    task = UncuedRuleSwitchTask()
    model = build(name, task.input_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    rng = np.random.default_rng(seed)
    best_accuracy, best_state, best_update, curve = -1.0, None, 0, []
    started = time.perf_counter()
    for update in range(1, UPDATES + 1):
        batch = task.generate("train", rng.integers(0, TRAIN_SIZE, BATCH).tolist())
        loss = loss_of(model, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % EVAL_EVERY == 0:
            validation = measure(model, task, "validation", VALIDATION_SIZE)
            curve.append({"update": update, **validation})
            if validation["accuracy"] > best_accuracy:
                best_accuracy = validation["accuracy"]
                best_state = copy.deepcopy(model.state_dict())
                best_update = update
    assert best_state is not None
    model.load_state_dict(best_state)
    result: dict[str, Any] = {
        "model": name,
        "seed": seed,
        "lr": lr,
        "parameters": parameter_count(model),
        "best_update": best_update,
        "validation": measure(model, task, "validation", VALIDATION_SIZE),
        "curve": curve,
        "train_seconds": round(time.perf_counter() - started, 1),
    }
    if not pilot:
        result["test"] = measure(model, task, "test", TEST_SIZE)
        result["ood"] = measure(model, task, "ood", OOD_SIZE)
    return result


def oracle_levels(task: UncuedRuleSwitchTask, split: DatasetSplit, size: int) -> dict[str, float]:
    batch = task.generate(split, list(range(size)))
    oracle = task.oracle(batch)[batch.loss_mask]
    target = batch.targets[batch.loss_mask].float()
    expected = oracle * target + (1 - oracle) * (1 - target)
    gaps = batch.auxiliary_targets["trials_since_switch"][batch.loss_mask]
    return {
        "accuracy": float(expected.mean()),
        "pre_switch": float(expected[gaps == -1].mean()),
        "switch_trial": float(expected[gaps == 0].mean()),
        "recovery": float(expected[torch.isin(gaps, torch.tensor(RECOVERY))].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="seed 999, train/validation only")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = (PILOT_SEED,) if args.pilot else SEEDS
    jobs = [(m, s, lr, args.pilot) for s in seeds for m in MODELS for lr in LEARNING_RATES]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        runs = pool.map(train, jobs)
    chosen: dict[int, dict[str, dict[str, Any]]] = {}
    for run in runs:
        current = chosen.setdefault(run["seed"], {}).get(run["model"])
        if current is None or run["validation"]["accuracy"] > current["validation"]["accuracy"]:
            chosen[run["seed"]][run["model"]] = run
    task = UncuedRuleSwitchTask()
    report: dict[str, Any] = {
        "protocol": "docs/transfer/T1-surprise-gate-protocol.md",
        "pilot": args.pilot,
        "oracle_validation": oracle_levels(task, "validation", VALIDATION_SIZE),
        "runs": runs,
    }
    if not args.pilot:
        report["oracle_test"] = oracle_levels(task, "test", TEST_SIZE)
        report["oracle_ood"] = oracle_levels(task, "ood", OOD_SIZE)
        per_seed = {
            seed: {
                f"{split}_{metric}/{model}": float(run[split][key])
                for model, run in by_model.items()
                for split, metric, key in (
                    ("test", "recovery", "recovery"),
                    ("test", "accuracy", "accuracy"),
                    ("ood", "accuracy", "accuracy"),
                    ("ood", "recovery", "recovery"),
                )
            }
            for seed, by_model in chosen.items()
        }
        report["per_seed"] = {str(s): v for s, v in per_seed.items()}
        report["summary"] = {
            key: float(np.mean([v[key] for v in per_seed.values()]))
            for key in next(iter(per_seed.values()))
        }
        report["contrasts"] = paired_contrasts(per_seed, SPECIFICATION)  # type: ignore[arg-type]
    for seed, by_model in sorted(chosen.items()):
        print(
            f"seed {seed}: "
            + "  ".join(
                f"{m} lr={r['lr']:g} val {r['validation']['accuracy']:.3f}/rec "
                f"{r['validation']['recovery']:.3f}"
                + (
                    f" test-rec {r['test']['recovery']:.3f} ood {r['ood']['accuracy']:.3f}"
                    if "test" in r
                    else ""
                )
                for m, r in by_model.items()
            ),
            flush=True,
        )
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    name = "t1_pilot.json" if args.pilot else "t1_report.json"
    (OUT / name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
