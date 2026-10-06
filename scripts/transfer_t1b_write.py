"""T1b: is the working-memory write structure why T1's memory trails the GRU?

Protocol docs/transfer/T1b-write-structure-protocol.md (CR-014, exploratory).
A 2x2 design crosses the write source (current vs read-enriched
representation) with the outcome-surprise gate (off vs on), plus the shared
GRU.  Task, splits, budget, learning-rate grid and selection rule are T1's;
model seeds are new.  ``--pilot`` runs seed 996 on train/validation only
(design check; no test, OOD or analysis access).
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

from neuromorphic.evaluation.statistics import percentile_bootstrap_mean
from neuromorphic.tasks.base import DatasetSplit, TaskBatch
from neuromorphic.transfer.statistics import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, paired_contrasts
from neuromorphic.transfer.surprise_wm import T1GRU, T1Network, parameter_count
from neuromorphic.transfer.uncued_switch import FEEDBACK, UncuedRuleSwitchTask

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/transfer"
CELLS = {  # name -> (write source, surprise mode)
    "cur_plain": ("current", "plain"),
    "cur_gate": ("current", "gate"),
    "enr_plain": ("enriched", "plain"),
    "enr_gate": ("enriched", "gate"),
}
MODELS = (*CELLS, "gru")
SEEDS = tuple(range(601, 611))
PILOT_SEED = 996
LEARNING_RATES = (1e-3, 3e-4)
UPDATES, BATCH, EVAL_EVERY = 5000, 64, 250
TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE, OOD_SIZE, ANALYSIS_SIZE = 8192, 1024, 2048, 2048, 2048
RECOVERY = (1, 2, 3, 4)  # queries this many trials after a switch
LEARNED = 0.70  # descriptive: test recovery at or above this counts as "learned"
SPECIFICATION = {
    "H1": ("test_recovery/enr_gate", "test_recovery/cur_gate", 0.03, "superiority"),
    "H2": ("test_recovery/enr_plain", "test_recovery/cur_plain", 0.05, "superiority"),
    "H3": ("test_recovery/enr_gate", "test_recovery/enr_plain", 0.03, "superiority"),
    "H4": ("ood_accuracy/enr_gate", "ood_accuracy/gru", 0.02, "superiority"),
    "H5": ("test_recovery/enr_gate", "test_recovery/gru", 0.02, "noninferiority"),
    "H6": ("ood_accuracy/enr_gate", "ood_accuracy/gru", 0.02, "noninferiority"),
}


def build(name: str, input_dim: int) -> nn.Module:
    if name == "gru":
        return T1GRU(input_dim)
    source, mode = CELLS[name]
    return T1Network(input_dim, mode=mode, write_source=source)  # type: ignore[arg-type]


def loss_of(model: nn.Module, batch: TaskBatch) -> Tensor:
    logits = model(batch)
    return F.cross_entropy(logits[batch.loss_mask], batch.targets[batch.loss_mask])


def grouped(values: Tensor, gaps: Tensor) -> dict[str, float]:
    recovery = torch.isin(gaps, torch.tensor(RECOVERY))
    return {
        "accuracy": float(values.mean()),
        "pre_switch": float(values[gaps == -1].mean()),
        "switch_trial": float(values[gaps == 0].mean()),
        "recovery": float(values[recovery].mean()),
    }


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
    return grouped(torch.cat(correct).float(), torch.cat(since))


@torch.no_grad()
def feedback_states(
    model: nn.Module, task: UncuedRuleSwitchTask, split: DatasetSplit, size: int
) -> dict[str, Tensor]:
    """Memory state right after every feedback step, with its episode, rule and gap."""

    model.eval()
    parts: dict[str, list[Tensor]] = {
        "state": [],
        "episode": [],
        "rule": [],
        "gap": [],
        "oracle": [],
    }
    for start in range(0, size, 256):
        indices = list(range(start, min(start + 256, size)))
        batch = task.generate(split, indices)
        _, states = model.run(batch)  # type: ignore[operator]
        events = batch.inputs[..., :4].argmax(dim=-1)
        feedback = events.eq(FEEDBACK) & batch.valid_mask
        rule = batch.auxiliary_targets["rule"]
        gap = batch.auxiliary_targets["trials_since_switch"].roll(1, dims=1)  # preceding query
        sets = task.consistent_rules(batch)
        chosen = sets.gather(-1, rule.clamp(min=0)[..., None]).squeeze(-1).float()
        parts["state"].append(states[feedback])
        parts["episode"].append(torch.tensor(indices)[:, None].expand_as(feedback)[feedback])
        parts["rule"].append(rule[feedback])
        parts["gap"].append(gap[feedback])
        parts["oracle"].append((chosen / sets.sum(dim=-1).clamp(min=1))[feedback])
    model.train()
    return {key: torch.cat(value) for key, value in parts.items()}


def rule_probe(model: nn.Module, task: UncuedRuleSwitchTask, split: DatasetSplit, size: int) -> Any:
    """Linear decodability of the current rule from the memory state after feedback.

    A multinomial logistic probe (standardized features, L2 1e-3, L-BFGS) is
    fit on the first half of the episodes and scored on the second half.
    ``oracle`` is the chance that a uniform pick from the ideal observer's
    consistent rule set is the true rule (a reference level, not a strict bound).
    """

    rows = feedback_states(model, task, split, size)
    first = rows["episode"] < size // 2
    fit = {key: value[first] for key, value in rows.items()}
    held = {key: value[~first] for key, value in rows.items()}
    mean, scale = fit["state"].mean(dim=0), fit["state"].std(dim=0) + 1e-6
    torch.manual_seed(0)
    probe = nn.Linear(fit["state"].shape[1], 4)
    optimizer = torch.optim.LBFGS(probe.parameters(), max_iter=200, line_search_fn="strong_wolfe")
    features = (fit["state"] - mean) / scale

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(probe(features), fit["rule"]) + 1e-3 * probe.weight.square().sum()
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    with torch.enable_grad():
        optimizer.step(closure)  # type: ignore[no-untyped-call]
    with torch.no_grad():
        hits = probe((held["state"] - mean) / scale).argmax(dim=-1).eq(held["rule"]).float()
    return {"probe": grouped(hits, held["gap"]), "oracle": grouped(held["oracle"], held["gap"])}


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
    if pilot:
        result["probe_validation"] = rule_probe(model, task, "validation", VALIDATION_SIZE)
    else:
        result["test"] = measure(model, task, "test", TEST_SIZE)
        result["ood"] = measure(model, task, "ood", OOD_SIZE)
        result["probe_analysis"] = rule_probe(model, task, "analysis", ANALYSIS_SIZE)
    return result


def oracle_levels(task: UncuedRuleSwitchTask, split: DatasetSplit, size: int) -> dict[str, float]:
    batch = task.generate(split, list(range(size)))
    oracle = task.oracle(batch)[batch.loss_mask]
    target = batch.targets[batch.loss_mask].float()
    expected = oracle * target + (1 - oracle) * (1 - target)
    return grouped(expected, batch.auxiliary_targets["trials_since_switch"][batch.loss_mask])


def describe(values: dict[int, float]) -> dict[str, Any]:
    """Mean and 95% percentile-bootstrap CI of a per-seed quantity (no verdict)."""

    seeds = sorted(values)
    interval = percentile_bootstrap_mean(
        [values[s] for s in seeds], samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    return {
        "per_seed": [round(values[s], 4) for s in seeds],
        "mean": interval.estimate,
        "ci95": [interval.lower, interval.upper],
    }


def descriptives(
    chosen: dict[int, dict[str, dict[str, Any]]], per_seed: dict[int, dict[str, float]]
) -> dict[str, Any]:
    seeds = sorted(per_seed)
    out: dict[str, Any] = {}
    for metric in ("test_recovery", "ood_accuracy"):
        effect = {
            s: (per_seed[s][f"{metric}/enr_gate"] - per_seed[s][f"{metric}/cur_gate"])
            - (per_seed[s][f"{metric}/enr_plain"] - per_seed[s][f"{metric}/cur_plain"])
            for s in seeds
        }
        out[f"interaction/{metric}"] = describe(effect)
        means = {m: float(np.mean([per_seed[s][f"{metric}/{m}"] for s in seeds])) for m in MODELS}
        gap = means["gru"] - means["cur_gate"]
        out[f"gap_closed/{metric}"] = (
            (means["enr_gate"] - means["cur_gate"]) / gap if abs(gap) > 1e-9 else None
        )
    for a, b in (("enr_gate", "cur_gate"), ("enr_plain", "cur_plain"), ("enr_gate", "gru")):
        delta = {
            s: per_seed[s][f"probe_recovery/{a}"] - per_seed[s][f"probe_recovery/{b}"]
            for s in seeds
        }
        out[f"probe_recovery/{a}-{b}"] = describe(delta)
    out["learned"] = {
        m: sum(per_seed[s][f"test_recovery/{m}"] >= LEARNED for s in seeds) for m in MODELS
    }
    out["lr_1e-3"] = {m: sum(chosen[s][m]["lr"] == 1e-3 for s in seeds) for m in MODELS}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="seed 996, train/validation only")
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
        "protocol": "docs/transfer/T1b-write-structure-protocol.md",
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
                    ("test", "pre_switch", "pre_switch"),
                    ("ood", "accuracy", "accuracy"),
                    ("ood", "recovery", "recovery"),
                )
            }
            | {
                f"probe_{group}/{model}": float(run["probe_analysis"]["probe"][group])
                for model, run in by_model.items()
                for group in ("pre_switch", "recovery")
            }
            for seed, by_model in chosen.items()
        }
        report["per_seed"] = {str(s): v for s, v in per_seed.items()}
        report["summary"] = {
            key: float(np.mean([v[key] for v in per_seed.values()]))
            for key in next(iter(per_seed.values()))
        }
        report["contrasts"] = paired_contrasts(per_seed, SPECIFICATION)  # type: ignore[arg-type]
        report["descriptive"] = descriptives(chosen, per_seed)
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
    name = "t1b_pilot.json" if args.pilot else "t1b_report.json"
    (OUT / name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
