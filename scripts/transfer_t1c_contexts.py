"""T1c: do addressable slots beat one shared GRU state when several rules must be held?

Protocol docs/transfer/T1c-multi-context-protocol.md (CR-014, exploratory).
Three contexts each carry their own rule; trials interleave the contexts and a
single context's rule switches silently.  Models: T1's gated working memory
(``cur_gate``), T1b's read-enriched gated memory (``enr_gate``) and the shared
GRU, with T1's learning-rate grid and selection rule, twice T1's budget and
new seeds.
``--pilot`` runs seed 991 on train/validation only (design check).
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
from neuromorphic.transfer.multi_context import CONTEXTS, FEEDBACK_BIT, MultiContextSwitchTask
from neuromorphic.transfer.statistics import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, paired_contrasts
from neuromorphic.transfer.surprise_wm import T1GRU, T1Network, parameter_count
from neuromorphic.transfer.uncued_switch import FEEDBACK

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/transfer"
CELLS = {"cur_gate": "current", "enr_gate": "enriched"}  # name -> write source
MODELS = (*CELLS, "gru")
SEEDS = tuple(range(701, 711))
PILOT_SEED = 991
LEARNING_RATES = (1e-3, 3e-4)
UPDATES, BATCH, EVAL_EVERY = 10_000, 64, 250
TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE, OOD_SIZE, ANALYSIS_SIZE = 8192, 1024, 2048, 2048, 2048
RECOVERY = (1, 2, 3, 4)  # occurrences of the switched context after its switch
AFTER_OTHER = (1, 2, 3)  # trials after a switch elsewhere (interference window)
LEARNED = 0.70
SPECIFICATION = {
    "H1": ("test_recovery/enr_gate", "test_recovery/gru", 0.02, "superiority"),
    "H2": ("ood_accuracy/enr_gate", "ood_accuracy/gru", 0.02, "superiority"),
    "H3": ("test_recovery/enr_gate", "test_recovery/cur_gate", 0.03, "superiority"),
    "H4": ("test_recovery/enr_gate", "test_recovery/gru", 0.02, "noninferiority"),
    "H5": ("ood_accuracy/enr_gate", "ood_accuracy/gru", 0.02, "noninferiority"),
}


def build(name: str, input_dim: int) -> nn.Module:
    if name == "gru":
        return T1GRU(input_dim)
    return T1Network(
        input_dim,
        mode="gate",
        write_source=CELLS[name],  # type: ignore[arg-type]
        feedback_bit=FEEDBACK_BIT,
    )


def loss_of(model: nn.Module, batch: TaskBatch) -> Tensor:
    logits = model(batch)
    return F.cross_entropy(logits[batch.loss_mask], batch.targets[batch.loss_mask])


def grouped(values: Tensor, gaps: Tensor, other: Tensor) -> dict[str, float]:
    """Overall, pre-switch, switch-trial, recovery and interference-window means.

    ``other`` marks queries of a context that has not switched, made 1-3
    trials after a switch in another context (``nan`` when absent).
    """

    def mean(mask: Tensor) -> float:
        return float(values[mask].mean()) if bool(mask.any()) else float("nan")

    return {
        "accuracy": float(values.mean()),
        "pre_switch": mean(gaps == -1),
        "switch_trial": mean(gaps == 0),
        "recovery": mean(torch.isin(gaps, torch.tensor(RECOVERY))),
        "other_after_switch": mean(other),
    }


def query_groups(batch: TaskBatch) -> tuple[Tensor, Tensor]:
    gaps = batch.auxiliary_targets["trials_since_switch"][batch.loss_mask]
    window = torch.isin(batch.auxiliary_targets["global_since"], torch.tensor(AFTER_OTHER))
    other = (window & batch.auxiliary_targets["trials_since_switch"].eq(-1))[batch.loss_mask]
    return gaps, other


@torch.no_grad()
def measure(
    model: nn.Module, task: MultiContextSwitchTask, split: DatasetSplit, size: int
) -> dict[str, float]:
    model.eval()
    correct: list[Tensor] = []
    gaps: list[Tensor] = []
    others: list[Tensor] = []
    for start in range(0, size, 256):
        batch = task.generate(split, list(range(start, min(start + 256, size))))
        hit = model(batch).argmax(dim=-1).eq(batch.targets)
        correct.append(hit[batch.loss_mask])
        gap, other = query_groups(batch)
        gaps.append(gap)
        others.append(other)
    model.train()
    return grouped(torch.cat(correct).float(), torch.cat(gaps), torch.cat(others))


@torch.no_grad()
def feedback_states(
    model: nn.Module, task: MultiContextSwitchTask, split: DatasetSplit, size: int
) -> dict[str, Tensor]:
    """State after every feedback step with every context's rule, gap and observer level."""

    model.eval()
    parts: dict[str, list[Tensor]] = {
        key: [] for key in ("state", "episode", "addressed", "rules", "since", "oracle")
    }
    for start in range(0, size, 256):
        indices = list(range(start, min(start + 256, size)))
        batch = task.generate(split, indices)
        _, states = model.run(batch)  # type: ignore[operator]
        feedback = batch.inputs[..., :4].argmax(dim=-1).eq(FEEDBACK) & batch.valid_mask
        rules = batch.auxiliary_targets["rules"]
        sets = task.consistent_rules(batch)
        chosen = sets.gather(-1, rules.clamp(min=0)[..., None]).squeeze(-1).float()
        context = batch.auxiliary_targets["context"]
        addressed = context[..., None].eq(torch.arange(CONTEXTS))
        parts["state"].append(states[feedback])
        parts["episode"].append(torch.tensor(indices)[:, None].expand_as(feedback)[feedback])
        parts["addressed"].append(addressed[feedback])
        parts["rules"].append(rules[feedback])
        parts["since"].append(batch.auxiliary_targets["context_since"][feedback])
        parts["oracle"].append((chosen / sets.sum(dim=-1).clamp(min=1))[feedback])
    model.train()
    return {key: torch.cat(value) for key, value in parts.items()}


def rule_probe(
    model: nn.Module, task: MultiContextSwitchTask, split: DatasetSplit, size: int
) -> dict[str, dict[str, float]]:
    """Linear decodability of every context's current rule after each feedback step.

    One multinomial logistic probe per context (standardized features, L2
    1e-3, L-BFGS) is fit on the first half of the episodes and scored on the
    second.  Rows are grouped by whether the feedback was about that context
    (``addressed``) or another one (``held``), and by whether that context is
    before its switch (``pre``) or has seen 1-4 of its feedbacks since the
    switch (``post``: the states its recovery queries rely on).  ``oracle``
    is a uniform pick from the observer's set for that context.
    """

    rows = feedback_states(model, task, split, size)
    first = rows["episode"] < size // 2
    hits: list[Tensor] = []
    for k in range(CONTEXTS):
        fit_x, fit_y = rows["state"][first], rows["rules"][first, k]
        mean, scale = fit_x.mean(dim=0), fit_x.std(dim=0) + 1e-6
        torch.manual_seed(k)
        probe = nn.Linear(fit_x.shape[1], 4)
        optimizer = torch.optim.LBFGS(
            probe.parameters(), max_iter=200, line_search_fn="strong_wolfe"
        )
        features = (fit_x - mean) / scale

        def closure(
            probe: nn.Linear = probe,
            optimizer: torch.optim.LBFGS = optimizer,
            features: Tensor = features,
            target: Tensor = fit_y,
        ) -> Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(probe(features), target) + 1e-3 * probe.weight.square().sum()
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        with torch.enable_grad():
            optimizer.step(closure)  # type: ignore[no-untyped-call]
        with torch.no_grad():
            predicted = probe((rows["state"][~first] - mean) / scale).argmax(dim=-1)
        hits.append(predicted.eq(rows["rules"][~first, k]).float())
    hit = torch.stack(hits, dim=1)  # [N, C]
    addressed = rows["addressed"][~first]
    since = rows["since"][~first]
    oracle = rows["oracle"][~first]
    post = torch.isin(since, torch.tensor((0, 1, 2, 3)))
    groups = {
        "addressed_pre": addressed & since.eq(-1),
        "addressed_post": addressed & post,
        "held_pre": ~addressed & since.eq(-1),
        "held_post": ~addressed & post,
    }
    return {
        "probe": {name: float(hit[mask].mean()) for name, mask in groups.items()},
        "oracle": {name: float(oracle[mask].mean()) for name, mask in groups.items()},
    }


def train(job: tuple[str, int, float, bool]) -> dict[str, Any]:
    name, seed, lr, pilot = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    task = MultiContextSwitchTask()
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


def oracle_levels(task: MultiContextSwitchTask, split: DatasetSplit, size: int) -> dict[str, float]:
    batch = task.generate(split, list(range(size)))
    oracle = task.oracle(batch)[batch.loss_mask]
    target = batch.targets[batch.loss_mask].float()
    expected = oracle * target + (1 - oracle) * (1 - target)
    return grouped(expected, *query_groups(batch))


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
    for model in MODELS:
        out[f"interference/{model}"] = describe(
            {
                s: chosen[s][model]["test"]["other_after_switch"]
                - chosen[s][model]["test"]["pre_switch"]
                for s in seeds
            }
        )
    for group in ("held_post", "addressed_post"):
        for a, b in (("enr_gate", "gru"), ("enr_gate", "cur_gate")):
            out[f"probe_{group}/{a}-{b}"] = describe(
                {
                    s: per_seed[s][f"probe_{group}/{a}"] - per_seed[s][f"probe_{group}/{b}"]
                    for s in seeds
                }
            )
    out["learned"] = {
        m: sum(per_seed[s][f"test_recovery/{m}"] >= LEARNED for s in seeds) for m in MODELS
    }
    out["lr_1e-3"] = {m: sum(chosen[s][m]["lr"] == 1e-3 for s in seeds) for m in MODELS}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="seed 991, train/validation only")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = (PILOT_SEED,) if args.pilot else SEEDS
    jobs = [(m, s, lr, args.pilot) for s in seeds for m in MODELS for lr in LEARNING_RATES]
    jobs.sort(key=lambda job: job[0] == "gru")  # long jobs first
    with mp.get_context("spawn").Pool(args.workers) as pool:
        runs = pool.map(train, jobs, chunksize=1)
    chosen: dict[int, dict[str, dict[str, Any]]] = {}
    for run in runs:
        current = chosen.setdefault(run["seed"], {}).get(run["model"])
        if current is None or run["validation"]["accuracy"] > current["validation"]["accuracy"]:
            chosen[run["seed"]][run["model"]] = run
    task = MultiContextSwitchTask()
    report: dict[str, Any] = {
        "protocol": "docs/transfer/T1c-multi-context-protocol.md",
        "pilot": args.pilot,
        "oracle_validation": oracle_levels(task, "validation", VALIDATION_SIZE),
        "runs": runs,
    }
    if not args.pilot:
        report["oracle_test"] = oracle_levels(task, "test", TEST_SIZE)
        report["oracle_ood"] = oracle_levels(task, "ood", OOD_SIZE)
        per_seed = {
            seed: {
                f"{split}_{metric}/{model}": float(run[split][metric])
                for model, run in by_model.items()
                for split in ("test", "ood")
                for metric in ("accuracy", "recovery", "pre_switch")
            }
            | {
                f"probe_{group}/{model}": float(run["probe_analysis"]["probe"][group])
                for model, run in by_model.items()
                for group in ("addressed_pre", "addressed_post", "held_pre", "held_post")
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
    name = "t1c_pilot.json" if args.pilot else "t1c_report.json"
    (OUT / name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
