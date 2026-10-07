"""T1d: does a content-addressed surprise gate let slot memory update the right context?

Protocol docs/transfer/T1d-addressed-gate-protocol.md (CR-014, exploratory).
Part ``multi`` reruns T1c's multi-context task; part ``single`` reruns T1b's
single-context task as a safety check.  Models: T1b/T1c's read-enriched
memory with the global surprise gate (``enr_gate``), the same memory with the
surprise bias weighted by the step's read attention (``enr_addr``) and the
shared GRU.  Evaluation functions are imported unchanged from the T1c and T1b
scripts; budgets, grids and selection rules are theirs; seeds are new.
``--pilot`` runs seed 996 on train/validation only (design check).
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
import transfer_t1b_write as t1b
import transfer_t1c_contexts as t1c
from torch import Tensor, nn
from torch.nn import functional as F

from neuromorphic.tasks.base import DatasetSplit, TaskBatch
from neuromorphic.transfer.multi_context import FEEDBACK_BIT, MultiContextSwitchTask
from neuromorphic.transfer.statistics import paired_contrasts
from neuromorphic.transfer.surprise_wm import T1GRU, T1Network, parameter_count
from neuromorphic.transfer.uncued_switch import FEEDBACK, UncuedRuleSwitchTask

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/transfer"
CELLS = {"enr_gate": "global", "enr_addr": "read"}  # name -> gate address
MODELS = (*CELLS, "gru")
PARTS = ("multi", "single")
SEEDS = tuple(range(901, 911))
PILOT_SEED = 996
LEARNING_RATES = (1e-3, 3e-4)
UPDATES = {"multi": t1c.UPDATES, "single": t1b.UPDATES}  # 10,000 and 5,000
BATCH, EVAL_EVERY = 64, 250
LEARNED = 0.70
SPECIFICATION = {
    "H1": ("multi_test_recovery/enr_addr", "multi_test_recovery/enr_gate", 0.03, "superiority"),
    "H2": ("multi_test_recovery/enr_addr", "multi_test_recovery/gru", 0.02, "superiority"),
    "H3": ("multi_test_recovery/enr_addr", "multi_test_recovery/gru", 0.02, "noninferiority"),
    "H4": ("multi_ood_accuracy/enr_addr", "multi_ood_accuracy/gru", 0.02, "noninferiority"),
    "H5": (
        "single_test_recovery/enr_addr",
        "single_test_recovery/enr_gate",
        0.02,
        "noninferiority",
    ),
    "H6": ("single_ood_accuracy/enr_addr", "single_ood_accuracy/gru", 0.02, "noninferiority"),
}
Task = MultiContextSwitchTask | UncuedRuleSwitchTask


def make_task(part: str) -> Task:
    return MultiContextSwitchTask() if part == "multi" else UncuedRuleSwitchTask()


def build(part: str, name: str, input_dim: int) -> nn.Module:
    if name == "gru":
        return T1GRU(input_dim)
    return T1Network(
        input_dim,
        mode="gate",
        write_source="enriched",
        feedback_bit=FEEDBACK_BIT if part == "multi" else 10,
        gate_address=CELLS[name],  # type: ignore[arg-type]
    )


def measure(part: str, model: nn.Module, task: Task, split: DatasetSplit, size: int) -> Any:
    if part == "multi":
        assert isinstance(task, MultiContextSwitchTask)
        return t1c.measure(model, task, split, size)
    assert isinstance(task, UncuedRuleSwitchTask)
    return t1b.measure(model, task, split, size)


def probe(part: str, model: nn.Module, task: Task, split: DatasetSplit, size: int) -> Any:
    if part == "multi":
        assert isinstance(task, MultiContextSwitchTask)
        return t1c.rule_probe(model, task, split, size)
    assert isinstance(task, UncuedRuleSwitchTask)
    return t1b.rule_probe(model, task, split, size)


@torch.no_grad()
def addressing(model: nn.Module, task: MultiContextSwitchTask, split: DatasetSplit) -> Any:
    """Read attention at feedback steps: peak weight and context -> slot consistency.

    Consistency is the share of feedback steps whose most-attended slot is the
    slot most often attended for that context within the same episode
    (1.0 = every context keeps its own slot).  It is only informative together
    with the peak weight: near-uniform attention can still have a stable argmax.
    """

    if not isinstance(model, T1Network):
        return None
    model.eval()
    model.trace_attention = True
    peaks: list[Tensor] = []
    agree: list[Tensor] = []
    for start in range(0, 1024, 256):
        batch = task.generate(split, list(range(start, start + 256)))
        model.run(batch)
        attention = torch.stack(model.attention_trace, dim=1)  # [B, T, S]
        feedback = batch.inputs[..., :4].argmax(dim=-1).eq(FEEDBACK) & batch.valid_mask
        context = batch.auxiliary_targets["context"]
        top = attention.argmax(dim=-1)
        peaks.append(attention.max(dim=-1).values[feedback])
        for b in range(batch.batch_size):
            for k in range(task.contexts):
                rows = feedback[b] & context[b].eq(k)
                if int(rows.sum()) > 1:
                    slots = top[b][rows]
                    agree.append(slots.eq(slots.mode().values).float())
    model.trace_attention = False
    model.train()
    return {
        "peak_attention": float(torch.cat(peaks).mean()),
        "slot_consistency": float(torch.cat(agree).mean()),
    }


def train(job: tuple[str, str, int, float, bool]) -> dict[str, Any]:
    part, name, seed, lr, pilot = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    task = make_task(part)
    sizes = {"validation": 1024, "test": 2048, "ood": 2048, "analysis": 2048}
    model = build(part, name, task.input_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    rng = np.random.default_rng(seed)
    best_accuracy, best_state, best_update, curve = -1.0, None, 0, []
    started = time.perf_counter()
    for update in range(1, UPDATES[part] + 1):
        batch: TaskBatch = task.generate("train", rng.integers(0, 8192, BATCH).tolist())
        logits = model(batch)
        loss = F.cross_entropy(logits[batch.loss_mask], batch.targets[batch.loss_mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % EVAL_EVERY == 0:
            validation = measure(part, model, task, "validation", sizes["validation"])
            curve.append({"update": update, **validation})
            if validation["accuracy"] > best_accuracy:
                best_accuracy = validation["accuracy"]
                best_state = copy.deepcopy(model.state_dict())
                best_update = update
    assert best_state is not None
    model.load_state_dict(best_state)
    result: dict[str, Any] = {
        "part": part,
        "model": name,
        "seed": seed,
        "lr": lr,
        "parameters": parameter_count(model),
        "best_update": best_update,
        "validation": measure(part, model, task, "validation", sizes["validation"]),
        "curve": curve,
        "train_seconds": round(time.perf_counter() - started, 1),
    }
    split: DatasetSplit = "validation" if pilot else "analysis"
    result[f"probe_{split}"] = probe(part, model, task, split, sizes[split])
    if part == "multi":
        assert isinstance(task, MultiContextSwitchTask)
        result[f"addressing_{split}"] = addressing(model, task, split)
    if not pilot:
        result["test"] = measure(part, model, task, "test", sizes["test"])
        result["ood"] = measure(part, model, task, "ood", sizes["ood"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="seed 996, train/validation only")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = (PILOT_SEED,) if args.pilot else SEEDS
    jobs = [
        (part, m, s, lr, args.pilot)
        for part in PARTS
        for s in seeds
        for m in MODELS
        for lr in LEARNING_RATES
    ]
    # Longest jobs first: multi-context memory, single-context memory, then the GRUs.
    order = {("multi", True): 0, ("single", True): 1, ("multi", False): 2, ("single", False): 3}
    jobs.sort(key=lambda job: order[(job[0], job[1] != "gru")])
    with mp.get_context("spawn").Pool(args.workers) as pool:
        runs = pool.map(train, jobs, chunksize=1)
    chosen: dict[str, dict[int, dict[str, dict[str, Any]]]] = {p: {} for p in PARTS}
    for run in runs:
        by_seed = chosen[run["part"]].setdefault(run["seed"], {})
        current = by_seed.get(run["model"])
        if current is None or run["validation"]["accuracy"] > current["validation"]["accuracy"]:
            by_seed[run["model"]] = run
    report: dict[str, Any] = {
        "protocol": "docs/transfer/T1d-addressed-gate-protocol.md",
        "pilot": args.pilot,
        "runs": runs,
    }
    if not args.pilot:
        per_seed: dict[int, dict[str, float]] = {}
        for part in PARTS:
            for seed, by_model in chosen[part].items():
                row = per_seed.setdefault(seed, {})
                for model, run in by_model.items():
                    for split in ("test", "ood"):
                        for metric in ("accuracy", "recovery"):
                            row[f"{part}_{split}_{metric}/{model}"] = float(run[split][metric])
        report["per_seed"] = {str(s): v for s, v in per_seed.items()}
        report["summary"] = {
            key: float(np.mean([v[key] for v in per_seed.values()]))
            for key in next(iter(per_seed.values()))
        }
        report["contrasts"] = paired_contrasts(per_seed, SPECIFICATION)  # type: ignore[arg-type]
        multi = chosen["multi"]
        report["descriptive"] = {
            **{
                f"interference/{m}": t1c.describe(
                    {
                        s: multi[s][m]["test"]["other_after_switch"]
                        - multi[s][m]["test"]["pre_switch"]
                        for s in multi
                    }
                )
                for m in MODELS
            },
            **{
                f"probe_{group}/{a}-{b}": t1c.describe(
                    {
                        s: multi[s][a]["probe_analysis"]["probe"][group]
                        - multi[s][b]["probe_analysis"]["probe"][group]
                        for s in multi
                    }
                )
                for group in ("held_post", "addressed_post")
                for a, b in (("enr_addr", "gru"), ("enr_addr", "enr_gate"))
            },
            "learned": {
                part: {
                    m: sum(chosen[part][s][m]["test"]["recovery"] >= LEARNED for s in chosen[part])
                    for m in MODELS
                }
                for part in PARTS
            },
        }
        report["oracle_test"] = {
            "multi": t1c.oracle_levels(MultiContextSwitchTask(), "test", 2048),
            "single": t1b.oracle_levels(UncuedRuleSwitchTask(), "test", 2048),
        }
    for part in PARTS:
        for seed, by_model in sorted(chosen[part].items()):
            print(
                f"{part} seed {seed}: "
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
    name = "t1d_pilot.json" if args.pilot else "t1d_report.json"
    (OUT / name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
