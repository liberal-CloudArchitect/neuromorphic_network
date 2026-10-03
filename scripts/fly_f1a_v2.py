"""F1a-v2: does KC homeostatic input normalization rescue the sparse code?

Protocol frozen in docs/fly/F1a-v2-mb-protocol.md (exploratory, CR-013).
Seeds 10-19; sigma fixed at the F1a calibration value 1.4.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import numpy as np
import torch
from fly_f1a_mb import (
    CLASSES_PER_TASK,
    N_CLASSES,
    OUT,
    Contrast,
    circuit,
    contrasts,
    task_for,
)

from neuromorphic.fly.continual import (
    AssociativeReadout,
    Learner,
    NearestClassMean,
    kc_code,
    run_class_incremental,
)
from neuromorphic.fly.mushroom_body import PNKCCircuit, degree_preserving, erdos_renyi

SEEDS = tuple(range(10, 20))
SIGMA = 1.4
TASKS = {"main": 0.0, "similar": 0.6}
NORMALIZATIONS = ("none", "kc_norm", "center", "kc_norm+center")
PRIMARY = "mb_assoc|real|0.05|kc_norm"
SPECIFICATION: dict[str, Contrast] = {
    "D1": (f"main/{PRIMARY}", "main/mb_assoc|real|0.05|none", 0.10, "superiority"),
    "D2": (f"main/{PRIMARY}", "main/pn_ncm", 0.02, "superiority"),
    "D3": (f"main/{PRIMARY}", "main/mb_assoc|real|0.50|kc_norm", 0.02, "superiority"),
    "D4": (
        f"main/{PRIMARY}",
        "main/mb_assoc|degree_preserving|0.05|kc_norm",
        0.01,
        "equivalence",
    ),
    "D5": (f"similar/{PRIMARY}", "similar/pn_ncm", 0.02, "superiority"),
}


def code(
    x: torch.Tensor, circuit_: PNKCCircuit, normalization: str, sparsity: float, *, binary: bool
) -> torch.Tensor:
    weights = torch.as_tensor(circuit_.dense(binary=binary), dtype=torch.float32)
    if "kc_norm" in normalization:
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
    if "center" in normalization:
        x = x - x.mean(dim=1, keepdim=True)
    return kc_code(x, weights, sparsity=sparsity)


def run_seed(real: PNKCCircuit, seed: int) -> dict[str, dict[str, object]]:
    rng = np.random.default_rng(10_000 + seed)
    topologies = {
        "real": real,
        "degree_preserving": degree_preserving(real, rng),
        "er": erdos_renyi(real, rng),
    }
    variants: list[tuple[str, float, str, bool]] = [
        ("real", s, n, False) for n in NORMALIZATIONS for s in (0.05, 0.50)
    ]
    variants += [("real", s, "kc_norm", False) for s in (0.01, 0.20)]
    variants += [(t, 0.05, "kc_norm", False) for t in ("degree_preserving", "er")]
    variants += [("real_binary", 0.05, n, True) for n in ("none", "kc_norm")]
    results: dict[str, dict[str, object]] = {}
    for task_name, mixing in TASKS.items():
        task = task_for(real, seed, SIGMA, mixing=mixing)
        runs: dict[str, tuple[Callable[[], Learner], torch.Tensor, torch.Tensor]] = {
            "pn_ncm": (lambda: NearestClassMean(real.n_pn, N_CLASSES), task.train_x, task.test_x)
        }
        for topology, sparsity, normalization, binary in variants:
            graph = topologies["real" if topology == "real_binary" else topology]
            runs[f"mb_assoc|{topology}|{sparsity:.2f}|{normalization}"] = (
                lambda: AssociativeReadout(real.n_kc, N_CLASSES),
                code(task.train_x, graph, normalization, sparsity, binary=binary),
                code(task.test_x, graph, normalization, sparsity, binary=binary),
            )
        for name, (build, train_x, test_x) in runs.items():
            result = run_class_incremental(
                build(),
                train_x,
                task.train_y,
                test_x,
                task.test_y,
                classes_per_task=CLASSES_PER_TASK,
            )
            results[f"{task_name}/{name}"] = {
                "final_accuracy": result.final_accuracy,
                "forgetting": result.forgetting,
            }
    return results


def main() -> None:
    started = time.perf_counter()
    real = circuit()
    per_seed = {}
    for seed in SEEDS:
        per_seed[seed] = run_seed(real, seed)
        print(f"seed {seed}: primary {per_seed[seed][f'main/{PRIMARY}']['final_accuracy']:.3f}")
    names = list(per_seed[SEEDS[0]])
    summary = {
        name: {
            metric: {
                "mean": float(np.mean([float(per_seed[s][name][metric]) for s in SEEDS])),  # type: ignore[arg-type]
                "sd": float(np.std([float(per_seed[s][name][metric]) for s in SEEDS], ddof=1)),  # type: ignore[arg-type]
            }
            for metric in ("final_accuracy", "forgetting")
        }
        for name in names
    }
    report = {
        "protocol": "docs/fly/F1a-v2-mb-protocol.md",
        "seeds": list(SEEDS),
        "sigma": SIGMA,
        "summary": summary,
        "contrasts": contrasts(per_seed, SPECIFICATION),
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (OUT / "f1a_v2_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
