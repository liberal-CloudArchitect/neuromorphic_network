"""F1a-v4: confirm the low-noise multi-modal advantage of sparse KC codes.

Protocol docs/fly/F1a-v4-confirmation-protocol.md (exploratory, CR-013).
Seeds 30-39; sigma fixed at 0.2 and 0.4 without calibration.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import numpy as np
import torch
from fly_f1a_mb import OUT, Contrast, circuit, contrasts
from fly_f1a_v3 import CLASSES_PER_TASK, N_CLASSES, oracle_accuracy, task_for

from neuromorphic.fly.continual import (
    AssociativeReadout,
    Learner,
    NearestClassMean,
    NearestNeighbor,
    kc_code,
    run_class_incremental,
)
from neuromorphic.fly.mushroom_body import PNKCCircuit, degree_preserving, erdos_renyi

SEEDS = tuple(range(30, 40))
SIGMAS = (0.2, 0.4)
PRIMARY = "mb_assoc|real_binary|0.05"
SPECIFICATION: dict[str, Contrast] = {
    "G1": (f"0.4/{PRIMARY}", "0.4/pn_ncm", 0.05, "superiority"),
    "G2": (f"0.2/{PRIMARY}", "0.2/pn_ncm", 0.05, "superiority"),
    "G3": (f"0.4/{PRIMARY}", "0.4/mb_assoc|real_binary|0.50", 0.02, "superiority"),
    "G4": (
        f"0.4/{PRIMARY}",
        "0.4/mb_assoc|degree_preserving_binary|0.05",
        0.01,
        "equivalence",
    ),
}


def mode_overlap(code: torch.Tensor, mode: torch.Tensor) -> tuple[float, float]:
    """Mean Jaccard overlap of KC codes for same-mode and different-mode test pairs."""

    inter = code @ code.T
    union = code.sum(1, keepdim=True) + code.sum(1) - inter
    jaccard = inter / union.clamp(min=1)
    same = mode[:, None] == mode[None, :]
    off_diagonal = ~torch.eye(mode.numel(), dtype=torch.bool)
    return float(jaccard[same & off_diagonal].mean()), float(jaccard[~same].mean())


def run_seed(real: PNKCCircuit, seed: int) -> dict[str, dict[str, object]]:
    rng = np.random.default_rng(10_000 + seed)
    graphs = {
        "real": real,
        "degree_preserving": degree_preserving(real, rng),
        "er": erdos_renyi(real, rng),
    }
    variants = [
        ("real_binary", "real", True, 0.05),
        ("real_binary", "real", True, 0.20),
        ("real_binary", "real", True, 0.50),
        ("real", "real", False, 0.05),
        ("degree_preserving_binary", "degree_preserving", True, 0.05),
        ("er_binary", "er", True, 0.05),
    ]
    results: dict[str, dict[str, object]] = {}
    for sigma in SIGMAS:
        task = task_for(real, seed, sigma)
        assert task.test_mode is not None
        runs: dict[str, tuple[Callable[[], Learner], torch.Tensor, torch.Tensor]] = {
            "pn_ncm": (lambda: NearestClassMean(real.n_pn, N_CLASSES), task.train_x, task.test_x),
            "pn_knn": (NearestNeighbor, task.train_x, task.test_x),
        }
        overlaps: dict[str, tuple[float, float]] = {}
        for label, graph, binary, sparsity in variants:
            weights = torch.as_tensor(graphs[graph].dense(binary=binary), dtype=torch.float32)
            name = f"mb_assoc|{label}|{sparsity:.2f}"
            train_code = kc_code(task.train_x, weights, sparsity=sparsity)
            test_code = kc_code(task.test_x, weights, sparsity=sparsity)
            overlaps[name] = mode_overlap(test_code, task.test_mode)
            runs[name] = (
                lambda: AssociativeReadout(real.n_kc, N_CLASSES),
                train_code,
                test_code,
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
            entry: dict[str, object] = {
                "final_accuracy": result.final_accuracy,
                "forgetting": result.forgetting,
            }
            if name in overlaps:
                entry["jaccard_same_mode"], entry["jaccard_other_mode"] = overlaps[name]
            results[f"{sigma}/{name}"] = entry
        results[f"{sigma}/oracle_mode_ncm"] = {
            "final_accuracy": oracle_accuracy(real, task),
            "forgetting": float("nan"),
        }
    return results


def main() -> None:
    started = time.perf_counter()
    real = circuit()
    per_seed = {}
    for seed in SEEDS:
        per_seed[seed] = run_seed(real, seed)
        print(f"seed {seed}: G1 primary {per_seed[seed][f'0.4/{PRIMARY}']['final_accuracy']:.3f}")
    names = list(per_seed[SEEDS[0]])
    summary = {
        name: {
            key: float(np.mean([float(per_seed[s][name][key]) for s in SEEDS]))  # type: ignore[arg-type]
            for key in per_seed[SEEDS[0]][name]
        }
        for name in names
    }
    report = {
        "protocol": "docs/fly/F1a-v4-confirmation-protocol.md",
        "seeds": list(SEEDS),
        "sigmas": list(SIGMAS),
        "summary": summary,
        "contrasts": contrasts(per_seed, SPECIFICATION),
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (OUT / "f1a_v4_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
