"""F1a-v3: multi-modal odor categories (protocol docs/fly/F1a-v3-multimodal-protocol.md).

Exploratory (CR-013).  Each category is the union of four unrelated odors.
Noise is calibrated on seed 999 with an oracle mode-aware NCM only; seeds 20-29.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import numpy as np
import torch
from fly_f1a_mb import OUT, SIGMA_GRID, Contrast, circuit, contrasts, glomerulus_index
from torch import nn

from neuromorphic.fly.continual import (
    AssociativeReadout,
    GradientLearner,
    Learner,
    NearestClassMean,
    NearestNeighbor,
    OdorTask,
    PerceptronReadout,
    kc_code,
    make_multimodal_task,
    run_class_incremental,
)
from neuromorphic.fly.mushroom_body import PNKCCircuit, degree_preserving, erdos_renyi

N_CLASSES, MODES, PER_MODE_TRAIN, PER_MODE_TEST, CLASSES_PER_TASK = 20, 4, 25, 25, 2
SEEDS = tuple(range(20, 30))
CALIBRATION_SEED, CALIBRATION_TARGET = 999, 0.90
PRIMARY = "mb_assoc|real|0.05"
SPECIFICATION: dict[str, Contrast] = {
    "E1": (PRIMARY, "pn_ncm", 0.05, "superiority"),
    "E2": (PRIMARY, "mb_assoc|real|0.50", 0.02, "superiority"),
    "E3": (PRIMARY, "mb_sgd|real|0.05", 0.05, "superiority"),
    "E4": (PRIMARY, "pn_sgd", 0.10, "superiority"),
    "E5": (PRIMARY, "mb_assoc|degree_preserving|0.05", 0.01, "equivalence"),
}


def task_for(real: PNKCCircuit, seed: int, sigma: float) -> OdorTask:
    index, n_glomeruli = glomerulus_index(real)
    return make_multimodal_task(
        index,
        n_glomeruli=n_glomeruli,
        n_classes=N_CLASSES,
        modes=MODES,
        n_train_per_mode=PER_MODE_TRAIN,
        n_test_per_mode=PER_MODE_TEST,
        sigma=sigma,
        rng=np.random.default_rng(seed),
    )


def oracle_accuracy(real: PNKCCircuit, task: OdorTask) -> float:
    """Joint mode-aware NCM; order-independent, so it equals the incremental end state."""

    assert task.train_mode is not None and task.test_mode is not None
    ncm = NearestClassMean(real.n_pn, N_CLASSES * MODES)
    ncm.fit_task(task.train_x, task.train_mode)
    predicted = ncm.predict(task.test_x, N_CLASSES * MODES) // MODES
    return float((predicted == task.test_y).float().mean())


def calibrate(real: PNKCCircuit) -> tuple[float, dict[str, float]]:
    curve = {str(s): oracle_accuracy(real, task_for(real, CALIBRATION_SEED, s)) for s in SIGMA_GRID}
    chosen = min(SIGMA_GRID, key=lambda s: (abs(curve[str(s)] - CALIBRATION_TARGET), s))
    return chosen, curve


def run_seed(real: PNKCCircuit, seed: int, sigma: float) -> dict[str, dict[str, object]]:
    task = task_for(real, seed, sigma)
    rng = np.random.default_rng(10_000 + seed)
    topologies = {
        "real": real,
        "degree_preserving": degree_preserving(real, rng),
        "er": erdos_renyi(real, rng),
    }
    kc = real.n_kc

    def coded(
        topology: str, sparsity: float, *, binary: bool = False, kc_norm: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.as_tensor(topologies[topology].dense(binary=binary), dtype=torch.float32)
        if kc_norm:
            w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return kc_code(task.train_x, w, sparsity=sparsity), kc_code(
            task.test_x, w, sparsity=sparsity
        )

    def assoc() -> Learner:
        return AssociativeReadout(kc, N_CLASSES)

    def gradient(model: Callable[[], nn.Module]) -> Callable[[], Learner]:
        def build() -> Learner:
            torch.manual_seed(seed)
            return GradientLearner(model(), seed=seed)

        return build

    conditions: dict[str, tuple[Callable[[], Learner], tuple[torch.Tensor, torch.Tensor]]] = {}
    for sparsity in (0.01, 0.05, 0.20, 0.50):
        conditions[f"mb_assoc|real|{sparsity:.2f}"] = (assoc, coded("real", sparsity))
    conditions["mb_assoc|real_binary|0.05"] = (assoc, coded("real", 0.05, binary=True))
    conditions["mb_assoc|real_kc_norm|0.05"] = (assoc, coded("real", 0.05, kc_norm=True))
    for topology in ("degree_preserving", "er"):
        conditions[f"mb_assoc|{topology}|0.05"] = (assoc, coded(topology, 0.05))
    real_code = conditions[PRIMARY][1]
    conditions["mb_perceptron|real|0.05"] = (
        lambda: PerceptronReadout(kc, N_CLASSES, epochs=5, seed=seed),
        real_code,
    )
    conditions["mb_sgd|real|0.05"] = (gradient(lambda: nn.Linear(kc, N_CLASSES)), real_code)
    raw = (task.train_x, task.test_x)
    hidden = round(N_CLASSES * kc / (real.n_pn + N_CLASSES))  # parameter-matched to the readout
    conditions["pn_ncm"] = (lambda: NearestClassMean(real.n_pn, N_CLASSES), raw)
    conditions["pn_knn"] = (NearestNeighbor, raw)
    conditions["pn_sgd"] = (gradient(lambda: nn.Linear(real.n_pn, N_CLASSES)), raw)
    conditions["mlp_sgd"] = (
        gradient(
            lambda: nn.Sequential(
                nn.Linear(real.n_pn, hidden), nn.ReLU(), nn.Linear(hidden, N_CLASSES)
            )
        ),
        raw,
    )
    results: dict[str, dict[str, object]] = {}
    for name, (build, (train_x, test_x)) in conditions.items():
        result = run_class_incremental(
            build(),
            train_x,
            task.train_y,
            test_x,
            task.test_y,
            classes_per_task=CLASSES_PER_TASK,
        )
        results[name] = {
            "final_accuracy": result.final_accuracy,
            "forgetting": result.forgetting,
            "accuracy_matrix": result.accuracy.round(4).tolist(),
        }
    results["oracle_mode_ncm"] = {
        "final_accuracy": oracle_accuracy(real, task),
        "forgetting": float("nan"),
    }
    return results


def main() -> None:
    started = time.perf_counter()
    real = circuit()
    sigma, curve = calibrate(real)
    print(f"sigma={sigma} (oracle {curve[str(sigma)]:.3f})", flush=True)
    per_seed = {}
    for seed in SEEDS:
        per_seed[seed] = run_seed(real, seed, sigma)
        print(f"seed {seed}: primary {per_seed[seed][PRIMARY]['final_accuracy']:.3f}", flush=True)
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
        "protocol": "docs/fly/F1a-v3-multimodal-protocol.md",
        "sigma": sigma,
        "calibration_curve": curve,
        "seeds": list(SEEDS),
        "summary": summary,
        "contrasts": contrasts(per_seed, SPECIFICATION),
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (OUT / "f1a_v3_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
