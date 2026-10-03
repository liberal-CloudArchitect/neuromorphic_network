"""F1a: mushroom-body class-incremental odor learning (protocol frozen in docs/fly).

Exploratory only (CR-013).  Extracts the right-hemisphere PN->KC circuit from
FlyWire 783 (cached), calibrates the odor noise on seed 999 with ``pn_ncm`` joint
training, then runs every condition on seeds 0-9 and the pre-registered
contrasts.  Writes JSON and Markdown to artifacts/fly/.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from neuromorphic.fly.continual import (
    AssociativeReadout,
    GradientLearner,
    Learner,
    NearestClassMean,
    OdorTask,
    PerceptronReadout,
    kc_code,
    make_odor_task,
    run_class_incremental,
)
from neuromorphic.fly.mushroom_body import (
    PNKCCircuit,
    degree_preserving,
    erdos_renyi,
    load_flywire_pn_kc,
)

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = (
    ROOT / "data/fly/flywire_annotations/v3.2.0/unzipped/flyconnectome-flywire_annotations-a83b277"
    "/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)
CONNECTIONS = ROOT / "data/fly/flywire_zenodo/783/proofread_connections_783.feather"
CACHE = ROOT / "artifacts/fly/mb_pn_kc_right_783.npz"
OUT = ROOT / "artifacts/fly"

N_CLASSES, N_TRAIN, N_TEST, CLASSES_PER_TASK = 100, 50, 50, 10
SEEDS = tuple(range(10))
CALIBRATION_SEED, CALIBRATION_TARGET = 999, 0.70
SIGMA_GRID = tuple(round(0.1 * i, 1) for i in range(1, 31))
BOOTSTRAP, BOOTSTRAP_SEED = 10_000, 20260715
MLP_HIDDEN = 1090


def circuit() -> PNKCCircuit:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        load_flywire_pn_kc(ANNOTATIONS, CONNECTIONS).save(CACHE)
    return PNKCCircuit.load(CACHE)


def glomerulus_index(real: PNKCCircuit) -> tuple[np.ndarray, int]:
    names = sorted(set(real.pn_glomerulus))
    return np.array([names.index(g) for g in real.pn_glomerulus], dtype=np.int64), len(names)


def task_for(real: PNKCCircuit, seed: int, sigma: float, *, mixing: float = 0.0) -> OdorTask:
    index, n_glomeruli = glomerulus_index(real)
    return make_odor_task(
        index,
        n_glomeruli=n_glomeruli,
        n_classes=N_CLASSES,
        n_train=N_TRAIN,
        n_test=N_TEST,
        sigma=sigma,
        rng=np.random.default_rng(seed),
        families=10 if mixing else 0,
        family_mixing=mixing,
    )


def calibrate(real: PNKCCircuit) -> tuple[float, dict[str, float]]:
    curve: dict[str, float] = {}
    for sigma in SIGMA_GRID:
        task = task_for(real, CALIBRATION_SEED, sigma)
        ncm = NearestClassMean(real.n_pn, N_CLASSES)
        ncm.fit_task(task.train_x, task.train_y)
        accuracy = (ncm.predict(task.test_x, N_CLASSES) == task.test_y).float().mean()
        curve[str(sigma)] = float(accuracy)
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

    def weights(topology: str, *, binary: bool = False) -> torch.Tensor:
        return torch.as_tensor(topologies[topology].dense(binary=binary), dtype=torch.float32)

    def coded(w: torch.Tensor, sparsity: float) -> tuple[torch.Tensor, torch.Tensor]:
        return kc_code(task.train_x, w, sparsity=sparsity), kc_code(
            task.test_x, w, sparsity=sparsity
        )

    conditions: dict[str, tuple[Callable[[], Learner], tuple[torch.Tensor, torch.Tensor]]] = {}
    kc = real.n_kc
    for sparsity in (0.01, 0.05, 0.20, 0.50):
        conditions[f"mb_assoc|real|{sparsity:.2f}"] = (
            lambda: AssociativeReadout(kc, N_CLASSES),
            coded(weights("real"), sparsity),
        )
    conditions["mb_assoc|real_binary|0.05"] = (
        lambda: AssociativeReadout(kc, N_CLASSES),
        coded(weights("real", binary=True), 0.05),
    )
    for topology in ("degree_preserving", "er"):
        conditions[f"mb_assoc|{topology}|0.05"] = (
            lambda: AssociativeReadout(kc, N_CLASSES),
            coded(weights(topology), 0.05),
        )
    real_code = coded(weights("real"), 0.05)
    conditions["mb_perceptron|real|0.05"] = (
        lambda: PerceptronReadout(kc, N_CLASSES, epochs=5, seed=seed),
        real_code,
    )

    def gradient(model: Callable[[], nn.Module]) -> Callable[[], Learner]:
        def build() -> Learner:
            torch.manual_seed(seed)
            return GradientLearner(model(), seed=seed)

        return build

    conditions["mb_sgd|real|0.05"] = (gradient(lambda: nn.Linear(kc, N_CLASSES)), real_code)
    raw = (task.train_x, task.test_x)
    conditions["pn_ncm"] = (lambda: NearestClassMean(real.n_pn, N_CLASSES), raw)
    conditions["pn_sgd"] = (gradient(lambda: nn.Linear(real.n_pn, N_CLASSES)), raw)
    conditions["mlp_sgd"] = (
        gradient(
            lambda: nn.Sequential(
                nn.Linear(real.n_pn, MLP_HIDDEN), nn.ReLU(), nn.Linear(MLP_HIDDEN, N_CLASSES)
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
    return results


CONTRASTS = {
    "C1": ("mb_assoc|real|0.05", "pn_sgd", 0.10, "superiority"),
    "C2": ("mb_assoc|real|0.05", "mb_sgd|real|0.05", 0.10, "superiority"),
    "C3": ("mb_assoc|real|0.05", "pn_ncm", 0.02, "superiority"),
    "C4": ("mb_assoc|real|0.05", "mb_assoc|degree_preserving|0.05", 0.01, "equivalence"),
    "C5": ("mb_assoc|real|0.05", "mb_assoc|real|0.50", 0.05, "superiority"),
}


Contrast = tuple[str, str, float, str]


def contrasts(
    per_seed: dict[int, dict[str, dict[str, object]]],
    specification: dict[str, Contrast] = CONTRASTS,
) -> dict[str, dict[str, object]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seeds = sorted(per_seed)
    draws = rng.integers(0, len(seeds), (BOOTSTRAP, len(seeds)))
    rows: dict[str, dict[str, object]] = {}
    for cid, (a, b, threshold, kind) in specification.items():
        diff = np.array(
            [
                float(per_seed[s][a]["final_accuracy"]) - float(per_seed[s][b]["final_accuracy"])  # type: ignore[arg-type]
                for s in seeds
            ]
        )
        boot = diff[draws].mean(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
        p = max(2 * min(float((boot <= 0).mean()), float((boot >= 0).mean())), 1 / BOOTSTRAP)
        rows[cid] = {
            "a": a,
            "b": b,
            "kind": kind,
            "threshold": threshold,
            "per_seed": diff.round(4).tolist(),
            "mean": float(diff.mean()),
            "ci95": [float(low), float(high)],
            "p": p,
        }
    order = sorted(rows, key=lambda c: float(rows[c]["p"]))  # type: ignore[arg-type]
    running = 0.0
    for rank, cid in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * float(rows[cid]["p"])))  # type: ignore[arg-type]
        rows[cid]["p_holm"] = running
    for row in rows.values():
        low, high = row["ci95"]  # type: ignore[misc]
        if row["kind"] == "equivalence":
            passed = -row["threshold"] < low and high < row["threshold"]  # type: ignore[operator]
            row["verdict"] = "EQUIVALENT" if passed else "UNDETERMINED"
        else:
            passed = (
                row["mean"] >= row["threshold"]  # type: ignore[operator]
                and low > 0
                and row["p_holm"] <= 0.05  # type: ignore[operator]
            )
            row["verdict"] = "SUPPORTED" if passed else "NOT_SUPPORTED"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = parser.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    started = time.perf_counter()
    real = circuit()
    sigma, curve = calibrate(real)
    print(f"circuit: {real.n_pn} PNs, {real.n_kc} KCs, {real.n_edges} edges; sigma={sigma}")
    per_seed: dict[int, dict[str, dict[str, object]]] = {}
    for seed in args.seeds:
        per_seed[seed] = run_seed(real, seed, sigma)
        primary = per_seed[seed]["mb_assoc|real|0.05"]["final_accuracy"]
        print(f"seed {seed} done; mb_assoc final={primary:.3f}", flush=True)
    names = list(per_seed[args.seeds[0]])
    summary = {
        name: {
            metric: {
                "mean": float(np.mean([per_seed[s][name][metric] for s in args.seeds])),
                "sd": float(np.std([per_seed[s][name][metric] for s in args.seeds], ddof=1)),
            }
            for metric in ("final_accuracy", "forgetting")
        }
        for name in names
    }
    report = {
        "protocol": "docs/fly/F1a-mb-protocol.md",
        "circuit": {
            "pn": real.n_pn,
            "kc": real.n_kc,
            "edges": real.n_edges,
            "glomeruli": len(set(real.pn_glomerulus)),
        },
        "sigma": sigma,
        "calibration_curve": curve,
        "seeds": args.seeds,
        "summary": summary,
        "contrasts": contrasts(per_seed) if len(args.seeds) > 1 else {},
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (OUT / "f1a_mb_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("sigma", "summary", "contrasts")}, indent=2))


if __name__ == "__main__":
    main()
