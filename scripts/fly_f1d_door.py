"""F1d: does real PN->KC glomerular sampling bias help discriminate real odors?

Protocol docs/fly/F1d-door-protocol.md (exploratory, CR-013).  DoOR consensus
responses (CC BY-SA 4.0) drive the right-hemisphere PN->KC circuit; seeds 70-79.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from fly_f1a_mb import OUT, Contrast, circuit, contrasts, glomerulus_index

from neuromorphic.fly.continual import AssociativeReadout, NearestClassMean, draw_samples, kc_code
from neuromorphic.fly.door import load_door_block
from neuromorphic.fly.mushroom_body import (
    PNKCCircuit,
    degree_preserving,
    erdos_renyi,
    glomerulus_uniform,
)

ROOT = Path(__file__).resolve().parents[1]
DOOR = ROOT / "data/fly/door/db323a4"
N_GLOMERULI, MIN_COVERAGE = 28, 0.9
N_TRAIN, N_TEST, SPARSITY = 20, 20, 0.05
CALIBRATION_SEED, CALIBRATION_TARGET = 999, 0.80
SIGMA_GRID = tuple(round(0.05 * i, 2) for i in range(1, 31))
SEEDS = tuple(range(70, 80))
SPECIFICATION: dict[str, Contrast] = {
    "L1": ("all/real", "all/glomerulus_uniform", 0.01, "superiority"),
    "L2": ("interaction/real", "interaction/glomerulus_uniform", 0.02, "superiority"),
    "L3": ("all/real", "all/degree_preserving", 0.01, "equivalence"),
}


def odor_prototypes(real: PNKCCircuit) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Prototypes over all circuit glomeruli and the per-odor bias score."""

    block = load_door_block(
        DOOR / "door_response_matrix.csv",
        DOOR / "door_mappings.csv",
        set(real.pn_glomerulus),
        n_glomeruli=N_GLOMERULI,
        min_coverage=MIN_COVERAGE,
    )
    _, n_glomeruli = glomerulus_index(real)
    names = sorted(set(real.pn_glomerulus))
    columns = [names.index(g) for g in block.glomeruli]
    prototypes = np.zeros((len(block.odors), n_glomeruli))
    prototypes[:, columns] = block.responses
    edge_glomerulus = np.array(real.pn_glomerulus)[real.edge_pn]
    counts = np.array([(edge_glomerulus == g).sum() for g in block.glomeruli], dtype=np.float64)
    bias = counts / counts.mean()
    total = block.responses.sum(axis=1)
    score = np.where(total > 0, (block.responses @ bias) / np.where(total > 0, total, 1.0), np.nan)
    valid = np.isfinite(score)
    low_cut, high_cut = np.quantile(score[valid], [1 / 3, 2 / 3])
    meta: dict[str, object] = {
        "glomeruli": list(block.glomeruli),
        "receptors": list(block.receptors),
        "n_odors": len(block.odors),
        "glomerulus_bias": dict(zip(block.glomeruli, bias.round(3).tolist(), strict=True)),
        "high_odors": int((valid & (score >= high_cut)).sum()),
        "low_odors": int((valid & (score <= low_cut)).sum()),
        "silent_odors": int((~valid).sum()),
    }
    group = np.full(len(block.odors), "mid", dtype=object)
    group[valid & (score >= high_cut)] = "high"
    group[valid & (score <= low_cut)] = "low"
    return prototypes, group, meta


def samples(
    real: PNKCCircuit, prototypes: np.ndarray, sigma: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    index, _ = glomerulus_index(real)
    rng = np.random.default_rng(seed)
    n = prototypes.shape[0]
    train_y = np.repeat(np.arange(n), N_TRAIN)
    test_y = np.repeat(np.arange(n), N_TEST)

    def draw(labels: np.ndarray) -> torch.Tensor:
        return draw_samples(
            prototypes, labels, sigma=sigma, pn_glomerulus_index=index, pn_noise=0.05, rng=rng
        )

    return draw(train_y), torch.as_tensor(train_y), draw(test_y), torch.as_tensor(test_y)


def calibrate(real: PNKCCircuit, prototypes: np.ndarray) -> tuple[float, dict[str, float]]:
    curve = {}
    n = prototypes.shape[0]
    for sigma in SIGMA_GRID:
        train_x, train_y, test_x, test_y = samples(real, prototypes, sigma, CALIBRATION_SEED)
        ncm = NearestClassMean(real.n_pn, n)
        ncm.fit_task(train_x, train_y)
        curve[str(sigma)] = float((ncm.predict(test_x, n) == test_y).float().mean())
    chosen = min(SIGMA_GRID, key=lambda s: (abs(curve[str(s)] - CALIBRATION_TARGET), s))
    return chosen, curve


def main() -> None:
    started = time.perf_counter()
    real = circuit()
    prototypes, group, meta = odor_prototypes(real)
    n = prototypes.shape[0]
    sigma, curve = calibrate(real, prototypes)
    print(
        f"{meta['n_odors']} odors, sigma={sigma}, "
        f"groups high/low {meta['high_odors']}/{meta['low_odors']}"
    )
    per_seed: dict[int, dict[str, dict[str, object]]] = {}
    for seed in SEEDS:
        train_x, train_y, test_x, test_y = samples(real, prototypes, sigma, seed)
        rng = np.random.default_rng(10_000 + seed)
        graphs = {
            "real": real,
            "degree_preserving": degree_preserving(real, rng),
            "glomerulus_uniform": glomerulus_uniform(real, rng),
            "er": erdos_renyi(real, rng),
        }
        variants = [(name, True) for name in graphs] + [
            ("real", False),
            ("glomerulus_uniform", False),
        ]
        test_group = group[test_y.numpy()]
        row: dict[str, dict[str, object]] = {}
        for name, binary in variants:
            weights = torch.as_tensor(graphs[name].dense(binary=binary), dtype=torch.float32)
            readout = AssociativeReadout(real.n_kc, n)
            readout.fit_task(kc_code(train_x, weights, sparsity=SPARSITY), train_y)
            correct = readout.predict(kc_code(test_x, weights, sparsity=SPARSITY), n) == test_y
            label = name if binary else f"{name}_counts"
            for subset, mask in (
                ("all", np.ones_like(test_group, dtype=bool)),
                ("high", test_group == "high"),
                ("low", test_group == "low"),
            ):
                row[f"{subset}/{label}"] = {
                    "final_accuracy": float(correct[torch.as_tensor(mask)].float().mean())
                }
            row[f"interaction/{label}"] = {
                "final_accuracy": float(row[f"high/{label}"]["final_accuracy"])  # type: ignore[arg-type]
                - float(row[f"low/{label}"]["final_accuracy"])  # type: ignore[arg-type]
            }
        ncm = NearestClassMean(real.n_pn, n)
        ncm.fit_task(train_x, train_y)
        row["all/pn_ncm"] = {
            "final_accuracy": float((ncm.predict(test_x, n) == test_y).float().mean())
        }
        per_seed[seed] = row
        print(
            f"seed {seed}: real {row['all/real']['final_accuracy']:.3f}  "
            f"glom_uniform {row['all/glomerulus_uniform']['final_accuracy']:.3f}  "
            f"dp {row['all/degree_preserving']['final_accuracy']:.3f}",
            flush=True,
        )
    names = list(per_seed[SEEDS[0]])
    summary = {
        name: float(np.mean([float(per_seed[s][name]["final_accuracy"]) for s in SEEDS]))  # type: ignore[arg-type]
        for name in names
    }
    report = {
        "protocol": "docs/fly/F1d-door-protocol.md",
        "data": "DoOR.data db323a4 (CC BY-SA 4.0)",
        "meta": meta,
        "sigma": sigma,
        "calibration_curve": curve,
        "seeds": list(SEEDS),
        "summary": summary,
        "contrasts": contrasts(per_seed, SPECIFICATION, holm_kinds=("superiority",)),
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (OUT / "f1d_door_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
