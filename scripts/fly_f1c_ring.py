"""F1c: does real central-complex wiring hold a cue-specific persistent state?

Protocol docs/fly/F1c-ring-protocol.md (exploratory, CR-013).  Each graph (real
and every null realization) is calibrated over the same gain x bias grid on
noise seeds 950-952; seeds 50-59 draw nulls and test noise.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from fly_f1a_mb import ANNOTATIONS, CONNECTIONS, OUT, Contrast, contrasts

from neuromorphic.fly.central_complex import (
    CXCircuit,
    MemoryProtocol,
    cue_bins,
    degree_preserving_directed,
    extract_cx,
    ring_embedding,
    simulate_memory,
    template_accuracy,
    type_preserving,
)
from neuromorphic.fly.flywire import _pandas

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts/fly/cx_core_783.npz"
N_CUES, TRIALS = 8, 10
GAINS = (0.5, 1.0, 2.0, 4.0, 8.0)
BIASES = (-0.2, -0.1, 0.0, 0.1)
CALIBRATION_SEEDS = (950, 951, 952)
SEEDS = tuple(range(50, 60))
PROTOCOL = MemoryProtocol()
SPECIFICATION: dict[str, Contrast] = {
    "J1": ("real", "chance50", 0.0, "superiority"),
    "J2": ("real", "type_preserving", 0.20, "superiority"),
    "J3": ("real", "degree_preserving", 0.20, "superiority"),
}


def circuit() -> CXCircuit:
    if not CACHE.exists():
        pd = _pandas()
        annotations = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
        connections = pd.read_feather(
            CONNECTIONS, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"]
        )
        extract_cx(annotations, connections).save(CACHE)
    return CXCircuit.load(CACHE)


def evaluate(
    graph: CXCircuit,
    cue_inputs: torch.Tensor,
    epg: np.ndarray,
    params: list[tuple[float, float]],
    noise_seed: int,
) -> list[tuple[float, float]]:
    """Accuracy and late/cue persistence for each (gain, bias) on one noise seed."""

    weights = torch.as_tensor(graph.weights(), dtype=torch.float32)
    per = 2 * N_CUES * TRIALS  # template + test trials
    cues = cue_inputs.repeat_interleave(TRIALS, dim=0).repeat(2, 1)
    labels = torch.arange(N_CUES).repeat_interleave(TRIALS)
    gain = torch.tensor([g for g, _ in params]).repeat_interleave(per)
    bias = torch.tensor([b for _, b in params]).repeat_interleave(per)
    late, during = simulate_memory(
        weights,
        cues.repeat(len(params), 1),
        gain,
        bias,
        protocol=PROTOCOL,
        generator=torch.Generator().manual_seed(noise_seed),
    )
    out = []
    for i in range(len(params)):
        block_late = late[i * per : (i + 1) * per, epg]
        block_cue = during[i * per : (i + 1) * per, epg]
        half = N_CUES * TRIALS
        templates = block_late[:half].reshape(N_CUES, TRIALS, -1).mean(dim=1)
        accuracy = template_accuracy(templates, block_late[half:], labels)
        x = block_late[half:] - block_late[half:].mean(dim=1, keepdim=True)
        y = block_cue[half:] - block_cue[half:].mean(dim=1, keepdim=True)
        corr = (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1)).clamp(min=1e-12)
        out.append((accuracy, float(corr.mean())))
    return out


def calibrate(graph: CXCircuit, cue_inputs: torch.Tensor, epg: np.ndarray) -> tuple[float, float]:
    grid = list(itertools.product(GAINS, BIASES))  # gain ascending, then bias ascending
    scores = np.zeros(len(grid))
    for seed in CALIBRATION_SEEDS:
        scores += np.array([acc for acc, _ in evaluate(graph, cue_inputs, epg, grid, seed)])
    return grid[int(np.argmax(scores))]  # first maximum = smallest gain, then bias


def main() -> None:
    started = time.perf_counter()
    real = circuit()
    epg = real.indices("EPG")
    angles, eigenvalues = ring_embedding(real)
    bins = cue_bins(angles, N_CUES)
    cue_inputs = torch.zeros(N_CUES, real.n)
    cue_inputs[torch.as_tensor(bins), torch.as_tensor(epg)] = 1.0
    print(f"EPGs per cue bin: {np.bincount(bins, minlength=N_CUES).tolist()}", flush=True)
    real_params = calibrate(real, cue_inputs, epg)
    print(f"real: gain={real_params[0]} bias={real_params[1]}", flush=True)
    per_seed: dict[int, dict[str, dict[str, object]]] = {}
    descriptive: dict[str, list[dict[str, object]]] = {
        "type_preserving": [],
        "degree_preserving": [],
    }
    for seed in SEEDS:
        rng = np.random.default_rng(10_000 + seed)
        graphs = {
            "real": (real, real_params),
            "type_preserving": (type_preserving(real, rng), None),
            "degree_preserving": (degree_preserving_directed(real, rng), None),
        }
        row: dict[str, dict[str, object]] = {"chance50": {"final_accuracy": 0.5}}
        for name, (graph, params) in graphs.items():
            chosen = params or calibrate(graph, cue_inputs, epg)
            accuracy, persistence = evaluate(graph, cue_inputs, epg, [chosen], seed)[0]
            row[name] = {
                "final_accuracy": accuracy,
                "persistence": persistence,
                "gain": chosen[0],
                "bias": chosen[1],
            }
            if name != "real":
                descriptive[name].append(
                    {"seed": seed, "eigenvalues": ring_embedding(graph)[1].round(4).tolist()}
                )
        per_seed[seed] = row
        print(
            f"seed {seed}: "
            + "  ".join(
                f"{k} {v['final_accuracy']:.3f}" for k, v in row.items() if k != "chance50"
            ),
            flush=True,
        )
    summary = {
        name: {
            key: float(np.mean([float(per_seed[s][name][key]) for s in SEEDS]))  # type: ignore[arg-type]
            for key in per_seed[SEEDS[0]][name]
        }
        for name in ("real", "type_preserving", "degree_preserving")
    }
    report = {
        "protocol": "docs/fly/F1c-ring-protocol.md",
        "circuit": {"neurons": real.n, "edges": int(real.edge_pre.shape[0]), "epg": int(epg.size)},
        "real_eigenvalues": eigenvalues.round(4).tolist(),
        "cue_bin_sizes": np.bincount(bins, minlength=N_CUES).tolist(),
        "real_params": {"gain": real_params[0], "bias": real_params[1]},
        "seeds": list(SEEDS),
        "summary": summary,
        "null_spectra": descriptive,
        "contrasts": contrasts(per_seed, SPECIFICATION, holm_kinds=("superiority",)),
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (OUT / "f1c_ring_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
