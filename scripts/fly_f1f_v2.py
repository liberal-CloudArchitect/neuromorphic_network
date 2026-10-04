"""F1f-v2: equal-budget joint search for memory and left/right rotation.

Protocol docs/fly/F1f-v2-joint-search-protocol.md (exploratory, CR-013).  Joint
score = min(memory accuracy, antisymmetry A); a rotating ideal ring must pass on
seed 990 before seeds 100-109 run.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
from fly_f1a_mb import OUT, Contrast, contrasts
from fly_f1c_ring import N_CUES, circuit
from fly_f1c_v4 import DEVICE, PROTOCOL, TYPES, accuracies, graphs_for

from neuromorphic.fly.central_complex import (
    CXCircuit,
    antisymmetry_index,
    cue_bins,
    decode_angle,
    ideal_rotating_ring,
    ring_embedding,
    simulate_rotation,
)

CANDIDATES, CHUNK = 1000, 100
SEARCH_TRIALS, EVAL_TRIALS = 5, 10
QUALIFICATION_SEED = 990
SEEDS = tuple(range(100, 110))
SPECIFICATION: dict[str, Contrast] = {
    "S1": ("real", "zero", 0.0, "superiority"),
    "S2": ("real", "type_preserving", 0.20, "superiority"),
    "S3": ("real", "degree_preserving", 0.20, "superiority"),
}


def candidates(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4000 + seed)
    gains = 2.0 ** rng.uniform(-2, 5, (CANDIDATES, len(TYPES)))
    biases = rng.uniform(-1.5, 0.5, (CANDIDATES, len(TYPES)))
    amplitudes = 10.0 ** rng.uniform(math.log10(0.05), math.log10(2.0), CANDIDATES)
    return gains, biases, amplitudes


def rotation_scores(
    graph: CXCircuit,
    cue: torch.Tensor,
    epg: np.ndarray,
    angles: np.ndarray,
    gains: torch.Tensor,
    biases: torch.Tensor,
    amplitudes: torch.Tensor,
    *,
    trials: int,
    generator: torch.Generator,
) -> list[tuple[float, float]]:
    pen = np.array([t.startswith("PEN") for t in graph.cell_type])
    side = np.array(graph.side)
    masks = torch.zeros(3, graph.n)
    masks[1, torch.as_tensor(np.flatnonzero(pen & (side == "left")))] = 1.0
    masks[2, torch.as_tensor(np.flatnonzero(pen & (side == "right")))] = 1.0
    per = 3 * N_CUES * trials
    cues = cue.repeat_interleave(trials, dim=0).repeat(3, 1)
    drive = masks.repeat_interleave(N_CUES * trials, dim=0)
    n = gains.shape[0]
    activity = simulate_rotation(
        torch.as_tensor(graph.weights(), dtype=torch.float32, device=DEVICE),
        cues.repeat(n, 1).to(DEVICE),
        (drive.repeat(n, 1) * amplitudes.repeat_interleave(per)[:, None]).to(DEVICE),
        gains.repeat_interleave(per, dim=0).to(DEVICE),
        biases.repeat_interleave(per, dim=0).to(DEVICE),
        protocol=PROTOCOL,
        generator=generator,
    ).cpu()
    theta = decode_angle(activity[:, epg], angles).reshape(n, 3, N_CUES, trials)
    return [antisymmetry_index(t[0], t[1], t[2]) for t in theta]


def search_and_evaluate(
    graph: CXCircuit, angles: np.ndarray, seed: int, eval_seed: int
) -> dict[str, object]:
    epg = graph.indices("EPG")
    bins = cue_bins(angles, N_CUES)
    cue = torch.zeros(N_CUES, graph.n)
    cue[torch.as_tensor(bins), torch.as_tensor(epg)] = 1.0
    code = torch.tensor([TYPES.index(t) for t in graph.cell_type])
    gain_table, bias_table, amplitude_table = candidates(seed)
    gains = torch.as_tensor(gain_table, dtype=torch.float32)[:, code]
    biases = torch.as_tensor(bias_table, dtype=torch.float32)[:, code]
    amplitudes = torch.as_tensor(amplitude_table, dtype=torch.float32)
    memory_generator = torch.Generator(device=DEVICE).manual_seed(950)
    rotation_generator = torch.Generator(device=DEVICE).manual_seed(951)
    memory: list[float] = []
    rotation: list[float] = []
    for start in range(0, CANDIDATES, CHUNK):
        part = slice(start, start + CHUNK)
        scores, _ = accuracies(
            graph,
            cue,
            epg,
            gains[part],
            biases[part],
            trials=SEARCH_TRIALS,
            generator=memory_generator,
        )
        memory.extend(scores)
        rotation.extend(
            a
            for a, _ in rotation_scores(
                graph,
                cue,
                epg,
                angles,
                gains[part],
                biases[part],
                amplitudes[part],
                trials=SEARCH_TRIALS,
                generator=rotation_generator,
            )
        )
    joint = np.minimum(memory, rotation)
    best = int(np.argmax(joint))
    pick = slice(best, best + 1)
    final_memory, _ = accuracies(
        graph,
        cue,
        epg,
        gains[pick],
        biases[pick],
        trials=EVAL_TRIALS,
        generator=torch.Generator(device=DEVICE).manual_seed(eval_seed),
    )
    ((final_a, shift),) = rotation_scores(
        graph,
        cue,
        epg,
        angles,
        gains[pick],
        biases[pick],
        amplitudes[pick],
        trials=EVAL_TRIALS,
        generator=torch.Generator(device=DEVICE).manual_seed(eval_seed + 1),
    )
    used = sorted(set(graph.cell_type))
    return {
        "final_accuracy": final_a,  # antisymmetry A, read by the shared contrast helper
        "memory_accuracy": final_memory[0],
        "mean_abs_shift_rad": shift,
        "search_joint_best": float(joint[best]),
        "search_fraction_joint_above_half": float(np.mean(joint >= 0.5)),
        "amplitude": float(amplitude_table[best]),
        "gain": {t: float(gain_table[best, TYPES.index(t)]) for t in used},
        "bias": {t: float(bias_table[best, TYPES.index(t)]) for t in used},
    }


def main() -> None:
    started = time.perf_counter()
    control = ideal_rotating_ring()
    control_angles = -math.pi + 2 * math.pi * np.arange(48) / 48
    qualification = search_and_evaluate(
        control, control_angles, QUALIFICATION_SEED, 200 + QUALIFICATION_SEED
    )
    passed = (
        float(qualification["memory_accuracy"]) >= 0.5  # type: ignore[arg-type]
        and float(qualification["final_accuracy"]) >= 0.5  # type: ignore[arg-type]
    )
    print(
        f"qualification: memory {qualification['memory_accuracy']:.3f} "
        f"A {qualification['final_accuracy']:+.3f} -> {passed}",
        flush=True,
    )
    report: dict[str, object] = {
        "protocol": "docs/fly/F1f-v2-joint-search-protocol.md",
        "device": str(DEVICE),
        "qualification": qualification,
        "gate_passed": passed,
    }
    if not passed:
        report["status"] = "INVALID"
    else:
        real = circuit()
        angles, _ = ring_embedding(real)
        per_seed: dict[int, dict[str, dict[str, object]]] = {}
        for seed in SEEDS:
            row: dict[str, dict[str, object]] = {"zero": {"final_accuracy": 0.0}}
            for name, graph in graphs_for(real, seed).items():
                row[name] = search_and_evaluate(graph, angles, seed, 200 + seed)
            per_seed[seed] = row
            print(
                f"seed {seed}: "
                + "  ".join(
                    f"{k} A={v['final_accuracy']:+.2f} mem={v['memory_accuracy']:.2f}"
                    for k, v in row.items()
                    if k != "zero"
                ),
                flush=True,
            )
        report["status"] = "COMPLETED"
        report["seeds"] = list(SEEDS)
        report["summary"] = {
            name: {
                key: float(np.mean([float(per_seed[s][name][key]) for s in SEEDS]))  # type: ignore[arg-type]
                for key in (
                    "final_accuracy",
                    "memory_accuracy",
                    "mean_abs_shift_rad",
                    "search_joint_best",
                    "search_fraction_joint_above_half",
                )
            }
            for name in ("real", "type_preserving", "degree_preserving")
        }
        report["contrasts"] = contrasts(per_seed, SPECIFICATION, holm_kinds=("superiority",))
        report["per_seed"] = {str(s): v for s, v in per_seed.items()}
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    (OUT / "f1f_v2_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
