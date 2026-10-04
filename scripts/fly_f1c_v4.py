"""F1c-v4: equal-budget random search over cell-type gains/biases (no gradients).

Protocol docs/fly/F1c-v4-search-ring-protocol.md (exploratory, CR-013).  Every
graph of a seed scores the identical 1,000 candidates on calibration noise 950;
the positive-control ring must pass on seed 990 before seeds 90-99 run.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from fly_f1a_mb import OUT, Contrast, contrasts
from fly_f1c_ring import N_CUES, circuit

from neuromorphic.fly.central_complex import (
    CXCircuit,
    MemoryProtocol,
    cue_bins,
    degree_preserving_directed,
    ideal_ring_circuit,
    ring_embedding,
    simulate_memory,
    template_accuracy,
    type_preserving,
)

TYPES = ("Delta7", "EPG", "EPGt", "PEG", "PEN_a(PEN1)", "PEN_b(PEN2)")
CANDIDATES, CHUNK = 1000, 100
SEARCH_TRIALS, EVAL_TRIALS = 5, 10
CALIBRATION_NOISE, QUALIFICATION_SEED = 950, 990
SEEDS = tuple(range(90, 100))
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PROTOCOL = MemoryProtocol()
SPECIFICATION: dict[str, Contrast] = {
    "P1": ("real", "chance50", 0.0, "superiority"),
    "P2": ("real", "type_preserving", 0.20, "superiority"),
    "P3": ("real", "degree_preserving", 0.20, "superiority"),
}


def candidates(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3000 + seed)
    return 2.0 ** rng.uniform(-2, 5, (CANDIDATES, len(TYPES))), rng.uniform(
        -1.5, 0.5, (CANDIDATES, len(TYPES))
    )


def setup(graph: CXCircuit, bins: np.ndarray) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    epg = graph.indices("EPG")
    cue_inputs = torch.zeros(N_CUES, graph.n)
    cue_inputs[torch.as_tensor(bins), torch.as_tensor(epg)] = 1.0
    code = torch.tensor([TYPES.index(t) for t in graph.cell_type])
    return cue_inputs, epg, code


def accuracies(
    graph: CXCircuit,
    cue_inputs: torch.Tensor,
    epg: np.ndarray,
    gains: torch.Tensor,
    biases: torch.Tensor,
    *,
    trials: int,
    generator: torch.Generator,
) -> tuple[list[float], list[float]]:
    """Template accuracy and mean late EPG rate for each row of per-neuron params."""

    weights = torch.as_tensor(graph.weights(), dtype=torch.float32, device=DEVICE)
    per = 2 * N_CUES * trials
    cues = cue_inputs.repeat_interleave(trials, dim=0).repeat(2, 1).to(DEVICE)
    labels = torch.arange(N_CUES).repeat_interleave(trials)
    n = gains.shape[0]
    late, _ = simulate_memory(
        weights,
        cues.repeat(n, 1),
        gains.to(DEVICE).repeat_interleave(per, dim=0),
        biases.to(DEVICE).repeat_interleave(per, dim=0),
        protocol=PROTOCOL,
        generator=generator,
    )
    late = late.cpu()[:, epg]
    half = N_CUES * trials
    scores, rates = [], []
    for i in range(n):
        block = late[i * per : (i + 1) * per]
        templates = block[:half].reshape(N_CUES, trials, -1).mean(dim=1)
        scores.append(template_accuracy(templates, block[half:], labels))
        rates.append(float(block[half:].mean()))
    return scores, rates


def search_and_evaluate(
    graph: CXCircuit, bins: np.ndarray, seed: int, eval_seed: int
) -> dict[str, object]:
    cue_inputs, epg, code = setup(graph, bins)
    gain_table, bias_table = candidates(seed)
    gains = torch.as_tensor(gain_table, dtype=torch.float32)[:, code]
    biases = torch.as_tensor(bias_table, dtype=torch.float32)[:, code]
    generator = torch.Generator(device=DEVICE).manual_seed(CALIBRATION_NOISE)
    scores: list[float] = []
    for start in range(0, CANDIDATES, CHUNK):
        chunk_scores, _ = accuracies(
            graph,
            cue_inputs,
            epg,
            gains[start : start + CHUNK],
            biases[start : start + CHUNK],
            trials=SEARCH_TRIALS,
            generator=generator,
        )
        scores.extend(chunk_scores)
    best = int(np.argmax(scores))
    final, rate = accuracies(
        graph,
        cue_inputs,
        epg,
        gains[best : best + 1],
        biases[best : best + 1],
        trials=EVAL_TRIALS,
        generator=torch.Generator(device=DEVICE).manual_seed(eval_seed),
    )
    used = sorted(set(graph.cell_type))
    return {
        "final_accuracy": final[0],
        "late_mean_rate": rate[0],
        "search_best": float(scores[best]),
        "search_mean": float(np.mean(scores)),
        "search_fraction_above_half": float(np.mean(np.array(scores) >= 0.5)),
        "candidate": best,
        "gain": {t: float(gain_table[best, TYPES.index(t)]) for t in used},
        "bias": {t: float(bias_table[best, TYPES.index(t)]) for t in used},
    }


def graphs_for(real: CXCircuit, seed: int) -> dict[str, CXCircuit]:
    rng = np.random.default_rng(10_000 + seed)
    return {
        "real": real,
        "type_preserving": type_preserving(real, rng),
        "degree_preserving": degree_preserving_directed(real, rng),
    }


def main() -> None:
    started = time.perf_counter()
    real = circuit()
    real_bins = cue_bins(ring_embedding(real)[0], N_CUES)
    control = ideal_ring_circuit()
    qualification = search_and_evaluate(
        control, np.arange(48) // 6, QUALIFICATION_SEED, QUALIFICATION_SEED
    )
    passed = float(qualification["final_accuracy"]) >= 0.5  # type: ignore[arg-type]
    print(
        f"qualification: ideal ring acc {qualification['final_accuracy']:.3f} "
        f"(search best {qualification['search_best']:.3f}) -> {passed}",
        flush=True,
    )
    report: dict[str, object] = {
        "protocol": "docs/fly/F1c-v4-search-ring-protocol.md",
        "device": str(DEVICE),
        "qualification": qualification,
        "gate_passed": passed,
    }
    if not passed:
        report["status"] = "INVALID"
    else:
        per_seed: dict[int, dict[str, dict[str, object]]] = {}
        for seed in SEEDS:
            row: dict[str, dict[str, object]] = {"chance50": {"final_accuracy": 0.5}}
            for name, graph in graphs_for(real, seed).items():
                row[name] = search_and_evaluate(graph, real_bins, seed, seed)
            per_seed[seed] = row
            print(
                f"seed {seed}: "
                + "  ".join(
                    f"{k} {v['final_accuracy']:.3f}/{v['search_best']:.3f}"
                    for k, v in row.items()
                    if k != "chance50"
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
                    "late_mean_rate",
                    "search_best",
                    "search_mean",
                    "search_fraction_above_half",
                )
            }
            for name in ("real", "type_preserving", "degree_preserving")
        }
        report["contrasts"] = contrasts(per_seed, SPECIFICATION, holm_kinds=("superiority",))
        report["per_seed"] = {str(s): v for s, v in per_seed.items()}
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    (OUT / "f1c_v4_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
