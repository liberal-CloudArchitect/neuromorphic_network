"""F1c-v2: fixed connectome, trained cell-type gains/biases (12 parameters).

Protocol docs/fly/F1c-v2-trained-ring-protocol.md (exploratory, CR-013).  Every
graph gets the identical BPTT budget; evaluation reuses the F1c template protocol.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from fly_f1a_mb import OUT, Contrast, contrasts
from fly_f1c_ring import N_CUES, PROTOCOL, TRIALS, circuit

from neuromorphic.fly.central_complex import (
    CXCircuit,
    cue_bins,
    degree_preserving_directed,
    ring_embedding,
    simulate_memory,
    template_accuracy,
    train_type_parameters,
    type_preserving,
)

SEEDS = tuple(range(60, 70))
STEPS, LR, TRIALS_PER_CUE = 200, 0.05, 4
SPECIFICATION: dict[str, Contrast] = {
    "K1": ("real", "chance50", 0.0, "superiority"),
    "K2": ("real", "type_preserving", 0.20, "superiority"),
    "K3": ("real", "degree_preserving", 0.20, "superiority"),
}


def evaluate(
    graph: CXCircuit,
    gain: torch.Tensor,
    bias: torch.Tensor,
    cue_inputs: torch.Tensor,
    epg: np.ndarray,
    noise_seed: int,
) -> tuple[float, float]:
    cues = cue_inputs.repeat_interleave(TRIALS, dim=0).repeat(2, 1)
    batch = cues.shape[0]
    with torch.no_grad():
        late, during = simulate_memory(
            torch.as_tensor(graph.weights(), dtype=torch.float32),
            cues,
            gain.expand(batch, -1),
            bias.expand(batch, -1),
            protocol=PROTOCOL,
            generator=torch.Generator().manual_seed(noise_seed),
        )
    half = N_CUES * TRIALS
    late, during = late[:, epg], during[:, epg]
    templates = late[:half].reshape(N_CUES, TRIALS, -1).mean(dim=1)
    labels = torch.arange(N_CUES).repeat_interleave(TRIALS)
    accuracy = template_accuracy(templates, late[half:], labels)
    x = late[half:] - late[half:].mean(dim=1, keepdim=True)
    y = during[half:] - during[half:].mean(dim=1, keepdim=True)
    corr = (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1)).clamp(min=1e-12)
    return accuracy, float(corr.mean())


def main() -> None:
    started = time.perf_counter()
    real = circuit()
    epg = real.indices("EPG")
    angles, _ = ring_embedding(real)
    bins = cue_bins(angles, N_CUES)
    cue_inputs = torch.zeros(N_CUES, real.n)
    cue_inputs[torch.as_tensor(bins), torch.as_tensor(epg)] = 1.0
    per_seed: dict[int, dict[str, dict[str, object]]] = {}
    for seed in SEEDS:
        rng = np.random.default_rng(10_000 + seed)
        graphs = {
            "real": real,
            "type_preserving": type_preserving(real, rng),
            "degree_preserving": degree_preserving_directed(real, rng),
        }
        row: dict[str, dict[str, object]] = {"chance50": {"final_accuracy": 0.5}}
        for name, graph in graphs.items():
            params, history = train_type_parameters(
                graph,
                cue_inputs,
                epg,
                seed=1000 + seed,
                steps=STEPS,
                lr=LR,
                trials_per_cue=TRIALS_PER_CUE,
                protocol=PROTOCOL,
            )
            gain, bias = params.per_neuron(graph)
            accuracy, persistence = evaluate(graph, gain, bias, cue_inputs, epg, seed)
            row[name] = {
                "final_accuracy": accuracy,
                "persistence": persistence,
                "final_loss": float(np.mean(history[-10:])),
                "gain": dict(zip(params.types, torch.exp(params.log_gain).tolist(), strict=True)),
                "bias": dict(zip(params.types, params.bias.tolist(), strict=True)),
            }
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
            for key in ("final_accuracy", "persistence", "final_loss")
        }
        for name in ("real", "type_preserving", "degree_preserving")
    }
    report = {
        "protocol": "docs/fly/F1c-v2-trained-ring-protocol.md",
        "seeds": list(SEEDS),
        "budget": {"steps": STEPS, "lr": LR, "trials_per_cue": TRIALS_PER_CUE},
        "summary": summary,
        "contrasts": contrasts(per_seed, SPECIFICATION, holm_kinds=("superiority",)),
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (OUT / "f1c_v2_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
