"""F1c-v3: qualified cell-type training of the CX ring (smooth activation, homeostasis).

Protocol docs/fly/F1c-v3-qualified-ring-protocol.md (exploratory, CR-013).
A qualification gate on seed 990 (positive-control ring >= 0.5, no silent graph)
must pass before formal seeds 80-89 run; otherwise the report is INVALID.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
from fly_f1a_mb import OUT, Contrast, contrasts
from fly_f1c_ring import N_CUES, TRIALS, circuit

from neuromorphic.fly.central_complex import (
    CXCircuit,
    MemoryProtocol,
    cue_bins,
    degree_preserving_directed,
    ideal_ring_circuit,
    ring_embedding,
    simulate_memory,
    template_accuracy,
    train_type_parameters,
    type_preserving,
)

PROTOCOL = MemoryProtocol(activation="tanh_softplus")
STEPS, LR, TRIALS_PER_CUE = 200, 0.05, 4
HOMEOSTASIS_WEIGHT, HOMEOSTASIS_TARGET = 1.0, 0.2
QUALIFICATION_SEED = 990
SEEDS = tuple(range(80, 90))
SPECIFICATION: dict[str, Contrast] = {
    "M1": ("real", "chance50", 0.0, "superiority"),
    "M2": ("real", "type_preserving", 0.20, "superiority"),
    "M3": ("real", "degree_preserving", 0.20, "superiority"),
}


def cues_for(graph: CXCircuit, bins: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    epg = graph.indices("EPG")
    cue_inputs = torch.zeros(N_CUES, graph.n)
    cue_inputs[torch.as_tensor(bins), torch.as_tensor(epg)] = 1.0
    return cue_inputs, epg


def train_and_evaluate(
    graph: CXCircuit, cue_inputs: torch.Tensor, epg: np.ndarray, seed: int
) -> dict[str, object]:
    params, history = train_type_parameters(
        graph,
        cue_inputs,
        epg,
        seed=1000 + seed,
        steps=STEPS,
        lr=LR,
        trials_per_cue=TRIALS_PER_CUE,
        protocol=PROTOCOL,
        homeostasis_weight=HOMEOSTASIS_WEIGHT,
        homeostasis_target=HOMEOSTASIS_TARGET,
    )
    gain, bias = params.per_neuron(graph)
    cues = cue_inputs.repeat_interleave(TRIALS, dim=0).repeat(2, 1)
    batch = cues.shape[0]
    with torch.no_grad():
        late, during = simulate_memory(
            torch.as_tensor(graph.weights(), dtype=torch.float32),
            cues,
            gain.expand(batch, -1),
            bias.expand(batch, -1),
            protocol=PROTOCOL,
            generator=torch.Generator().manual_seed(seed),
        )
    half = N_CUES * TRIALS
    late, during = late[:, epg], during[:, epg]
    templates = late[:half].reshape(N_CUES, TRIALS, -1).mean(dim=1)
    labels = torch.arange(N_CUES).repeat_interleave(TRIALS)
    x = late[half:] - late[half:].mean(dim=1, keepdim=True)
    y = during[half:] - during[half:].mean(dim=1, keepdim=True)
    corr = (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1)).clamp(min=1e-12)
    return {
        "final_accuracy": template_accuracy(templates, late[half:], labels),
        "persistence": float(corr.mean()),
        "late_mean_rate": float(late[half:].mean()),
        "final_loss": float(np.mean(history[-10:])),
        "loss_finite": bool(np.isfinite(history).all()),
        "gain": dict(zip(params.types, torch.exp(params.log_gain).tolist(), strict=True)),
        "bias": dict(zip(params.types, params.bias.tolist(), strict=True)),
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
    angles, _ = ring_embedding(real)
    real_bins = cue_bins(angles, N_CUES)
    real_cues, real_epg = cues_for(real, real_bins)

    control = ideal_ring_circuit()
    control_bins = np.arange(48) // 6
    control_cues, control_epg = cues_for(control, control_bins)
    qualification: dict[str, dict[str, object]] = {
        "ideal_ring": train_and_evaluate(control, control_cues, control_epg, QUALIFICATION_SEED)
    }
    for name, graph in graphs_for(real, QUALIFICATION_SEED).items():
        qualification[name] = train_and_evaluate(graph, real_cues, real_epg, QUALIFICATION_SEED)
    gate = {
        "positive_control": float(qualification["ideal_ring"]["final_accuracy"]) >= 0.5,  # type: ignore[arg-type]
        "no_silent_graph": all(
            bool(outcome["loss_finite"]) and float(outcome["late_mean_rate"]) > 0.01  # type: ignore[arg-type]
            for name, outcome in qualification.items()
            if name != "ideal_ring"
        ),
    }
    print(f"qualification: {gate}", flush=True)
    for name, outcome in qualification.items():
        print(
            f"  {name}: acc {outcome['final_accuracy']:.3f} late {outcome['late_mean_rate']:.3f} "
            f"loss {outcome['final_loss']:.3f}",
            flush=True,
        )
    report: dict[str, object] = {
        "protocol": "docs/fly/F1c-v3-qualified-ring-protocol.md",
        "qualification_seed": QUALIFICATION_SEED,
        "qualification": qualification,
        "gate": gate,
    }
    if not all(gate.values()):
        report["status"] = "INVALID"
    else:
        per_seed: dict[int, dict[str, dict[str, object]]] = {}
        for seed in SEEDS:
            row: dict[str, dict[str, object]] = {"chance50": {"final_accuracy": 0.5}}
            for name, graph in graphs_for(real, seed).items():
                row[name] = train_and_evaluate(graph, real_cues, real_epg, seed)
            per_seed[seed] = row
            print(
                f"seed {seed}: "
                + "  ".join(
                    f"{k} {v['final_accuracy']:.3f}" for k, v in row.items() if k != "chance50"
                ),
                flush=True,
            )
        report["status"] = "COMPLETED"
        report["seeds"] = list(SEEDS)
        report["summary"] = {
            name: {
                key: float(np.mean([float(per_seed[s][name][key]) for s in SEEDS]))  # type: ignore[arg-type]
                for key in ("final_accuracy", "persistence", "late_mean_rate", "final_loss")
            }
            for name in ("real", "type_preserving", "degree_preserving")
        }
        report["contrasts"] = contrasts(per_seed, SPECIFICATION, holm_kinds=("superiority",))
        report["per_seed"] = {str(s): v for s, v in per_seed.items()}
    report["chance_loss"] = math.log(N_CUES)
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    (OUT / "f1c_v3_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
