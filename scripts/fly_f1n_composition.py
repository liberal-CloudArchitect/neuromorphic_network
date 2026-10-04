"""F1n: does type-level input composition explain the FlyWire/hemibrain memory gap?

Protocol docs/fly/F1n-composition-swap-protocol.md (exploratory, CR-013).  F1m
search with the activity floor, synapse-count weights.  Each dataset keeps its
own neuron-level wiring while every (pre type -> post type) block is rescaled so
each post type's input mix equals its own or the other dataset's.  ``--pilot``
runs the positive control on seed 991 only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace

import numpy as np
import torch
from fly_f1a_mb import OUT, Contrast, contrasts
from fly_f1c_ring import N_CUES, circuit
from fly_f1c_v4 import DEVICE, PROTOCOL, TYPES, graphs_for
from fly_f1g_pi import CUE_CENTERS
from fly_f1l_pb_phase import HEMIBRAIN, TRANSMITTERS
from fly_f1m_floor import (
    CONTROL_FACTOR,
    candidates,
    cue_inputs,
    search_and_evaluate,
    weights_for,
)

from neuromorphic.fly.central_complex import (
    CORE_TYPES,
    CXCircuit,
    cue_bins,
    decode_angle,
    extract_cx,
    ideal_ring_circuit,
    match_composition,
    ring_embedding,
    simulate_memory,
    type_composition,
)
from neuromorphic.fly.hemibrain import read_tables

QUALIFICATION_SEED, PILOT_SEED = 990, 991
SEEDS = tuple(range(190, 200))
WIRING = ("flywire", "hemibrain")
ARMS = tuple(f"{w}/{c}" for w in WIRING for c in ("own", "swapped"))
GRAPHS = ("real", "type_preserving")


def advantage(row: dict[str, dict[str, object]], arm: str) -> float:
    return float(row[f"{arm}/real"]["memory"]) - float(row[f"{arm}/type_preserving"]["memory"])  # type: ignore[arg-type]


SPECIFICATION: dict[str, Contrast] = {
    "N1": ("hemibrain/swapped/real", "hemibrain/swapped/type_preserving", 0.10, "superiority"),
    "N2": ("advantage/flywire/own", "advantage/flywire/swapped", 0.10, "superiority"),
    "N3": ("advantage/hemibrain/swapped", "advantage/hemibrain/own", 0.10, "superiority"),
}


def describe(
    graph: CXCircuit, bins: np.ndarray, angles: np.ndarray, index: int, seed: int
) -> dict[str, float]:
    """Bump coherence and anchoring error of the selected candidate (evaluation noise)."""

    code = torch.tensor([TYPES.index(t) for t in graph.cell_type])
    gain_table, bias_table = candidates(seed)
    gains = torch.as_tensor(gain_table[index : index + 1], dtype=torch.float32)[:, code]
    biases = torch.as_tensor(bias_table[index : index + 1], dtype=torch.float32)[:, code]
    trials = 10
    cues = cue_inputs(graph, bins).repeat_interleave(trials, dim=0).to(DEVICE)
    late, _ = simulate_memory(
        weights_for(graph, None),
        cues,
        gains.to(DEVICE).expand(cues.shape[0], -1),
        biases.to(DEVICE).expand(cues.shape[0], -1),
        protocol=PROTOCOL,
        generator=torch.Generator(device=DEVICE).manual_seed(300 + seed),
    )
    epg = late.cpu()[:, graph.indices("EPG")]
    phase = torch.exp(1j * torch.as_tensor(angles, dtype=torch.float64))
    vector = (epg.to(torch.float64) * phase).sum(dim=1)
    coherence = (vector.abs() / epg.sum(dim=1).clamp(min=1e-12)).mean()
    centers = torch.as_tensor(CUE_CENTERS, dtype=torch.float32)[
        torch.arange(N_CUES).repeat_interleave(trials)
    ]
    theta = decode_angle(epg, angles)
    error = torch.remainder(theta - centers + math.pi, 2 * math.pi) - math.pi
    return {"bump_coherence": float(coherence), "anchor_error_rad": float(error.abs().mean())}


def build() -> dict[str, dict[str, object]]:
    """Per wiring source: real circuit, spectral cue bins/angles and both compositions."""

    flywire = circuit()
    annotations, connections, _ = read_tables(HEMIBRAIN, TRANSMITTERS)
    hemibrain = extract_cx(annotations, connections)
    reals = {"flywire": flywire, "hemibrain": hemibrain}
    mixes = {name: type_composition(graph) for name, graph in reals.items()}
    out: dict[str, dict[str, object]] = {}
    for name, graph in reals.items():
        angles, _ = ring_embedding(graph)
        other = "hemibrain" if name == "flywire" else "flywire"
        out[name] = {
            "graph": graph,
            "angles": angles,
            "bins": cue_bins(angles, N_CUES),
            "own": mixes[name],
            "swapped": mixes[other],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="positive control on seed 991")
    args = parser.parse_args()
    started = time.perf_counter()
    ring = replace(
        ideal_ring_circuit(), edge_count=ideal_ring_circuit().edge_count * CONTROL_FACTOR
    )
    control_seed = PILOT_SEED if args.pilot else QUALIFICATION_SEED
    control = search_and_evaluate(ring, None, np.arange(48) // 6, control_seed, 300 + control_seed)
    passed = float(control["memory"]) >= 0.5  # type: ignore[arg-type]
    print(f"control: memory {control['memory']:.3f} ({control['floor_status']}) -> {passed}")
    if args.pilot:
        return
    data = build()
    report: dict[str, object] = {
        "protocol": "docs/fly/F1n-composition-swap-protocol.md",
        "device": str(DEVICE),
        "qualification": control,
        "gate_passed": passed,
        "types": list(CORE_TYPES),
        "composition": {
            name: np.round(d["own"], 4).tolist()  # type: ignore[call-overload]
            for name, d in data.items()
        },
    }
    if not passed:
        report["status"] = "INVALID"
    else:
        achieved: dict[str, float] = {}
        per_seed: dict[int, dict[str, dict[str, object]]] = {}
        for seed in SEEDS:
            row: dict[str, dict[str, object]] = {}
            for wiring in WIRING:
                d = data[wiring]
                graphs = graphs_for(d["graph"], seed)  # type: ignore[arg-type]
                for mix in ("own", "swapped"):
                    arm = f"{wiring}/{mix}"
                    for name in GRAPHS:
                        graph = match_composition(graphs[name], d[mix])  # type: ignore[arg-type]
                        error = float(np.abs(type_composition(graph) - d[mix]).max())  # type: ignore[operator]
                        achieved[f"{arm}/{name}"] = max(achieved.get(f"{arm}/{name}", 0.0), error)
                        result = search_and_evaluate(graph, None, d["bins"], seed, 300 + seed)  # type: ignore[arg-type]
                        result["final_accuracy"] = result["memory"]
                        if result.get("floor_candidate") is not None:
                            result.update(
                                describe(
                                    graph,
                                    d["bins"],  # type: ignore[arg-type]
                                    d["angles"],  # type: ignore[arg-type]
                                    int(result["floor_candidate"]),  # type: ignore[call-overload]
                                    seed,
                                )
                            )
                        row[f"{arm}/{name}"] = result
                    row[f"advantage/{arm}"] = {"final_accuracy": advantage(row, arm)}
            per_seed[seed] = row
            print(
                f"seed {seed}: "
                + "  ".join(
                    f"{arm} {row[f'{arm}/real']['memory']:.2f}/"
                    f"{row[f'{arm}/type_preserving']['memory']:.2f}"
                    for arm in ARMS
                ),
                flush=True,
            )
        report["status"] = "COMPLETED"
        report["seeds"] = list(SEEDS)
        report["composition_max_abs_error"] = achieved
        keys = ("memory", "memory_nofloor", "valid_fraction", "bump_coherence", "anchor_error_rad")
        report["summary"] = {
            f"{arm}/{name}": {
                key: float(
                    np.mean(
                        [
                            float(per_seed[s][f"{arm}/{name}"][key])  # type: ignore[arg-type]
                            for s in SEEDS
                            if per_seed[s][f"{arm}/{name}"].get(key) is not None
                        ]
                    )
                )
                for key in keys
            }
            for arm in ARMS
            for name in GRAPHS
        }
        report["advantage"] = {
            arm: float(np.mean([advantage(per_seed[s], arm) for s in SEEDS])) for arm in ARMS
        }
        report["contrasts"] = contrasts(per_seed, SPECIFICATION, holm_kinds=("superiority",))
        report["per_seed"] = {str(s): v for s, v in per_seed.items()}
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    (OUT / "f1n_composition_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
