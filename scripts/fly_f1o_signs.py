"""F1o: hemibrain CX memory advantage with one transmitter sign per cell type.

Protocol docs/fly/F1o-consensus-signs-protocol.md (exploratory, CR-013).  F1m
search with the activity floor.  Every neuron gets its cell type's majority sign
in FlyWire (consistent with the experimental ``known_nt`` annotations); the
hemibrain body-mean predictions are kept as a comparator arm.  ``--pilot`` runs
the positive control on seed 991 only.
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
from fly_f1l_pb_phase import HEMIBRAIN, TRANSMITTERS, wrap
from fly_f1m_floor import (
    CONTROL_FACTOR,
    HEMIBRAIN_MATCHED_THRESHOLD,
    candidates,
    cue_inputs,
    search_and_evaluate,
    weights_for,
)

from neuromorphic.fly.central_complex import (
    CXCircuit,
    consensus_signs,
    cue_bins,
    decode_angle,
    extract_cx,
    ideal_ring_circuit,
    ring_embedding,
    simulate_memory,
)
from neuromorphic.fly.hemibrain import circuit_phases, read_tables

QUALIFICATION_SEED, PILOT_SEED = 990, 991
SEEDS = tuple(range(200, 210))
GRAPHS = ("real", "type_preserving", "degree_preserving")
# arm -> (circuit key, binary threshold or None, coordinates)
ARMS: dict[str, tuple[str, float | None, str]] = {
    "hemibrain_consensus/counts": ("hemibrain_consensus", None, "spectral"),
    "hemibrain_consensus/binary": ("hemibrain_consensus", HEMIBRAIN_MATCHED_THRESHOLD, "spectral"),
    "hemibrain_consensus/pb_binary": ("hemibrain_consensus", HEMIBRAIN_MATCHED_THRESHOLD, "pb"),
    "hemibrain_predicted/counts": ("hemibrain_predicted", None, "spectral"),
    "flywire_consensus/counts": ("flywire_consensus", None, "spectral"),
}
SPECIFICATION: dict[str, Contrast] = {
    "O1": (
        "hemibrain_consensus/counts/real",
        "hemibrain_consensus/counts/type_preserving",
        0.10,
        "superiority",
    ),
    "O2": (
        "hemibrain_consensus/binary/real",
        "hemibrain_consensus/binary/type_preserving",
        0.10,
        "superiority",
    ),
    "O3": (
        "advantage/hemibrain_consensus/counts",
        "advantage/hemibrain_predicted/counts",
        0.10,
        "superiority",
    ),
}


def pinning(
    graph: CXCircuit,
    threshold: float | None,
    bins: np.ndarray,
    angles: np.ndarray,
    index: int,
    seed: int,
) -> dict[str, float]:
    """Cue alignment and effective number of settled bump positions (evaluation noise)."""

    code = torch.tensor([TYPES.index(t) for t in graph.cell_type])
    gain_table, bias_table = candidates(seed)
    gains = torch.as_tensor(gain_table[index : index + 1], dtype=torch.float32)[:, code]
    biases = torch.as_tensor(bias_table[index : index + 1], dtype=torch.float32)[:, code]
    trials = 10
    cues = cue_inputs(graph, bins).repeat_interleave(trials, dim=0).to(DEVICE)
    late, _ = simulate_memory(
        weights_for(graph, threshold),
        cues,
        gains.to(DEVICE).expand(cues.shape[0], -1),
        biases.to(DEVICE).expand(cues.shape[0], -1),
        protocol=PROTOCOL,
        generator=torch.Generator(device=DEVICE).manual_seed(300 + seed),
    )
    theta = decode_angle(late.cpu()[:, graph.indices("EPG")], angles).numpy()
    cue = CUE_CENTERS[np.repeat(np.arange(N_CUES), trials)]
    alignment = max(float(abs(np.exp(1j * (theta - k * cue)).mean())) for k in (1.0, -1.0))
    occupancy = (
        np.bincount(
            ((theta + math.pi) // (2 * math.pi / N_CUES)).astype(int) % N_CUES, minlength=N_CUES
        )
        / theta.size
    )
    used = occupancy[occupancy > 0]
    return {
        "cue_alignment": alignment,
        "settled_positions": float(np.exp(-(used * np.log(used)).sum())),
    }


def circuits() -> dict[str, tuple[CXCircuit, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    """Circuits with their cue bins and decoding angles per coordinate system."""

    flywire = circuit()
    annotations, connections, _ = read_tables(HEMIBRAIN, TRANSMITTERS)
    hemibrain = extract_cx(annotations, connections)
    phases = circuit_phases(annotations, hemibrain)[hemibrain.indices("EPG")]
    pb = (np.argmin(np.abs(wrap(phases[:, None] - CUE_CENTERS[None, :])), axis=1), phases)

    def spectral(graph: CXCircuit) -> tuple[np.ndarray, np.ndarray]:
        angles, _ = ring_embedding(graph)
        return cue_bins(angles, N_CUES), angles

    return {
        "hemibrain_consensus": (
            consensus_signs(hemibrain, flywire),
            {"spectral": spectral(hemibrain), "pb": pb},
        ),
        "hemibrain_predicted": (hemibrain, {"spectral": spectral(hemibrain)}),
        "flywire_consensus": (consensus_signs(flywire, flywire), {"spectral": spectral(flywire)}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="positive control on seed 991")
    args = parser.parse_args()
    started = time.perf_counter()
    ring = ideal_ring_circuit()
    ring = replace(ring, edge_count=ring.edge_count * CONTROL_FACTOR)
    control_seed = PILOT_SEED if args.pilot else QUALIFICATION_SEED
    control = search_and_evaluate(ring, None, np.arange(48) // 6, control_seed, 300 + control_seed)
    passed = float(control["memory"]) >= 0.5  # type: ignore[arg-type]
    print(f"control: memory {control['memory']:.3f} ({control['floor_status']}) -> {passed}")
    if args.pilot:
        return
    data = circuits()
    report: dict[str, object] = {
        "protocol": "docs/fly/F1o-consensus-signs-protocol.md",
        "device": str(DEVICE),
        "qualification": control,
        "gate_passed": passed,
        "inhibitory_by_type": {
            key: {
                t: f"{int((g.sign[g.indices(t)] < 0).sum())}/{g.indices(t).size}"
                for t in sorted(set(g.cell_type))
            }
            for key, (g, _) in data.items()
        },
    }
    if not passed:
        report["status"] = "INVALID"
    else:
        per_seed: dict[int, dict[str, dict[str, object]]] = {}
        for seed in SEEDS:
            graphs = {key: graphs_for(g, seed) for key, (g, _) in data.items()}
            row: dict[str, dict[str, object]] = {}
            for arm, (key, threshold, coordinates) in ARMS.items():
                bins, angles = data[key][1][coordinates]
                for name in GRAPHS:
                    graph = graphs[key][name]
                    result = search_and_evaluate(graph, threshold, bins, seed, 300 + seed)
                    result["final_accuracy"] = result["memory"]
                    if name == "real" and result.get("floor_candidate") is not None:
                        result.update(
                            pinning(
                                graph,
                                threshold,
                                bins,
                                angles,
                                int(result["floor_candidate"]),  # type: ignore[call-overload]
                                seed,
                            )
                        )
                    row[f"{arm}/{name}"] = result
                row[f"advantage/{arm}"] = {
                    "final_accuracy": float(row[f"{arm}/real"]["memory"])  # type: ignore[arg-type]
                    - float(row[f"{arm}/type_preserving"]["memory"])  # type: ignore[arg-type]
                }
            per_seed[seed] = row
            print(
                f"seed {seed}: "
                + "  ".join(
                    f"{arm} {row[f'{arm}/real']['memory']:.2f}/"
                    f"{row[f'{arm}/type_preserving']['memory']:.2f}/"
                    f"{row[f'{arm}/degree_preserving']['memory']:.2f}"
                    for arm in ARMS
                ),
                flush=True,
            )
        keys = (
            "memory",
            "memory_nofloor",
            "valid_fraction",
            "floor_eval_rate",
            "cue_alignment",
            "settled_positions",
        )
        report["status"] = "COMPLETED"
        report["seeds"] = list(SEEDS)
        report["summary"] = {
            f"{arm}/{name}": {
                key: float(np.mean(values))
                for key in keys
                if (
                    values := [
                        float(per_seed[s][f"{arm}/{name}"][key])  # type: ignore[arg-type]
                        for s in SEEDS
                        if per_seed[s][f"{arm}/{name}"].get(key) is not None
                    ]
                )
            }
            for arm in ARMS
            for name in GRAPHS
        }
        report["contrasts"] = contrasts(per_seed, SPECIFICATION, holm_kinds=("superiority",))
        report["per_seed"] = {str(s): v for s, v in per_seed.items()}
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    (OUT / "f1o_signs_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
