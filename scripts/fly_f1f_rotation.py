"""F1f: do left/right PEN drives rotate the bump in opposite directions?

Protocol docs/fly/F1f-path-integration-protocol.md (exploratory, CR-013).  Reuses
each graph's memory-selected F1c-v4 cell-type parameters; rotation is never
selected for.  Amplitude is chosen per graph on calibration noise 951.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from fly_f1a_mb import OUT, Contrast, contrasts
from fly_f1c_ring import N_CUES, circuit
from fly_f1c_v4 import DEVICE, PROTOCOL, graphs_for

from neuromorphic.fly.central_complex import (
    CXCircuit,
    antisymmetry_index,
    cue_bins,
    decode_angle,
    ring_embedding,
    simulate_rotation,
)

TRIALS = 10
AMPLITUDES = (0.1, 0.3, 1.0)
CALIBRATION_NOISE = 951
SEEDS = tuple(range(90, 100))
SPECIFICATION: dict[str, Contrast] = {
    "R1": ("real", "zero", 0.0, "superiority"),
    "R2": ("real", "type_preserving", 0.20, "superiority"),
    "R3": ("real", "degree_preserving", 0.20, "superiority"),
}


def rotation(
    graph: CXCircuit,
    params: dict[str, dict[str, float]],
    bins: np.ndarray,
    angles: np.ndarray,
    amplitude: float,
    noise_seed: int,
) -> tuple[float, float]:
    epg = graph.indices("EPG")
    gain = torch.tensor([params["gain"][t] for t in graph.cell_type], dtype=torch.float32)
    bias = torch.tensor([params["bias"][t] for t in graph.cell_type], dtype=torch.float32)
    cue = torch.zeros(N_CUES, graph.n)
    cue[torch.as_tensor(bins), torch.as_tensor(epg)] = 1.0
    pen = np.array([t.startswith("PEN") for t in graph.cell_type])
    side = np.array(graph.side)
    drives = torch.zeros(3, graph.n)  # none, left, right
    drives[1, torch.as_tensor(np.flatnonzero(pen & (side == "left")))] = amplitude
    drives[2, torch.as_tensor(np.flatnonzero(pen & (side == "right")))] = amplitude
    cues = cue.repeat_interleave(TRIALS, dim=0).repeat(3, 1)
    drive = drives.repeat_interleave(N_CUES * TRIALS, dim=0)
    batch = cues.shape[0]
    activity = simulate_rotation(
        torch.as_tensor(graph.weights(), dtype=torch.float32, device=DEVICE),
        cues.to(DEVICE),
        drive.to(DEVICE),
        gain.to(DEVICE).expand(batch, -1),
        bias.to(DEVICE).expand(batch, -1),
        protocol=PROTOCOL,
        generator=torch.Generator(device=DEVICE).manual_seed(noise_seed),
    ).cpu()
    theta = decode_angle(activity[:, epg], angles).reshape(3, N_CUES, TRIALS)
    return antisymmetry_index(theta[0], theta[1], theta[2])


def main() -> None:
    started = time.perf_counter()
    real = circuit()
    angles, _ = ring_embedding(real)
    bins = cue_bins(angles, N_CUES)
    fitted = json.loads((OUT / "f1c_v4_report.json").read_text(encoding="utf-8"))["per_seed"]
    per_seed: dict[int, dict[str, dict[str, object]]] = {}
    for seed in SEEDS:
        row: dict[str, dict[str, object]] = {"zero": {"final_accuracy": 0.0}}
        for name, graph in graphs_for(real, seed).items():
            params = fitted[str(seed)][name]
            calibration = [
                rotation(graph, params, bins, angles, a, CALIBRATION_NOISE)[0] for a in AMPLITUDES
            ]
            amplitude = AMPLITUDES[int(np.argmax(calibration))]
            index, shift = rotation(graph, params, bins, angles, amplitude, 100 + seed)
            row[name] = {
                "final_accuracy": index,  # contrasts() reads this key; here it is A
                "mean_abs_shift_rad": shift,
                "amplitude": amplitude,
                "memory_accuracy_f1c_v4": float(params["final_accuracy"]),
            }
        per_seed[seed] = row
        print(
            f"seed {seed}: "
            + "  ".join(
                f"{k} A={v['final_accuracy']:+.2f} "
                f"|d|={v['mean_abs_shift_rad']:.2f} a={v['amplitude']}"
                for k, v in row.items()
                if k != "zero"
            ),
            flush=True,
        )
    summary = {
        name: {
            key: float(np.mean([float(per_seed[s][name][key]) for s in SEEDS]))  # type: ignore[arg-type]
            for key in ("final_accuracy", "mean_abs_shift_rad")
        }
        for name in ("real", "type_preserving", "degree_preserving")
    }
    report = {
        "protocol": "docs/fly/F1f-path-integration-protocol.md",
        "metric": "antisymmetry index A (stored as final_accuracy for the shared contrast helper)",
        "device": str(DEVICE),
        "seeds": list(SEEDS),
        "summary": summary,
        "contrasts": contrasts(per_seed, SPECIFICATION, holm_kinds=("superiority",)),
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (OUT / "f1f_rotation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
