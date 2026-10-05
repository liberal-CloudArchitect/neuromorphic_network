"""F1r: does the hemibrain CX integrate (rotate continuously) or only shift?

Protocol docs/fly/F1r-integration-protocol.md (exploratory, CR-013).  Re-simulates
the F1p real-graph solutions (joint-best candidates, seeds 210-219) with a 1 s
one-sided drive at five amplitudes, decoding heading every 10 ms.  A true
integrator keeps rotating at a constant rate (persistence ~1); a displacement
saturates (persistence ~0).  ``--pilot`` runs the ideal-ring control on seed 991.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from typing import Any

import numpy as np
import torch
from fly_f1a_mb import OUT, Contrast, contrasts
from fly_f1c_ring import N_CUES, circuit
from fly_f1c_v4 import DEVICE, PROTOCOL
from fly_f1g_pi import LNO_TYPES
from fly_f1l_pb_phase import CONTROL_FACTOR, HEMIBRAIN, TRANSMITTERS
from fly_f1m_floor import FLOOR_RATE, FLOOR_SPREAD
from fly_f1p_pi_signs import (
    MODES,
    Rotating,
    candidates,
    lno_with_type_signs,
    search_and_evaluate,
)

from neuromorphic.fly.central_complex import (
    consensus_signs,
    extract_cx,
    extract_input_pathway,
    ideal_input_pathways,
    ideal_rotating_ring,
    simulate_trace,
)
from neuromorphic.fly.hemibrain import circuit_phases, read_tables

FACTORS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
LOW = FACTORS.index(0.25)
REFERENCE = FACTORS.index(1.0)
TRIALS, RECORD_MS = 5, 10
HOLD_MS, DRIVE_MS = 500, 1000
QUALIFICATION_SEED, PILOT_SEED = 990, 991
SEEDS = tuple(range(210, 220))
SPECIFICATION: dict[str, Contrast] = {
    "T1": ("persistence/lno", "zero", 0.5, "superiority"),
    "T2": ("graded/lno", "zero", 0.5, "superiority"),
    "T3": ("persistence/pen", "zero", 0.5, "superiority"),
    "T4": ("graded/pen", "zero", 0.5, "superiority"),
}


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (average ranks for ties); NaN when a side is constant."""

    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        r = np.empty(v.size)
        r[order] = np.arange(v.size, dtype=float)
        for value in np.unique(v):
            tied = v == value
            r[tied] = r[tied].mean()
        return r

    a, b = ranks(np.asarray(x, dtype=float)), ranks(np.asarray(y, dtype=float))
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def unwrap(theta: torch.Tensor) -> torch.Tensor:
    """Unwrap angles along the last axis."""

    step = torch.remainder(torch.diff(theta, dim=-1) + math.pi, 2 * math.pi) - math.pi
    return torch.cat([theta[..., :1], theta[..., :1] + torch.cumsum(step, dim=-1)], dim=-1)


def time_course(
    setup: Rotating,
    mode: str,
    pick: dict[str, Any],
    gains: torch.Tensor,
    biases: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    """Drift-corrected antisymmetric rotation s(t) per amplitude factor, plus summaries."""

    conditions = [(0, 0.0)] + [
        (row, f) for f in FACTORS for row in (1, 2)
    ]  # 0 none, 1 left, 2 right
    per = N_CUES * TRIALS
    cue = setup.cue.repeat_interleave(TRIALS, dim=0) * float(pick["cue_gain"])
    velocity = torch.as_tensor(setup.velocity[mode], dtype=torch.float32)
    cues, drives = [], []
    for row, factor in conditions:
        cues.append(cue)
        drives.append(velocity[row].expand(per, -1) * float(pick["amplitude"]) * factor)
    cue_batch, drive_batch = torch.cat(cues), torch.cat(drives)
    batch = cue_batch.shape[0]
    trace = simulate_trace(
        setup.weights.to(DEVICE),
        gains.expand(batch, -1).to(DEVICE),
        biases.expand(batch, -1).to(DEVICE),
        [
            (200, None),
            (300, cue_batch.to(DEVICE)),
            (HOLD_MS, None),
            (DRIVE_MS, drive_batch.to(DEVICE)),
        ],
        record_every=RECORD_MS,
        neurons=setup.epg,
        protocol=PROTOCOL,
        generator=torch.Generator(device=DEVICE).manual_seed(400 + seed),
    ).cpu()
    start = (200 + 300 + HOLD_MS) // RECORD_MS - 1  # last record before the drive
    window = trace[:, start:]  # [batch, 1 + drive records, EPG]
    phase = torch.as_tensor(setup.angles, dtype=torch.float32)
    theta = torch.atan2(window @ torch.sin(phase), window @ torch.cos(phase))
    delta = unwrap(theta) - theta[:, :1]  # [batch, time]
    curves = delta.reshape(len(conditions), per, -1).mean(dim=1)  # [conditions, time]
    rates = window[:, 1:].reshape(len(conditions), per, -1, window.shape[-1])
    valid = [bool(r.mean() >= FLOOR_RATE and r.std(dim=-1).mean() >= FLOOR_SPREAD) for r in rates]
    times = np.arange(curves.shape[1]) * RECORD_MS / 1000.0  # seconds since drive onset
    half = curves.shape[1] // 2
    signed: list[np.ndarray] = []
    for k in range(len(FACTORS)):
        left, right = curves[1 + 2 * k] - curves[0], curves[2 + 2 * k] - curves[0]
        signed.append((0.5 * (left - right)).numpy())
    reference = signed[REFERENCE]
    direction = math.copysign(1.0, reference[-1]) if reference[-1] != 0 else 1.0
    early = direction * (reference[half] - reference[0]) / times[half]
    late = direction * (reference[-1] - reference[half]) / (times[-1] - times[half])
    persistence = float(np.clip(late / early, -1.0, 2.0)) if abs(early) > 0.05 else 0.0
    late_by_factor = [
        float(direction * (s[-1] - s[half]) / (times[-1] - times[half])) for s in signed
    ]
    rho = spearman(list(FACTORS), late_by_factor)
    fit = np.polyfit(times, direction * reference, 1)
    residual = direction * reference - np.polyval(fit, times)
    r2 = 1.0 - float(residual.var() / max((direction * reference).var(), 1e-12))
    all_valid = all(valid)
    return {
        "valid": all_valid,
        "valid_by_condition": valid,
        "persistence": persistence if all_valid else 0.0,
        "monotonic": float(rho) if np.isfinite(rho) else 0.0,
        "graded": late_by_factor[REFERENCE] - late_by_factor[LOW] if all_valid else 0.0,
        "early_velocity": float(early),
        "late_velocity": float(late),
        "late_velocity_by_factor": late_by_factor,
        "linear_r2": r2,
        "total_rotation_rad": float(direction * reference[-1]),
        "curve_reference": np.round(direction * reference[::5], 4).tolist(),
    }


def picks_from_report(seed: int, mode: str) -> dict[str, Any]:
    report = json.loads((OUT / "f1p_pi_signs_report.json").read_text(encoding="utf-8"))
    pick: dict[str, Any] = report["per_seed"][str(seed)]["real"][mode]
    return pick


def per_neuron(setup: Rotating, seed: int, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    table = candidates(seed)
    gains = torch.as_tensor(table["gain"][index : index + 1], dtype=torch.float32)[:, setup.code]
    biases = torch.as_tensor(table["bias"][index : index + 1], dtype=torch.float32)[:, setup.code]
    return gains, biases


def control(seed: int) -> dict[str, dict[str, Any]]:
    """Ideal rotating ring (counts x5, ideal LNO): search as in F1p, then time courses."""

    ring = ideal_rotating_ring()
    angles = np.full(ring.n, np.nan)
    ring_angles = -math.pi + 2 * math.pi * np.arange(48) / 48
    angles[ring.indices("EPG")] = ring_angles
    angles[np.array([t.startswith("PEN") for t in ring.cell_type])] = np.tile(ring_angles, 2)
    _, lno = ideal_input_pathways(ring)
    ring = dataclasses.replace(ring, edge_count=ring.edge_count * CONTROL_FACTOR)
    lno = dataclasses.replace(lno, counts=lno.counts * CONTROL_FACTOR)
    setup = Rotating(ring, lno, angles)
    found = search_and_evaluate(setup, seed, 200 + seed)
    out = {}
    for mode in MODES:
        pick = found[mode]
        gains, biases = per_neuron(setup, seed, int(pick["candidate"]))
        result = time_course(setup, mode, pick, gains, biases, seed)
        result["passed"] = result["persistence"] >= 0.5 and result["graded"] >= 0.5
        out[mode] = result
        print(
            f"control {mode}: persistence {result['persistence']:+.2f} graded "
            f"{result['graded']:+.2f} R2 {result['linear_r2']:.2f} total "
            f"{result['total_rotation_rad']:+.2f} rad valid {result['valid']} "
            f"-> {result['passed']}",
            flush=True,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="ideal-ring control on seed 991")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.pilot:
        (OUT / "f1r_pilot.json").write_text(
            json.dumps({"seed": PILOT_SEED, "control": control(PILOT_SEED)}, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    qualification = control(QUALIFICATION_SEED)
    valid_modes = {m: bool(qualification[m]["passed"]) for m in MODES}
    annotations, connections, _ = read_tables(HEMIBRAIN, TRANSMITTERS)
    real = consensus_signs(extract_cx(annotations, connections), circuit())
    lno = lno_with_type_signs(
        extract_input_pathway(annotations, connections, real, source_types=LNO_TYPES)
    )
    setup = Rotating(real, lno, circuit_phases(annotations, real))
    report: dict[str, Any] = {
        "protocol": "docs/fly/F1r-integration-protocol.md",
        "device": str(DEVICE),
        "qualification": qualification,
        "valid_modes": valid_modes,
    }
    if not any(valid_modes.values()):
        report["status"] = "INVALID"
    else:
        per_seed: dict[int, dict[str, dict[str, Any]]] = {}
        for seed in SEEDS:
            row: dict[str, dict[str, Any]] = {"zero": {"final_accuracy": 0.0}}
            for mode in MODES:
                pick = picks_from_report(seed, mode)
                gains, biases = per_neuron(setup, seed, int(pick["candidate"]))
                result = time_course(setup, mode, pick, gains, biases, seed)
                row[mode] = result
                row[f"persistence/{mode}"] = {"final_accuracy": result["persistence"]}
                row[f"graded/{mode}"] = {"final_accuracy": result["graded"]}
            per_seed[seed] = row
            print(
                f"seed {seed}: "
                + "  ".join(
                    f"{m}: persistence {row[m]['persistence']:+.2f} graded "
                    f"{row[m]['graded']:+.2f} R2 {row[m]['linear_r2']:.2f} "
                    f"v_early {row[m]['early_velocity']:+.2f} "
                    f"v_late {row[m]['late_velocity']:+.2f} "
                    f"total {row[m]['total_rotation_rad']:+.2f} valid {row[m]['valid']}"
                    for m in MODES
                ),
                flush=True,
            )
        specification = {
            cid: spec for cid, spec in SPECIFICATION.items() if valid_modes[spec[0].split("/")[1]]
        }
        report["status"] = "COMPLETED"
        report["seeds"] = list(SEEDS)
        report["summary"] = {
            mode: {
                key: float(np.mean([float(per_seed[s][mode][key]) for s in SEEDS]))
                for key in (
                    "persistence",
                    "monotonic",
                    "graded",
                    "linear_r2",
                    "early_velocity",
                    "late_velocity",
                    "total_rotation_rad",
                )
            }
            | {
                "valid": int(sum(per_seed[s][mode]["valid"] for s in SEEDS)),
                "late_velocity_by_factor": np.mean(
                    [per_seed[s][mode]["late_velocity_by_factor"] for s in SEEDS], axis=0
                )
                .round(4)
                .tolist(),
            }
            for mode in MODES
        }
        report["contrasts"] = contrasts(per_seed, specification, holm_kinds=("superiority",))
        report["per_seed"] = {str(s): {m: row[m] for m in MODES} for s, row in per_seed.items()}
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    (OUT / "f1r_integration_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
