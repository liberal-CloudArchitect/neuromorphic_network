"""F1s: search for integrators; does real hemibrain wiring integrate better than nulls?

Protocol docs/fly/F1s-integration-search-protocol.md (exploratory, CR-013).  F1p
model (hemibrain, one sign per cell type, count weights, activity floor).  Every
graph gets the same 1,000 candidates; the search maximizes an integration score
min(memory, persistence, graded velocity / 1 rad/s) under a 1 s one-sided drive.
The selected candidate is re-measured with independent noise and a finer drive
grid.  ``--pilot`` runs the ideal-ring controls on seed 991 only.
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
from fly_f1c_v4 import DEVICE, PROTOCOL, TYPES, graphs_for
from fly_f1g_pi import LNO_TYPES
from fly_f1l_pb_phase import CONTROL_FACTOR, HEMIBRAIN, TRANSMITTERS
from fly_f1m_floor import FLOOR_RATE, FLOOR_SPREAD
from fly_f1p_pi_signs import MODES, Rotating, floor, lno_with_type_signs
from fly_f1r_integration import unwrap

from neuromorphic.fly.central_complex import (
    CXCircuit,
    InputPathway,
    consensus_signs,
    extract_cx,
    extract_input_pathway,
    ideal_input_pathways,
    ideal_rotating_ring,
    shuffle_block,
    simulate_trace,
)
from neuromorphic.fly.hemibrain import circuit_phases, read_tables

CHANCE = 1.0 / N_CUES
CANDIDATES, CHUNK = 1000, 50
SEARCH_FACTORS, EVAL_FACTORS = (0.25, 1.0), (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
SEARCH_TRIALS, EVAL_TRIALS = 2, 5
SEARCH_RECORD_MS, EVAL_RECORD_MS = 100, 10
HOLD_MS, DRIVE_MS = 500, 1000
FULL_GRADE = 1.0  # rad/s of graded velocity for full credit
QUALIFICATION_SEED, PILOT_SEED = 990, 991
SEEDS = tuple(range(230, 240))
GRAPHS = ("real", "type_preserving", "degree_preserving", "pen_shuffled")
SPECIFICATION: dict[str, Contrast] = {
    "S1": ("score/lno/real", "zero", 0.5, "superiority"),
    "S2": ("score/lno/real", "score/lno/type_preserving", 0.3, "superiority"),
    "S3": ("score/lno/real", "score/lno/pen_shuffled", 0.3, "superiority"),
    "S4": ("score/pen/real", "zero", 0.5, "superiority"),
    "S5": ("score/pen/real", "score/pen/type_preserving", 0.3, "superiority"),
    "S6": ("score/pen/real", "score/pen/pen_shuffled", 0.3, "superiority"),
}


def candidates(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(16_000 + seed)
    return {
        "gain": 2.0 ** rng.uniform(-2, 5, (CANDIDATES, len(TYPES))),
        "bias": rng.uniform(-1.5, 0.5, (CANDIDATES, len(TYPES))),
        "cue_gain": 10.0 ** rng.uniform(-1, 1, CANDIDATES),
        "amplitude": 10.0 ** rng.uniform(math.log10(0.05), math.log10(5.0), CANDIDATES),
    }


def integration(
    setup: Rotating,
    mode: str,
    gains: torch.Tensor,
    biases: torch.Tensor,
    cue_gain: torch.Tensor,
    amplitude: torch.Tensor,
    *,
    factors: tuple[float, ...],
    trials: int,
    record_ms: int,
    generator: torch.Generator,
) -> list[dict[str, Any]]:
    """Persistence, graded velocity and velocity curve per candidate (direction-agnostic)."""

    n = gains.shape[0]
    rows = [0] + [r for _ in factors for r in (1, 2)]
    scale = [0.0] + [f for f in factors for _ in (1, 2)]
    per = N_CUES * trials
    conditions = len(rows)
    cue = setup.cue.repeat_interleave(trials, dim=0)  # [per, N]
    velocity = torch.as_tensor(setup.velocity[mode], dtype=torch.float32)
    drive_rows = torch.stack([velocity[r] * s for r, s in zip(rows, scale, strict=True)])
    cue_batch = (cue[None, None] * cue_gain[:, None, None, None]).expand(n, conditions, -1, -1)
    drive = drive_rows[None, :, None, :] * amplitude[:, None, None, None]
    drive_batch = drive.expand(n, conditions, per, -1)
    batch = n * conditions * per
    trace = simulate_trace(
        setup.weights.to(DEVICE),
        gains.repeat_interleave(conditions * per, dim=0).to(DEVICE),
        biases.repeat_interleave(conditions * per, dim=0).to(DEVICE),
        [
            (200, None),
            (300, cue_batch.reshape(batch, -1).to(DEVICE)),
            (HOLD_MS, None),
            (DRIVE_MS, drive_batch.reshape(batch, -1).to(DEVICE)),
        ],
        record_every=record_ms,
        neurons=setup.epg,
        protocol=PROTOCOL,
        generator=generator,
    ).cpu()
    start = (200 + 300 + HOLD_MS) // record_ms - 1
    window = trace[:, start:]
    phase = torch.as_tensor(setup.angles, dtype=torch.float32)
    theta = torch.atan2(window @ torch.sin(phase), window @ torch.cos(phase))
    delta = (unwrap(theta) - theta[:, :1]).reshape(n, conditions, per, -1).mean(dim=2)
    rates = window[:, 1:].reshape(n, conditions, per, window.shape[1] - 1, -1)
    mean_rate = rates.mean(dim=(2, 3, 4))
    spread = rates.std(dim=-1).mean(dim=(2, 3))
    steps = delta.shape[-1] - 1
    half = steps // 2
    seconds = record_ms / 1000.0
    out = []
    for i in range(n):
        valid = bool(((mean_rate[i] >= FLOOR_RATE) & (spread[i] >= FLOOR_SPREAD)).all())
        signed = [
            (0.5 * (delta[i, 1 + 2 * k] - delta[i, 2 + 2 * k])).numpy() for k in range(len(factors))
        ]
        reference = signed[factors.index(1.0)]
        direction = math.copysign(1.0, reference[-1]) if reference[-1] != 0 else 1.0
        early = direction * (reference[half] - reference[0]) / (half * seconds)
        late_by_factor = [
            float(direction * (s[-1] - s[half]) / ((steps - half) * seconds)) for s in signed
        ]
        late = late_by_factor[factors.index(1.0)]
        persistence = float(np.clip(late / early, -1.0, 2.0)) if abs(early) > 0.05 else 0.0
        graded = late - late_by_factor[factors.index(0.25)]
        times = np.arange(steps + 1) * seconds
        curve = direction * reference
        residual = curve - np.polyval(np.polyfit(times, curve, 1), times)
        out.append(
            {
                "valid": valid,
                "persistence": persistence if valid else 0.0,
                "graded": graded if valid else 0.0,
                "early_velocity": float(early),
                "late_velocity": late,
                "late_velocity_by_factor": late_by_factor,
                "linear_r2": 1.0 - float(residual.var() / max(curve.var(), 1e-12)),
                "total_rotation_rad": float(curve[-1]),
            }
        )
    return out


def score(memory: float, metrics: dict[str, Any]) -> float:
    return min(
        memory,
        float(np.clip(metrics["persistence"], 0.0, 1.0)),
        float(np.clip(metrics["graded"] / FULL_GRADE, 0.0, 1.0)),
    )


def search_and_evaluate(setup: Rotating, seed: int, eval_seed: int) -> dict[str, Any]:
    table = candidates(seed)
    gains = torch.as_tensor(table["gain"], dtype=torch.float32)[:, setup.code]
    biases = torch.as_tensor(table["bias"], dtype=torch.float32)[:, setup.code]
    cue_gain = torch.as_tensor(table["cue_gain"], dtype=torch.float32)
    amplitude = torch.as_tensor(table["amplitude"], dtype=torch.float32)
    memory_gen = torch.Generator(device=DEVICE).manual_seed(950)
    memory: list[tuple[float, float, float]] = []
    for start in range(0, CANDIDATES, 100):
        part = slice(start, start + 100)
        memory += setup.memory(
            gains[part], biases[part], cue_gain[part], trials=5, generator=memory_gen
        )
    accuracy = np.array([m[0] for m in memory])
    valid_memory = np.array([floor(m[1], m[2]) for m in memory])
    results: dict[str, Any] = {"valid_memory_fraction": float(valid_memory.mean())}
    for mode in MODES:
        generator = torch.Generator(device=DEVICE).manual_seed(952)
        scores = np.full(CANDIDATES, -np.inf)
        for start in range(0, CANDIDATES, CHUNK):
            part = slice(start, start + CHUNK)
            metrics = integration(
                setup,
                mode,
                gains[part],
                biases[part],
                cue_gain[part],
                amplitude[part],
                factors=SEARCH_FACTORS,
                trials=SEARCH_TRIALS,
                record_ms=SEARCH_RECORD_MS,
                generator=generator,
            )
            for offset, m in enumerate(metrics):
                index = start + offset
                if valid_memory[index]:
                    scores[index] = score(float(accuracy[index]), m)
        if not np.isfinite(scores).any():
            results[mode] = {"score": 0.0, "status": "no_valid_candidate"}
            continue
        best = int(np.argmax(scores))
        pick = slice(best, best + 1)
        ((acc, rate, spread),) = setup.memory(
            gains[pick],
            biases[pick],
            cue_gain[pick],
            trials=10,
            generator=torch.Generator(device=DEVICE).manual_seed(eval_seed),
        )
        memory_eval = acc if floor(rate, spread) else CHANCE
        (metrics_eval,) = integration(
            setup,
            mode,
            gains[pick],
            biases[pick],
            cue_gain[pick],
            amplitude[pick],
            factors=EVAL_FACTORS,
            trials=EVAL_TRIALS,
            record_ms=EVAL_RECORD_MS,
            generator=torch.Generator(device=DEVICE).manual_seed(eval_seed + 1),
        )
        results[mode] = {
            **metrics_eval,
            "memory": memory_eval,
            "score": score(memory_eval, metrics_eval),
            "search_best": float(scores[best]),
            "search_fraction_above_half": float(np.mean(scores >= 0.5)),
            "candidate": best,
            "amplitude": float(table["amplitude"][best]),
        }
    return results


def control(seed: int) -> dict[str, Any]:
    ring = ideal_rotating_ring()
    angles = np.full(ring.n, np.nan)
    ring_angles = -math.pi + 2 * math.pi * np.arange(48) / 48
    angles[ring.indices("EPG")] = ring_angles
    angles[np.array([t.startswith("PEN") for t in ring.cell_type])] = np.tile(ring_angles, 2)
    _, lno = ideal_input_pathways(ring)
    ring = dataclasses.replace(ring, edge_count=ring.edge_count * CONTROL_FACTOR)
    lno = dataclasses.replace(lno, counts=lno.counts * CONTROL_FACTOR)
    graphs = {
        "intact": ring,
        "pen_shuffled": shuffle_block(ring, np.random.default_rng(15_000 + seed)),
    }
    results = {
        name: search_and_evaluate(Rotating(g, lno, angles), seed, 200 + seed)
        for name, g in graphs.items()
    }
    gates = {}
    for mode in MODES:
        intact, shuffled = results["intact"][mode], results["pen_shuffled"][mode]
        intact_ok = float(intact["score"]) >= 0.5
        gates[mode] = {
            "intact": intact_ok,
            "shuffled": intact_ok and float(shuffled["score"]) <= float(intact["score"]) - 0.3,
        }
        for name in graphs:
            r = results[name][mode]
            print(
                f"control {name}/{mode}: score {r['score']:.3f} memory {r.get('memory', 0):.3f} "
                f"persistence {r.get('persistence', 0):+.2f} graded {r.get('graded', 0):+.2f}",
                flush=True,
            )
        print(f"control gates {mode}: {gates[mode]}", flush=True)
    return {"results": results, "gates": gates}


def hemibrain() -> tuple[CXCircuit, InputPathway, np.ndarray]:
    annotations, connections, _ = read_tables(HEMIBRAIN, TRANSMITTERS)
    real = consensus_signs(extract_cx(annotations, connections), circuit())
    lno = lno_with_type_signs(
        extract_input_pathway(annotations, connections, real, source_types=LNO_TYPES)
    )
    return real, lno, circuit_phases(annotations, real)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="ideal-ring controls on seed 991")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.pilot:
        pilot = control(PILOT_SEED)
        (OUT / "f1s_pilot.json").write_text(
            json.dumps({"seed": PILOT_SEED, **pilot}, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return
    qualification = control(QUALIFICATION_SEED)
    gates = qualification["gates"]
    real, lno, phases = hemibrain()
    report: dict[str, Any] = {
        "protocol": "docs/fly/F1s-integration-search-protocol.md",
        "device": str(DEVICE),
        "qualification": qualification,
    }
    if not any(g["intact"] for g in gates.values()):
        report["status"] = "INVALID"
    else:
        per_seed: dict[int, dict[str, dict[str, Any]]] = {}
        for seed in SEEDS:
            graphs = {
                **graphs_for(real, seed),
                "pen_shuffled": shuffle_block(real, np.random.default_rng(15_000 + seed)),
            }
            row: dict[str, dict[str, Any]] = {"zero": {"final_accuracy": 0.0}}
            for name in GRAPHS:
                result = search_and_evaluate(Rotating(graphs[name], lno, phases), seed, 200 + seed)
                row[name] = result
                for mode in MODES:
                    row[f"score/{mode}/{name}"] = {"final_accuracy": result[mode]["score"]}
            per_seed[seed] = row
            print(
                f"seed {seed}: "
                + "  ".join(
                    f"{name} "
                    + " ".join(
                        f"{m}:I={row[name][m]['score']:.2f}"
                        f"/P={row[name][m].get('persistence', 0):+.2f}"
                        f"/G={row[name][m].get('graded', 0):+.2f}"
                        for m in MODES
                    )
                    for name in GRAPHS
                ),
                flush=True,
            )
        specification = {
            cid: spec
            for cid, spec in SPECIFICATION.items()
            if gates[spec[0].split("/")[1]]["shuffled" if "pen_shuffled" in spec[1] else "intact"]
        }
        report["status"] = "COMPLETED"
        report["seeds"] = list(SEEDS)
        report["summary"] = {
            name: {
                f"{mode}/{key}": float(
                    np.mean([float(per_seed[s][name][mode].get(key, 0.0)) for s in SEEDS])
                )
                for mode in MODES
                for key in (
                    "score",
                    "memory",
                    "persistence",
                    "graded",
                    "linear_r2",
                    "late_velocity",
                    "search_best",
                    "search_fraction_above_half",
                )
            }
            for name in GRAPHS
        }
        report["contrasts"] = contrasts(per_seed, specification, holm_kinds=("superiority",))
        report["per_seed"] = {
            str(s): {k: v for k, v in row.items() if k in GRAPHS} for s, row in per_seed.items()
        }
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    (OUT / "f1s_integration_search_report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
