"""F1q: clean causal tests of the PEN->EPG mapping in hemibrain path integration.

Protocol docs/fly/F1q-pen-causal-protocol.md (exploratory, CR-013).  F1p model,
search and activity floor (hemibrain, one sign per cell type, count weights).
Only the PEN->EPG block is manipulated: offsets zeroed or mirrored exactly on the
PB phase grid, or targets shuffled within the block.  Rotation direction is a
primary metric.  ``--pilot`` runs the positive controls on seed 991 only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from typing import Any

import numpy as np
from fly_f1a_mb import OUT, Contrast, contrasts
from fly_f1c_ring import circuit
from fly_f1c_v4 import DEVICE, graphs_for
from fly_f1g_pi import LNO_TYPES
from fly_f1l_pb_phase import CONTROL_FACTOR, HEMIBRAIN, TRANSMITTERS
from fly_f1p_pi_signs import MODES, Rotating, lno_with_type_signs, search_and_evaluate

from neuromorphic.fly.central_complex import (
    CXCircuit,
    InputPathway,
    consensus_signs,
    extract_cx,
    extract_input_pathway,
    ideal_input_pathways,
    ideal_rotating_ring,
    remap_offsets,
    shuffle_block,
)
from neuromorphic.fly.hemibrain import circuit_phases, read_tables

QUALIFICATION_SEED, PILOT_SEED = 990, 991
SEEDS = tuple(range(220, 230))
GRAPHS = ("real", "type_preserving", "offset_zeroed", "offset_mirrored", "pen_shuffled")
ZEROED_DROP = 0.3  # control: zeroed J must fall at least this far below intact J


SPECIFICATION: dict[str, Contrast] = {
    "R1": ("dir/lno/real", "zero", 0.30, "superiority"),
    "R2": ("joint/lno/real", "joint/lno/offset_zeroed", 0.20, "superiority"),
    "R3": ("joint/lno/real", "joint/lno/pen_shuffled", 0.20, "superiority"),
    "R4": ("mirror/lno", "zero", 0.30, "superiority"),
    "R5": ("joint/pen/real", "joint/pen/offset_zeroed", 0.20, "superiority"),
    "R6": ("joint/pen/real", "joint/pen/pen_shuffled", 0.20, "superiority"),
    "R7": ("mirror/pen", "zero", 0.30, "superiority"),
}
MODE_OF = {
    "R1": "lno",
    "R2": "lno",
    "R3": "lno",
    "R4": "lno",
    "R5": "pen",
    "R6": "pen",
    "R7": "pen",
}
NEEDS = {"R2": "zeroed", "R5": "zeroed", "R4": "mirrored", "R7": "mirrored"}


def signed_shift(result: dict[str, Any]) -> float:
    """Antisymmetric shift (radians); positive = left drive moves the bump forward."""

    return 0.5 * (
        float(result.get("rotation_d_left", 0.0)) - float(result.get("rotation_d_right", 0.0))
    )


def directional(result: dict[str, Any], expected: float) -> float:
    """Joint score signed by whether rotation runs in the ``expected`` direction (+1/-1)."""

    joint = max(float(result.get("joint", 0.0)), 0.0)
    return math.copysign(joint, expected * signed_shift(result)) if joint > 0 else 0.0


def signed_joint(result: dict[str, Any]) -> float:
    """Joint score signed by the rotation direction (0 without rotation)."""

    joint = max(float(result.get("joint", 0.0)), 0.0)
    return math.copysign(joint, signed_shift(result)) if joint > 0 else 0.0


def mirror_reversal(real: dict[str, Any], mirrored: dict[str, Any]) -> float:
    """Mirrored graph's joint score, positive when it rotates opposite to the real graph."""

    reference = signed_joint(real)
    return -math.copysign(1.0, reference) * signed_joint(mirrored) if reference != 0 else 0.0


def manipulations(real: CXCircuit, angles: np.ndarray, seed: int | None) -> dict[str, CXCircuit]:
    out = {
        "offset_zeroed": remap_offsets(real, angles, mode="zero"),
        "offset_mirrored": remap_offsets(real, angles, mode="mirror"),
    }
    if seed is not None:
        out["pen_shuffled"] = shuffle_block(real, np.random.default_rng(15_000 + seed))
    return out


def control(seed: int) -> dict[str, Any]:
    """Ideal rotating ring (counts x5) with ideal LNO: intact, zeroed and mirrored."""

    ring = ideal_rotating_ring()
    angles = np.full(ring.n, np.nan)
    ring_angles = -math.pi + 2 * math.pi * np.arange(48) / 48
    angles[ring.indices("EPG")] = ring_angles
    angles[np.array([t.startswith("PEN") for t in ring.cell_type])] = np.tile(ring_angles, 2)
    _, lno = ideal_input_pathways(ring)
    ring = dataclasses.replace(ring, edge_count=ring.edge_count * CONTROL_FACTOR)
    lno = dataclasses.replace(lno, counts=lno.counts * CONTROL_FACTOR)
    graphs = {"intact": ring, **manipulations(ring, angles, None)}
    results = {
        name: search_and_evaluate(Rotating(graph, lno, angles), seed, 200 + seed)
        for name, graph in graphs.items()
    }
    gates: dict[str, dict[str, bool]] = {}
    for mode in MODES:
        intact = results["intact"][mode]
        zeroed = results["offset_zeroed"][mode]
        mirrored = results["offset_mirrored"][mode]
        intact_ok = float(intact.get("memory", 0.0)) >= 0.5 and float(intact["joint"]) >= 0.5
        gates[mode] = {
            "intact": intact_ok,
            "zeroed": intact_ok and float(zeroed["joint"]) <= float(intact["joint"]) - ZEROED_DROP,
            "mirrored": intact_ok
            and float(mirrored["joint"]) >= 0.5
            and signed_shift(mirrored) * signed_shift(intact) < 0,
        }
        for name in graphs:
            r = results[name][mode]
            print(
                f"control {name}/{mode}: memory {r.get('memory', 0):.3f} joint {r['joint']:+.3f} "
                f"signed shift {signed_shift(r):+.3f}",
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
    parser.add_argument("--pilot", action="store_true", help="positive controls on seed 991")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.pilot:
        pilot = control(PILOT_SEED)
        (OUT / "f1q_pilot.json").write_text(
            json.dumps({"seed": PILOT_SEED, **pilot}, indent=2, default=float) + "\n",
            encoding="utf-8",
        )
        return
    qualification = control(QUALIFICATION_SEED)
    gates = qualification["gates"]
    real, lno, phases = hemibrain()
    fixed = manipulations(real, phases, None)
    report: dict[str, Any] = {
        "protocol": "docs/fly/F1q-pen-causal-protocol.md",
        "device": str(DEVICE),
        "qualification": qualification,
        "expected_direction": {"lno": 1.0, "pen": 1.0},
        "edges": {
            "real": int(real.edge_pre.size),
            **{k: int(v.edge_pre.size) for k, v in fixed.items()},
        },
    }
    if not any(g["intact"] for g in gates.values()):
        report["status"] = "INVALID"
    else:
        per_seed: dict[int, dict[str, dict[str, Any]]] = {}
        for seed in SEEDS:
            graphs = {
                "real": real,
                "type_preserving": graphs_for(real, seed)["type_preserving"],
                **manipulations(real, phases, seed),
            }
            row: dict[str, dict[str, Any]] = {"zero": {"final_accuracy": 0.0}}
            for name in GRAPHS:
                result = search_and_evaluate(Rotating(graphs[name], lno, phases), seed, 200 + seed)
                row[name] = result
                row[f"memory/{name}"] = {"final_accuracy": result["memory"]["memory"]}
                for mode in MODES:
                    expected = report["expected_direction"][mode]
                    result[mode]["signed_shift"] = signed_shift(result[mode])
                    result[mode]["directional"] = directional(result[mode], expected)
                    row[f"joint/{mode}/{name}"] = {"final_accuracy": result[mode]["joint"]}
                    row[f"dir/{mode}/{name}"] = {"final_accuracy": result[mode]["directional"]}
            for mode in MODES:
                row[f"mirror/{mode}"] = {
                    "final_accuracy": mirror_reversal(
                        row["real"][mode], row["offset_mirrored"][mode]
                    )
                }
            per_seed[seed] = row
            print(
                f"seed {seed}: "
                + "  ".join(
                    f"{name} mem={row[name]['memory']['memory']:.2f} "
                    + " ".join(
                        f"{m}:J={row[name][m]['joint']:+.2f}/dir={row[name][m]['directional']:+.2f}"
                        for m in MODES
                    )
                    for name in GRAPHS
                ),
                flush=True,
            )
        specification = {
            cid: spec
            for cid, spec in SPECIFICATION.items()
            if gates[MODE_OF[cid]][NEEDS.get(cid, "intact")]
        }
        report["status"] = "COMPLETED"
        report["seeds"] = list(SEEDS)
        report["summary"] = {
            name: {
                "memory": float(np.mean([per_seed[s][name]["memory"]["memory"] for s in SEEDS])),
                **{
                    f"{mode}/{key}": float(
                        np.mean([float(per_seed[s][name][mode].get(key, 0.0)) for s in SEEDS])
                    )
                    for mode in MODES
                    for key in ("joint", "directional", "signed_shift", "rotation_A", "memory")
                },
                **{
                    f"{mode}/direction_positive": int(
                        sum(per_seed[s][name][mode]["signed_shift"] > 0 for s in SEEDS)
                    )
                    for mode in MODES
                },
            }
            for name in GRAPHS
        }
        report["mirror_reversal"] = {
            mode: [float(per_seed[s][f"mirror/{mode}"]["final_accuracy"]) for s in SEEDS]
            for mode in MODES
        }
        report["contrasts"] = contrasts(per_seed, specification, holm_kinds=("superiority",))
        report["per_seed"] = {
            str(s): {k: v for k, v in row.items() if k in GRAPHS} for s, row in per_seed.items()
        }
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    (OUT / "f1q_pen_causal_report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
