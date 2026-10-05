"""F1p: hemibrain path integration with one transmitter sign per cell type.

Protocol docs/fly/F1p-pi-consensus-protocol.md (exploratory, CR-013).  F1l
redone with consistent signs (CX core and LNO inputs), synapse-count weights and
the F1m activity floor on both the delay and the velocity readouts.  Two
velocity arms: one-sided LNO input (F1l) and direct one-sided PEN drive.
``--pilot`` runs the positive controls on seed 991 only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from fly_f1a_mb import ANNOTATIONS, OUT, Contrast, contrasts
from fly_f1c_ring import N_CUES, circuit
from fly_f1c_v4 import DEVICE, PROTOCOL, TYPES, graphs_for
from fly_f1g_pi import CUE_CENTERS, LNO_TYPES
from fly_f1l_pb_phase import (
    CONTROL_FACTOR,
    FULL_SHIFT,
    HEMIBRAIN,
    TRANSMITTERS,
    ablate_offsets,
    offsets_by_group,
    wrap,
)
from fly_f1m_floor import FLOOR_RATE, FLOOR_SPREAD

from neuromorphic.fly.central_complex import (
    INHIBITORY_TRANSMITTERS,
    CXCircuit,
    InputPathway,
    consensus_signs,
    decode_angle,
    extract_cx,
    extract_input_pathway,
    ideal_input_pathways,
    ideal_rotating_ring,
    simulate_phases,
    template_accuracy,
)
from neuromorphic.fly.hemibrain import circuit_phases, read_tables

MODES = ("lno", "pen")
CHANCE = 1.0 / N_CUES
CANDIDATES, CHUNK = 1000, 100
SEARCH_TRIALS, EVAL_TRIALS = 5, 10
QUALIFICATION_SEED, PILOT_SEED = 990, 991
SEEDS = tuple(range(210, 220))
GRAPHS = ("real", "type_preserving", "degree_preserving", "offset_ablated")
SPECIFICATION: dict[str, Contrast] = {
    "Q1": ("joint/lno/real", "zero", 0.30, "superiority"),
    "Q2": ("joint/lno/real", "joint/lno/offset_ablated", 0.20, "superiority"),
    "Q3": ("joint/lno/real", "joint/lno/type_preserving", 0.20, "superiority"),
    "Q4": ("joint/pen/real", "zero", 0.30, "superiority"),
    "Q5": ("joint/pen/real", "joint/pen/offset_ablated", 0.20, "superiority"),
}


def floor(rate: float, spread: float) -> bool:
    return rate >= FLOOR_RATE and spread >= FLOOR_SPREAD


def gated(index: float, shift: float) -> float:
    return index * min(1.0, shift / FULL_SHIFT)


class Rotating:
    """Count weights, direct EPG cue on phase bins, LNO and PEN velocity inputs."""

    def __init__(self, graph: CXCircuit, lno: InputPathway, angles: np.ndarray) -> None:
        self.graph = graph
        self.epg = graph.indices("EPG")
        self.angles = angles[self.epg]
        self.weights = torch.as_tensor(graph.weights(), dtype=torch.float32)
        bins = np.argmin(np.abs(wrap(self.angles[:, None] - CUE_CENTERS[None, :])), axis=1)
        cue = torch.zeros(N_CUES, graph.n)
        cue[torch.as_tensor(bins), torch.as_tensor(self.epg)] = 1.0
        self.cue = cue
        side = np.array(graph.side)
        lno_side = np.array(lno.side)
        lno_w = lno.weights(graph.weight_scale())
        pen = np.array([t.startswith("PEN") for t in graph.cell_type])
        zero = np.zeros(graph.n)
        self.velocity = {
            "lno": np.stack(
                [
                    zero,
                    lno_w @ (lno_side == "left").astype(np.float64),
                    lno_w @ (lno_side == "right").astype(np.float64),
                ]
            ),
            "pen": np.stack(
                [
                    zero,
                    (pen & (side == "left")).astype(float),
                    (pen & (side == "right")).astype(float),
                ]
            ),
        }
        self.code = torch.tensor([TYPES.index(t) for t in graph.cell_type])

    def memory(
        self,
        gains: torch.Tensor,
        biases: torch.Tensor,
        cue_gain: torch.Tensor,
        *,
        trials: int,
        generator: torch.Generator,
    ) -> list[tuple[float, float, float]]:
        """Template accuracy, late EPG mean rate and spatial s.d. per candidate."""

        n = gains.shape[0]
        per = 2 * N_CUES * trials
        cue = self.cue.repeat_interleave(trials, dim=0).repeat(2, 1).repeat(n, 1)
        cue = cue * cue_gain.repeat_interleave(per)[:, None]
        late = simulate_phases(
            self.weights.to(DEVICE),
            gains.repeat_interleave(per, dim=0).to(DEVICE),
            biases.repeat_interleave(per, dim=0).to(DEVICE),
            [(200, None), (300, cue.to(DEVICE)), (1000, None)],
            readout_phase=2,
            readout_ms=200,
            protocol=PROTOCOL,
            generator=generator,
        ).cpu()[:, self.epg]
        labels = torch.arange(N_CUES).repeat_interleave(trials)
        half = N_CUES * trials
        out = []
        for i in range(n):
            block = late[i * per : (i + 1) * per]
            templates = block[:half].reshape(N_CUES, trials, -1).mean(dim=1)
            test = block[half:]
            out.append(
                (
                    template_accuracy(templates, test, labels),
                    float(test.mean()),
                    float(test.std(dim=1).mean()),
                )
            )
        return out

    def rotation(
        self,
        mode: str,
        gains: torch.Tensor,
        biases: torch.Tensor,
        cue_gain: torch.Tensor,
        amplitude: torch.Tensor,
        *,
        trials: int,
        generator: torch.Generator,
    ) -> list[dict[str, float]]:
        n = gains.shape[0]
        per = 3 * N_CUES * trials
        cue = self.cue.repeat_interleave(trials, dim=0).repeat(3, 1).repeat(n, 1)
        cue = cue * cue_gain.repeat_interleave(per)[:, None]
        velocity = torch.as_tensor(self.velocity[mode], dtype=torch.float32)
        drive = velocity.repeat_interleave(N_CUES * trials, dim=0).repeat(n, 1)
        drive = drive * amplitude.repeat_interleave(per)[:, None]
        activity = simulate_phases(
            self.weights.to(DEVICE),
            gains.repeat_interleave(per, dim=0).to(DEVICE),
            biases.repeat_interleave(per, dim=0).to(DEVICE),
            [(200, None), (300, cue.to(DEVICE)), (500, None), (300, drive.to(DEVICE))],
            readout_phase=3,
            readout_ms=100,
            protocol=PROTOCOL,
            generator=generator,
        ).cpu()[:, self.epg]
        theta = decode_angle(activity, self.angles).reshape(n, 3, N_CUES, trials)
        blocks = activity.reshape(n, per, -1)
        out = []
        for i in range(n):
            mean = torch.atan2(torch.sin(theta[i]).mean(-1), torch.cos(theta[i]).mean(-1))
            d_left = torch.remainder(mean[1] - mean[0] + math.pi, 2 * math.pi) - math.pi
            d_right = torch.remainder(mean[2] - mean[0] + math.pi, 2 * math.pi) - math.pi
            energy = 0.5 * float((d_left**2 + d_right**2).sum())
            index = -float((d_left * d_right).sum()) / energy if energy > 0 else 0.0
            out.append(
                {
                    "A": index,
                    "shift": float(torch.cat([d_left.abs(), d_right.abs()]).mean()),
                    "d_left": float(d_left.mean()),
                    "d_right": float(d_right.mean()),
                    "rate": float(blocks[i].mean()),
                    "spread": float(blocks[i].std(dim=1).mean()),
                }
            )
        return out


def candidates(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(13_000 + seed)
    return {
        "gain": 2.0 ** rng.uniform(-2, 5, (CANDIDATES, len(TYPES))),
        "bias": rng.uniform(-1.5, 0.5, (CANDIDATES, len(TYPES))),
        "cue_gain": 10.0 ** rng.uniform(-1, 1, CANDIDATES),
        "amplitude": 10.0 ** rng.uniform(math.log10(0.05), math.log10(5.0), CANDIDATES),
    }


def rotation_score(r: dict[str, float]) -> float:
    return gated(r["A"], r["shift"]) if floor(r["rate"], r["spread"]) else 0.0


def search_and_evaluate(setup: Rotating, seed: int, eval_seed: int) -> dict[str, dict[str, Any]]:
    table = candidates(seed)
    code = setup.code
    gains = torch.as_tensor(table["gain"], dtype=torch.float32)[:, code]
    biases = torch.as_tensor(table["bias"], dtype=torch.float32)[:, code]
    cue_gain = torch.as_tensor(table["cue_gain"], dtype=torch.float32)
    amplitude = torch.as_tensor(table["amplitude"], dtype=torch.float32)
    memory_gen = torch.Generator(device=DEVICE).manual_seed(950)
    memory: list[tuple[float, float, float]] = []
    for start in range(0, CANDIDATES, CHUNK):
        part = slice(start, start + CHUNK)
        memory += setup.memory(
            gains[part], biases[part], cue_gain[part], trials=SEARCH_TRIALS, generator=memory_gen
        )
    accuracy = np.array([m[0] for m in memory])
    valid = np.array([floor(m[1], m[2]) for m in memory])

    def evaluate(index: int, mode: str | None) -> dict[str, Any]:
        pick = slice(index, index + 1)
        ((acc, rate, spread),) = setup.memory(
            gains[pick],
            biases[pick],
            cue_gain[pick],
            trials=EVAL_TRIALS,
            generator=torch.Generator(device=DEVICE).manual_seed(eval_seed),
        )
        out: dict[str, Any] = {
            "memory": acc if floor(rate, spread) else CHANCE,
            "memory_valid": floor(rate, spread),
            "rate": rate,
        }
        if mode is not None:
            ((rot,),) = [
                setup.rotation(
                    mode,
                    gains[pick],
                    biases[pick],
                    cue_gain[pick],
                    amplitude[pick],
                    trials=EVAL_TRIALS,
                    generator=torch.Generator(device=DEVICE).manual_seed(eval_seed + 1),
                )
            ]
            score = rotation_score(rot)
            out.update({f"rotation_{k}": v for k, v in rot.items()})
            out["rotation_score"] = score
            out["joint"] = min(float(out["memory"]), score)
        return out

    results: dict[str, dict[str, Any]] = {}
    best_memory = int(np.argmax(np.where(valid, accuracy, -np.inf))) if valid.any() else None
    results["memory"] = (
        {**evaluate(best_memory, None), "valid_fraction": float(valid.mean())}
        if best_memory is not None
        else {"memory": CHANCE, "valid_fraction": 0.0}
    )
    for mode in MODES:
        rotation_gen = torch.Generator(device=DEVICE).manual_seed(951)
        scores: list[float] = []
        for start in range(0, CANDIDATES, CHUNK):
            part = slice(start, start + CHUNK)
            scores += [
                rotation_score(r)
                for r in setup.rotation(
                    mode,
                    gains[part],
                    biases[part],
                    cue_gain[part],
                    amplitude[part],
                    trials=SEARCH_TRIALS,
                    generator=rotation_gen,
                )
            ]
        joint = np.where(valid, np.minimum(accuracy, scores), -np.inf)
        if not valid.any():
            results[mode] = {"joint": 0.0, "status": "no_valid_candidate"}
            continue
        best = int(np.argmax(joint))
        result = evaluate(best, mode)
        result.update(
            {
                "search_joint_best": float(joint[best]),
                "search_fraction_joint_above_half": float(np.mean(joint >= 0.5)),
                "candidate": best,
                "amplitude": float(table["amplitude"][best]),
                "cue_gain": float(table["cue_gain"][best]),
            }
        )
        results[mode] = result
    return results


def lno_with_type_signs(lno: InputPathway) -> InputPathway:
    """Each LNO-family neuron gets its type's majority transmitter sign in FlyWire."""

    fw = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False, usecols=["cell_type", "top_nt"])
    majority = {}
    for t in set(lno.cell_type):
        nts = fw[fw.cell_type == t].top_nt.astype(str).str.lower()
        majority[t] = -1.0 if nts.isin(INHIBITORY_TRANSMITTERS).mean() >= 0.5 else 1.0
    return dataclasses.replace(lno, sign=np.array([majority[t] for t in lno.cell_type]))


def control(seed: int) -> dict[str, dict[str, Any]]:
    ring = ideal_rotating_ring()
    angles = np.full(ring.n, np.nan)
    ring_angles = -math.pi + 2 * math.pi * np.arange(48) / 48
    angles[ring.indices("EPG")] = ring_angles
    angles[np.array([t.startswith("PEN") for t in ring.cell_type])] = np.tile(ring_angles, 2)
    _, lno = ideal_input_pathways(ring)
    ring = dataclasses.replace(ring, edge_count=ring.edge_count * CONTROL_FACTOR)
    lno = dataclasses.replace(lno, counts=lno.counts * CONTROL_FACTOR)
    offsets = offsets_by_group(ring, angles)
    offsets = {k: v for k, v in offsets.items() if v["n"]}
    out = {}
    for name, graph in (
        ("intact", ring),
        ("offset_ablated", ablate_offsets(ring, angles, offsets)),
    ):
        result = search_and_evaluate(Rotating(graph, lno, angles), seed, 200 + seed)
        for mode in MODES:
            r = result[mode]
            r["passed"] = float(r.get("memory", 0.0)) >= 0.5 and float(r["joint"]) >= 0.5
            print(
                f"control {name}/{mode}: memory {r.get('memory', 0):.3f} joint {r['joint']:+.3f} "
                f"A {r.get('rotation_A', 0):+.3f} shift {r.get('rotation_shift', 0):.3f}",
                flush=True,
            )
        out[name] = result
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="positive controls on seed 991")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.pilot:
        (OUT / "f1p_pilot.json").write_text(
            json.dumps(
                {"seed": PILOT_SEED, "control": control(PILOT_SEED)}, indent=2, default=float
            )
            + "\n",
            encoding="utf-8",
        )
        return
    qualification = control(QUALIFICATION_SEED)
    valid_modes = {m: bool(qualification["intact"][m]["passed"]) for m in MODES}
    annotations, connections, _ = read_tables(HEMIBRAIN, TRANSMITTERS)
    real = consensus_signs(extract_cx(annotations, connections), circuit())
    phases = circuit_phases(annotations, real)
    lno = lno_with_type_signs(
        extract_input_pathway(annotations, connections, real, source_types=LNO_TYPES)
    )
    offsets = offsets_by_group(real, phases)
    report: dict[str, Any] = {
        "protocol": "docs/fly/F1p-pi-consensus-protocol.md",
        "device": str(DEVICE),
        "qualification": qualification,
        "valid_modes": valid_modes,
        "pen_offsets_pb": offsets,
        "inputs": {
            "lno_neurons": int(lno.ids.size),
            "lno_inhibitory": int((lno.sign < 0).sum()),
            "lno_sides": {s: lno.side.count(s) for s in sorted(set(lno.side))},
        },
    }
    if not any(valid_modes.values()):
        report["status"] = "INVALID"
    else:
        ablated = ablate_offsets(real, phases, offsets)
        per_seed: dict[int, dict[str, dict[str, Any]]] = {}
        for seed in SEEDS:
            graphs = {**graphs_for(real, seed), "offset_ablated": ablated}
            row: dict[str, dict[str, Any]] = {"zero": {"final_accuracy": 0.0}}
            for name in GRAPHS:
                result = search_and_evaluate(Rotating(graphs[name], lno, phases), seed, 200 + seed)
                row[name] = result
                row[f"memory/{name}"] = {"final_accuracy": result["memory"]["memory"]}
                for mode in MODES:
                    row[f"joint/{mode}/{name}"] = {"final_accuracy": result[mode]["joint"]}
            per_seed[seed] = row
            print(
                f"seed {seed}: "
                + "  ".join(
                    f"{name} mem={row[name]['memory']['memory']:.2f} "
                    + " ".join(
                        f"{m}:J={row[name][m]['joint']:+.2f}"
                        f"/A={row[name][m].get('rotation_A', 0):+.2f}"
                        f"/s={row[name][m].get('rotation_shift', 0):.2f}"
                        for m in MODES
                    )
                    for name in GRAPHS
                ),
                flush=True,
            )
        specification = {
            cid: spec
            for cid, spec in SPECIFICATION.items()
            if valid_modes["lno" if cid in ("Q1", "Q2", "Q3") else "pen"]
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
                    for key in (
                        "joint",
                        "memory",
                        "rotation_A",
                        "rotation_shift",
                        "rotation_d_left",
                        "rotation_d_right",
                        "search_joint_best",
                    )
                },
            }
            for name in GRAPHS
        }
        report["contrasts"] = contrasts(per_seed, specification, holm_kinds=("superiority",))
        report["per_seed"] = {
            str(s): {k: v for k, v in row.items() if k in GRAPHS} for s, row in per_seed.items()
        }
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    (OUT / "f1p_pi_signs_report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
