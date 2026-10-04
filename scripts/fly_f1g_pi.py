"""F1g/F1g-v2: path integration with real LNO velocity and ER visual input pathways.

Protocols docs/fly/F1g-detailed-pi-protocol.md (gated out) and
docs/fly/F1g-v2-detailed-pi-protocol.md (control counts x5, seeds 130-139;
``--v2``), exploratory under CR-013.  Binary
weights (>= 5 synapses) are the primary arm, counts secondary.  The cue is an ER
disinhibition landmark; velocity is one-sided LNO-family activity in darkness.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time

import numpy as np
import torch
from fly_f1a_mb import ANNOTATIONS, CONNECTIONS, OUT, Contrast, contrasts
from fly_f1c_ring import N_CUES, circuit
from fly_f1c_v4 import DEVICE, PROTOCOL, TYPES, graphs_for

from neuromorphic.fly.central_complex import (
    CXCircuit,
    InputPathway,
    antisymmetry_index,
    decode_angle,
    extract_input_pathway,
    ideal_input_pathways,
    ideal_rotating_ring,
    pen_offsets,
    ring_embedding,
    simulate_phases,
    template_accuracy,
)
from neuromorphic.fly.flywire import _pandas

LNO_TYPES = ("GLNO", "LNO1", "LNO2", "LNOa")
CANDIDATES, CHUNK = 1000, 100
SEARCH_TRIALS, EVAL_TRIALS = 5, 10
QUALIFICATION_SEED = 990
SEEDS = tuple(range(110, 120))
CUE_CENTERS = -math.pi + (np.arange(N_CUES) + 0.5) * 2 * math.pi / N_CUES
SPECIFICATION: dict[str, Contrast] = {
    "V1": ("real", "zero", 0.0, "superiority"),
    "V2": ("real", "type_preserving", 0.20, "superiority"),
    "V3": ("real", "degree_preserving", 0.20, "superiority"),
}


class Setup:
    """Recurrent and input weights plus cue/velocity patterns for one graph and arm."""

    def __init__(
        self,
        graph: CXCircuit,
        er: InputPathway,
        lno: InputPathway,
        er_angles: np.ndarray,
        ring_angles: np.ndarray,
        *,
        binary: bool,
    ) -> None:
        scale = graph.weight_scale(binary=binary)
        self.graph = graph
        self.ring_angles = ring_angles
        self.epg = graph.indices("EPG")
        self.weights = torch.as_tensor(graph.weights(binary=binary), dtype=torch.float32)
        er_w = er.weights(scale, binary=binary)
        lno_w = lno.weights(scale, binary=binary)
        distance = np.abs(np.angle(np.exp(1j * (er_angles[None, :] - CUE_CENTERS[:, None]))))
        er_rates = (distance > math.pi / 4).astype(np.float64)  # silence ER near the cue
        self.cue = torch.as_tensor(er_rates @ er_w.T, dtype=torch.float32)  # [cues, N]
        side = np.array(lno.side)
        velocity = np.stack(
            [
                np.zeros(graph.n),
                lno_w @ (side == "left").astype(np.float64),
                lno_w @ (side == "right").astype(np.float64),
            ]
        )
        self.velocity = torch.as_tensor(velocity, dtype=torch.float32)  # none, left, right
        self.code = torch.tensor([TYPES.index(t) for t in graph.cell_type])

    def per_neuron(
        self, gains: np.ndarray, biases: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(gains, dtype=torch.float32)[:, self.code],
            torch.as_tensor(biases, dtype=torch.float32)[:, self.code],
        )

    def memory(
        self,
        gains: torch.Tensor,
        biases: torch.Tensor,
        er_gain: torch.Tensor,
        *,
        trials: int,
        generator: torch.Generator,
    ) -> list[float]:
        per = 2 * N_CUES * trials
        n = gains.shape[0]
        cue = self.cue.repeat_interleave(trials, dim=0).repeat(2, 1).repeat(n, 1)
        cue = cue * er_gain.repeat_interleave(per)[:, None]
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
        scores = []
        for i in range(n):
            block = late[i * per : (i + 1) * per]
            templates = block[:half].reshape(N_CUES, trials, -1).mean(dim=1)
            scores.append(template_accuracy(templates, block[half:], labels))
        return scores

    def rotation(
        self,
        gains: torch.Tensor,
        biases: torch.Tensor,
        er_gain: torch.Tensor,
        amplitude: torch.Tensor,
        *,
        trials: int,
        generator: torch.Generator,
    ) -> list[tuple[float, float]]:
        per = 3 * N_CUES * trials
        n = gains.shape[0]
        cue = self.cue.repeat_interleave(trials, dim=0).repeat(3, 1).repeat(n, 1)
        cue = cue * er_gain.repeat_interleave(per)[:, None]
        drive = self.velocity.repeat_interleave(N_CUES * trials, dim=0).repeat(n, 1)
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
        ).cpu()
        theta = decode_angle(activity[:, self.epg], self.ring_angles).reshape(n, 3, N_CUES, trials)
        return [antisymmetry_index(t[0], t[1], t[2]) for t in theta]


def candidates(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5000 + seed)
    return {
        "gain": 2.0 ** rng.uniform(-2, 5, (CANDIDATES, len(TYPES))),
        "bias": rng.uniform(-1.5, 0.5, (CANDIDATES, len(TYPES))),
        "er_gain": 10.0 ** rng.uniform(-1, 1, CANDIDATES),
        "amplitude": 10.0 ** rng.uniform(math.log10(0.05), math.log10(5.0), CANDIDATES),
    }


def search_and_evaluate(setup: Setup, seed: int, eval_seed: int) -> dict[str, object]:
    table = candidates(seed)
    gains, biases = setup.per_neuron(table["gain"], table["bias"])
    er_gain = torch.as_tensor(table["er_gain"], dtype=torch.float32)
    amplitude = torch.as_tensor(table["amplitude"], dtype=torch.float32)
    memory_gen = torch.Generator(device=DEVICE).manual_seed(950)
    rotation_gen = torch.Generator(device=DEVICE).manual_seed(951)
    memory: list[float] = []
    rotation: list[float] = []
    for start in range(0, CANDIDATES, CHUNK):
        part = slice(start, start + CHUNK)
        memory += setup.memory(
            gains[part], biases[part], er_gain[part], trials=SEARCH_TRIALS, generator=memory_gen
        )
        rotation += [
            a
            for a, _ in setup.rotation(
                gains[part],
                biases[part],
                er_gain[part],
                amplitude[part],
                trials=SEARCH_TRIALS,
                generator=rotation_gen,
            )
        ]
    joint = np.minimum(memory, rotation)
    best = int(np.argmax(joint))
    pick = slice(best, best + 1)
    final_memory = setup.memory(
        gains[pick],
        biases[pick],
        er_gain[pick],
        trials=EVAL_TRIALS,
        generator=torch.Generator(device=DEVICE).manual_seed(eval_seed),
    )[0]
    ((final_a, shift),) = setup.rotation(
        gains[pick],
        biases[pick],
        er_gain[pick],
        amplitude[pick],
        trials=EVAL_TRIALS,
        generator=torch.Generator(device=DEVICE).manual_seed(eval_seed + 1),
    )
    used = sorted(set(setup.graph.cell_type))
    return {
        "final_accuracy": final_a,  # antisymmetry A for the shared contrast helper
        "memory_accuracy": final_memory,
        "mean_abs_shift_rad": shift,
        "search_joint_best": float(joint[best]),
        "search_memory_best": float(np.max(memory)),
        "search_rotation_best": float(np.max(rotation)),
        "search_fraction_joint_above_half": float(np.mean(joint >= 0.5)),
        "er_gain": float(table["er_gain"][best]),
        "amplitude": float(table["amplitude"][best]),
        "gain": {t: float(table["gain"][best, TYPES.index(t)]) for t in used},
        "bias": {t: float(table["bias"][best, TYPES.index(t)]) for t in used},
    }


def structure(real: CXCircuit, er: InputPathway, angles: np.ndarray) -> dict[str, object]:
    counts = real.counts()
    epg = real.indices("EPG")
    delta7 = real.indices("Delta7")
    phase = np.exp(1j * angles)
    own = np.angle((counts[np.ix_(delta7, epg)] * phase[None, :]).sum(axis=1))
    difference = np.abs(np.angle(np.exp(1j * (angles[None, :] - own[:, None]))))
    weights = counts[np.ix_(epg, delta7)].T  # Delta7 -> EPG
    bins = np.minimum((difference / (math.pi / 8)).astype(int), 7)
    profile = [float(weights[bins == b].mean()) for b in range(8)]
    _, strength = er.preferred_angles(real, angles)
    return {
        "pen_offsets_rad": pen_offsets(real, angles),
        "delta7_to_epg_by_angle_bin": profile,
        "er_tuning_strength": {
            "median": float(np.median(strength)),
            "fraction_above_0.5": float(np.mean(strength > 0.5)),
            "n_er": int(strength.size),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v2", action="store_true", help="F1g-v2: control counts x5, seeds 130-139"
    )
    args = parser.parse_args()
    seeds = tuple(range(130, 140)) if args.v2 else SEEDS
    factor = 5.0 if args.v2 else 1.0
    started = time.perf_counter()
    control = ideal_rotating_ring()
    control_angles = -math.pi + 2 * math.pi * np.arange(48) / 48
    ideal_er, ideal_lno = ideal_input_pathways(control)
    control = dataclasses.replace(control, edge_count=control.edge_count * factor)
    ideal_er = dataclasses.replace(ideal_er, counts=ideal_er.counts * factor)
    ideal_lno = dataclasses.replace(ideal_lno, counts=ideal_lno.counts * factor)
    control_setup = Setup(
        control,
        ideal_er,
        ideal_lno,
        ideal_er.preferred_angles(control, control_angles)[0],
        control_angles,
        binary=True,
    )
    qualification = search_and_evaluate(control_setup, QUALIFICATION_SEED, 200 + QUALIFICATION_SEED)
    passed = (
        float(qualification["memory_accuracy"]) >= 0.5  # type: ignore[arg-type]
        and float(qualification["final_accuracy"]) >= 0.5  # type: ignore[arg-type]
    )
    print(
        f"qualification: memory {qualification['memory_accuracy']:.3f} "
        f"A {qualification['final_accuracy']:+.3f} -> {passed}",
        flush=True,
    )
    report: dict[str, object] = {
        "protocol": "docs/fly/F1g-v2-detailed-pi-protocol.md"
        if args.v2
        else "docs/fly/F1g-detailed-pi-protocol.md",
        "device": str(DEVICE),
        "qualification": qualification,
        "gate_passed": passed,
    }
    real = circuit()
    angles, _ = ring_embedding(real)
    pd = _pandas()
    annotations = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    connections = pd.read_feather(
        CONNECTIONS, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"]
    )
    er_types = sorted(t for t in annotations.cell_type.dropna().unique() if str(t).startswith("ER"))
    er = extract_input_pathway(annotations, connections, real, source_types=er_types)
    lno = extract_input_pathway(annotations, connections, real, source_types=LNO_TYPES)
    er_angles, _ = er.preferred_angles(real, angles)
    report["structure"] = structure(real, er, angles)
    report["inputs"] = {
        "er_neurons": int(er.ids.size),
        "er_synapses_to_core": float(er.counts.sum()),
        "lno_neurons": int(lno.ids.size),
        "lno_synapses_to_core": float(lno.counts.sum()),
        "lno_sides": {s: lno.side.count(s) for s in sorted(set(lno.side))},
    }
    print(json.dumps(report["structure"], indent=1), flush=True)
    if not passed:
        report["status"] = "INVALID"
    else:
        per_seed: dict[int, dict[str, dict[str, object]]] = {}
        for seed in seeds:
            row: dict[str, dict[str, object]] = {"zero": {"final_accuracy": 0.0}}
            for name, graph in graphs_for(real, seed).items():
                for binary in (True, False):
                    setup = Setup(graph, er, lno, er_angles, angles, binary=binary)
                    key = name if binary else f"counts/{name}"
                    row[key] = search_and_evaluate(setup, seed, 200 + seed)
            per_seed[seed] = row
            print(
                f"seed {seed}: "
                + "  ".join(
                    f"{k} A={v['final_accuracy']:+.2f} mem={v['memory_accuracy']:.2f}"
                    for k, v in row.items()
                    if k != "zero"
                ),
                flush=True,
            )
        names = [k for k in per_seed[seeds[0]] if k != "zero"]
        report["status"] = "COMPLETED"
        report["seeds"] = list(seeds)
        report["summary"] = {
            name: {
                key: float(np.mean([float(per_seed[s][name][key]) for s in seeds]))  # type: ignore[arg-type]
                for key in (
                    "final_accuracy",
                    "memory_accuracy",
                    "search_joint_best",
                    "search_memory_best",
                    "search_rotation_best",
                    "search_fraction_joint_above_half",
                )
            }
            for name in names
        }
        report["contrasts"] = contrasts(per_seed, SPECIFICATION, holm_kinds=("superiority",))
        report["per_seed"] = {str(s): v for s, v in per_seed.items()}
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    name = "f1g_v2_pi_report.json" if args.v2 else "f1g_pi_report.json"
    (OUT / name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
