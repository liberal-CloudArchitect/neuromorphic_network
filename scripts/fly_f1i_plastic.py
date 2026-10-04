"""F1i: plastic ER->EPG visual map, then LNO-driven rotation in darkness.

Protocol docs/fly/F1i-plastic-er-protocol.md (exploratory, CR-013).  Ring
neurons get synthetic von Mises visual tuning unrelated to anatomy; anatomical
binary ER->EPG synapses are depressed by ER-EPG co-activity under a teacher bump
(``learned``), left unlearned (``anatomical``) or bypassed by a direct EPG cue
(``direct``).  Binary recurrent weights; equal-budget random search per arm.
``--pilot`` runs only the positive control on a non-qualification seed.
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
from fly_f1g_pi import CUE_CENTERS, LNO_TYPES

from neuromorphic.fly.central_complex import (
    CXCircuit,
    InputPathway,
    antisymmetry_index,
    coactivity_map,
    cue_bins,
    decode_angle,
    extract_input_pathway,
    ideal_input_pathways,
    ideal_rotating_ring,
    ring_embedding,
    simulate_phases,
    template_accuracy,
    visual_responses,
)
from neuromorphic.fly.flywire import _pandas

KAPPA_VISUAL, KAPPA_BUMP = 2.0, 2.5
CANDIDATES, CHUNK = 1000, 100
SEARCH_TRIALS, EVAL_TRIALS = 5, 10
QUALIFICATION_SEED, PILOT_SEED = 990, 991
SEEDS = tuple(range(140, 150))
CONTROL_FACTOR = 5.0
ARMS = ("learned", "anatomical", "direct")
SPECIFICATION: dict[str, Contrast] = {
    "I1": ("memory/learned/real", "memory/anatomical/real", 0.20, "superiority"),
    "I2": ("memory/learned/real", "memory/learned/type_preserving", 0.20, "superiority"),
    "I3": ("joint/learned/real", "zero", 0.30, "superiority"),
    "I4": ("joint/direct/real", "zero", 0.30, "superiority"),
    "I5": ("joint/learned/real", "joint/learned/type_preserving", 0.10, "superiority"),
}
LEARNED_CONTRASTS = ("I1", "I2", "I3", "I5")


class Plastic:
    """Recurrent weights, per-arm cue banks and LNO velocity inputs for one graph."""

    def __init__(
        self,
        graph: CXCircuit,
        er: InputPathway,
        lno: InputPathway,
        visual_preferred: np.ndarray,
        epg_angles: np.ndarray,
    ) -> None:
        scale = graph.weight_scale(binary=True)
        self.graph = graph
        self.angles = epg_angles
        self.epg = graph.indices("EPG")
        self.weights = torch.as_tensor(graph.weights(binary=True), dtype=torch.float32)
        er_w = er.weights(scale, binary=True)  # [N, ER]
        visual = visual_responses(visual_preferred, CUE_CENTERS, KAPPA_VISUAL)  # [cues, ER]
        rest = er_w.copy()
        rest[self.epg] = 0.0
        self.fixed = torch.as_tensor(visual @ rest.T, dtype=torch.float32)  # [cues, N]
        self.visual_epg = torch.as_tensor(
            visual[:, None, :] * er_w[self.epg][None, :, :], dtype=torch.float32
        )  # [cues, EPG, ER]
        self.coactivity = torch.as_tensor(
            coactivity_map(
                epg_angles, visual_preferred, kappa_visual=KAPPA_VISUAL, kappa_bump=KAPPA_BUMP
            ),
            dtype=torch.float32,
        )  # [EPG, ER]
        direct = torch.zeros(N_CUES, graph.n)
        direct[torch.as_tensor(cue_bins(epg_angles, N_CUES)), torch.as_tensor(self.epg)] = 1.0
        self.direct = direct
        side = np.array(lno.side)
        lno_w = lno.weights(scale, binary=True)
        self.velocity = torch.as_tensor(
            np.stack(
                [
                    np.zeros(graph.n),
                    lno_w @ (side == "left").astype(np.float64),
                    lno_w @ (side == "right").astype(np.float64),
                ]
            ),
            dtype=torch.float32,
        )  # none, left, right
        self.code = torch.tensor([TYPES.index(t) for t in graph.cell_type])

    def cues(self, arm: str, beta: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        """Cue inputs ``[candidates, cues, N]`` for one arm."""

        n = beta.shape[0]
        if arm == "direct":
            bank = self.direct.expand(n, -1, -1).clone()
        else:
            if arm == "learned":
                factor = torch.exp(-beta[:, None, None] * self.coactivity[None])
            else:
                factor = torch.ones(n, *self.coactivity.shape)
            epg_part = torch.einsum("cej,nej->nce", self.visual_epg, factor)
            bank = self.fixed.expand(n, -1, -1).clone()
            bank[:, :, torch.as_tensor(self.epg)] += epg_part
        return bank * gain[:, None, None]

    def per_neuron(
        self, gains: np.ndarray, biases: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(gains, dtype=torch.float32)[:, self.code],
            torch.as_tensor(biases, dtype=torch.float32)[:, self.code],
        )

    def memory(
        self,
        cues: torch.Tensor,
        gains: torch.Tensor,
        biases: torch.Tensor,
        *,
        trials: int,
        generator: torch.Generator,
    ) -> list[tuple[float, float]]:
        """Template accuracy and mean |decoded - cue| angle (radians) per candidate."""

        n = gains.shape[0]
        per = 2 * N_CUES * trials
        cue = cues.repeat_interleave(trials, dim=1).repeat(1, 2, 1).reshape(n * per, -1)
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
        centers = torch.as_tensor(CUE_CENTERS, dtype=torch.float32)[labels]
        half = N_CUES * trials
        out = []
        for i in range(n):
            block = late[i * per : (i + 1) * per]
            templates = block[:half].reshape(N_CUES, trials, -1).mean(dim=1)
            theta = decode_angle(block[half:], self.angles)
            error = torch.remainder(theta - centers + math.pi, 2 * math.pi) - math.pi
            out.append(
                (template_accuracy(templates, block[half:], labels), float(error.abs().mean()))
            )
        return out

    def rotation(
        self,
        cues: torch.Tensor,
        gains: torch.Tensor,
        biases: torch.Tensor,
        amplitude: torch.Tensor,
        *,
        trials: int,
        generator: torch.Generator,
    ) -> list[tuple[float, float]]:
        n = gains.shape[0]
        per = 3 * N_CUES * trials
        cue = cues.repeat_interleave(trials, dim=1).repeat(1, 3, 1).reshape(n * per, -1)
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
        theta = decode_angle(activity[:, self.epg], self.angles).reshape(n, 3, N_CUES, trials)
        return [antisymmetry_index(t[0], t[1], t[2]) for t in theta]


def candidates(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(8000 + seed)
    return {
        "gain": 2.0 ** rng.uniform(-2, 5, (CANDIDATES, len(TYPES))),
        "bias": rng.uniform(-1.5, 0.5, (CANDIDATES, len(TYPES))),
        "cue_gain": 10.0 ** rng.uniform(-1, 1, CANDIDATES),
        "amplitude": 10.0 ** rng.uniform(math.log10(0.05), math.log10(5.0), CANDIDATES),
        "beta": 10.0 ** rng.uniform(math.log10(0.5), math.log10(20.0), CANDIDATES),
    }


def search_and_evaluate(setup: Plastic, arm: str, seed: int, eval_seed: int) -> dict[str, object]:
    table = candidates(seed)
    gains, biases = setup.per_neuron(table["gain"], table["bias"])
    cue_gain = torch.as_tensor(table["cue_gain"], dtype=torch.float32)
    amplitude = torch.as_tensor(table["amplitude"], dtype=torch.float32)
    beta = torch.as_tensor(table["beta"], dtype=torch.float32)
    memory_gen = torch.Generator(device=DEVICE).manual_seed(950)
    rotation_gen = torch.Generator(device=DEVICE).manual_seed(951)
    memory: list[float] = []
    rotation: list[float] = []
    for start in range(0, CANDIDATES, CHUNK):
        part = slice(start, start + CHUNK)
        cues = setup.cues(arm, beta[part], cue_gain[part])
        memory += [
            m
            for m, _ in setup.memory(
                cues, gains[part], biases[part], trials=SEARCH_TRIALS, generator=memory_gen
            )
        ]
        rotation += [
            a
            for a, _ in setup.rotation(
                cues,
                gains[part],
                biases[part],
                amplitude[part],
                trials=SEARCH_TRIALS,
                generator=rotation_gen,
            )
        ]
    joint = np.minimum(memory, rotation)

    def evaluate(index: int) -> tuple[float, float, float, float]:
        pick = slice(index, index + 1)
        cues = setup.cues(arm, beta[pick], cue_gain[pick])
        ((accuracy, anchor),) = setup.memory(
            cues,
            gains[pick],
            biases[pick],
            trials=EVAL_TRIALS,
            generator=torch.Generator(device=DEVICE).manual_seed(eval_seed),
        )
        ((index_a, shift),) = setup.rotation(
            cues,
            gains[pick],
            biases[pick],
            amplitude[pick],
            trials=EVAL_TRIALS,
            generator=torch.Generator(device=DEVICE).manual_seed(eval_seed + 1),
        )
        return accuracy, anchor, index_a, shift

    best_memory = int(np.argmax(memory))
    best_joint = int(np.argmax(joint))
    memory_accuracy, memory_anchor, _, _ = evaluate(best_memory)
    joint_memory, joint_anchor, joint_a, joint_shift = evaluate(best_joint)
    used = sorted(set(setup.graph.cell_type))
    return {
        "memory": memory_accuracy,
        "anchor_error_rad": memory_anchor,
        "joint": min(joint_memory, joint_a),
        "joint_memory": joint_memory,
        "joint_anchor_error_rad": joint_anchor,
        "joint_A": joint_a,
        "joint_mean_abs_shift_rad": joint_shift,
        "search_memory_best": float(np.max(memory)),
        "search_rotation_best": float(np.max(rotation)),
        "search_joint_best": float(joint[best_joint]),
        "search_fraction_joint_above_half": float(np.mean(joint >= 0.5)),
        "chosen_memory": {
            "cue_gain": float(table["cue_gain"][best_memory]),
            "beta": float(table["beta"][best_memory]),
        },
        "chosen_joint": {
            "cue_gain": float(table["cue_gain"][best_joint]),
            "beta": float(table["beta"][best_joint]),
            "amplitude": float(table["amplitude"][best_joint]),
            "gain": {t: float(table["gain"][best_joint, TYPES.index(t)]) for t in used},
            "bias": {t: float(table["bias"][best_joint, TYPES.index(t)]) for t in used},
        },
    }


def control_setup(seed: int) -> Plastic:
    """Ideal rotating ring, all-to-all ideal ER and ideal LNO; all counts x5."""

    ring = ideal_rotating_ring()
    angles = -math.pi + 2 * math.pi * np.arange(48) / 48
    _, lno = ideal_input_pathways(ring)
    counts = np.zeros((ring.n, 48))
    counts[ring.indices("EPG")] = 6.0
    er = InputPathway(
        ids=np.arange(1000, 1048, dtype=np.int64),
        cell_type=("ER",) * 48,
        side=("right",) * 48,
        sign=-np.ones(48),
        counts=counts * CONTROL_FACTOR,
    )
    ring = dataclasses.replace(ring, edge_count=ring.edge_count * CONTROL_FACTOR)
    lno = dataclasses.replace(lno, counts=lno.counts * CONTROL_FACTOR)
    preferred = np.random.default_rng(6000 + seed).uniform(-math.pi, math.pi, 48)
    return Plastic(ring, er, lno, preferred, angles)


def qualify(seed: int) -> dict[str, dict[str, object]]:
    setup = control_setup(seed)
    out = {}
    for arm in ARMS:
        result = search_and_evaluate(setup, arm, seed, 200 + seed)
        result["passed"] = (
            float(result["joint_memory"]) >= 0.5  # type: ignore[arg-type]
            and float(result["joint_A"]) >= 0.5  # type: ignore[arg-type]
        )
        out[arm] = result
        print(
            f"control {arm}: memory {result['memory']:.3f} joint_memory "
            f"{result['joint_memory']:.3f} A {result['joint_A']:+.3f} -> {result['passed']}",
            flush=True,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="positive control on seed 991 only")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.pilot:
        pilot = qualify(PILOT_SEED)
        (OUT / "f1i_pilot.json").write_text(
            json.dumps({"seed": PILOT_SEED, "control": pilot}, indent=2) + "\n", encoding="utf-8"
        )
        return
    qualification = qualify(QUALIFICATION_SEED)
    valid = {arm: bool(qualification[arm]["passed"]) for arm in ("learned", "direct")}
    report: dict[str, object] = {
        "protocol": "docs/fly/F1i-plastic-er-protocol.md",
        "device": str(DEVICE),
        "qualification": qualification,
        "valid_arms": valid,
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
    mask = er.counts[real.indices("EPG")] >= 5
    report["inputs"] = {
        "er_neurons": int(er.ids.size),
        "er_with_epg_targets": int(mask.any(axis=0).sum()),
        "er_epg_mask_density": float(mask[:, mask.any(axis=0)].mean()),
        "lno_neurons": int(lno.ids.size),
    }
    if not any(valid.values()):
        report["status"] = "INVALID"
    else:
        per_seed: dict[int, dict[str, dict[str, object]]] = {}
        for seed in SEEDS:
            preferred = np.random.default_rng(6000 + seed).uniform(-math.pi, math.pi, er.ids.size)
            row: dict[str, dict[str, object]] = {"zero": {"final_accuracy": 0.0}}
            for name, graph in graphs_for(real, seed).items():
                setup = Plastic(graph, er, lno, preferred, angles)
                for arm in ARMS:
                    if arm == "anatomical" and name != "real":
                        continue
                    result = search_and_evaluate(setup, arm, seed, 200 + seed)
                    row[f"{arm}/{name}"] = result
                    row[f"memory/{arm}/{name}"] = {"final_accuracy": result["memory"]}
                    row[f"joint/{arm}/{name}"] = {"final_accuracy": result["joint"]}
            per_seed[seed] = row
            print(
                f"seed {seed}: "
                + "  ".join(
                    f"{k} mem={v['memory']:.2f} J={v['joint']:.2f}"
                    for k, v in row.items()
                    if "memory" in v and "joint" in v
                ),
                flush=True,
            )
        names = [k for k in per_seed[SEEDS[0]] if k.count("/") == 1]
        keys = (
            "memory",
            "anchor_error_rad",
            "joint",
            "joint_memory",
            "joint_A",
            "search_memory_best",
            "search_rotation_best",
            "search_joint_best",
            "search_fraction_joint_above_half",
        )
        report["summary"] = {
            name: {
                key: float(np.mean([float(per_seed[s][name][key]) for s in SEEDS]))  # type: ignore[arg-type]
                for key in keys
            }
            for name in names
        }
        specification = {
            cid: spec
            for cid, spec in SPECIFICATION.items()
            if valid["learned" if cid in LEARNED_CONTRASTS else "direct"]
        }
        report["status"] = "COMPLETED"
        report["seeds"] = list(SEEDS)
        report["contrasts"] = contrasts(per_seed, specification, holm_kinds=("superiority",))
        report["per_seed"] = {
            str(s): {k: v for k, v in row.items() if k.count("/") == 1}
            for s, row in per_seed.items()
        }
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    (OUT / "f1i_plastic_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
