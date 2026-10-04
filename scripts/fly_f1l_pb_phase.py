"""F1l: path integration on the hemibrain CX with PB-glomerulus phases.

Protocol docs/fly/F1l-pb-phase-protocol.md (exploratory, CR-013).  Structure
(S1-S4, deterministic) on hemibrain v1.2, then dynamics: a direct EPG cue on the
eight PB phases, one-sided LNO velocity, decoding with PB phases and a rotation
score gated by shift magnitude.  ``--pilot`` runs the positive controls on seed
991 only; ``--structure`` prints S1-S4 only.
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
from fly_f1a_mb import OUT, ROOT, Contrast, contrasts
from fly_f1c_ring import N_CUES
from fly_f1c_v4 import DEVICE, PROTOCOL, TYPES
from fly_f1g_pi import CUE_CENTERS, LNO_TYPES

from neuromorphic.fly.central_complex import (
    CORE_TYPES,
    CXCircuit,
    InputPathway,
    antisymmetry_index,
    decode_angle,
    degree_preserving_directed,
    extract_cx,
    extract_input_pathway,
    ideal_input_pathways,
    ideal_rotating_ring,
    pen_offsets,
    ring_embedding,
    simulate_phases,
    template_accuracy,
    type_preserving,
)
from neuromorphic.fly.hemibrain import N_PHASES, circuit_phases, read_tables

HEMIBRAIN = ROOT / "data/fly/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz"
TRANSMITTERS = ROOT / "data/fly/hemibrain/v1.2/hemibrain-v1.2-body-mean-neurotransmitters.feather"
PEN_TYPES = ("PEN_a(PEN1)", "PEN_b(PEN2)")
FULL_SHIFT = 0.2  # radians; rotation credit is scaled down below this mean |shift|
CANDIDATES, CHUNK = 1000, 100
SEARCH_TRIALS, EVAL_TRIALS = 5, 10
QUALIFICATION_SEED, PILOT_SEED = 990, 991
SEEDS = tuple(range(160, 170))
CONTROL_FACTOR = 5.0
SPECIFICATION: dict[str, Contrast] = {
    "P1": ("memory/real", "memory/type_preserving", 0.20, "superiority"),
    "P2": ("joint/real", "zero", 0.30, "superiority"),
    "P3": ("joint/real", "joint/offset_ablated", 0.20, "superiority"),
    "P4": ("joint/real", "joint/type_preserving", 0.20, "superiority"),
}
assert N_CUES == N_PHASES


def wrap(x: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * x))


# --------------------------------------------------------------------------- structure


def eb_similarity_profile(
    annotations: Any, roi: Any, circuit: CXCircuit, phases: np.ndarray
) -> dict[str, object]:
    """S1: EB-connectivity cosine similarity of EPG pairs by PB phase difference."""

    epg = circuit.indices("EPG")
    ids = circuit.ids[epg]
    eb = roi[roi.roi == "EB"]
    partners = np.union1d(eb.pre_pt_root_id.to_numpy(), eb.post_pt_root_id.to_numpy())
    profile = np.zeros((ids.size, 2 * partners.size))
    for column, (mine, other) in enumerate(
        (("pre_pt_root_id", "post_pt_root_id"), ("post_pt_root_id", "pre_pt_root_id"))
    ):
        rows = eb[eb[mine].isin(ids)]
        profile[
            np.searchsorted(ids, rows[mine].to_numpy()),
            column * partners.size + np.searchsorted(partners, rows[other].to_numpy()),
        ] = rows.syn_count.to_numpy()
    profile /= np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-12)
    similarity = profile @ profile.T
    step = np.rint(
        np.abs(wrap(phases[epg][:, None] - phases[epg][None, :])) / (2 * math.pi / N_PHASES)
    ).astype(int)
    upper = np.triu(np.ones_like(similarity, dtype=bool), 1)
    by_step = [float(similarity[upper & (step == d)].mean()) for d in range(N_PHASES // 2 + 1)]
    # Cross-side rule check: pair L_i with R_j when i + j == 9 + k.
    lookup = dict(zip(annotations.root_id, annotations.instance, strict=True))
    glomeruli = [str(lookup[int(b)]).rsplit("_", 1)[1] for b in ids]
    side = np.array([g[0] for g in glomeruli])
    index = np.array([int(g[1]) for g in glomeruli])
    rule = {}
    for k in range(-4, 4):
        mask = (
            (side[:, None] == "L")
            & (side[None, :] == "R")
            & (index[:, None] + index[None, :] == 9 + k)
        )
        rule[k] = float(similarity[mask].mean()) if mask.any() else float("nan")
    best_rule = max((v, k) for k, v in rule.items() if not math.isnan(v))[1]
    return {
        "similarity_by_phase_step": by_step,
        "cross_side_rule_similarity": rule,
        "best_cross_side_rule_k": best_rule,
        "passed": by_step[0] == max(by_step) and by_step[-1] == min(by_step) and best_rule == 0,
    }


def offsets_by_group(circuit: CXCircuit, phases: np.ndarray) -> dict[str, dict[str, float]]:
    """S2: per (PEN type, side), circular mean of EPG-target phase minus own PB phase."""

    counts = circuit.counts()
    epg = circuit.indices("EPG")
    out: dict[str, dict[str, float]] = {}
    for kind in PEN_TYPES:
        for side in ("left", "right"):
            members = [
                i
                for i, (t, s) in enumerate(zip(circuit.cell_type, circuit.side, strict=True))
                if t == kind and s == side and not math.isnan(phases[i])
            ]
            values = []
            for i in members:
                target = (counts[epg, i] * np.exp(1j * phases[epg])).sum()
                if abs(target) > 0:
                    values.append(np.angle(target) - phases[i])
            if not values:
                out[f"{kind}|{side}"] = {"offset_deg": math.nan, "consistency": math.nan, "n": 0}
                continue
            mean = np.exp(1j * np.array(values)).mean()
            out[f"{kind}|{side}"] = {
                "offset_deg": float(np.degrees(np.angle(mean))),
                "consistency": float(abs(mean)),
                "n": len(values),
            }
    return out


def offsets_pass(offsets: dict[str, dict[str, float]]) -> bool:
    ok = True
    for kind in PEN_TYPES:
        left = offsets[f"{kind}|left"]["offset_deg"]
        right = offsets[f"{kind}|right"]["offset_deg"]
        ok &= left * right < 0 and all(22.5 <= abs(x) <= 67.5 for x in (left, right))
    return bool(ok)


def structure(annotations: Any, roi: Any, real: CXCircuit, phases: np.ndarray) -> dict[str, object]:
    epg = real.indices("EPG")
    spectral, eigenvalues = ring_embedding(real)
    alignment = max(
        float(abs(np.exp(1j * (spectral - s * phases[epg])).mean())) for s in (1.0, -1.0)
    )
    offsets = offsets_by_group(real, phases)
    counts = real.counts()
    delta7 = real.indices("Delta7")
    own = np.angle((counts[np.ix_(delta7, epg)] * np.exp(1j * phases[epg])[None, :]).sum(axis=1))
    difference = np.abs(wrap(phases[epg][None, :] - own[:, None]))
    weights = counts[np.ix_(epg, delta7)].T
    bins = np.minimum((difference / (math.pi / 8)).astype(int), 7)
    return {
        "s1_eb_similarity": eb_similarity_profile(annotations, roi, real, phases),
        "s2_pen_offsets_pb": offsets,
        "s2_passed": offsets_pass(offsets),
        "s3_spectral_vs_pb_alignment": alignment,
        "s3_spectral_eigenvalues": eigenvalues.round(4).tolist(),
        "s3_pen_offsets_spectral_deg": {
            k: float(np.degrees(v[0])) for k, v in pen_offsets(real, spectral).items()
        },
        "s4_delta7_to_epg_by_angle_bin": [float(weights[bins == b].mean()) for b in range(8)],
        "counts": {t: int(real.indices(t).size) for t in CORE_TYPES},
    }


# --------------------------------------------------------------------------- graphs


def ablate_offsets(
    circuit: CXCircuit, angles: np.ndarray, offsets: dict[str, dict[str, float]]
) -> CXCircuit:
    """Move every PEN->EPG edge back by its group's mean offset (nearest EPG angle).

    Among EPGs at the new angle the one with the same within-angle rank is chosen,
    so per-EPG input counts change only through the remapping; duplicates merge.
    """

    epg = circuit.indices("EPG")
    epg_angles = angles[epg]
    levels = np.unique(np.round(epg_angles, 6))
    groups = {
        float(level): epg[np.round(epg_angles, 6) == level] for level in levels
    }  # angle -> EPG indices (sorted)
    level_of = {int(e): float(np.round(angles[e], 6)) for e in epg}
    is_epg = np.zeros(circuit.n, dtype=bool)
    is_epg[epg] = True
    shift = np.zeros(circuit.n)
    for i, (t, s) in enumerate(zip(circuit.cell_type, circuit.side, strict=True)):
        key = f"{t}|{s}"
        if key in offsets and math.isfinite(offsets[key]["offset_deg"]):
            shift[i] = math.radians(offsets[key]["offset_deg"])
    pre, post, count = [], [], []
    for p, q, c in zip(
        circuit.edge_pre.tolist(),
        circuit.edge_post.tolist(),
        circuit.edge_count.tolist(),
        strict=True,
    ):
        if is_epg[q] and shift[p] != 0.0:
            group = groups[level_of[q]]
            rank = int(np.flatnonzero(group == q)[0])
            target_angle = float(wrap(np.array(angles[q] - shift[p])))
            nearest = levels[np.argmin(np.abs(wrap(levels - target_angle)))]
            destination = groups[float(nearest)]
            q = int(destination[rank % destination.size])
        pre.append(p)
        post.append(q)
        count.append(c)
    merged: dict[tuple[int, int], float] = {}
    for p, q, c in zip(pre, post, count, strict=True):
        merged[(p, q)] = merged.get((p, q), 0.0) + c
    keys = sorted(merged)
    return dataclasses.replace(
        circuit,
        edge_pre=np.array([k[0] for k in keys], dtype=np.int64),
        edge_post=np.array([k[1] for k in keys], dtype=np.int64),
        edge_count=np.array([merged[k] for k in keys]),
        topology="offset_ablated",
    )


# --------------------------------------------------------------------------- dynamics


class Phased:
    """Direct EPG cue on phase bins, LNO velocity, decoding with given EPG angles."""

    def __init__(self, graph: CXCircuit, lno: InputPathway, angles: np.ndarray) -> None:
        scale = graph.weight_scale(binary=True)
        self.graph = graph
        self.epg = graph.indices("EPG")
        self.angles = angles[self.epg]
        self.weights = torch.as_tensor(graph.weights(binary=True), dtype=torch.float32)
        bins = np.argmin(
            np.abs(wrap(self.angles[:, None] - CUE_CENTERS[None, :])), axis=1
        )  # nearest cue center
        cue = torch.zeros(N_CUES, graph.n)
        cue[torch.as_tensor(bins), torch.as_tensor(self.epg)] = 1.0
        self.cue = cue
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
        )
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
        cue_gain: torch.Tensor,
        *,
        trials: int,
        generator: torch.Generator,
    ) -> list[tuple[float, float]]:
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
        gains: torch.Tensor,
        biases: torch.Tensor,
        cue_gain: torch.Tensor,
        amplitude: torch.Tensor,
        *,
        trials: int,
        generator: torch.Generator,
    ) -> list[tuple[float, float]]:
        n = gains.shape[0]
        per = 3 * N_CUES * trials
        cue = self.cue.repeat_interleave(trials, dim=0).repeat(3, 1).repeat(n, 1)
        cue = cue * cue_gain.repeat_interleave(per)[:, None]
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


def gated(index: float, shift: float) -> float:
    """Antisymmetry credited only in proportion to shift magnitude (full at FULL_SHIFT)."""

    return index * min(1.0, shift / FULL_SHIFT)


def candidates(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(9000 + seed)
    return {
        "gain": 2.0 ** rng.uniform(-2, 5, (CANDIDATES, len(TYPES))),
        "bias": rng.uniform(-1.5, 0.5, (CANDIDATES, len(TYPES))),
        "cue_gain": 10.0 ** rng.uniform(-1, 1, CANDIDATES),
        "amplitude": 10.0 ** rng.uniform(math.log10(0.05), math.log10(5.0), CANDIDATES),
    }


def search_and_evaluate(setup: Phased, seed: int, eval_seed: int) -> dict[str, object]:
    table = candidates(seed)
    gains, biases = setup.per_neuron(table["gain"], table["bias"])
    cue_gain = torch.as_tensor(table["cue_gain"], dtype=torch.float32)
    amplitude = torch.as_tensor(table["amplitude"], dtype=torch.float32)
    memory_gen = torch.Generator(device=DEVICE).manual_seed(950)
    rotation_gen = torch.Generator(device=DEVICE).manual_seed(951)
    memory: list[float] = []
    rotation: list[float] = []
    for start in range(0, CANDIDATES, CHUNK):
        part = slice(start, start + CHUNK)
        memory += [
            m
            for m, _ in setup.memory(
                gains[part],
                biases[part],
                cue_gain[part],
                trials=SEARCH_TRIALS,
                generator=memory_gen,
            )
        ]
        rotation += [
            gated(a, s)
            for a, s in setup.rotation(
                gains[part],
                biases[part],
                cue_gain[part],
                amplitude[part],
                trials=SEARCH_TRIALS,
                generator=rotation_gen,
            )
        ]
    joint = np.minimum(memory, rotation)

    def evaluate(index: int) -> tuple[float, float, float, float]:
        pick = slice(index, index + 1)
        ((accuracy, anchor),) = setup.memory(
            gains[pick],
            biases[pick],
            cue_gain[pick],
            trials=EVAL_TRIALS,
            generator=torch.Generator(device=DEVICE).manual_seed(eval_seed),
        )
        ((index_a, shift),) = setup.rotation(
            gains[pick],
            biases[pick],
            cue_gain[pick],
            amplitude[pick],
            trials=EVAL_TRIALS,
            generator=torch.Generator(device=DEVICE).manual_seed(eval_seed + 1),
        )
        return accuracy, anchor, index_a, shift

    best_memory, best_joint = int(np.argmax(memory)), int(np.argmax(joint))
    memory_accuracy, memory_anchor, _, _ = evaluate(best_memory)
    joint_memory, _, joint_a, joint_shift = evaluate(best_joint)
    used = sorted(set(setup.graph.cell_type))
    return {
        "memory": memory_accuracy,
        "anchor_error_rad": memory_anchor,
        "joint": min(joint_memory, gated(joint_a, joint_shift)),
        "joint_memory": joint_memory,
        "joint_A": joint_a,
        "joint_mean_abs_shift_rad": joint_shift,
        "search_memory_best": float(np.max(memory)),
        "search_rotation_best": float(np.max(rotation)),
        "search_joint_best": float(joint[best_joint]),
        "search_fraction_joint_above_half": float(np.mean(joint >= 0.5)),
        "chosen_joint": {
            "cue_gain": float(table["cue_gain"][best_joint]),
            "amplitude": float(table["amplitude"][best_joint]),
            "gain": {t: float(table["gain"][best_joint, TYPES.index(t)]) for t in used},
            "bias": {t: float(table["bias"][best_joint, TYPES.index(t)]) for t in used},
        },
    }


def control(seed: int) -> dict[str, dict[str, object]]:
    """Ideal rotating ring (counts x5) with ideal LNO; intact and offset-ablated."""

    ring = ideal_rotating_ring()
    angles = np.full(ring.n, np.nan)
    ring_angles = -math.pi + 2 * math.pi * np.arange(48) / 48
    angles[ring.indices("EPG")] = ring_angles
    pen = np.array([t.startswith("PEN") for t in ring.cell_type])
    angles[pen] = np.tile(ring_angles, 2)  # each PEN_i reads EPG_i
    _, lno = ideal_input_pathways(ring)
    ring = dataclasses.replace(ring, edge_count=ring.edge_count * CONTROL_FACTOR)
    lno = dataclasses.replace(lno, counts=lno.counts * CONTROL_FACTOR)
    offsets = {
        f"PEN_a(PEN1)|{side}": {"offset_deg": float(np.degrees(sign * 2 * math.pi / 48))}
        for side, sign in (("left", 1.0), ("right", -1.0))
    }
    out = {}
    for name, graph in (
        ("intact", ring),
        ("offset_ablated", ablate_offsets(ring, angles, offsets)),
    ):
        result = search_and_evaluate(Phased(graph, lno, angles), seed, 200 + seed)
        result["passed"] = float(result["joint_memory"]) >= 0.5 and float(result["joint"]) >= 0.5  # type: ignore[arg-type]
        out[name] = result
        print(
            f"control {name}: memory {result['memory']:.3f} joint {result['joint']:+.3f} "
            f"A {result['joint_A']:+.3f} shift {result['joint_mean_abs_shift_rad']:.3f}",
            flush=True,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="positive controls on seed 991")
    parser.add_argument("--structure", action="store_true", help="print S1-S4 and exit")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.pilot:
        (OUT / "f1l_pilot.json").write_text(
            json.dumps({"seed": PILOT_SEED, "control": control(PILOT_SEED)}, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    annotations, connections, roi = read_tables(HEMIBRAIN, TRANSMITTERS)
    real = extract_cx(annotations, connections)
    phases = circuit_phases(annotations, real)
    report: dict[str, object] = {"protocol": "docs/fly/F1l-pb-phase-protocol.md"}
    report["structure"] = structure(annotations, roi, real, phases)
    print(json.dumps(report["structure"], indent=1), flush=True)
    if args.structure:
        return
    qualification = control(QUALIFICATION_SEED)
    report["qualification"] = qualification
    passed = bool(qualification["intact"]["passed"])
    lno = extract_input_pathway(annotations, connections, real, source_types=LNO_TYPES)
    report["inputs"] = {
        "lno_neurons": int(lno.ids.size),
        "lno_sides": {s: lno.side.count(s) for s in sorted(set(lno.side))},
        "lno_synapses_to_core": float(lno.counts.sum()),
    }
    if not passed:
        report["status"] = "INVALID"
    else:
        offsets = report["structure"]["s2_pen_offsets_pb"]  # type: ignore[index]
        ablated = ablate_offsets(real, phases, offsets)
        per_seed: dict[int, dict[str, dict[str, object]]] = {}
        for seed in SEEDS:
            rng = np.random.default_rng(10_000 + seed)
            graphs = {
                "real": real,
                "type_preserving": type_preserving(real, rng),
                "degree_preserving": degree_preserving_directed(real, rng),
                "offset_ablated": ablated,
            }
            row: dict[str, dict[str, object]] = {"zero": {"final_accuracy": 0.0}}
            for name, graph in graphs.items():
                result = search_and_evaluate(Phased(graph, lno, phases), seed, 200 + seed)
                row[name] = result
                row[f"memory/{name}"] = {"final_accuracy": result["memory"]}
                row[f"joint/{name}"] = {"final_accuracy": result["joint"]}
            per_seed[seed] = row
            print(
                f"seed {seed}: "
                + "  ".join(
                    f"{k} mem={v['memory']:.2f} J={v['joint']:+.2f} A={v['joint_A']:+.2f} "
                    f"shift={v['joint_mean_abs_shift_rad']:.2f}"
                    for k, v in row.items()
                    if "joint_A" in v
                ),
                flush=True,
            )
        specification = {
            cid: spec
            for cid, spec in SPECIFICATION.items()
            if cid != "P3" or report["structure"]["s2_passed"]  # type: ignore[index]
        }
        names = [k for k in per_seed[SEEDS[0]] if "/" not in k and k != "zero"]
        keys = (
            "memory",
            "anchor_error_rad",
            "joint",
            "joint_memory",
            "joint_A",
            "joint_mean_abs_shift_rad",
            "search_memory_best",
            "search_joint_best",
            "search_fraction_joint_above_half",
        )
        report["status"] = "COMPLETED"
        report["seeds"] = list(SEEDS)
        report["summary"] = {
            name: {
                key: float(np.mean([float(per_seed[s][name][key]) for s in SEEDS]))  # type: ignore[arg-type]
                for key in keys
            }
            for name in names
        }
        report["contrasts"] = contrasts(per_seed, specification, holm_kinds=("superiority",))
        report["per_seed"] = {
            str(s): {k: v for k, v in row.items() if "/" not in k and k != "zero"}
            for s, row in per_seed.items()
        }
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    (OUT / "f1l_pb_phase_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
