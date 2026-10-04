"""F1m: CX memory advantage on FlyWire and hemibrain with an activity floor.

Protocol docs/fly/F1m-activity-floor-protocol.md (exploratory, CR-013).  The
F1c-v4 memory task and type-parameter search, plus a validity floor on the late
EPG activity (mean and spatial spread) in search and evaluation.  Each dataset
uses its own spectral ring angles; weights are synapse counts or binary edges
(hemibrain threshold density-matched to FlyWire >= 5).  The unfloored F1c-v4
rule is applied to the same candidates for comparison.  ``--pilot`` runs the
positive control on seed 991 only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
import torch
from fly_f1a_mb import OUT, Contrast, contrasts
from fly_f1c_ring import N_CUES, circuit
from fly_f1c_v4 import DEVICE, PROTOCOL, TYPES, graphs_for
from fly_f1g_pi import CUE_CENTERS
from fly_f1l_pb_phase import HEMIBRAIN, TRANSMITTERS, wrap

from neuromorphic.fly.central_complex import (
    CXCircuit,
    cue_bins,
    extract_cx,
    ideal_ring_circuit,
    ring_embedding,
    simulate_memory,
    template_accuracy,
)
from neuromorphic.fly.hemibrain import circuit_phases, read_tables

FLOOR_RATE, FLOOR_SPREAD, CHANCE = 0.05, 0.05, 1.0 / N_CUES
CANDIDATES, CHUNK = 1000, 100
SEARCH_TRIALS, EVAL_TRIALS = 5, 10
SEARCH_NOISE, QUALIFICATION_SEED, PILOT_SEED = 950, 990, 991
SEEDS = tuple(range(180, 190))
FLYWIRE_THRESHOLD, HEMIBRAIN_MATCHED_THRESHOLD = 5.0, 17.0
CONTROL_FACTOR = 5.0
# arm -> (dataset, binary threshold or None for counts, coordinates)
ARMS: dict[str, tuple[str, float | None, str]] = {
    "flywire/counts": ("flywire", None, "spectral"),
    "flywire/binary": ("flywire", FLYWIRE_THRESHOLD, "spectral"),
    "hemibrain/counts": ("hemibrain", None, "spectral"),
    "hemibrain/binary": ("hemibrain", HEMIBRAIN_MATCHED_THRESHOLD, "spectral"),
    "hemibrain/binary5": ("hemibrain", FLYWIRE_THRESHOLD, "spectral"),
    "hemibrain/pb_binary": ("hemibrain", HEMIBRAIN_MATCHED_THRESHOLD, "pb"),
}
SPECIFICATION: dict[str, Contrast] = {
    f"M{i}": (f"{arm}/real", f"{arm}/type_preserving", 0.10, "superiority")
    for i, arm in enumerate(
        ("flywire/counts", "flywire/binary", "hemibrain/counts", "hemibrain/binary"), start=1
    )
}


def candidates(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11_000 + seed)
    return 2.0 ** rng.uniform(-2, 5, (CANDIDATES, len(TYPES))), rng.uniform(
        -1.5, 0.5, (CANDIDATES, len(TYPES))
    )


def weights_for(graph: CXCircuit, threshold: float | None) -> torch.Tensor:
    matrix = (
        graph.weights() if threshold is None else graph.weights(binary=True, threshold=threshold)
    )
    return torch.as_tensor(matrix, dtype=torch.float32, device=DEVICE)


def cue_inputs(graph: CXCircuit, bins: np.ndarray) -> torch.Tensor:
    cues = torch.zeros(N_CUES, graph.n)
    cues[torch.as_tensor(bins), torch.as_tensor(graph.indices("EPG"))] = 1.0
    return cues


def scores(
    weights: torch.Tensor,
    cues: torch.Tensor,
    epg: np.ndarray,
    gains: torch.Tensor,
    biases: torch.Tensor,
    *,
    trials: int,
    generator: torch.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Template accuracy, mean late EPG rate and mean spatial s.d. per candidate."""

    n = gains.shape[0]
    per = 2 * N_CUES * trials
    batch = cues.repeat_interleave(trials, dim=0).repeat(2, 1).to(DEVICE)
    late, _ = simulate_memory(
        weights,
        batch.repeat(n, 1),
        gains.to(DEVICE).repeat_interleave(per, dim=0),
        biases.to(DEVICE).repeat_interleave(per, dim=0),
        protocol=PROTOCOL,
        generator=generator,
    )
    late = late.cpu()[:, epg]
    labels = torch.arange(N_CUES).repeat_interleave(trials)
    half = N_CUES * trials
    accuracy, rate, spread = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        block = late[i * per : (i + 1) * per]
        templates = block[:half].reshape(N_CUES, trials, -1).mean(dim=1)
        accuracy[i] = template_accuracy(templates, block[half:], labels)
        rate[i] = float(block[half:].mean())
        spread[i] = float(block[half:].std(dim=1).mean())
    return accuracy, rate, spread


def valid(rate: np.ndarray, spread: np.ndarray) -> np.ndarray:
    return (rate >= FLOOR_RATE) & (spread >= FLOOR_SPREAD)


def search_and_evaluate(
    graph: CXCircuit, threshold: float | None, bins: np.ndarray, seed: int, eval_seed: int
) -> dict[str, object]:
    weights = weights_for(graph, threshold)
    cues = cue_inputs(graph, bins)
    epg = graph.indices("EPG")
    code = torch.tensor([TYPES.index(t) for t in graph.cell_type])
    gain_table, bias_table = candidates(seed)
    gains = torch.as_tensor(gain_table, dtype=torch.float32)[:, code]
    biases = torch.as_tensor(bias_table, dtype=torch.float32)[:, code]
    generator = torch.Generator(device=DEVICE).manual_seed(SEARCH_NOISE)
    parts = [
        scores(
            weights,
            cues,
            epg,
            gains[start : start + CHUNK],
            biases[start : start + CHUNK],
            trials=SEARCH_TRIALS,
            generator=generator,
        )
        for start in range(0, CANDIDATES, CHUNK)
    ]
    accuracy, rate, spread = (np.concatenate([p[k] for p in parts]) for k in range(3))
    ok = valid(rate, spread)
    picks = {"floor": int(np.argmax(np.where(ok, accuracy, -np.inf))) if ok.any() else None}
    picks["nofloor"] = int(np.argmax(accuracy))
    out: dict[str, object] = {
        "valid_fraction": float(ok.mean()),
        "search_best_floor": float(accuracy[ok].max()) if ok.any() else None,
        "search_best_nofloor": float(accuracy.max()),
    }
    for rule, index in picks.items():
        if index is None:
            out.update({"memory" if rule == "floor" else f"memory_{rule}": CHANCE})
            out[f"{rule}_status"] = "no_valid_candidate"
            continue
        final, final_rate, final_spread = scores(
            weights,
            cues,
            epg,
            gains[index : index + 1],
            biases[index : index + 1],
            trials=EVAL_TRIALS,
            generator=torch.Generator(device=DEVICE).manual_seed(eval_seed),
        )
        passes = bool(valid(final_rate, final_spread)[0])
        if rule == "floor":
            out["memory"] = float(final[0]) if passes else CHANCE
            out["floor_status"] = "valid" if passes else "invalid_at_evaluation"
        else:
            out["memory_nofloor"] = float(final[0])
            out["nofloor_status"] = "valid" if passes else "silent_or_flat"
        out[f"{rule}_eval_rate"] = float(final_rate[0])
        out[f"{rule}_eval_spread"] = float(final_spread[0])
        out[f"{rule}_candidate"] = index
    return out


def control(seed: int) -> dict[str, dict[str, object]]:
    ring = ideal_ring_circuit()
    ring = dataclasses.replace(ring, edge_count=ring.edge_count * CONTROL_FACTOR)
    bins = np.arange(48) // 6
    out = {}
    for name, threshold in (("counts", None), ("binary", FLYWIRE_THRESHOLD)):
        result = search_and_evaluate(ring, threshold, bins, seed, 300 + seed)
        result["passed"] = float(result["memory"]) >= 0.5  # type: ignore[arg-type]
        out[name] = result
        print(
            f"control {name}: memory {result['memory']:.3f} ({result['floor_status']}) "
            f"nofloor {result['memory_nofloor']:.3f} valid {result['valid_fraction']:.3f}",
            flush=True,
        )
    return out


def datasets() -> dict[str, tuple[CXCircuit, dict[str, np.ndarray]]]:
    """Real circuit and cue bins per coordinate system for each dataset."""

    flywire = circuit()
    annotations, connections, _ = read_tables(HEMIBRAIN, TRANSMITTERS)
    hemibrain = extract_cx(annotations, connections)
    phases = circuit_phases(annotations, hemibrain)[hemibrain.indices("EPG")]
    pb_bins = np.argmin(np.abs(wrap(phases[:, None] - CUE_CENTERS[None, :])), axis=1)
    return {
        "flywire": (flywire, {"spectral": cue_bins(ring_embedding(flywire)[0], N_CUES)}),
        "hemibrain": (
            hemibrain,
            {"spectral": cue_bins(ring_embedding(hemibrain)[0], N_CUES), "pb": pb_bins},
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="positive control on seed 991")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.pilot:
        (OUT / "f1m_pilot.json").write_text(
            json.dumps({"seed": PILOT_SEED, "control": control(PILOT_SEED)}, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    qualification = control(QUALIFICATION_SEED)
    passed = all(bool(v["passed"]) for v in qualification.values())
    report: dict[str, object] = {
        "protocol": "docs/fly/F1m-activity-floor-protocol.md",
        "device": str(DEVICE),
        "floor": {"rate": FLOOR_RATE, "spread": FLOOR_SPREAD},
        "qualification": qualification,
        "gate_passed": passed,
    }
    data = datasets()
    report["structure"] = {
        name: {
            "neurons": graph.n,
            "edges": int(graph.edge_pre.size),
            "edges_kept_binary": {
                arm: int((graph.edge_count >= threshold).sum())
                for arm, (dataset, threshold, _) in ARMS.items()
                if dataset == name and threshold is not None
            },
            "cue_bin_sizes": {
                k: np.bincount(v, minlength=N_CUES).tolist() for k, v in bins.items()
            },
        }
        for name, (graph, bins) in data.items()
    }
    if not passed:
        report["status"] = "INVALID"
    else:
        per_seed: dict[int, dict[str, dict[str, object]]] = {}
        for seed in SEEDS:
            graphs = {name: graphs_for(graph, seed) for name, (graph, _) in data.items()}
            row: dict[str, dict[str, object]] = {}
            for arm, (dataset, threshold, coordinates) in ARMS.items():
                bins = data[dataset][1][coordinates]
                for name, graph in graphs[dataset].items():
                    result = search_and_evaluate(graph, threshold, bins, seed, 300 + seed)
                    result["final_accuracy"] = result["memory"]
                    row[f"{arm}/{name}"] = result
            per_seed[seed] = row
            print(
                f"seed {seed}: "
                + "  ".join(
                    f"{arm} real {row[f'{arm}/real']['memory']:.2f}/"
                    f"tp {row[f'{arm}/type_preserving']['memory']:.2f}/"
                    f"dp {row[f'{arm}/degree_preserving']['memory']:.2f}"
                    for arm in ARMS
                ),
                flush=True,
            )
        keys = (
            "memory",
            "memory_nofloor",
            "valid_fraction",
            "floor_eval_rate",
            "nofloor_eval_rate",
        )
        names = list(per_seed[SEEDS[0]])
        report["status"] = "COMPLETED"
        report["seeds"] = list(SEEDS)
        report["summary"] = {
            name: {
                key: float(
                    np.mean(
                        [
                            float(per_seed[s][name][key])  # type: ignore[arg-type]
                            for s in SEEDS
                            if per_seed[s][name].get(key) is not None
                        ]
                    )
                )
                for key in keys
            }
            for name in names
        }
        report["silent_nofloor_picks"] = {
            name: sum(per_seed[s][name].get("nofloor_status") == "silent_or_flat" for s in SEEDS)
            for name in names
        }
        report["contrasts"] = contrasts(per_seed, SPECIFICATION, holm_kinds=("superiority",))
        report["per_seed"] = {str(s): v for s, v in per_seed.items()}
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    (OUT / "f1m_floor_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
