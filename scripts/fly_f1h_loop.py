"""F1h: do real MBON->DAN loops compute useful compartment teaching signals?

Protocol docs/fly/F1h-compartment-loop-protocol.md (exploratory, CR-013).  F1b
task and KC code; every variant tunes a 36-point grid on seeds 900-904
(schedule-averaged accuracy); seeds 120-129 test.
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
import torch
from fly_f1a_mb import ANNOTATIONS, CONNECTIONS, OUT, Contrast, circuit, contrasts
from fly_f1b_surprise import SCHEDULES, features

from neuromorphic.fly.compartment import CompartmentLearner, CompartmentReadout, build_readout
from neuromorphic.fly.flywire import _pandas
from neuromorphic.fly.mb_output import extract_output_loop, rewire_feedback
from neuromorphic.fly.valence import ValenceSchedule

ETAS = (0.03, 0.1, 0.3, 1.0)
RECOVERIES = (0.0, 0.003, 0.03)
GAINS = (0.3, 1.0, 3.0)
FEEDBACK_GAINS = (1.0, 3.0, 10.0)
CALIBRATION_SEEDS = tuple(range(900, 905))
SEEDS = tuple(range(120, 130))
VARIANTS = ("none", "ideal", "loop_real", "loop_rewired")
SPECIFICATION: dict[str, Contrast] = {
    "W1": ("volatile/loop_real", "volatile/none", 0.05, "superiority"),
    "W2": ("volatile/loop_real", "volatile/loop_rewired", 0.03, "superiority"),
    "W3": ("volatile/loop_real", "volatile/ideal", 0.05, "noninferiority"),
}


def grid(variant: str) -> list[dict[str, float]]:
    third = (
        [{"gain": g, "feedback_gain": 0.0} for g in GAINS]
        if variant in {"none", "ideal"}
        else [{"gain": 1.0, "feedback_gain": f} for f in FEEDBACK_GAINS]
    )
    return [
        {"eta": e, "recovery": r, **t} for e, r, t in itertools.product(ETAS, RECOVERIES, third)
    ]


def run(
    readout: CompartmentReadout,
    variant: str,
    params: list[dict[str, float]],
    code: torch.Tensor,
    schedule: ValenceSchedule,
) -> np.ndarray:
    """Correct-decision matrix ``[P, trials]`` for a batch of parameter settings."""

    def column(key: str) -> torch.Tensor:
        return torch.tensor([p[key] for p in params])

    learner = CompartmentLearner(
        readout,
        "loop" if variant.startswith("loop") else variant,  # type: ignore[arg-type]
        eta=column("eta"),
        recovery=column("recovery"),
        gain=column("gain"),
        feedback_gain=column("feedback_gain"),
    )
    correct = np.empty((len(params), code.shape[0]), dtype=bool)
    for trial in range(code.shape[0]):
        value = learner.step(code[trial], int(schedule.reward[trial]))
        correct[:, trial] = ((value > 0) == (schedule.value[trial] > 0)).numpy()
    return correct


def main() -> None:
    started = time.perf_counter()
    pd = _pandas()
    annotations = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    connections = pd.read_feather(
        CONNECTIONS, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"]
    )
    loop = extract_output_loop(annotations, connections)
    mb = circuit()
    edges = connections[
        connections.pre_pt_root_id.isin(mb.kc_ids) & connections.post_pt_root_id.isin(loop.mbon_ids)
    ]
    pairs = edges.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False).syn_count.sum()
    kc_mbon = np.zeros((loop.mbon_ids.size, mb.kc_ids.size))
    kc_mbon[
        np.searchsorted(loop.mbon_ids, pairs.post_pt_root_id.to_numpy()),
        np.searchsorted(mb.kc_ids, pairs.pre_pt_root_id.to_numpy()),
    ] = pairs.syn_count.to_numpy()
    real_readout = build_readout(loop, kc_mbon)

    def readout_for(variant: str, seed: int) -> CompartmentReadout:
        if variant != "loop_rewired":
            return real_readout
        rewired = rewire_feedback(loop.feedback, np.random.default_rng(7000 + seed))
        return build_readout(loop, kc_mbon, feedback=rewired)

    chosen: dict[str, dict[str, float]] = {}
    calibration_scores: dict[str, list[float]] = {}
    calibration_data = {seed: features(mb, seed) for seed in CALIBRATION_SEEDS}
    for variant in VARIANTS:
        params = grid(variant)
        totals = np.zeros(len(params))
        for seed, data in calibration_data.items():
            readout = readout_for(variant, seed)
            for kind in SCHEDULES:
                code, _, schedule = data[kind]
                totals += run(readout, variant, params, code, schedule).mean(axis=1)  # type: ignore[arg-type]
        scores = totals / (len(CALIBRATION_SEEDS) * len(SCHEDULES))
        best = int(np.argmax(scores))
        chosen[variant] = params[best]
        calibration_scores[variant] = scores.round(4).tolist()
        print(f"{variant}: chose {params[best]} (calibration {scores[best]:.3f})", flush=True)

    per_seed: dict[int, dict[str, dict[str, object]]] = {}
    for seed in SEEDS:
        data = features(mb, seed)
        row: dict[str, dict[str, object]] = {}
        for variant in VARIANTS:
            readout = readout_for(variant, seed)
            for kind in SCHEDULES:
                code, _, schedule = data[kind]
                correct = run(readout, variant, [chosen[variant]], code, schedule)[0]  # type: ignore[arg-type]
                after = schedule.after_reversal  # type: ignore[attr-defined]
                row[f"{kind}/{variant}"] = {
                    "final_accuracy": float(correct.mean()),
                    "recovery_accuracy": float(correct[after].mean())
                    if after.any()
                    else float("nan"),
                }
        per_seed[seed] = row
        print(
            f"seed {seed}: "
            + "  ".join(f"{v} {row[f'volatile/{v}']['final_accuracy']:.3f}" for v in VARIANTS),
            flush=True,
        )
    names = list(per_seed[SEEDS[0]])
    summary = {
        name: float(np.mean([float(per_seed[s][name]["final_accuracy"]) for s in SEEDS]))  # type: ignore[arg-type]
        for name in names
    }
    pam = np.array([t.startswith("PAM") for t in loop.dan_type])
    report = {
        "protocol": "docs/fly/F1h-compartment-loop-protocol.md",
        "structure": {
            "mbons": int(loop.mbon_ids.size),
            "kc": int(mb.kc_ids.size),
            "kc_mbon_pairs": int((kc_mbon > 0).sum()),
            "valence_counts": {
                str(v): int((real_readout.valence == v).sum()) for v in (-1.0, 0.0, 1.0)
            },
            "pam_dans": int(pam.sum()),
        },
        "chosen": chosen,
        "calibration_scores": calibration_scores,
        "seeds": list(SEEDS),
        "summary": summary,
        "contrasts": contrasts(per_seed, SPECIFICATION, holm_kinds=("superiority",)),
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (OUT / "f1h_loop_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
