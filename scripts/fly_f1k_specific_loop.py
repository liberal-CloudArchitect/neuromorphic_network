"""F1k: can synaptic tags or excitatory glutamate make the MBON->DAN loop memory-specific?

Protocol docs/fly/F1k-specific-loop-protocol.md (exploratory, CR-013).  F1j tasks,
probes and novel-referenced indices; every variant runs the same 24-point grid of
(eta, recovery, feedback gain, tag decay).  ``--smoke SEED`` prints one seed.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import time

import numpy as np
import torch
from fly_f1a_mb import OUT, Contrast, contrasts
from fly_f1j_positive_loop import TASKS, Episode, episodes, indices, load
from torch import Tensor

from neuromorphic.fly.compartment import (
    CompartmentLearner,
    CompartmentReadout,
    build_readout,
    normalize_columns,
)
from neuromorphic.fly.mb_output import OutputLoop, rewire_feedback
from neuromorphic.fly.mushroom_body import PNKCCircuit

ETAS = (0.1, 0.3, 1.0)
RECOVERIES = (0.001, 0.003)
FEEDBACK_GAINS = (1.0, 3.0)
TAG_DECAYS = (0.99, 0.999)
GRID = [
    {"eta": e, "recovery": r, "gain": 1.0, "feedback_gain": f, "tag_decay": t}
    for e, r, f, t in itertools.product(ETAS, RECOVERIES, FEEDBACK_GAINS, TAG_DECAYS)
]
# variant -> (readout, learner mode, tagged)
VARIANTS: dict[str, tuple[str, str, bool]] = {
    "none": ("real", "none", False),
    "replay_oracle": ("real", "none", False),
    "ideal": ("real", "ideal", False),
    "loop_real": ("real", "loop", False),
    "loop_real_tag": ("real", "loop", True),
    "loop_rewired_tag": ("rewired", "loop", True),
    "loop_glu": ("glutamate_excitatory", "loop", False),
    "loop_glu_tag": ("glutamate_excitatory", "loop", True),
    "loop_positive_tag": ("positive", "loop", True),
}
QUALIFICATION_SEEDS = tuple(range(990, 995))
SEEDS = tuple(range(170, 180))
SPECIFICATION: dict[str, Contrast] = {
    "K1": ("retention/app/loop_real_tag", "retention/app/none", 0.02, "superiority"),
    "K2": ("retention/bias/loop_real", "retention/bias/loop_real_tag", 0.03, "superiority"),
    "K3": ("retention/app/loop_real_tag", "retention/app/loop_rewired_tag", 0.02, "superiority"),
    "K4": ("retention/bias/loop_real", "retention/bias/loop_glu", 0.03, "superiority"),
    "K5": ("retention/bias/loop_glu", "retention/bias/none", 0.02, "equivalence"),
    "K6": ("retention/bias/loop_rewired_tag", "retention/bias/loop_real_tag", 0.03, "superiority"),
    "K7": ("reversal/app/none", "reversal/app/loop_real_tag", 0.02, "superiority"),
}
TAG_CONTRASTS = ("K1", "K3")


def readouts(loop: OutputLoop, kc_mbon: np.ndarray, seed: int) -> dict[str, CompartmentReadout]:
    real = build_readout(loop, kc_mbon)
    gaba_only = torch.as_tensor(
        [-1.0 if t == "gaba" else 1.0 for t in loop.mbon_transmitter], dtype=torch.float32
    )
    return {
        "real": real,
        "rewired": build_readout(
            loop,
            kc_mbon,
            feedback=rewire_feedback(loop.feedback, np.random.default_rng(7000 + seed)),
        ),
        "glutamate_excitatory": dataclasses.replace(real, mbon_sign=gaba_only),
        "positive": dataclasses.replace(
            real,
            feedback=torch.as_tensor(normalize_columns(loop.teach.T), dtype=torch.float32),
            mbon_sign=-torch.ones_like(real.mbon_sign),
        ),
    }


def run(readout: CompartmentReadout, variant: str, episode: Episode) -> dict[str, np.ndarray]:
    """Per-grid-point metrics ``[P]`` after acquisition (``*_acq``) and at the end."""

    def column(key: str) -> Tensor:
        return torch.tensor([p[key] for p in GRID])

    _, mode, tagged = VARIANTS[variant]
    learner = CompartmentLearner(
        readout,
        mode,  # type: ignore[arg-type]
        eta=column("eta"),
        recovery=column("recovery"),
        gain=column("gain"),
        feedback_gain=column("feedback_gain"),
        tag_decay=column("tag_decay") if tagged else None,
    )
    rewards = episode.replay_reward if variant == "replay_oracle" else episode.reward
    n_learning = rewards.shape[0]
    probes = episode.code[n_learning:]

    def probe() -> np.ndarray:
        return torch.stack([learner.value(z) for z in probes], dim=1).numpy()

    out: dict[str, np.ndarray] = {}
    for trial in range(n_learning):
        if trial == episode.acquisition:
            acquired = indices(probe(), dataclasses.replace(episode, reversed=False))
            out.update({f"{k}_acq": v for k, v in acquired.items()})
        learner.step(episode.code[trial], int(rewards[trial]))
    out.update(indices(probe(), episode))
    return out


def run_seed(
    loop: OutputLoop, mb: PNKCCircuit, kc_mbon: np.ndarray, seed: int
) -> dict[str, dict[str, np.ndarray]]:
    data = episodes(mb, seed)
    banks = readouts(loop, kc_mbon, seed)
    return {
        f"{task}/{variant}": run(banks[VARIANTS[variant][0]], variant, data[task])
        for task in TASKS
        for variant in VARIANTS
    }


def margin(results: dict[int, dict[str, dict[str, np.ndarray]]], variant: str) -> float:
    return float(
        np.mean(
            [
                r[f"retention/{variant}"]["app"].mean() - r["retention/none"]["app"].mean()
                for r in results.values()
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=int, help="print one non-formal seed and exit")
    args = parser.parse_args()
    started = time.perf_counter()
    loop, mb, kc_mbon = load()
    if args.smoke is not None:
        result = run_seed(loop, mb, kc_mbon, args.smoke)
        for name, metrics in result.items():
            print(
                f"{name:34s} "
                + " ".join(
                    f"{k} {metrics[k].mean():+.3f}" for k in ("app_acq", "app", "av", "bias")
                ),
                flush=True,
            )
        return
    qualification = {seed: run_seed(loop, mb, kc_mbon, seed) for seed in QUALIFICATION_SEEDS}
    task_margin = margin(qualification, "replay_oracle")
    mechanism_margin = margin(qualification, "loop_positive_tag")
    task_ok, mechanism_ok = task_margin >= 0.02, mechanism_margin >= 0.02
    print(
        f"qualification: replay_oracle {task_margin:+.3f} -> {task_ok}; "
        f"loop_positive_tag {mechanism_margin:+.3f} -> {mechanism_ok}",
        flush=True,
    )
    report: dict[str, object] = {
        "protocol": "docs/fly/F1k-specific-loop-protocol.md",
        "grid": GRID,
        "qualification": {
            "replay_oracle_margin": task_margin,
            "loop_positive_tag_margin": mechanism_margin,
            "task_passed": task_ok,
            "mechanism_passed": mechanism_ok,
        },
    }
    if not task_ok:
        report["status"] = "INVALID"
    else:
        per_seed: dict[int, dict[str, dict[str, object]]] = {}
        grids: dict[str, list[dict[str, np.ndarray]]] = {}
        for seed in SEEDS:
            result = run_seed(loop, mb, kc_mbon, seed)
            row: dict[str, dict[str, object]] = {}
            for name, metrics in result.items():
                task, variant = name.split("/")
                grids.setdefault(name, []).append(metrics)
                for key, value in metrics.items():
                    row[f"{task}/{key}/{variant}"] = {"final_accuracy": float(value.mean())}
            per_seed[seed] = row
            print(
                f"seed {seed}: retention app/bias "
                + "  ".join(
                    f"{v} {row[f'retention/app/{v}']['final_accuracy']:+.3f}"
                    f"/{row[f'retention/bias/{v}']['final_accuracy']:+.3f}"
                    for v in VARIANTS
                ),
                flush=True,
            )
        specification = {
            cid: spec
            for cid, spec in SPECIFICATION.items()
            if mechanism_ok or cid not in TAG_CONTRASTS
        }
        report["status"] = "COMPLETED"
        report["seeds"] = list(SEEDS)
        report["summary"] = {
            name: float(np.mean([float(per_seed[s][name]["final_accuracy"]) for s in SEEDS]))  # type: ignore[arg-type]
            for name in per_seed[SEEDS[0]]
        }
        report["per_grid_point"] = {
            name: {
                key: np.mean([m[key] for m in runs], axis=0).round(4).tolist()
                for key in ("app", "av", "bias", "app_acq")
            }
            for name, runs in grids.items()
        }
        report["contrasts"] = contrasts(per_seed, specification, holm_kinds=("superiority",))
        report["per_seed"] = {str(s): v for s, v in per_seed.items()}
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    (OUT / "f1k_specific_loop_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
