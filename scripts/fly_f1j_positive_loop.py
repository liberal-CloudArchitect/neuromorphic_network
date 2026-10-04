"""F1j: does the positive MBON->DAN loop consolidate memories or bias all odors?

Protocol docs/fly/F1j-positive-loop-protocol.md (exploratory, CR-013).  F1h
compartment readout; acquisition, then unreinforced retention or reversal, then
learning-free probes.  Memory indices are referenced to never-reinforced odors so
a global valence bias cannot pose as memory.  Every variant runs the same 12-point
grid of (eta, recovery, feedback gain) - no per-variant calibration (F1h lesson).
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import time

import numpy as np
import torch
from fly_f1a_mb import ANNOTATIONS, CONNECTIONS, OUT, Contrast, circuit, contrasts, glomerulus_index
from torch import Tensor

from neuromorphic.fly.compartment import (
    CompartmentLearner,
    CompartmentReadout,
    build_readout,
    normalize_columns,
)
from neuromorphic.fly.continual import draw_prototypes, draw_samples, kc_code
from neuromorphic.fly.flywire import _pandas
from neuromorphic.fly.mb_output import OutputLoop, extract_output_loop, rewire_feedback
from neuromorphic.fly.mushroom_body import PNKCCircuit

N_TRAINED, N_NOVEL = 20, 20
ACQUISITION_REPEATS, REVERSAL_REPEATS, TEST_REPEATS = 3, 6, 5
RETENTION_TRIALS = 400
SIGMA, SPARSITY, PN_NOISE = 0.4, 0.05, 0.05
ETAS = (0.1, 0.3, 1.0)
RECOVERIES = (0.001, 0.003)
FEEDBACK_GAINS = (1.0, 3.0)
TASKS = ("retention", "retention_novel", "reversal")
VARIANTS = (
    "none",
    "ideal",
    "loop_real",
    "loop_rewired",
    "loop_flipped",
    "loop_positive",
    "replay_oracle",
)
QUALIFICATION_SEEDS = tuple(range(990, 995))
SEEDS = tuple(range(150, 160))
SPECIFICATION: dict[str, Contrast] = {
    "J1": ("retention/app/loop_real", "retention/app/none", 0.02, "superiority"),
    "J2": ("retention/bias/loop_real", "retention/bias/none", 0.03, "superiority"),
    "J3": ("retention/bias/loop_real", "retention/bias/loop_flipped", 0.03, "superiority"),
    "J4": ("retention/bias/loop_real", "retention/bias/loop_rewired", 0.03, "superiority"),
    "J5": ("reversal/app/none", "reversal/app/loop_real", 0.02, "superiority"),
    "J6": ("retention/av/loop_real", "retention/av/none", 0.02, "equivalence"),
}
GRID = [
    {"eta": e, "recovery": r, "gain": 1.0, "feedback_gain": f}
    for e, r, f in itertools.product(ETAS, RECOVERIES, FEEDBACK_GAINS)
]


@dataclasses.dataclass(frozen=True)
class Episode:
    """Learning trials then learning-free probes; ``target`` is the probe's correct sign."""

    code: Tensor  # [learning + probes, KC]
    reward: np.ndarray  # [learning]
    replay_reward: np.ndarray  # [learning]; re-exposures reinforced as in acquisition
    acquisition: int  # number of acquisition trials
    original: np.ndarray  # valence each probe odor was acquired with (0 for novel)
    reversed: bool


def episodes(mb: PNKCCircuit, seed: int) -> dict[str, Episode]:
    index, n_glomeruli = glomerulus_index(mb)
    prototypes = draw_prototypes(
        np.random.default_rng([seed, 0]), N_TRAINED + N_NOVEL, n_glomeruli, 0.3
    )
    valence = np.random.default_rng([seed, 2]).permutation(
        np.array([1] * (N_TRAINED // 2) + [-1] * (N_TRAINED // 2))
    )
    weights = torch.as_tensor(mb.dense(binary=True), dtype=torch.float32)
    trained = np.arange(N_TRAINED)
    novel = np.arange(N_TRAINED, N_TRAINED + N_NOVEL)
    probes = np.repeat(np.arange(N_TRAINED + N_NOVEL), TEST_REPEATS)
    original = np.concatenate([valence, np.zeros(N_NOVEL, dtype=np.int64)])[probes]
    out: dict[str, Episode] = {}
    for i, task in enumerate(TASKS):
        rng = np.random.default_rng([seed, 10 + i])
        acquisition = rng.permutation(np.repeat(trained, ACQUISITION_REPEATS))
        if task == "retention":
            again = rng.integers(0, N_TRAINED, RETENTION_TRIALS)
            distractor = rng.integers(N_TRAINED, N_TRAINED + N_NOVEL, RETENTION_TRIALS)
            second = np.where(rng.random(RETENTION_TRIALS) < 0.5, again, distractor)
            second_reward = np.zeros(RETENTION_TRIALS, dtype=np.int64)
        elif task == "retention_novel":
            second = rng.choice(novel, RETENTION_TRIALS)
            second_reward = np.zeros(RETENTION_TRIALS, dtype=np.int64)
        else:
            second = rng.permutation(np.repeat(trained, REVERSAL_REPEATS))
            second_reward = -valence[second]
        learning = np.concatenate([acquisition, second])
        reward = np.concatenate([valence[acquisition], second_reward])
        replay = reward.copy()
        if task == "retention":
            again_trained = second < N_TRAINED
            replay[acquisition.size :][again_trained] = valence[second[again_trained]]
        pn = draw_samples(
            prototypes,
            np.concatenate([learning, probes]),
            sigma=SIGMA,
            pn_glomerulus_index=index,
            pn_noise=PN_NOISE,
            rng=rng,
        )
        out[task] = Episode(
            kc_code(pn, weights, sparsity=SPARSITY),
            reward,
            replay,
            acquisition.size,
            original,
            task == "reversal",
        )
    return out


def readouts(loop: OutputLoop, kc_mbon: np.ndarray, seed: int) -> dict[str, CompartmentReadout]:
    real = build_readout(loop, kc_mbon)
    rewired = build_readout(
        loop, kc_mbon, feedback=rewire_feedback(loop.feedback, np.random.default_rng(7000 + seed))
    )
    return {
        "none": real,
        "replay_oracle": real,
        "ideal": real,
        "loop_real": real,
        "loop_rewired": rewired,
        "loop_flipped": dataclasses.replace(real, mbon_sign=-real.mbon_sign),
        "loop_positive": dataclasses.replace(
            real,
            feedback=torch.as_tensor(normalize_columns(loop.teach.T), dtype=torch.float32),
            mbon_sign=-torch.ones_like(real.mbon_sign),
        ),
    }


def indices(values: np.ndarray, episode: Episode) -> dict[str, np.ndarray]:
    """Novel-referenced memory indices and bias ``[P]`` from probe values ``[P, probes]``.

    ``app``/``av``: mean value of the odors acquired as appetitive/aversive minus
    that of never-reinforced odors, signed so that the currently correct valence
    (reversed in ``reversal``) is positive.  ``bias``: mean value of novel odors.
    """

    novel = values[:, episode.original == 0].mean(axis=1)
    flip = -1.0 if episode.reversed else 1.0
    out = {
        "app": flip * (values[:, episode.original == 1].mean(axis=1) - novel),
        "av": flip * (novel - values[:, episode.original == -1].mean(axis=1)),
        "bias": novel,
        "novel_false_approach": (values[:, episode.original == 0] > 0.1).mean(axis=1),
    }
    return out


def run(readout: CompartmentReadout, variant: str, episode: Episode) -> dict[str, np.ndarray]:
    """Per-grid-point metrics ``[P]`` after acquisition (``*_acq``) and at the end."""

    def column(key: str) -> Tensor:
        return torch.tensor([p[key] for p in GRID])

    learner = CompartmentLearner(
        readout,
        "loop" if variant.startswith("loop") else "none" if variant == "replay_oracle" else variant,  # type: ignore[arg-type]
        eta=column("eta"),
        recovery=column("recovery"),
        gain=column("gain"),
        feedback_gain=column("feedback_gain"),
    )
    rewards = episode.replay_reward if variant == "replay_oracle" else episode.reward
    n_learning = rewards.shape[0]
    probes = episode.code[n_learning:]

    def probe() -> np.ndarray:
        return torch.stack([learner.value(z) for z in probes], dim=1).numpy()  # [P, probes]

    out: dict[str, np.ndarray] = {}
    for trial in range(n_learning):
        if trial == episode.acquisition:
            acquired = indices(probe(), dataclasses.replace(episode, reversed=False))
            out.update({f"{k}_acq": v for k, v in acquired.items()})
        learner.step(episode.code[trial], int(rewards[trial]))
    out.update(indices(probe(), episode))
    return out


def load() -> tuple[OutputLoop, PNKCCircuit, np.ndarray]:
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
    return loop, mb, kc_mbon


def run_seed(
    loop: OutputLoop, mb: PNKCCircuit, kc_mbon: np.ndarray, seed: int
) -> dict[str, dict[str, np.ndarray]]:
    data = episodes(mb, seed)
    banks = readouts(loop, kc_mbon, seed)
    return {
        f"{task}/{variant}": run(banks[variant], variant, data[task])
        for task in TASKS
        for variant in VARIANTS
    }


def main() -> None:
    started = time.perf_counter()
    loop, mb, kc_mbon = load()
    qualification = {seed: run_seed(loop, mb, kc_mbon, seed) for seed in QUALIFICATION_SEEDS}
    margin = float(
        np.mean(
            [
                q["retention/replay_oracle"]["app"].mean() - q["retention/none"]["app"].mean()
                for q in qualification.values()
            ]
        )
    )
    passed = margin >= 0.02
    print(f"qualification: replay_oracle - none (appetitive retention) {margin:+.3f} -> {passed}")
    report: dict[str, object] = {
        "protocol": "docs/fly/F1j-positive-loop-protocol.md",
        "grid": GRID,
        "qualification": {"margin": margin, "passed": passed},
    }
    if not passed:
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
        names = list(per_seed[SEEDS[0]])
        report["status"] = "COMPLETED"
        report["seeds"] = list(SEEDS)
        report["summary"] = {
            name: float(np.mean([float(per_seed[s][name]["final_accuracy"]) for s in SEEDS]))  # type: ignore[arg-type]
            for name in names
        }
        report["per_grid_point"] = {
            name: {
                key: np.mean([m[key] for m in runs], axis=0).round(4).tolist()
                for key in ("app", "av", "bias", "app_acq")
            }
            for name, runs in grids.items()
        }

        def favoring(key: str) -> str:
            real = np.mean([m[key] for m in grids["retention/loop_real"]], axis=0)
            none = np.mean([m[key] for m in grids["retention/none"]], axis=0)
            return f"{int((real > none).sum())}/{len(GRID)}"

        report["grid_points_loop_real_above_none"] = {k: favoring(k) for k in ("app", "bias")}
        report["contrasts"] = contrasts(per_seed, SPECIFICATION, holm_kinds=("superiority",))
        report["per_seed"] = {str(s): v for s, v in per_seed.items()}
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    (OUT / "f1j_positive_loop_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
