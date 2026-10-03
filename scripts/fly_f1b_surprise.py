"""F1b: prediction-error and surprise gated valence learning on the MB code.

Protocol docs/fly/F1b-surprise-protocol.md (exploratory, CR-013).  Every learner
is tuned on seeds 900-904 with the schedule-averaged accuracy; seeds 40-49 test.
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable

import numpy as np
import torch
from fly_f1a_mb import OUT, Contrast, circuit, contrasts, glomerulus_index
from torch import Tensor

from neuromorphic.fly.continual import draw_prototypes, draw_samples, kc_code
from neuromorphic.fly.mushroom_body import PNKCCircuit
from neuromorphic.fly.valence import (
    DeltaValence,
    HebbianValence,
    ScheduleKind,
    SurpriseValence,
    ValenceLearner,
    make_schedule,
    run_valence,
)

N_ODORS, N_TRIALS, SIGMA, SPARSITY = 20, 2000, 0.4, 0.05
SCHEDULES: tuple[ScheduleKind, ...] = ("stable", "volatile", "noisy")
CALIBRATION_SEEDS = tuple(range(900, 905))
SEEDS = tuple(range(40, 50))
ETAS = (0.03, 0.1, 0.3, 1.0)
GAMMAS = (0.05, 0.2)
SPECIFICATION: dict[str, Contrast] = {
    "H1": ("volatile/kc_delta", "volatile/kc_hebb", 0.10, "superiority"),
    "H2": ("volatile/kc_surprise", "volatile/kc_delta", 0.03, "superiority"),
    "H3": ("noisy/kc_surprise", "noisy/kc_delta", 0.02, "noninferiority"),
    "H4": ("volatile/kc_delta", "volatile/pn_delta", 0.03, "superiority"),
}

Factory = Callable[[int, float], ValenceLearner]


def grid() -> dict[str, list[tuple[dict[str, float], Factory]]]:
    def hebb(eta: float) -> Factory:
        return lambda n, scale: HebbianValence(n, eta=eta, scale=scale)

    def delta(eta: float) -> Factory:
        return lambda n, scale: DeltaValence(n, eta=eta, scale=scale)

    def surprise(eta: float, gamma: float) -> Factory:
        return lambda n, scale: SurpriseValence(n, eta=eta, scale=scale, gamma=gamma)

    return {
        "kc_hebb": [({"eta": e}, hebb(e)) for e in ETAS],
        "kc_delta": [({"eta": e}, delta(e)) for e in ETAS],
        "kc_surprise": [
            ({"eta": e, "gamma": g}, surprise(e, g)) for e, g in itertools.product(ETAS, GAMMAS)
        ],
        "pn_delta": [({"eta": e}, delta(e)) for e in ETAS],
    }


def features(real: PNKCCircuit, seed: int) -> dict[str, tuple[Tensor, Tensor, object]]:
    """Per schedule: KC code, normalized PN input and the schedule itself."""

    index, n_glomeruli = glomerulus_index(real)
    prototypes = draw_prototypes(np.random.default_rng([seed, 0]), N_ODORS, n_glomeruli, 0.3)
    weights = torch.as_tensor(real.dense(binary=True), dtype=torch.float32)
    out: dict[str, tuple[Tensor, Tensor, object]] = {}
    for i, kind in enumerate(SCHEDULES):
        rng = np.random.default_rng([seed, 1 + i])
        schedule = make_schedule(kind, n_odors=N_ODORS, n_trials=N_TRIALS, rng=rng)
        pn = draw_samples(
            prototypes,
            schedule.odor,
            sigma=SIGMA,
            pn_glomerulus_index=index,
            pn_noise=0.05,
            rng=rng,
        )
        kc = kc_code(pn, weights, sparsity=SPARSITY)
        out[kind] = (kc, torch.nn.functional.normalize(pn, dim=1), schedule)
    return out


def evaluate(
    learner: str, factory: Factory, data: dict[str, tuple[Tensor, Tensor, object]], kc_scale: float
) -> dict[str, tuple[float, float]]:
    results: dict[str, tuple[float, float]] = {}
    for kind, (kc, pn, schedule) in data.items():
        x, scale = (pn, 1.0) if learner.startswith("pn_") else (kc, kc_scale)
        result = run_valence(factory(x.shape[1], scale), x, schedule)  # type: ignore[arg-type]
        results[kind] = (result.accuracy, result.recovery_accuracy)
    return results


def main() -> None:
    started = time.perf_counter()
    real = circuit()
    kc_scale = float(round(SPARSITY * real.n_kc))
    calibration_data = {seed: features(real, seed) for seed in CALIBRATION_SEEDS}
    chosen: dict[str, tuple[dict[str, float], Factory]] = {}
    calibration: dict[str, list[dict[str, object]]] = {}
    for learner, options in grid().items():
        scored = []
        for params, factory in options:
            scores = [
                np.mean([acc for acc, _ in evaluate(learner, factory, data, kc_scale).values()])
                for data in calibration_data.values()
            ]
            scored.append((float(np.mean(scores)), params, factory))
        best = max(scored, key=lambda item: item[0])  # first maximum in grid order
        chosen[learner] = (best[1], best[2])
        calibration[learner] = [{"params": p, "score": s} for s, p, _ in scored]
        print(f"{learner}: chose {best[1]} (calibration {best[0]:.3f})", flush=True)

    per_seed: dict[int, dict[str, dict[str, object]]] = {}
    for seed in SEEDS:
        data = features(real, seed)
        row: dict[str, dict[str, object]] = {}
        for learner, (_, factory) in chosen.items():
            for kind, (accuracy, recovery) in evaluate(learner, factory, data, kc_scale).items():
                row[f"{kind}/{learner}"] = {
                    "final_accuracy": accuracy,
                    "recovery_accuracy": recovery,
                }
        per_seed[seed] = row
    names = list(per_seed[SEEDS[0]])
    summary = {
        name: {
            key: float(np.nanmean([float(per_seed[s][name][key]) for s in SEEDS]))  # type: ignore[arg-type]
            for key in ("final_accuracy", "recovery_accuracy")
            if not (key == "recovery_accuracy" and not name.startswith("volatile/"))
        }
        for name in names
    }
    report = {
        "protocol": "docs/fly/F1b-surprise-protocol.md",
        "chosen": {k: v[0] for k, v in chosen.items()},
        "calibration": calibration,
        "seeds": list(SEEDS),
        "summary": summary,
        "contrasts": contrasts(per_seed, SPECIFICATION, holm_kinds=("superiority",)),
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (OUT / "f1b_surprise_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
