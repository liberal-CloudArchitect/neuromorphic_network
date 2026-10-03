"""Post-hoc exploratory sensitivity for F1a (not pre-registered; descriptive only).

Varies odor noise and within-family odor similarity for the key learners on
seeds 0-4 and writes artifacts/fly/f1a_sensitivity.json.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import torch
from fly_f1a_mb import CLASSES_PER_TASK, N_CLASSES, N_TEST, N_TRAIN, OUT, circuit, glomerulus_index

from neuromorphic.fly.continual import (
    AssociativeReadout,
    Learner,
    NearestClassMean,
    kc_code,
    make_odor_task,
    run_class_incremental,
)
from neuromorphic.fly.mushroom_body import degree_preserving

SEEDS = range(5)
SIGMAS = (0.3, 0.7, 1.4)
MIXINGS = (0.0, 0.6, 0.85)


def main() -> None:
    real = circuit()
    index, n_glomeruli = glomerulus_index(real)
    rows = []
    for sigma in SIGMAS:
        for mixing in MIXINGS:
            scores: dict[str, list[float]] = {}
            for seed in SEEDS:
                task = make_odor_task(
                    index,
                    n_glomeruli=n_glomeruli,
                    n_classes=N_CLASSES,
                    n_train=N_TRAIN,
                    n_test=N_TEST,
                    sigma=sigma,
                    rng=np.random.default_rng(seed),
                    families=10 if mixing else 0,
                    family_mixing=mixing,
                )
                null = degree_preserving(real, np.random.default_rng(10_000 + seed))
                weights = {
                    "real": torch.as_tensor(real.dense(), dtype=torch.float32),
                    "real_binary": torch.as_tensor(real.dense(binary=True), dtype=torch.float32),
                    "degree_preserving": torch.as_tensor(null.dense(), dtype=torch.float32),
                }
                runs: dict[str, tuple[Callable[[], Learner], torch.Tensor, torch.Tensor]] = {
                    "pn_ncm": (
                        lambda: NearestClassMean(real.n_pn, N_CLASSES),
                        task.train_x,
                        task.test_x,
                    )
                }
                for topology, w in weights.items():
                    for sparsity in (0.05, 0.50):
                        runs[f"mb_assoc|{topology}|{sparsity:.2f}"] = (
                            lambda: AssociativeReadout(real.n_kc, N_CLASSES),
                            kc_code(task.train_x, w, sparsity=sparsity),
                            kc_code(task.test_x, w, sparsity=sparsity),
                        )
                for name, (build, train_x, test_x) in runs.items():
                    result = run_class_incremental(
                        build(),
                        train_x,
                        task.train_y,
                        test_x,
                        task.test_y,
                        classes_per_task=CLASSES_PER_TASK,
                    )
                    scores.setdefault(name, []).append(result.final_accuracy)
            row = {
                "sigma": sigma,
                "family_mixing": mixing,
                "final_accuracy": {k: round(float(np.mean(v)), 4) for k, v in scores.items()},
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    (OUT / "f1a_sensitivity.json").write_text(
        json.dumps({"exploratory": True, "seeds": list(SEEDS), "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
