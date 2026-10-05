"""Paired seed-bootstrap contrasts for T-track protocols (main-line conventions)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np

from neuromorphic.evaluation.statistics import holm_adjust, percentile_bootstrap_mean

Kind = Literal["superiority", "noninferiority"]
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_715


def paired_contrasts(
    per_seed: Mapping[int, Mapping[str, float]],
    specification: Mapping[str, tuple[str, str, float, Kind]],
) -> dict[str, dict[str, object]]:
    """``(a, b, threshold, kind)`` contrasts on per-seed values; Holm over superiority rows.

    Superiority passes when the mean difference reaches ``threshold``, the 95%
    percentile-bootstrap CI excludes 0 and the Holm-adjusted two-sided p is at
    most 0.05; non-inferiority passes when the CI lower bound exceeds
    ``-threshold``.  Use ``b = "zero"`` to test a single condition against 0.
    """

    seeds = sorted(per_seed)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(seeds), size=(BOOTSTRAP_SAMPLES, len(seeds))
    )
    rows: dict[str, dict[str, object]] = {}
    for cid, (a, b, threshold, kind) in specification.items():
        diff = np.array([per_seed[s][a] - (0.0 if b == "zero" else per_seed[s][b]) for s in seeds])
        interval = percentile_bootstrap_mean(
            diff.tolist(), samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
        )
        boot = diff[draws].mean(axis=1)
        p = max(
            2 * min(float((boot <= 0).mean()), float((boot >= 0).mean())), 1 / BOOTSTRAP_SAMPLES
        )
        rows[cid] = {
            "a": a,
            "b": b,
            "kind": kind,
            "threshold": threshold,
            "per_seed": diff.round(4).tolist(),
            "mean": interval.estimate,
            "ci95": [interval.lower, interval.upper],
            "p": min(p, 1.0),
            "p_holm": None,
        }
    family = [cid for cid, row in rows.items() if row["kind"] == "superiority"]
    if family:
        adjusted = holm_adjust([float(rows[c]["p"]) for c in family])  # type: ignore[arg-type]
        for cid, value in zip(family, adjusted, strict=True):
            rows[cid]["p_holm"] = value
    for row in rows.values():
        low = float(row["ci95"][0])  # type: ignore[index]
        threshold = float(row["threshold"])  # type: ignore[arg-type]
        if row["kind"] == "noninferiority":
            row["verdict"] = "NON_INFERIOR" if low > -threshold else "NOT_NON_INFERIOR"
        else:
            passed = (
                float(row["mean"]) >= threshold  # type: ignore[arg-type]
                and low > 0
                and float(row["p_holm"]) <= 0.05  # type: ignore[arg-type]
            )
            row["verdict"] = "SUPPORTED" if passed else "NOT_SUPPORTED"
    return rows


__all__ = ["paired_contrasts"]
