"""F1e: is MBON->DAN feedback enriched on reciprocal (teaching) pairs?

Protocol docs/fly/F1e-feedback-loop-protocol.md (exploratory, CR-013).  Purely
structural: FlyWire 783 right hemisphere, 1,000 degree-preserving rewirings.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from fly_f1a_mb import ANNOTATIONS, CONNECTIONS, OUT

from neuromorphic.fly.flywire import _pandas
from neuromorphic.fly.mb_output import (
    extract_output_loop,
    inhibitory_enrichment,
    reciprocal_fraction,
    reciprocal_mask,
    rewire_feedback,
)

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 5.0
NULLS, PERMUTATIONS, RNG_SEED = 1000, 1000, 20261004


def main() -> None:
    started = time.perf_counter()
    pd = _pandas()
    annotations = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    connections = pd.read_feather(
        CONNECTIONS, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"]
    )
    loop = extract_output_loop(annotations, connections)
    rng = np.random.default_rng(RNG_SEED)
    mask = reciprocal_mask(loop, THRESHOLD)
    real = reciprocal_fraction(loop.feedback, mask)
    null = np.array(
        [reciprocal_fraction(rewire_feedback(loop.feedback, rng), mask) for _ in range(NULLS)]
    )
    p_o1 = float((np.sum(null >= real) + 1) / (NULLS + 1))
    ratio = real / float(null.mean())
    o1 = bool(real > np.percentile(null, 99.9) and ratio >= 2.0)

    enrichment = inhibitory_enrichment(loop.feedback, mask, loop.mbon_inhibitory)
    permuted = np.array(
        [
            inhibitory_enrichment(loop.feedback, mask, rng.permutation(loop.mbon_inhibitory))
            for _ in range(PERMUTATIONS)
        ]
    )
    p_o2 = float((np.sum(permuted >= enrichment) + 1) / (PERMUTATIONS + 1))
    o2 = bool(p_o2 < 0.05)

    dan_family = np.array([t[:4] if t.startswith("PPL1") else t[:3] for t in loop.dan_type])
    descriptive: dict[str, object] = {
        "dans": int(loop.dan_ids.size),
        "mbons": int(loop.mbon_ids.size),
        "teach_synapses": float(loop.teach.sum()),
        "feedback_synapses": float(loop.feedback.sum()),
        "reciprocal_pairs": int(mask.sum()),
        "feedback_pairs": int((loop.feedback > 0).sum()),
        "fraction_by_threshold": {
            str(t): reciprocal_fraction(loop.feedback, reciprocal_mask(loop, t)) for t in (1, 5, 10)
        },
        "fraction_by_dan_family": {
            family: reciprocal_fraction(
                loop.feedback[:, dan_family == family], mask[:, dan_family == family]
            )
            for family in sorted(set(dan_family))
        },
        "inhibitory_mbons": int(loop.mbon_inhibitory.sum()),
    }
    report = {
        "protocol": "docs/fly/F1e-feedback-loop-protocol.md",
        "O1": {
            "S_real": real,
            "S_null_mean": float(null.mean()),
            "S_null_p99_9": float(np.percentile(null, 99.9)),
            "ratio": ratio,
            "p": p_o1,
            "verdict": "SUPPORTED" if o1 else "NOT_SUPPORTED",
        },
        "O2": {
            "enrichment": enrichment,
            "permutation_mean": float(permuted.mean()),
            "p": p_o2,
            "verdict": "SUPPORTED" if o2 else "NOT_SUPPORTED",
        },
        "descriptive": descriptive,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (OUT / "f1e_feedback_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("O1", "O2", "descriptive")}, indent=2))


if __name__ == "__main__":
    main()
