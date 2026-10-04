"""F1h post-hoc (exploratory, not pre-registered): is the loop advantage a parameter confound?

Runs every variant at the same two (eta, recovery) settings - the one chosen for
loop_real and the one chosen for the others - on seeds 120-124, all schedules.
"""

from __future__ import annotations

import json

import numpy as np
from fly_f1a_mb import OUT
from fly_f1b_surprise import SCHEDULES, features
from fly_f1h_loop import VARIANTS, run

from neuromorphic.fly.compartment import build_readout
from neuromorphic.fly.mb_output import rewire_feedback

SETTINGS = {
    "fast (loop_real choice)": {"eta": 1.0, "recovery": 0.03},
    "slow (others' choice)": {"eta": 0.03, "recovery": 0.003},
}
SEEDS = tuple(range(120, 125))


def main() -> None:
    from fly_f1a_mb import ANNOTATIONS, CONNECTIONS, circuit

    from neuromorphic.fly.flywire import _pandas
    from neuromorphic.fly.mb_output import extract_output_loop

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
    real = build_readout(loop, kc_mbon)
    results: dict[str, dict[str, float]] = {}
    for label, setting in SETTINGS.items():
        for variant in VARIANTS:
            third = (
                {"gain": 1.0, "feedback_gain": 0.0}
                if variant in {"none", "ideal"}
                else {"gain": 1.0, "feedback_gain": 3.0}
            )
            scores: dict[str, list[float]] = {kind: [] for kind in SCHEDULES}
            for seed in SEEDS:
                readout = (
                    build_readout(
                        loop,
                        kc_mbon,
                        feedback=rewire_feedback(loop.feedback, np.random.default_rng(7000 + seed)),
                    )
                    if variant == "loop_rewired"
                    else real
                )
                data = features(mb, seed)
                for kind in SCHEDULES:
                    code, _, schedule = data[kind]
                    correct = run(readout, variant, [{**setting, **third}], code, schedule)  # type: ignore[arg-type]
                    scores[kind].append(float(correct.mean()))
            results[f"{label} | {variant}"] = {
                k: round(float(np.mean(v)), 3) for k, v in scores.items()
            }
            print(f"{label:24s} {variant:13s}", results[f"{label} | {variant}"], flush=True)
    (OUT / "f1h_posthoc.json").write_text(
        json.dumps({"exploratory": True, "seeds": list(SEEDS), "results": results}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
