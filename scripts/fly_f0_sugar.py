"""F0 reproduction: sugar GRN stimulation -> MN9 (Shiu et al. 2024, Fig. 1).

Runs the PyTorch LIF on the v630 connectome with the paper's 21 sugar GRNs and
compares per-neuron firing rates with the upstream Brian2 reference outputs.
Exploratory only (CR-013); writes JSON to artifacts/fly/.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch

from neuromorphic.fly.flywire import load_shiu_connectome
from neuromorphic.fly.lif import LIFParameters, simulate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/fly/shiu2024/91bdd1e"
SUGAR_V630 = (
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
)  # fmt: skip
MN9 = 720575940660219265
MN9_OTHER = 720575940645521262
REFERENCES = {200.0: "sugarR.parquet", 100.0: "sugarR_100Hz.parquet"}


def reference_rates(path: Path) -> pd.Series:
    spikes = pd.read_parquet(path)
    trials = spikes["trial"].nunique()
    return spikes.groupby("flywire_id").size() / trials  # 1 s trials -> Hz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, choices=sorted(REFERENCES), required=True)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    graph = load_shiu_connectome(
        DATA / "2023_03_23_connectivity_630_final.parquet",
        DATA / "2023_03_23_completeness_630_final.csv",
        version="630",
    )
    params = LIFParameters(poisson_rate_hz=args.rate)
    started = time.perf_counter()
    result = simulate(
        graph,
        graph.index_of(SUGAR_V630),
        parameters=params,
        trials=args.trials,
        seed=args.seed,
        device=torch.device(args.device),
    )
    elapsed = time.perf_counter() - started
    ours = pd.Series(result.rates_hz().numpy(), index=graph.neuron_ids.numpy())
    ours = ours[ours > 0]
    ref = reference_rates(DATA / REFERENCES[args.rate])
    union = ref.index.union(ours.index)
    a = ref.reindex(union, fill_value=0.0)
    b = ours.reindex(union, fill_value=0.0)
    pearson = float(a.corr(b))
    log_pearson = float(a.map(math.log1p).corr(b.map(math.log1p)))
    report = {
        "experiment": "sugar_grn_to_mn9",
        "connectome": "flywire-630 (Shiu 2024 signed)",
        "poisson_rate_hz": args.rate,
        "trials": args.trials,
        "seed": args.seed,
        "device": args.device,
        "wall_seconds": round(elapsed, 1),
        "mn9_hz": {"ours": float(ours.get(MN9, 0.0)), "reference": float(ref.get(MN9, 0.0))},
        "mn9_other_hz": {
            "ours": float(ours.get(MN9_OTHER, 0.0)),
            "reference": float(ref.get(MN9_OTHER, 0.0)),
        },
        "active_neurons": {"ours": len(ours), "reference": len(ref)},
        "active_overlap": len(ref.index.intersection(ours.index)),
        "rate_pearson": pearson,
        "rate_log1p_pearson": log_pearson,
        "mean_abs_rate_diff_hz": float((a - b).abs().mean()),
    }
    out = ROOT / f"artifacts/fly/f0_sugar_{int(args.rate)}hz_{args.device}_s{args.seed}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
