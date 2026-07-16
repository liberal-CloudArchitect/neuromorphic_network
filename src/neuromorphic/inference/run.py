"""Run bounded JSON inference from a network-mvp-v1 bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neuromorphic.inference.bundle import load_network_mvp
from neuromorphic.tasks import create_task


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument(
        "--split", choices=("validation", "analysis", "test", "ood"), default="test"
    )
    parser.add_argument("--indices", default="0")
    parser.add_argument("--device", default="cpu")
    parsed = parser.parse_args(arguments)
    indices = [int(value) for value in parsed.indices.split(",")]
    if len(indices) > 64 or any(index < 0 for index in indices):
        raise ValueError("inference accepts 1-64 non-negative indices")
    task = create_task(parsed.task_id, profile="smoke")
    device = torch.device(parsed.device)
    network = load_network_mvp(parsed.bundle, device)
    batch = task.generate(parsed.split, indices, device=device)
    output = network.predict_batch(batch)
    result = {
        "schema_version": "network-mvp-inference-v1",
        "task_id": parsed.task_id,
        "split": parsed.split,
        "sample_indices": indices,
        "predictions": output.logits.argmax(dim=-1).to("cpu").tolist(),
        "valid_mask": batch.valid_mask.to("cpu").tolist(),
        "routing": {
            "steps": len(output.routing_trace),
            "capacity_drops": sum(item.capacity_drops for item in output.routing_trace),
        },
        "qualification_only": network.manifest.get("qualification_only"),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
