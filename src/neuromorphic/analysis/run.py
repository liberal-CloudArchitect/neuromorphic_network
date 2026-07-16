"""Verify and summarize a frozen P3 registry without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuromorphic.training.p3_suite import verify_p3_run


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    directory = (
        parsed.registry.parent if parsed.registry.name == "registry.json" else parsed.registry
    )
    print(json.dumps(verify_p3_run(directory), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
