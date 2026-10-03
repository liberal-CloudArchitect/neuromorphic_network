"""Download or verify one reviewed set of public fly datasets (CR-013)."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuromorphic.fly.datasets import fetch, load_manifest, verify_file

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("download_set", help="manifest set, e.g. T0-min, T0-paper, T0-full, T1")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/fly/datasets.yaml")
    parser.add_argument("--root", type=Path, default=ROOT / "data/fly")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    files = load_manifest(args.manifest, args.download_set)
    total = sum(spec.size_bytes for spec in files)
    print(f"{args.download_set}: {len(files)} files, {total / 1e6:.1f} MB")
    for spec in files:
        if args.verify_only:
            verify_file(spec.local_path(args.root), spec)
            path = spec.local_path(args.root)
        else:
            path = fetch(spec, args.root)
        print(f"ok  {spec.checksum_algorithm:<13} {path.relative_to(args.root)}")


if __name__ == "__main__":
    main()
