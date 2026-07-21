"""Recompute and write the formal P3 report."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuromorphic.analysis.p3_report import summarize_p3_run, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--json", type=Path, default=Path("reports/p3/formal.json"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/p3/formal.md"))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    arguments = parser.parse_args()
    report = summarize_p3_run(
        arguments.run_directory, bootstrap_samples=arguments.bootstrap_samples
    )
    write_report(report, arguments.json, arguments.markdown)
    print(arguments.json)
    print(arguments.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
