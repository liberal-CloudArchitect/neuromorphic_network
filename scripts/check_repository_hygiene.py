"""Check text-file hygiene without downloading pre-commit hook environments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


def check_file(path: Path) -> list[str]:
    """Return hygiene errors for one existing text file."""
    if not path.exists() or not path.is_file():
        return []

    raw = path.read_bytes()
    if b"\x00" in raw:
        return []

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return []

    errors: list[str] = []
    if text and not text.endswith("\n"):
        errors.append(f"{path}: missing newline at end of file")

    for line_number, line in enumerate(text.splitlines(), start=1):
        if line != line.rstrip(" \t"):
            errors.append(f"{path}:{line_number}: trailing whitespace")
        if line.startswith(("<<<<<<< ", "=======", ">>>>>>> ")):
            errors.append(f"{path}:{line_number}: merge-conflict marker")

    try:
        if path.suffix == ".json":
            json.loads(text)
        elif path.suffix in {".yaml", ".yml"}:
            list(yaml.safe_load_all(text))
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        errors.append(f"{path}: invalid structured data: {error}")

    return errors


def main(arguments: list[str] | None = None) -> int:
    """Validate files supplied by pre-commit and report all failures."""
    paths = [Path(argument) for argument in (arguments or sys.argv[1:])]
    errors: list[str] = []
    casefolded: dict[str, Path] = {}

    for path in paths:
        normalized = str(path).casefold()
        previous = casefolded.get(normalized)
        if previous is not None and previous != path:
            errors.append(f"case-conflicting paths: {previous} and {path}")
        casefolded[normalized] = path
        errors.extend(check_file(path))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
