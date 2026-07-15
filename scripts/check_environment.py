#!/usr/bin/env python3
"""Validate the local PyTorch runtime and report the result as JSON."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import Any

import torch


def _mps_available() -> bool:
    """Return whether PyTorch can create tensors on an MPS device."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_built() and backend.is_available())


def _select_accelerator(requirement: str) -> tuple[str, str | None]:
    if requirement == "cpu":
        return "cpu", None
    if requirement == "mps":
        if _mps_available():
            return "mps", None
        return "mps", "MPS was required but is not available"
    return ("mps", None) if _mps_available() else ("cpu", None)


def _smoke_test(accelerator: str) -> tuple[bool, str | None]:
    try:
        device = torch.device(accelerator)
        features = torch.tensor([[1.0, -2.0], [0.5, 3.0]], device=device)
        weights = torch.tensor([[0.25], [-0.5]], device=device, requires_grad=True)
        loss = (features @ weights).square().mean()
        loss.backward()  # type: ignore[no-untyped-call]
        if weights.grad is None or not bool(torch.isfinite(weights.grad).all().item()):
            return False, "backward pass did not produce finite gradients"
        return True, None
    except Exception as exc:  # pragma: no cover - device/runtime dependent
        return False, f"{type(exc).__name__}: {exc}"


def build_report(requirement: str) -> dict[str, Any]:
    """Build an environment report for the requested accelerator."""
    accelerator, selection_error = _select_accelerator(requirement)
    smoke_ok, smoke_error = (
        (False, selection_error) if selection_error else _smoke_test(accelerator)
    )
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "accelerator": accelerator,
        "smoke_ok": smoke_ok,
        "error": smoke_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="Accelerator that must pass forward/backward smoke validation.",
    )
    args = parser.parse_args()
    report = build_report(args.require)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["smoke_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
