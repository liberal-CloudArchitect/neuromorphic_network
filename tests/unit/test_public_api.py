from __future__ import annotations

import neuromorphic


def test_package_version_is_gate_1_release() -> None:
    assert neuromorphic.__version__ == "0.2.0"
