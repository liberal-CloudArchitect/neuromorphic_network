from __future__ import annotations

import neuromorphic


def test_package_version_is_gate_2_release() -> None:
    assert neuromorphic.__version__ == "0.3.0"
