from __future__ import annotations

import neuromorphic


def test_package_version_is_initial_release() -> None:
    assert neuromorphic.__version__ == "0.1.0"
