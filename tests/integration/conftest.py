from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

_AMPLE_FREE_BYTES = 1024 * 1024**3

# Loaded by worker subprocesses through PYTHONPATH, so the same disk view
# applies to the isolated P4 workers that the suites spawn.
_SITECUSTOMIZE = f"""
import shutil

_real_disk_usage = shutil.disk_usage


def _ample_disk_usage(path):
    usage = _real_disk_usage(path)
    free = max(usage.free, {_AMPLE_FREE_BYTES})
    return shutil._ntuple_diskusage(max(usage.total, free), usage.used, free)


shutil.disk_usage = _ample_disk_usage
"""


@pytest.fixture(scope="session")
def _ample_disk_site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("ample-disk-site")
    (directory / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    return directory


@pytest.fixture(autouse=True)
def ample_suite_disk(monkeypatch: pytest.MonkeyPatch, _ample_disk_site: Path) -> None:
    """Keep suite tests independent of how full the host's temp volume is.

    The P3/P4 suites stop with ``RESOURCE_LIMIT`` when their run directory has
    less than 20 GiB free.  ``tmp_path`` lives on the host system volume, so on a
    nearly full disk the suite tests would fail for reasons unrelated to the
    behaviour under test.  The frozen guard itself is unchanged and covered by
    ``test_suite_disk_guard.py``, which patches a low value explicitly.
    """

    real_disk_usage: Callable[..., shutil._ntuple_diskusage] = shutil.disk_usage

    def disk_usage(path: object) -> shutil._ntuple_diskusage:
        usage = real_disk_usage(path)
        free = max(usage.free, _AMPLE_FREE_BYTES)
        return shutil._ntuple_diskusage(max(usage.total, free), usage.used, free)

    monkeypatch.setattr(shutil, "disk_usage", disk_usage)
    existing = os.environ.get("PYTHONPATH")
    paths = [str(_ample_disk_site), *([existing] if existing else [])]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(paths))
