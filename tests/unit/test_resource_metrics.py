from __future__ import annotations

from neuromorphic.training.resource_metrics import process_peak_rss_bytes


def test_process_peak_rss_is_positive_and_identifies_source() -> None:
    value, source = process_peak_rss_bytes()

    assert value > 0
    assert source
