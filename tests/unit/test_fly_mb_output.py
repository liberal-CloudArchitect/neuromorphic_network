from __future__ import annotations

import numpy as np
import pytest

from neuromorphic.fly.mb_output import (
    OutputLoop,
    extract_output_loop,
    inhibitory_enrichment,
    reciprocal_fraction,
    reciprocal_mask,
    rewire_feedback,
)

pd = pytest.importorskip("pandas")


def test_extract_builds_teach_and_feedback_matrices() -> None:
    annotations = pd.DataFrame(
        {
            "root_id": [1, 2, 10, 11],
            "cell_class": ["DAN", "DAN", "MBON", "MBON"],
            "cell_type": ["PAM01", "PPL101", "MBON01", "MBON11"],
            "side": ["right"] * 4,
            "top_nt": ["dopamine", "dopamine", "glutamate", "acetylcholine"],
        }
    )
    connections = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 2, 10, 11],
            "post_pt_root_id": [10, 10, 11, 1, 1],
            "syn_count": [3, 4, 6, 9, 2],
        }
    )
    loop = extract_output_loop(annotations, connections)
    assert loop.teach.tolist() == [[7.0, 0.0], [0.0, 6.0]]
    assert loop.feedback.tolist() == [[9.0, 0.0], [2.0, 0.0]]
    assert loop.mbon_inhibitory.tolist() == [True, False]
    assert loop.mbon_transmitter == ("glutamate", "acetylcholine")
    mask = reciprocal_mask(loop, 5)
    assert reciprocal_fraction(loop.feedback, mask) == pytest.approx(9 / 11)
    assert inhibitory_enrichment(loop.feedback, mask, loop.mbon_inhibitory) == pytest.approx(1.0)


def test_rewire_preserves_degrees_and_weights() -> None:
    rng = np.random.default_rng(0)
    feedback = (rng.random((12, 15)) < 0.3) * rng.integers(1, 9, (12, 15)).astype(float)
    null = rewire_feedback(feedback, np.random.default_rng(1))
    np.testing.assert_array_equal((null > 0).sum(1), (feedback > 0).sum(1))
    np.testing.assert_array_equal((null > 0).sum(0), (feedback > 0).sum(0))
    np.testing.assert_array_equal(np.sort(null[null > 0]), np.sort(feedback[feedback > 0]))
    np.testing.assert_allclose(null.sum(1), feedback.sum(1))
    assert not np.array_equal(null, feedback)


def test_loop_validates_shapes() -> None:
    with pytest.raises(ValueError, match="teach"):
        OutputLoop(
            dan_ids=np.arange(2),
            dan_type=("a", "b"),
            mbon_ids=np.arange(3),
            mbon_type=("x", "y", "z"),
            mbon_inhibitory=np.zeros(3, dtype=bool),
            teach=np.zeros((3, 2)),
            feedback=np.zeros((3, 2)),
        )


def test_loop_validates_transmitter_names() -> None:
    with pytest.raises(ValueError, match="transmitter name"):
        OutputLoop(
            dan_ids=np.arange(2),
            dan_type=("a", "b"),
            mbon_ids=np.arange(2),
            mbon_type=("x", "y"),
            mbon_inhibitory=np.zeros(2, dtype=bool),
            teach=np.zeros((2, 2)),
            feedback=np.zeros((2, 2)),
            mbon_transmitter=("gaba",),
        )
