from __future__ import annotations

import numpy as np
import pytest
import torch

from neuromorphic.fly.compartment import (
    CompartmentLearner,
    build_readout,
    normalize_columns,
)
from neuromorphic.fly.mb_output import OutputLoop


def _loop(feedback_sign: float = 1.0) -> OutputLoop:
    # DAN 0 = PAM teaches MBON 0 (avoidance); DAN 1 = PPL1 teaches MBON 1 (approach).
    return OutputLoop(
        dan_ids=np.array([1, 2]),
        dan_type=("PAM01", "PPL101"),
        mbon_ids=np.array([10, 11]),
        mbon_type=("MBON01", "MBON11"),
        mbon_inhibitory=np.array([feedback_sign < 0, False]),
        teach=np.array([[9.0, 0.0], [0.0, 9.0]]),
        feedback=np.array([[4.0, 0.0], [0.0, 4.0]]),
    )


def _kc_mbon() -> np.ndarray:
    return np.ones((2, 6))


def _learner(variant: str, readout_loop: OutputLoop | None = None) -> CompartmentLearner:
    readout = build_readout(readout_loop or _loop(), _kc_mbon())
    one = torch.ones(1)
    return CompartmentLearner(
        readout,
        variant,  # type: ignore[arg-type]
        eta=one * 0.5,
        recovery=one * 0.0,
        gain=one,
        feedback_gain=one,
    )


def test_readout_assigns_valence_from_teaching_family() -> None:
    readout = build_readout(_loop(), _kc_mbon())
    assert readout.valence.tolist() == [-1.0, 1.0]
    assert readout.reward_family.tolist() == [1.0, -1.0]
    np.testing.assert_allclose(
        normalize_columns(np.array([[1.0, 0.0], [3.0, 0.0]])), [[0.25, 0], [0.75, 0]]
    )


def test_reward_depresses_avoidance_and_raises_value() -> None:
    z = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    for variant in ("none", "ideal", "loop"):
        learner = _learner(variant)
        assert learner.step(z, 1).item() == pytest.approx(0.0)
        assert learner.step(z, 1).item() > 0.0
        assert learner.weights[0, 0, :3].max() < 1.0  # avoidance MBON depressed
        assert torch.equal(learner.weights[0, 1], torch.ones(6))  # approach untouched


def test_ideal_rpe_stops_learning_once_reward_is_predicted() -> None:
    z = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    ideal = _learner("ideal")
    none = _learner("none")
    for _ in range(30):
        ideal.step(z, 1)
        none.step(z, 1)
    assert ideal.step(z, 1).item() == pytest.approx(1.0, abs=0.05)
    assert none.weights[0, 0, :2].max() == pytest.approx(0.0)  # pure association saturates


def test_reversal_flips_sign_with_opponent_compartments() -> None:
    z = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    learner = _learner("ideal")
    for _ in range(10):
        learner.step(z, 1)
    for _ in range(20):
        learner.step(z, -1)
    assert learner.step(z, -1).item() < 0.0


def test_invalid_variant_rejected() -> None:
    with pytest.raises(ValueError, match="variant"):
        _learner("other")


def test_value_probe_does_not_learn() -> None:
    z = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    learner = _learner("none")
    learner.step(z, 1)
    before = learner.weights.clone()
    probe = learner.value(z)
    assert torch.equal(learner.weights, before)
    assert probe.item() == pytest.approx(learner.step(z, 0).item())


def test_unreinforced_positive_loop_keeps_depressing() -> None:
    # Avoidance MBON 0 inhibits its own PAM: depression disinhibits PAM (positive loop).
    z = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    learner = _learner("loop", _loop(feedback_sign=-1.0))
    learner.step(z, 1)
    after_reward = learner.weights[0, 0, :3].clone()
    learner.step(z, 0)
    assert (learner.weights[0, 0, :3] < after_reward).all()
    plain = _learner("none", _loop(feedback_sign=-1.0))
    plain.step(z, 1)
    held = plain.weights[0, 0, :3].clone()
    plain.step(z, 0)
    assert torch.equal(plain.weights[0, 0, :3], held)


def _tagged(decay: float | None) -> CompartmentLearner:
    readout = build_readout(_loop(feedback_sign=-1.0), _kc_mbon())
    one = torch.ones(1)
    return CompartmentLearner(
        readout,
        "loop",
        eta=one * 0.5,
        recovery=one * 0.0,
        gain=one,
        feedback_gain=one,
        tag_decay=None if decay is None else one * decay,
    )


def test_tags_confine_loop_plasticity_to_reinforced_synapses() -> None:
    trained = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    mixture = torch.ones(6)  # shares KCs 0-2 with the trained odor
    plain, tagged = _tagged(None), _tagged(0.9)
    for learner in (plain, tagged):
        learner.step(trained, 1)
    assert tagged.tags is not None
    assert (tagged.tags[0, 0, :3] > 0).all() and (tagged.tags[0, 0, 3:] == 0).all()
    assert (tagged.tags[0, 1] == 0).all()  # the PPL1-taught MBON was not reinforced
    before = tagged.weights.clone()
    for learner in (plain, tagged):
        learner.step(mixture, 0)
    assert (plain.weights[0, 0, 3:] < 1.0).all()  # untagged synapses depressed by the loop
    assert torch.equal(tagged.weights[0, 0, 3:], before[0, 0, 3:])
    assert (tagged.weights[0, 0, :3] < before[0, 0, :3]).all()


def test_tags_decay_and_require_the_loop_variant() -> None:
    learner = _tagged(0.5)
    learner.step(torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), 1)
    assert learner.tags is not None
    first = learner.tags[0, 0, 0].item()
    learner.step(torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), 0)
    assert learner.tags[0, 0, 0].item() == pytest.approx(0.5 * first)
    one = torch.ones(1)
    with pytest.raises(ValueError, match="loop variant"):
        CompartmentLearner(
            build_readout(_loop(), _kc_mbon()),
            "none",
            eta=one,
            recovery=one,
            gain=one,
            feedback_gain=one,
            tag_decay=one,
        )
