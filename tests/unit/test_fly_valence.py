from __future__ import annotations

import numpy as np
import pytest
import torch

from neuromorphic.fly.valence import (
    DeltaValence,
    HebbianValence,
    SurpriseValence,
    ValenceSchedule,
    make_schedule,
    run_valence,
)


def test_volatile_schedule_flips_without_cue_and_tags_recovery() -> None:
    schedule = make_schedule("volatile", n_odors=4, n_trials=600, rng=np.random.default_rng(0))
    assert schedule.odor.shape == (600,)
    first = {int(o): int(v) for o, v in zip(schedule.odor[:200], schedule.value[:200], strict=True)}
    second = {
        int(o): int(v) for o, v in zip(schedule.odor[200:400], schedule.value[200:400], strict=True)
    }
    assert any(first[o] != second[o] for o in first if o in second)
    assert not schedule.after_reversal[:200].any()
    assert schedule.after_reversal[200:250].any()
    assert not schedule.after_reversal[250:400].any()
    assert np.array_equal(schedule.reward, schedule.value)


def test_noisy_schedule_keeps_values_and_corrupts_some_rewards() -> None:
    schedule = make_schedule("noisy", n_odors=3, n_trials=2000, rng=np.random.default_rng(1))
    per_odor: dict[int, set[int]] = {int(o): set() for o in schedule.odor}
    for o, v in zip(schedule.odor, schedule.value, strict=True):
        per_odor[int(o)].add(int(v))
    assert all(len(v) == 1 for v in per_odor.values())
    agreement = float((schedule.reward == schedule.value).mean())
    assert 0.75 < agreement < 0.85
    with pytest.raises(ValueError, match="unknown"):
        make_schedule("chaotic", n_odors=2, n_trials=2, rng=np.random.default_rng(0))  # type: ignore[arg-type]


def test_delta_rule_unlearns_a_reversal_but_hebbian_lags() -> None:
    x = torch.tensor([1.0, 0.0])
    delta = DeltaValence(2, eta=0.5, scale=1.0)
    hebb = HebbianValence(2, eta=0.5, scale=1.0)
    for learner in (delta, hebb):
        for _ in range(20):
            learner.update(x, 1, learner.predict(x))
        for _ in range(3):
            learner.update(x, -1, learner.predict(x))
    assert delta.predict(x) < 0
    assert hebb.predict(x) > 0


def test_one_shot_update_with_unit_step_matches_reward() -> None:
    x = torch.tensor([0.0, 1.0, 1.0])
    learner = DeltaValence(3, eta=1.0, scale=2.0)
    learner.update(x, -1, learner.predict(x))
    assert learner.predict(x) == pytest.approx(-1.0)


def test_surprise_raises_learning_rate_after_large_errors() -> None:
    learner = SurpriseValence(1, eta=0.1, scale=1.0, gamma=0.5)
    x = torch.tensor([1.0])
    for _ in range(50):
        learner.update(x, 1, learner.predict(x))
    calm = learner.alpha
    learner.update(x, -1, learner.predict(x))
    assert learner.alpha > calm
    with pytest.raises(ValueError, match="gamma"):
        SurpriseValence(1, eta=0.1, scale=1.0, gamma=0.0)


def test_run_valence_scores_decisions_against_true_value() -> None:
    schedule = ValenceSchedule(
        odor=np.array([0, 0, 0]),
        value=np.array([1, 1, 1]),
        reward=np.array([1, 1, 1]),
        after_reversal=np.zeros(3, dtype=bool),
    )
    result = run_valence(DeltaValence(1, eta=1.0, scale=1.0), torch.ones(3, 1), schedule)
    assert result.correct.tolist() == [False, True, True]
    assert np.isnan(result.recovery_accuracy)
