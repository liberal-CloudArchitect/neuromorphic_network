"""Trial-by-trial odor valence learning with unannounced reversals (F1b).

A learner predicts the valence ``v`` of each odor sample from its features,
acts (approach if ``v > 0``), receives reward ``r`` and updates.  Feature
vectors are scaled so that ``x . x / k == 1``; with step size 1 a single
error-driven update therefore moves the prediction for the same sample by the
full error.  The protocol is frozen in ``docs/fly/F1b-surprise-protocol.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
ScheduleKind = Literal["stable", "volatile", "noisy"]


@dataclass(frozen=True, slots=True)
class ValenceSchedule:
    """Per-trial odor, true current valence, delivered reward and reversal tags."""

    odor: IntArray
    value: IntArray
    reward: IntArray
    after_reversal: BoolArray

    def __post_init__(self) -> None:
        trials = self.odor.shape[0]
        for name in ("value", "reward", "after_reversal"):
            if getattr(self, name).shape != (trials,):
                raise ValueError(f"{name} must have one entry per trial")
        if not np.all(np.abs(self.value) == 1) or not np.all(np.abs(self.reward) == 1):
            raise ValueError("values and rewards must be +-1")


def make_schedule(
    kind: ScheduleKind,
    *,
    n_odors: int,
    n_trials: int,
    rng: np.random.Generator,
    block: int = 200,
    flip_fraction: float = 0.5,
    reliability: float = 0.8,
    recovery_window: int = 50,
) -> ValenceSchedule:
    """Draw an odor sequence and its rewards.

    ``volatile`` flips the valence of ``flip_fraction`` of odors at every
    ``block`` boundary without any cue; ``after_reversal`` marks trials within
    ``recovery_window`` of a reversal whose odor was flipped at it.  ``noisy``
    keeps valences fixed but delivers the opposite reward with probability
    ``1 - reliability``.
    """

    if kind not in {"stable", "volatile", "noisy"}:
        raise ValueError(f"unknown schedule kind: {kind}")
    values = rng.choice(np.array([-1, 1]), size=n_odors)
    odor = rng.integers(0, n_odors, n_trials)
    value = np.empty(n_trials, dtype=np.int64)
    after = np.zeros(n_trials, dtype=np.bool_)
    flipped = np.zeros(n_odors, dtype=np.bool_)
    last_reversal = -recovery_window - 1
    for trial in range(n_trials):
        if kind == "volatile" and trial > 0 and trial % block == 0:
            flipped = rng.random(n_odors) < flip_fraction
            values = np.where(flipped, -values, values)
            last_reversal = trial
        value[trial] = values[odor[trial]]
        after[trial] = trial - last_reversal < recovery_window and bool(flipped[odor[trial]])
    reward = value.copy()
    if kind == "noisy":
        reward = np.where(rng.random(n_trials) < reliability, value, -value)
    return ValenceSchedule(odor.astype(np.int64), value, reward.astype(np.int64), after)


class ValenceLearner(Protocol):
    def predict(self, x: Tensor) -> float: ...

    def update(self, x: Tensor, reward: int, prediction: float) -> None: ...


class HebbianValence:
    """Associative only: reward strengthens an approach, punishment an avoid MBON."""

    def __init__(self, n_features: int, *, eta: float, scale: float) -> None:
        self.approach = torch.zeros(n_features)
        self.avoid = torch.zeros(n_features)
        self.step = eta / scale

    def predict(self, x: Tensor) -> float:
        return float((self.approach - self.avoid) @ x)

    def update(self, x: Tensor, reward: int, prediction: float) -> None:
        del prediction
        target = self.approach if reward > 0 else self.avoid
        target.add_(x, alpha=self.step)


class DeltaValence:
    """Prediction-error gated plasticity (Rescorla-Wagner; DAN = r - v)."""

    def __init__(self, n_features: int, *, eta: float, scale: float) -> None:
        self.weight = torch.zeros(n_features)
        self.step = eta / scale

    def predict(self, x: Tensor) -> float:
        return float(self.weight @ x)

    def update(self, x: Tensor, reward: int, prediction: float) -> None:
        self.weight.add_(x, alpha=self.step * (reward - prediction))


class SurpriseValence(DeltaValence):
    """Delta rule whose global learning rate tracks recent unsigned surprise.

    Pearce-Hall style associability ``alpha`` (a tonic, global signal) is the
    running mean of ``|r - v|`` and scales every update; it starts at 1.
    """

    def __init__(self, n_features: int, *, eta: float, scale: float, gamma: float) -> None:
        super().__init__(n_features, eta=eta, scale=scale)
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = gamma
        self.alpha = 1.0

    def update(self, x: Tensor, reward: int, prediction: float) -> None:
        error = reward - prediction
        self.weight.add_(x, alpha=self.step * self.alpha * error)
        self.alpha = (1.0 - self.gamma) * self.alpha + self.gamma * abs(error)


@dataclass(frozen=True, slots=True)
class ValenceResult:
    correct: BoolArray
    after_reversal: BoolArray

    @property
    def accuracy(self) -> float:
        return float(self.correct.mean())

    @property
    def recovery_accuracy(self) -> float:
        """Accuracy on flipped odors shortly after reversals (NaN if none)."""

        if not self.after_reversal.any():
            return float("nan")
        return float(self.correct[self.after_reversal].mean())


def run_valence(
    learner: ValenceLearner, features: Tensor, schedule: ValenceSchedule
) -> ValenceResult:
    """Run one schedule; ``features[t]`` is the input of trial ``t``."""

    if features.shape[0] != schedule.odor.shape[0]:
        raise ValueError("one feature row is required per trial")
    correct = np.empty(schedule.odor.shape[0], dtype=np.bool_)
    for trial in range(features.shape[0]):
        x = features[trial]
        prediction = learner.predict(x)
        correct[trial] = (prediction > 0) == (schedule.value[trial] > 0)
        learner.update(x, int(schedule.reward[trial]), prediction)
    return ValenceResult(correct, schedule.after_reversal.copy())
