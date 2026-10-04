"""Dynamic mushroom-body compartment readout with DAN teaching signals (F1h).

KCs drive MBONs through real binary connectivity; DANs teach compartments
through the real DAN->MBON matrix; plasticity is dopamine-gated depression of
active KC->MBON synapses plus recovery.  DAN activity is reward only (``none``),
an ideal opponent RPE (``ideal``) or computed by the signed MBON->DAN feedback
loop (``loop``).  Protocol: docs/fly/F1h-compartment-loop-protocol.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from neuromorphic.fly.mb_output import OutputLoop

FloatArray = NDArray[np.float64]
Variant = Literal["none", "ideal", "loop"]


@dataclass(frozen=True, slots=True)
class CompartmentReadout:
    connectivity: Tensor  # [M, K] binary KC->MBON
    valence: Tensor  # [M] -1 avoidance (PAM-taught), +1 approach (PPL1-taught), 0 untaught
    mbon_sign: Tensor  # [M] +1 excitatory, -1 inhibitory transmitter
    teach: Tensor  # [D, M] DAN->MBON, each taught MBON column sums to 1
    feedback: Tensor  # [M, D] MBON->DAN, each DAN column with input sums to 1
    reward_family: Tensor  # [D] +1 PAM (reward), -1 PPL1 (punishment), 0 other


def _float(x: FloatArray | NDArray[np.bool_]) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def normalize_columns(matrix: FloatArray) -> FloatArray:
    total = matrix.sum(axis=0, keepdims=True)
    return np.divide(matrix, total, out=np.zeros_like(matrix), where=total > 0)


def build_readout(
    loop: OutputLoop,
    kc_mbon_counts: FloatArray,
    *,
    feedback: FloatArray | None = None,
    teach_threshold: float = 5.0,
) -> CompartmentReadout:
    """Assemble the readout; ``feedback`` overrides the real MBON->DAN matrix (nulls)."""

    family = np.array(
        [
            1.0 if t.startswith("PAM") else -1.0 if t.startswith("PPL1") else 0.0
            for t in loop.dan_type
        ]
    )
    pam = loop.teach[family > 0].sum(axis=0)
    ppl1 = loop.teach[family < 0].sum(axis=0)
    total = loop.teach.sum(axis=0)
    valence = np.where(total < teach_threshold, 0.0, np.where(pam >= ppl1, -1.0, 1.0))
    raw_feedback = loop.feedback if feedback is None else feedback
    return CompartmentReadout(
        connectivity=_float(kc_mbon_counts > 0),
        valence=_float(valence),
        mbon_sign=_float(np.where(loop.mbon_inhibitory, -1.0, 1.0)),
        teach=_float(normalize_columns(loop.teach)),
        feedback=_float(normalize_columns(raw_feedback)),
        reward_family=_float(family),
    )


class CompartmentLearner:
    """A batch of ``P`` parameter settings learning in parallel on the same trials."""

    def __init__(
        self,
        readout: CompartmentReadout,
        variant: Variant,
        *,
        eta: Tensor,
        recovery: Tensor,
        gain: Tensor,
        feedback_gain: Tensor,
    ) -> None:
        if variant not in {"none", "ideal", "loop"}:
            raise ValueError(f"unknown variant: {variant}")
        self.readout = readout
        self.variant = variant
        self.eta = eta[:, None, None]
        self.recovery = recovery[:, None, None]
        self.gain = gain[:, None]
        self.feedback_gain = feedback_gain[:, None]
        batch = eta.shape[0]
        self.base = readout.connectivity
        self.weights = readout.connectivity.expand(batch, -1, -1).clone()

    def _readout(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """MBON deviations from baseline ``[P, M]`` and decision values ``[P]``."""

        r = self.readout
        active = z.sum().clamp(min=1.0)
        output = (self.weights @ z) / active  # [P, M]
        baseline = (self.base @ z) / active  # [M]
        denominator = (r.valence.abs() * baseline).sum().clamp(min=1e-9)
        deviation = output - baseline
        return deviation, 2.0 * (deviation * r.valence).sum(dim=1) / denominator

    def value(self, z: Tensor) -> Tensor:
        """Predicted values ``v`` [P] for code ``z`` [K] without learning (test probes)."""

        return self._readout(z)[1]

    def step(self, z: Tensor, reward: int) -> Tensor:
        """Predict values for code ``z`` [K], learn from ``reward``; returns ``v`` [P].

        ``reward`` 0 (no reinforcement) leaves only the feedback loop to drive DANs.
        """

        r = self.readout
        deviation, value = self._readout(z)
        reward_drive = torch.where(
            r.reward_family > 0,
            torch.tensor(float(max(reward, 0))),
            torch.where(
                r.reward_family < 0, torch.tensor(float(max(-reward, 0))), torch.tensor(0.0)
            ),
        )  # [D]
        if self.variant == "none":
            dan = self.gain * reward_drive[None, :]
        elif self.variant == "ideal":
            error = (reward - value)[:, None]
            dan = self.gain * torch.where(
                r.reward_family[None, :] > 0,
                error.clamp(min=0.0),
                torch.where(r.reward_family[None, :] < 0, (-error).clamp(min=0.0), 0.0),
            )
        else:
            loop_drive = (deviation * r.mbon_sign) @ r.feedback  # [P, D]
            dan = (reward_drive[None, :] + self.feedback_gain * loop_drive).clamp(min=0.0)
        teaching = dan @ r.teach  # [P, M]
        update = self.weights - self.eta * teaching[:, :, None] * z[None, None, :]
        update = update + self.recovery * (self.base - update)
        self.weights = torch.minimum(update.clamp(min=0.0), 2.0 * self.base)
        return value
