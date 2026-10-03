"""Batched leaky integrate-and-fire simulation over a signed connectome.

The model follows the structure of the whole-brain fly LIF model of Shiu et al.
(2024): each neuron has a membrane potential ``v`` driven by a synaptic variable
``g`` that decays exponentially; every presynaptic spike adds
``signed_count * synapse_weight`` to ``g`` after a fixed axonal delay.  Selected
neurons additionally receive independent Poisson drive.  Linear dynamics are
integrated exactly between steps, so results do not depend on Euler error.

Defaults and the per-step order mirror the upstream Brian2 implementation
(philshiu/Drosophila_brain_model @ 91bdd1e, MIT): state update of
non-refractory neurons, threshold, synaptic and Poisson delivery, then reset of
``v`` and ``g``.  Driven neurons have no refractory period, and silencing zeroes
a neuron's outgoing weights rather than its own spiking.

Synaptic input that arrives while a neuron is refractory is discarded by
default.  This choice was fixed empirically: it reproduces the upstream Brian2
reference spikes (per-neuron rate correlation 1.000, MN9 93.9 vs 93.3 Hz for
200 Hz sugar GRN drive), whereas accumulating that input inflates downstream
rates by about 20%.

This is a qualitative simulator of an annotated wiring diagram, not a
physiological emulation of a fly brain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from neuromorphic.fly.connectome import Connectome


@dataclass(frozen=True, slots=True)
class LIFParameters:
    """Simulation constants in millivolts, milliseconds and hertz."""

    v_rest_mv: float = -52.0
    v_reset_mv: float = -52.0
    v_threshold_mv: float = -45.0
    membrane_tau_ms: float = 20.0
    synaptic_tau_ms: float = 5.0
    refractory_ms: float = 2.2
    driven_refractory_ms: float = 0.0
    delay_ms: float = 1.8
    synapse_weight_mv: float = 0.275
    poisson_rate_hz: float = 150.0
    poisson_weight_factor: float = 250.0
    poisson_target: Literal["v", "g"] = "v"
    reset_synaptic: bool = True
    refractory_input: Literal["discard", "accumulate"] = "discard"
    dt_ms: float = 0.1
    duration_ms: float = 1000.0

    def __post_init__(self) -> None:
        positive = (
            self.membrane_tau_ms,
            self.synaptic_tau_ms,
            self.dt_ms,
            self.duration_ms,
            self.delay_ms,
        )
        if min(positive) <= 0:
            raise ValueError("time constants, delay, dt and duration must be positive")
        if min(self.refractory_ms, self.driven_refractory_ms, self.poisson_rate_hz) < 0:
            raise ValueError("refractory period and Poisson rate must be non-negative")
        if math.isclose(self.membrane_tau_ms, self.synaptic_tau_ms):
            raise ValueError("exact integration requires distinct membrane and synaptic taus")
        if self.v_threshold_mv <= self.v_reset_mv:
            raise ValueError("threshold must exceed reset potential")
        if self.poisson_target not in {"v", "g"}:
            raise ValueError("poisson_target must be 'v' or 'g'")
        if self.refractory_input not in {"discard", "accumulate"}:
            raise ValueError("refractory_input must be 'discard' or 'accumulate'")

    @property
    def steps(self) -> int:
        return round(self.duration_ms / self.dt_ms)

    @property
    def delay_steps(self) -> int:
        return max(1, round(self.delay_ms / self.dt_ms))

    @property
    def refractory_steps(self) -> int:
        """Steps after a spike during which a neuron neither integrates nor fires."""

        return max(0, round(self.refractory_ms / self.dt_ms) - 1)

    @property
    def driven_refractory_steps(self) -> int:
        return max(0, round(self.driven_refractory_ms / self.dt_ms) - 1)


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Per-trial spike counts for every neuron, on CPU."""

    spike_counts: Tensor
    duration_ms: float

    def rates_hz(self) -> Tensor:
        """Mean firing rate across trials, shape ``[neurons]``."""

        return self.spike_counts.to(torch.float64).mean(dim=0) * (1000.0 / self.duration_ms)


def simulate(
    connectome: Connectome,
    stimulated: Tensor,
    *,
    parameters: LIFParameters | None = None,
    trials: int = 1,
    seed: int = 0,
    device: torch.device | None = None,
    silenced: Tensor | None = None,
) -> SimulationResult:
    """Run ``trials`` independent simulations in one batch.

    ``stimulated`` and ``silenced`` hold connectome row indices.  Silenced
    neurons may still fire but their spikes reach no postsynaptic partner,
    which implements the silencing screens used to probe pathways.  Each trial
    draws independent Poisson input from a generator seeded with ``seed``.
    """

    params = parameters or LIFParameters()
    device = device or torch.device("cpu")
    if trials <= 0:
        raise ValueError("trials must be positive")
    neurons = connectome.n_neurons
    stim = _validated_indices(stimulated, neurons, name="stimulated").to(device)
    silence_mask = torch.zeros(neurons, dtype=torch.bool, device=device)
    if silenced is not None:
        silence_mask[_validated_indices(silenced, neurons, name="silenced").to(device)] = True

    propagator = _EventPropagator(connectome, params.synapse_weight_mv, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    decay_g = math.exp(-params.dt_ms / params.synaptic_tau_ms)
    decay_v = math.exp(-params.dt_ms / params.membrane_tau_ms)
    coupling = params.synaptic_tau_ms / (params.synaptic_tau_ms - params.membrane_tau_ms)
    poisson_probability = params.poisson_rate_hz * params.dt_ms / 1000.0
    poisson_weight = params.synapse_weight_mv * params.poisson_weight_factor
    if poisson_probability > 1.0:
        raise ValueError("Poisson rate is too high for dt")

    shape = (trials, neurons)
    refractory_steps = torch.full(
        (neurons,), params.refractory_steps, dtype=torch.int32, device=device
    )
    refractory_steps[stim] = params.driven_refractory_steps
    # u = v - v_rest keeps the rest potential at zero for the exact update.
    u = torch.full(shape, params.v_reset_mv - params.v_rest_mv, device=device)
    g = torch.zeros(shape, device=device)
    since_spike = torch.full(
        shape, torch.iinfo(torch.int32).max // 2, dtype=torch.int32, device=device
    )
    pending = torch.zeros((params.delay_steps, *shape), dtype=torch.bool, device=device)
    counts = torch.zeros(shape, dtype=torch.int32, device=device)
    threshold = params.v_threshold_mv - params.v_rest_mv
    reset = params.v_reset_mv - params.v_rest_mv

    for step in range(params.steps):
        # 1. State update of non-refractory neurons.
        since_spike += 1
        refractory = since_spike <= refractory_steps
        u = torch.where(refractory, u, u * decay_v + g * coupling * (decay_g - decay_v))
        g = torch.where(refractory, g, g * decay_g)
        # 2. Threshold.
        spikes = (u > threshold) & ~refractory
        # 3. Delayed synaptic delivery and Poisson input.
        slot = step % params.delay_steps
        if params.refractory_input == "discard":
            g = g + torch.where(refractory, 0.0, propagator.deliver(pending[slot], g.shape))
        else:
            g = g + propagator.deliver(pending[slot], g.shape)
        if stim.numel() and poisson_probability > 0:
            events = torch.rand((trials, stim.numel()), generator=generator, device=device)
            drive = (events < poisson_probability).to(g.dtype) * poisson_weight
            if params.poisson_target == "v":
                u[:, stim] += drive
            else:
                g[:, stim] += drive
        # 4. Reset.
        u = torch.where(spikes, torch.full_like(u, reset), u)
        if params.reset_synaptic:
            g = torch.where(spikes, torch.zeros_like(g), g)
        since_spike = torch.where(spikes, torch.zeros_like(since_spike), since_spike)
        counts += spikes.to(torch.int32)
        pending[slot] = spikes & ~silence_mask
    return SimulationResult(spike_counts=counts.cpu(), duration_ms=params.steps * params.dt_ms)


class _EventPropagator:
    """Event-driven delivery of delayed spikes along outgoing edges.

    Spikes are sparse (a few hundred active neurons out of ~10^5), so gathering
    only the outgoing edges of neurons that fired is far cheaper than a dense
    or sparse matrix product over all neurons at every step.
    """

    def __init__(self, connectome: Connectome, scale: float, device: torch.device) -> None:
        order = torch.argsort(connectome.pre * connectome.n_neurons + connectome.post)
        pre = connectome.pre[order]
        self.neurons = connectome.n_neurons
        self.post = connectome.post[order].to(device)
        self.weight = (connectome.signed_count[order].to(torch.float32) * scale).to(device)
        degree = torch.bincount(pre, minlength=self.neurons)
        self.offsets = torch.cat((torch.zeros(1, dtype=torch.int64), degree.cumsum(0))).to(device)

    def deliver(self, spikes: Tensor, shape: torch.Size) -> Tensor:
        """Return ``(W @ spikes.T).T`` for a ``[trials, neurons]`` spike matrix."""

        drive = torch.zeros(shape, device=spikes.device)
        trial, source = spikes.nonzero(as_tuple=True)
        if trial.numel() == 0:
            return drive
        start = self.offsets[source]
        count = self.offsets[source + 1] - start
        total = int(count.sum())
        if total == 0:
            return drive
        first = torch.repeat_interleave(count.cumsum(0) - count, count, output_size=total)
        edge = torch.repeat_interleave(start, count, output_size=total)
        edge = edge + torch.arange(total, device=drive.device) - first
        row = torch.repeat_interleave(trial, count, output_size=total)
        drive.view(-1).index_add_(0, row * self.neurons + self.post[edge], self.weight[edge])
        return drive


def _validated_indices(indices: Tensor, neurons: int, *, name: str) -> Tensor:
    if indices.dim() != 1 or indices.is_floating_point() or indices.dtype == torch.bool:
        raise ValueError(f"{name} must be a one-dimensional integer index tensor")
    indices = indices.to(device="cpu", dtype=torch.int64)
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= neurons):
        raise ValueError(f"{name} contains indices outside the connectome")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError(f"{name} must not contain duplicates")
    return indices
