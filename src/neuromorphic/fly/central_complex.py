"""Central-complex heading circuit: extraction, ring inference, nulls and dynamics.

The core is the EPG/EPGt/PEN/PEG/Delta7 network (F1c).  Ring angles are
inferred from connectivity alone, and a signed rate model tests whether a cue
leaves a persistent, cue-specific state.  Protocol: docs/fly/F1c-ring-protocol.md.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

CORE_TYPES = ("EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7")
INHIBITORY_TRANSMITTERS = frozenset({"gaba", "glutamate"})


@dataclass(frozen=True, slots=True)
class CXCircuit:
    """Directed signed graph; ``sign`` belongs to the presynaptic neuron."""

    ids: IntArray
    cell_type: tuple[str, ...]
    side: tuple[str, ...]
    sign: FloatArray
    edge_pre: IntArray
    edge_post: IntArray
    edge_count: FloatArray
    topology: str = "real"

    def __post_init__(self) -> None:
        n = self.ids.shape[0]
        if len(self.cell_type) != n or len(self.side) != n or self.sign.shape != (n,):
            raise ValueError("neuron annotations must align with ids")
        if not np.all(np.abs(self.sign) == 1):
            raise ValueError("signs must be +-1")
        edges = self.edge_pre.shape[0]
        if self.edge_post.shape != (edges,) or self.edge_count.shape != (edges,):
            raise ValueError("edge arrays must have equal length")
        if np.unique(self.edge_pre * n + self.edge_post).size != edges:
            raise ValueError("edges must be unique (pre, post) pairs")

    @property
    def n(self) -> int:
        return int(self.ids.shape[0])

    def indices(self, cell_type: str) -> IntArray:
        return np.flatnonzero(np.array(self.cell_type) == cell_type).astype(np.int64)

    def counts(self) -> FloatArray:
        """Unsigned synapse counts ``C[post, pre]``."""

        matrix = np.zeros((self.n, self.n))
        matrix[self.edge_post, self.edge_pre] = self.edge_count
        return matrix

    def signed_matrix(self, *, binary: bool = False, threshold: float = 5.0) -> FloatArray:
        """``W[post, pre]`` as signed counts, or signed 0/1 for counts >= ``threshold``."""

        counts = self.counts()
        if binary:
            counts = (counts >= threshold).astype(np.float64)
        return counts * self.sign[None, :]

    def weight_scale(self, *, binary: bool = False, threshold: float = 5.0) -> float:
        """Median absolute input sum; shared by recurrent and external pathways."""

        signed = self.signed_matrix(binary=binary, threshold=threshold)
        return float(np.median(np.abs(signed).sum(axis=1)))

    def weights(self, *, binary: bool = False, threshold: float = 5.0) -> FloatArray:
        """Signed weights ``W[post, pre]`` scaled by the median absolute input."""

        signed = self.signed_matrix(binary=binary, threshold=threshold)
        scale = self.weight_scale(binary=binary, threshold=threshold)
        return signed / scale if scale > 0 else signed

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            ids=self.ids,
            cell_type=np.array(self.cell_type),
            side=np.array(self.side),
            sign=self.sign,
            edge_pre=self.edge_pre,
            edge_post=self.edge_post,
            edge_count=self.edge_count,
            topology=np.array(self.topology),
        )

    @classmethod
    def load(cls, path: Path) -> CXCircuit:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                ids=data["ids"].astype(np.int64),
                cell_type=tuple(str(x) for x in data["cell_type"]),
                side=tuple(str(x) for x in data["side"]),
                sign=data["sign"].astype(np.float64),
                edge_pre=data["edge_pre"].astype(np.int64),
                edge_post=data["edge_post"].astype(np.int64),
                edge_count=data["edge_count"].astype(np.float64),
                topology=str(data["topology"]),
            )


def extract_cx(
    annotations: Any, connections: Any, *, types: tuple[str, ...] = CORE_TYPES
) -> CXCircuit:
    """Build the signed core circuit; synapse counts are summed over neuropils."""

    core = annotations[annotations.cell_type.isin(types)].sort_values("root_id")
    ids = core.root_id.to_numpy().astype(np.int64)
    edges = connections[
        connections.pre_pt_root_id.isin(ids) & connections.post_pt_root_id.isin(ids)
    ]
    pairs = edges.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False).syn_count.sum()
    transmitter = core.top_nt.astype(str).str.lower()
    sign = np.where(transmitter.isin(INHIBITORY_TRANSMITTERS), -1.0, 1.0)
    return CXCircuit(
        ids=ids,
        cell_type=tuple(core.cell_type.astype(str)),
        side=tuple(core.side.astype(str)),
        sign=sign,
        edge_pre=np.searchsorted(ids, pairs.pre_pt_root_id.to_numpy()).astype(np.int64),
        edge_post=np.searchsorted(ids, pairs.post_pt_root_id.to_numpy()).astype(np.int64),
        edge_count=pairs.syn_count.to_numpy().astype(np.float64),
    )


def ring_embedding(circuit: CXCircuit, ring_type: str = "EPG") -> tuple[FloatArray, FloatArray]:
    """Infer ring angles of ``ring_type`` neurons from shared partners.

    Each neuron is described by its normalized input and output count vectors;
    the angle comes from the two smallest non-trivial eigenvectors of the
    normalized Laplacian of their cosine-similarity graph.  Returns angles and
    the six smallest eigenvalues (a near-degenerate second/third pair separated
    from the fourth is the spectral signature of a ring).
    """

    counts = circuit.counts()
    ring = circuit.indices(ring_type)
    profile = np.concatenate([counts[ring, :], counts[:, ring].T], axis=1)
    norms = np.linalg.norm(profile, axis=1, keepdims=True)
    profile = profile / np.where(norms > 0, norms, 1.0)
    similarity = profile @ profile.T
    degree = similarity.sum(axis=1)
    degree = np.where(degree > 0, degree, 1.0)
    laplacian = np.eye(ring.size) - similarity / np.sqrt(np.outer(degree, degree))
    values, vectors = np.linalg.eigh(laplacian)
    return np.arctan2(vectors[:, 2], vectors[:, 1]), values[:6]


def _swap_targets(
    circuit: CXCircuit,
    rng: np.random.Generator,
    *,
    group_of_edge: IntArray | None,
    swaps_per_edge: int,
    topology: str,
) -> CXCircuit:
    pre = circuit.edge_pre
    post = circuit.edge_post.copy()
    n = circuit.n
    existing = set((pre * n + post).tolist())
    edges = pre.shape[0]
    members: dict[int, list[int]] = defaultdict(list)
    if group_of_edge is not None:
        for edge, group in enumerate(group_of_edge.tolist()):
            members[group].append(edge)
    attempts = swaps_per_edge * edges
    first = rng.integers(0, edges, attempts)
    pick = rng.random(attempts)
    for a, u in zip(first.tolist(), pick.tolist(), strict=True):
        if group_of_edge is None:
            b = int(u * edges)
        else:
            pool = members[int(group_of_edge[a])]
            b = pool[int(u * len(pool))]
        pa, pb, qa, qb = int(pre[a]), int(pre[b]), int(post[a]), int(post[b])
        if a == b or pa in (pb, qb) or qa in (qb, pb):
            continue
        new_a, new_b = pa * n + qb, pb * n + qa
        if new_a in existing or new_b in existing:
            continue
        existing.difference_update((pa * n + qa, pb * n + qb))
        existing.update((new_a, new_b))
        post[a], post[b] = qb, qa
    return replace(circuit, edge_post=post, topology=topology)


def degree_preserving_directed(
    circuit: CXCircuit, rng: np.random.Generator, *, swaps_per_edge: int = 10
) -> CXCircuit:
    """Swap targets of random edge pairs; in/out degrees and per-source weights kept."""

    return _swap_targets(
        circuit,
        rng,
        group_of_edge=None,
        swaps_per_edge=swaps_per_edge,
        topology="degree_preserving",
    )


def type_preserving(
    circuit: CXCircuit, rng: np.random.Generator, *, swaps_per_edge: int = 10
) -> CXCircuit:
    """Swap targets only within the same (pre type, post type) edge group.

    Type-to-type synapse counts and every neuron's in/out degree are preserved;
    which individual neurons connect (the angular topography) is scrambled.
    """

    names = sorted(set(circuit.cell_type))
    code = np.array([names.index(t) for t in circuit.cell_type])
    group = code[circuit.edge_pre] * len(names) + code[circuit.edge_post]
    return _swap_targets(
        circuit,
        rng,
        group_of_edge=group.astype(np.int64),
        swaps_per_edge=swaps_per_edge,
        topology="type_preserving",
    )


@dataclass(frozen=True, slots=True)
class MemoryProtocol:
    """Trial timing in milliseconds and the noise/cue amplitudes."""

    baseline_ms: int = 200
    cue_ms: int = 300
    delay_ms: int = 1000
    readout_ms: int = 200
    tau_ms: float = 20.0
    dt_ms: float = 1.0
    cue_amplitude: float = 1.0
    noise_sd: float = 0.05
    activation: Literal["relu_tanh", "tanh_softplus"] = "relu_tanh"

    def __post_init__(self) -> None:
        if self.activation not in {"relu_tanh", "tanh_softplus"}:
            raise ValueError("activation must be 'relu_tanh' or 'tanh_softplus'")


def _activate(x: Tensor, activation: str) -> Tensor:
    """``relu_tanh`` = max(0, tanh x) (F1c); ``tanh_softplus`` = tanh(softplus x).

    The second is smooth and strictly positive, so gradients never vanish on a
    silent network (the F1c-v2 failure mode).
    """

    if activation == "relu_tanh":
        return torch.tanh(x).clamp(min=0.0)
    return torch.tanh(torch.nn.functional.softplus(x))


def simulate_memory(
    weights: Tensor,
    cue_inputs: Tensor,
    gain: Tensor,
    bias: Tensor,
    *,
    protocol: MemoryProtocol,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Simulate a batch of trials; returns late-delay and cue-period mean rates.

    ``cue_inputs`` is ``[batch, neurons]`` (zero outside cued neurons);
    ``gain`` and ``bias`` are either per-trial ``[batch]`` (so parameter grids
    can share one batch) or per-neuron ``[batch, neurons]``.  Dynamics:
    ``tau dr/dt = -r + max(0, tanh(g * (W r) + b + I + noise))``; the function
    is differentiable with respect to ``gain`` and ``bias``.
    """

    batch, neurons = cue_inputs.shape
    device = cue_inputs.device
    rate = torch.zeros(batch, neurons, device=device)
    step = protocol.dt_ms / protocol.tau_ms
    phases = (
        (protocol.baseline_ms, False),
        (protocol.cue_ms, True),
        (protocol.delay_ms, False),
    )
    cue_sum = torch.zeros(batch, neurons, device=device)
    late_sum = torch.zeros(batch, neurons, device=device)
    late_start = protocol.delay_ms - protocol.readout_ms
    g = gain[:, None] if gain.dim() == 1 else gain
    b = bias[:, None] if bias.dim() == 1 else bias
    for phase, (duration, cued) in enumerate(phases):
        steps = round(duration / protocol.dt_ms)
        for t in range(steps):
            drive = g * (rate @ weights.T) + b
            if cued:
                drive = drive + protocol.cue_amplitude * cue_inputs
            noise = (
                torch.randn(batch, neurons, generator=generator, device=device) * protocol.noise_sd
            )
            rate = rate + step * (-rate + _activate(drive + noise, protocol.activation))
            if cued:
                cue_sum = cue_sum + rate
            if phase == 2 and t * protocol.dt_ms >= late_start:
                late_sum = late_sum + rate
    readout_steps = round(protocol.readout_ms / protocol.dt_ms)
    cue_steps = round(protocol.cue_ms / protocol.dt_ms)
    return late_sum / readout_steps, cue_sum / cue_steps


def cue_bins(angles: FloatArray, n_bins: int) -> IntArray:
    width = 2.0 * math.pi / n_bins
    return np.minimum(((angles + math.pi) // width).astype(np.int64), n_bins - 1)


def template_accuracy(templates: Tensor, tests: Tensor, labels: Tensor) -> float:
    """Nearest-template (Pearson correlation) accuracy; silent or flat tests count as wrong."""

    def centered(x: Tensor) -> tuple[Tensor, Tensor]:
        x = x - x.mean(dim=-1, keepdim=True)
        norm = x.norm(dim=-1, keepdim=True)
        return x / norm.clamp(min=1e-12), norm.squeeze(-1) > 1e-9

    t, t_ok = centered(templates)
    x, x_ok = centered(tests)
    scores = x @ t.T
    scores[:, ~t_ok] = -math.inf
    predicted = scores.argmax(dim=1)
    correct = (predicted == labels) & x_ok & t_ok.any()
    return float(correct.float().mean())


@dataclass(frozen=True, slots=True)
class TypeParameters:
    """Per-cell-type gain ``exp(log_gain)`` and bias for postsynaptic neurons."""

    types: tuple[str, ...]
    log_gain: Tensor
    bias: Tensor

    def per_neuron(self, circuit: CXCircuit) -> tuple[Tensor, Tensor]:
        code = torch.tensor([self.types.index(t) for t in circuit.cell_type])
        return torch.exp(self.log_gain)[code], self.bias[code]


def train_type_parameters(
    circuit: CXCircuit,
    cue_inputs: Tensor,
    readout: IntArray,
    *,
    seed: int,
    steps: int = 200,
    lr: float = 0.05,
    trials_per_cue: int = 4,
    clip: float = 1.0,
    protocol: MemoryProtocol | None = None,
    homeostasis_weight: float = 0.0,
    homeostasis_target: float = 0.2,
) -> tuple[TypeParameters, list[float]]:
    """Fit cell-type gains/biases by BPTT on a cue-memory objective.

    The connectome (topology and synapse counts) stays fixed.  An auxiliary
    linear decoder reads the late-delay activity of ``readout`` neurons during
    training only; evaluation must not use it.  With ``homeostasis_weight > 0``
    the loss adds ``w * (mean late readout rate - target)^2``.
    """

    protocol = protocol or MemoryProtocol()
    types = tuple(sorted(set(circuit.cell_type)))
    code = torch.tensor([types.index(t) for t in circuit.cell_type])
    weights = torch.as_tensor(circuit.weights(), dtype=torch.float32)
    log_gain = torch.zeros(len(types), requires_grad=True)
    bias = torch.zeros(len(types), requires_grad=True)
    torch.manual_seed(seed)
    decoder = torch.nn.Linear(readout.size, cue_inputs.shape[0])
    parameters = [log_gain, bias, *decoder.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(cue_inputs.shape[0]).repeat_interleave(trials_per_cue)
    cues = cue_inputs.repeat_interleave(trials_per_cue, dim=0)
    index = torch.as_tensor(readout)
    history: list[float] = []
    for _ in range(steps):
        batch = cues.shape[0]
        gain = torch.exp(log_gain)[code].expand(batch, -1)
        offset = bias[code].expand(batch, -1)
        late, _ = simulate_memory(
            weights, cues, gain, offset, protocol=protocol, generator=generator
        )
        loss = torch.nn.functional.cross_entropy(decoder(late[:, index]), labels)
        if homeostasis_weight:
            loss = loss + homeostasis_weight * (late[:, index].mean() - homeostasis_target) ** 2
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(parameters, clip)
        optimizer.step()
        history.append(float(loss))
    return TypeParameters(types, log_gain.detach().clone(), bias.detach().clone()), history


def ideal_ring_circuit(
    n_ring: int = 48, n_inhibitory: int = 8, *, near: int = 10, far: int = 4
) -> CXCircuit:
    """Textbook ring attractor used as a positive control (F1c-v3).

    Excitatory "EPG" units couple to ring neighbours at distance 1 (``near``
    synapses) and 2 (``far``); inhibitory "Delta7" units receive 2 synapses from
    every EPG and return 3 synapses to every EPG.  Ring order equals index order.
    """

    pre: list[int] = []
    post: list[int] = []
    count: list[float] = []
    for i in range(n_ring):
        for distance, synapses in ((1, near), (2, far)):
            for j in ((i + distance) % n_ring, (i - distance) % n_ring):
                pre.append(i)
                post.append(j)
                count.append(float(synapses))
    for k in range(n_inhibitory):
        inhibitory = n_ring + k
        for i in range(n_ring):
            pre += [i, inhibitory]
            post += [inhibitory, i]
            count += [2.0, 3.0]
    n = n_ring + n_inhibitory
    return CXCircuit(
        ids=np.arange(n, dtype=np.int64),
        cell_type=("EPG",) * n_ring + ("Delta7",) * n_inhibitory,
        side=("right",) * n,
        sign=np.array([1.0] * n_ring + [-1.0] * n_inhibitory),
        edge_pre=np.array(pre, dtype=np.int64),
        edge_post=np.array(post, dtype=np.int64),
        edge_count=np.array(count),
        topology="ideal_ring",
    )


def simulate_rotation(
    weights: Tensor,
    cue_inputs: Tensor,
    drive_inputs: Tensor,
    gain: Tensor,
    bias: Tensor,
    *,
    protocol: MemoryProtocol,
    generator: torch.Generator,
    hold_ms: int = 500,
    drive_ms: int = 300,
    readout_ms: int = 100,
) -> Tensor:
    """Cue, hold, then apply ``drive_inputs``; return mean rates over the last ``readout_ms``.

    ``gain``/``bias`` are per-neuron ``[batch, neurons]``.  Dynamics and noise
    follow :func:`simulate_memory` (F1f path integration).
    """

    batch, neurons = cue_inputs.shape
    device = cue_inputs.device
    rate = torch.zeros(batch, neurons, device=device)
    step = protocol.dt_ms / protocol.tau_ms
    phases = (
        (protocol.baseline_ms, None),
        (protocol.cue_ms, cue_inputs),
        (hold_ms, None),
        (drive_ms, drive_inputs),
    )
    readout = torch.zeros(batch, neurons, device=device)
    readout_steps = round(readout_ms / protocol.dt_ms)
    for phase, (duration, extra) in enumerate(phases):
        steps = round(duration / protocol.dt_ms)
        for t in range(steps):
            drive = gain * (rate @ weights.T) + bias
            if extra is not None:
                drive = drive + (protocol.cue_amplitude if phase == 1 else 1.0) * extra
            noise = (
                torch.randn(batch, neurons, generator=generator, device=device) * protocol.noise_sd
            )
            rate = rate + step * (-rate + _activate(drive + noise, protocol.activation))
            if phase == 3 and t >= steps - readout_steps:
                readout = readout + rate
    return readout / readout_steps


def decode_angle(activity: Tensor, angles: FloatArray) -> Tensor:
    """Population-vector angle of ``activity[:, ring]`` given ring angles."""

    phi = torch.as_tensor(angles, dtype=activity.dtype, device=activity.device)
    return torch.atan2(activity @ torch.sin(phi), activity @ torch.cos(phi))


def _wrap(x: Tensor) -> Tensor:
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


def _circular_mean(theta: Tensor, dim: int) -> Tensor:
    return torch.atan2(torch.sin(theta).mean(dim), torch.cos(theta).mean(dim))


def antisymmetry_index(none: Tensor, left: Tensor, right: Tensor) -> tuple[float, float]:
    """Antisymmetry of left/right-driven bump shifts; inputs are ``[cues, trials]`` angles.

    Returns ``A = -sum dL*dR / (0.5 * sum(dL^2 + dR^2))`` over cues (1 for equal
    and opposite shifts, about 0 for unrelated, -1 for same-direction) and the
    mean absolute shift in radians.
    """

    base = _circular_mean(none, 1)
    d_left = _wrap(_circular_mean(left, 1) - base)
    d_right = _wrap(_circular_mean(right, 1) - base)
    energy = 0.5 * float((d_left**2 + d_right**2).sum())
    index = -float((d_left * d_right).sum()) / energy if energy > 0 else 0.0
    return index, float(torch.cat([d_left.abs(), d_right.abs()]).mean())


def ideal_rotating_ring(n_ring: int = 48, n_inhibitory: int = 8) -> CXCircuit:
    """Ring attractor with left/right PENs offset by +-1 (F1f-v2 positive control).

    Indices: EPG ``0..n-1`` (ring order), left PEN ``n..2n-1``, right PEN
    ``2n..3n-1``, Delta7 after that.  PEN_i receives 6 synapses from EPG_i and
    sends 6 to EPG_{i+1} (left) or EPG_{i-1} (right).
    """

    base = ideal_ring_circuit(n_ring, n_inhibitory)
    shift = 2 * n_ring  # PENs are inserted between EPGs and Delta7
    pre = [int(p) + (shift if p >= n_ring else 0) for p in base.edge_pre]
    post = [int(q) + (shift if q >= n_ring else 0) for q in base.edge_post]
    count = base.edge_count.tolist()
    for i in range(n_ring):
        for offset, pen in ((1, n_ring + i), (-1, 2 * n_ring + i)):
            pre += [i, pen]
            post += [pen, (i + offset) % n_ring]
            count += [6.0, 6.0]
    n = 3 * n_ring + n_inhibitory
    return CXCircuit(
        ids=np.arange(n, dtype=np.int64),
        cell_type=("EPG",) * n_ring + ("PEN_a(PEN1)",) * (2 * n_ring) + ("Delta7",) * n_inhibitory,
        side=("right",) * n_ring
        + ("left",) * n_ring
        + ("right",) * n_ring
        + ("right",) * n_inhibitory,
        sign=np.array([1.0] * (3 * n_ring) + [-1.0] * n_inhibitory),
        edge_pre=np.array(pre, dtype=np.int64),
        edge_post=np.array(post, dtype=np.int64),
        edge_count=np.array(count),
        topology="ideal_rotating_ring",
    )


@dataclass(frozen=True, slots=True)
class InputPathway:
    """External neurons projecting into a core circuit (no own dynamics).

    ``counts[core, source]`` are synapse counts onto core neurons; ``sign`` is
    per source neuron (from its transmitter).
    """

    ids: IntArray
    cell_type: tuple[str, ...]
    side: tuple[str, ...]
    sign: FloatArray
    counts: FloatArray

    def __post_init__(self) -> None:
        n = self.ids.shape[0]
        if len(self.cell_type) != n or len(self.side) != n or self.sign.shape != (n,):
            raise ValueError("source annotations must align with ids")
        if self.counts.ndim != 2 or self.counts.shape[1] != n:
            raise ValueError("counts must be [core, source]")

    def weights(self, scale: float, *, binary: bool = False, threshold: float = 5.0) -> FloatArray:
        counts = (self.counts >= threshold).astype(np.float64) if binary else self.counts
        return counts * self.sign[None, :] / scale

    def preferred_angles(
        self, circuit: CXCircuit, ring_angles: FloatArray, ring_type: str = "EPG"
    ) -> tuple[FloatArray, FloatArray]:
        """Synapse-weighted circular mean of target ring angles and its tuning strength."""

        ring = circuit.indices(ring_type)
        w = self.counts[ring, :]
        vector = (w * np.exp(1j * ring_angles)[:, None]).sum(axis=0)
        total = w.sum(axis=0)
        strength = np.where(total > 0, np.abs(vector) / np.where(total > 0, total, 1.0), 0.0)
        return np.angle(vector), strength


def extract_input_pathway(
    annotations: Any, connections: Any, circuit: CXCircuit, *, source_types: Iterable[str]
) -> InputPathway:
    sources = annotations[annotations.cell_type.isin(list(source_types))].sort_values("root_id")
    ids = sources.root_id.to_numpy().astype(np.int64)
    edges = connections[
        connections.pre_pt_root_id.isin(ids) & connections.post_pt_root_id.isin(circuit.ids)
    ]
    pairs = edges.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False).syn_count.sum()
    counts = np.zeros((circuit.n, ids.size))
    counts[
        np.searchsorted(circuit.ids, pairs.post_pt_root_id.to_numpy()),
        np.searchsorted(ids, pairs.pre_pt_root_id.to_numpy()),
    ] = pairs.syn_count.to_numpy()
    transmitter = sources.top_nt.astype(str).str.lower()
    return InputPathway(
        ids=ids,
        cell_type=tuple(sources.cell_type.astype(str)),
        side=tuple(sources.side.astype(str)),
        sign=np.where(transmitter.isin(INHIBITORY_TRANSMITTERS), -1.0, 1.0),
        counts=counts,
    )


def simulate_phases(
    weights: Tensor,
    gain: Tensor,
    bias: Tensor,
    phases: list[tuple[int, Tensor | None]],
    *,
    readout_phase: int,
    readout_ms: int,
    protocol: MemoryProtocol,
    generator: torch.Generator,
) -> Tensor:
    """Run ``(duration_ms, external_input [batch, neurons] or None)`` phases.

    Returns mean rates over the last ``readout_ms`` of ``readout_phase``.  Gain
    and bias are per-neuron ``[batch, neurons]``; dynamics as :func:`simulate_memory`.
    """

    batch, neurons = gain.shape
    device = gain.device
    rate = torch.zeros(batch, neurons, device=device)
    step = protocol.dt_ms / protocol.tau_ms
    readout = torch.zeros(batch, neurons, device=device)
    readout_steps = round(readout_ms / protocol.dt_ms)
    for index, (duration, external) in enumerate(phases):
        steps = round(duration / protocol.dt_ms)
        for t in range(steps):
            drive = gain * (rate @ weights.T) + bias
            if external is not None:
                drive = drive + external
            noise = (
                torch.randn(batch, neurons, generator=generator, device=device) * protocol.noise_sd
            )
            rate = rate + step * (-rate + _activate(drive + noise, protocol.activation))
            if index == readout_phase and t >= steps - readout_steps:
                readout = readout + rate
    return readout / readout_steps


def ideal_input_pathways(ring: CXCircuit, n_ring: int = 48) -> tuple[InputPathway, InputPathway]:
    """Ideal ER (ER_j inhibits EPG_j) and LNO (one per side, inhibits all same-side PENs)."""

    er_counts = np.zeros((ring.n, n_ring))
    er_counts[np.arange(n_ring), np.arange(n_ring)] = 6.0
    er = InputPathway(
        ids=np.arange(1000, 1000 + n_ring, dtype=np.int64),
        cell_type=("ER",) * n_ring,
        side=("right",) * n_ring,
        sign=-np.ones(n_ring),
        counts=er_counts,
    )
    pen = np.array([t.startswith("PEN") for t in ring.cell_type])
    side = np.array(ring.side)
    lno_counts = np.zeros((ring.n, 2))
    lno_counts[pen & (side == "left"), 0] = 6.0
    lno_counts[pen & (side == "right"), 1] = 6.0
    lno = InputPathway(
        ids=np.array([2000, 2001], dtype=np.int64),
        cell_type=("LNO", "LNO"),
        side=("left", "right"),
        sign=-np.ones(2),
        counts=lno_counts,
    )
    return er, lno


def pen_offsets(circuit: CXCircuit, ring_angles: FloatArray) -> dict[str, tuple[float, float]]:
    """Per (PEN type, side): circular mean of output-minus-input EPG angle and its length."""

    counts = circuit.counts()
    epg = circuit.indices("EPG")
    phase = np.exp(1j * ring_angles)
    out: dict[str, tuple[float, float]] = {}
    for kind in ("PEN_a(PEN1)", "PEN_b(PEN2)"):
        for side in ("left", "right"):
            members = [
                i
                for i, (t, s) in enumerate(zip(circuit.cell_type, circuit.side, strict=True))
                if t == kind and s == side
            ]
            offsets = []
            for i in members:
                inbound = (counts[i, epg] * phase).sum()
                outbound = (counts[epg, i] * phase).sum()
                if abs(inbound) > 0 and abs(outbound) > 0:
                    offsets.append(np.angle(outbound / inbound))
            if offsets:
                mean = np.exp(1j * np.array(offsets)).mean()
                out[f"{kind}|{side}"] = (float(np.angle(mean)), float(abs(mean)))
    return out
