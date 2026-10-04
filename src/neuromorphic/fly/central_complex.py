"""Central-complex heading circuit: extraction, ring inference, nulls and dynamics.

The core is the EPG/EPGt/PEN/PEG/Delta7 network (F1c).  Ring angles are
inferred from connectivity alone, and a signed rate model tests whether a cue
leaves a persistent, cue-specific state.  Protocol: docs/fly/F1c-ring-protocol.md.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

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

    def weights(self) -> FloatArray:
        """Signed weights ``W[post, pre]`` scaled by the median absolute input."""

        signed = self.counts() * self.sign[None, :]
        scale = float(np.median(np.abs(signed).sum(axis=1)))
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
    rate = torch.zeros(batch, neurons)
    step = protocol.dt_ms / protocol.tau_ms
    phases = (
        (protocol.baseline_ms, False),
        (protocol.cue_ms, True),
        (protocol.delay_ms, False),
    )
    cue_sum = torch.zeros(batch, neurons)
    late_sum = torch.zeros(batch, neurons)
    late_start = protocol.delay_ms - protocol.readout_ms
    g = gain[:, None] if gain.dim() == 1 else gain
    b = bias[:, None] if bias.dim() == 1 else bias
    for phase, (duration, cued) in enumerate(phases):
        steps = round(duration / protocol.dt_ms)
        for t in range(steps):
            drive = g * (rate @ weights.T) + b
            if cued:
                drive = drive + protocol.cue_amplitude * cue_inputs
            noise = torch.randn(batch, neurons, generator=generator) * protocol.noise_sd
            rate = rate + step * (-rate + torch.tanh(drive + noise).clamp(min=0.0))
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
) -> tuple[TypeParameters, list[float]]:
    """Fit cell-type gains/biases by BPTT on a cue-memory objective.

    The connectome (topology and synapse counts) stays fixed.  An auxiliary
    linear decoder reads the late-delay activity of ``readout`` neurons during
    training only; evaluation must not use it.
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
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(parameters, clip)
        optimizer.step()
        history.append(float(loss))
    return TypeParameters(types, log_gain.detach().clone(), bias.detach().clone()), history
