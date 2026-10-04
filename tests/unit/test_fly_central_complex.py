from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from neuromorphic.fly.central_complex import (
    CXCircuit,
    MemoryProtocol,
    cue_bins,
    degree_preserving_directed,
    extract_cx,
    ideal_ring_circuit,
    ring_embedding,
    simulate_memory,
    template_accuracy,
    train_type_parameters,
    type_preserving,
)

pd = pytest.importorskip("pandas")


def _ring(n: int = 16) -> CXCircuit:
    """EPG ring with nearest-neighbour coupling plus an inhibitory hub type."""

    pre, post, count = [], [], []
    for i in range(n):
        for d in (1, 2):
            for j in ((i + d) % n, (i - d) % n):
                pre.append(i)
                post.append(j)
                count.append(10 if d == 1 else 4)
        pre += [i, n + i % 4]
        post += [n + i % 4, i]
        count += [3, 2]
    types = ("EPG",) * n + ("Delta7",) * 4
    sign = np.array([1.0] * n + [-1.0] * 4)
    keys: dict[tuple[int, int], int] = {}
    for a, b, c in zip(pre, post, count, strict=True):
        keys[(a, b)] = keys.get((a, b), 0) + c
    return CXCircuit(
        ids=np.arange(n + 4, dtype=np.int64),
        cell_type=types,
        side=("right",) * (n + 4),
        sign=sign,
        edge_pre=np.array([k[0] for k in keys], dtype=np.int64),
        edge_post=np.array([k[1] for k in keys], dtype=np.int64),
        edge_count=np.array(list(keys.values()), dtype=np.float64),
    )


def _degrees(c: CXCircuit) -> tuple[np.ndarray, np.ndarray]:
    return np.bincount(c.edge_pre, minlength=c.n), np.bincount(c.edge_post, minlength=c.n)


def test_nulls_preserve_degrees_and_type_null_preserves_type_matrix() -> None:
    real = _ring()
    for null in (
        degree_preserving_directed(real, np.random.default_rng(0)),
        type_preserving(real, np.random.default_rng(1)),
    ):
        for a, b in zip(_degrees(null), _degrees(real), strict=True):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(
            np.sort(null.counts().sum(axis=0)), np.sort(real.counts().sum(axis=0))
        )
        assert not np.array_equal(null.counts(), real.counts())
    typed = type_preserving(real, np.random.default_rng(2))
    t = np.array(real.cell_type)

    def type_matrix(c: CXCircuit) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        for a, b, w in zip(c.edge_pre, c.edge_post, c.edge_count, strict=True):
            out[(t[a], t[b])] = out.get((t[a], t[b]), 0.0) + w
        return out

    assert type_matrix(typed) == type_matrix(real)


def test_ring_embedding_recovers_cyclic_order() -> None:
    angles, eigenvalues = ring_embedding(_ring())
    order = np.argsort(angles)
    steps = np.diff(np.concatenate([order, order[:1]]))
    assert np.all(np.isin(steps % 16, (1, 15)))  # neighbours stay neighbours
    assert eigenvalues[1] == pytest.approx(eigenvalues[2], rel=1e-6)


def test_extract_signs_follow_presynaptic_transmitter() -> None:
    annotations = pd.DataFrame(
        {
            "root_id": [5, 3, 9],
            "cell_type": ["EPG", "Delta7", "Other"],
            "side": ["left", "right", "left"],
            "top_nt": ["acetylcholine", "glutamate", "gaba"],
        }
    )
    connections = pd.DataFrame(
        {"pre_pt_root_id": [3, 3, 5, 9], "post_pt_root_id": [5, 5, 3, 5], "syn_count": [2, 1, 4, 7]}
    )
    circuit = extract_cx(annotations, connections, types=("EPG", "Delta7"))
    assert circuit.ids.tolist() == [3, 5]
    assert circuit.sign.tolist() == [-1.0, 1.0]
    assert circuit.counts().tolist() == [[0.0, 4.0], [3.0, 0.0]]
    assert circuit.weights()[1, 0] < 0


def test_simulation_without_recurrence_forgets_the_cue() -> None:
    protocol = MemoryProtocol(baseline_ms=10, cue_ms=50, delay_ms=400, readout_ms=50, noise_sd=0.0)
    cue = torch.zeros(2, 3)
    cue[0, 0] = 1.0
    late, during = simulate_memory(
        torch.zeros(3, 3),
        cue,
        torch.ones(2),
        torch.zeros(2),
        protocol=protocol,
        generator=torch.Generator().manual_seed(0),
    )
    assert during[0, 0] > 0.3
    assert late.abs().max() < 1e-4


def test_template_accuracy_and_cue_bins() -> None:
    templates = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    tests = torch.tensor([[0.9, 0.1, 0.0], [0.0, 0.0, 0.0], [0.1, 0.8, 0.0]])
    assert template_accuracy(templates, tests, torch.tensor([0, 1, 1])) == pytest.approx(2 / 3)
    bins = cue_bins(np.array([-math.pi, -0.1, 0.1, math.pi - 1e-9]), 4)
    assert bins.tolist() == [0, 1, 2, 3]


def test_type_training_keeps_topology_and_reduces_loss() -> None:
    ring = _ring()
    protocol = MemoryProtocol(baseline_ms=5, cue_ms=20, delay_ms=60, readout_ms=20, noise_sd=0.01)
    epg = ring.indices("EPG")
    cues = torch.zeros(4, ring.n)
    for k in range(4):
        cues[k, epg[k * 4 : k * 4 + 4]] = 1.0
    before = ring.counts().copy()
    params, history = train_type_parameters(
        ring, cues, epg, seed=0, steps=25, lr=0.1, trials_per_cue=2, protocol=protocol
    )
    np.testing.assert_array_equal(ring.counts(), before)
    assert params.types == ("Delta7", "EPG")
    assert np.isfinite(history).all()
    assert np.mean(history[-5:]) < np.mean(history[:5])
    gain, bias = params.per_neuron(ring)
    assert gain.shape == (ring.n,) and bias.shape == (ring.n,)
    assert torch.all(gain > 0)


def test_smooth_activation_never_silences_gradients() -> None:
    protocol = MemoryProtocol(
        baseline_ms=5, cue_ms=10, delay_ms=40, readout_ms=10, activation="tanh_softplus"
    )
    bias = torch.full((2, 3), -20.0, requires_grad=True)
    late, _ = simulate_memory(
        torch.zeros(3, 3),
        torch.zeros(2, 3),
        torch.ones(2),
        bias,
        protocol=protocol,
        generator=torch.Generator().manual_seed(0),
    )
    late.sum().backward()  # type: ignore[no-untyped-call]
    assert bias.grad is not None and torch.all(bias.grad > 0)
    with pytest.raises(ValueError, match="activation"):
        MemoryProtocol(activation="sigmoid")  # type: ignore[arg-type]


def test_ideal_ring_is_a_ring() -> None:
    ring = ideal_ring_circuit()
    assert ring.n == 56
    angles, eigenvalues = ring_embedding(ring)
    order = np.argsort(angles)
    steps = np.diff(np.concatenate([order, order[:1]])) % 48
    assert np.all(np.isin(steps, (1, 47)))
    assert eigenvalues[1] == pytest.approx(eigenvalues[2], rel=1e-6)  # exact ring symmetry
