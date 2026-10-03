from __future__ import annotations

import pytest
import torch

from neuromorphic.fly import Connectome


def _toy() -> Connectome:
    # Root IDs are deliberately large and unsorted, as in FlyWire tables.
    return Connectome.from_edges(
        dataset="toy",
        version="1",
        pre_ids=[900, 100, 100, 500, 500, 900],
        post_ids=[100, 500, 500, 900, 100, 500],
        signed_count=[3, 2, 4, -6, 6, -1],
        neuron_ids=[100, 500, 900, 700],
    )


def test_duplicate_pairs_are_summed_and_isolated_neurons_kept() -> None:
    graph = _toy()
    assert graph.neuron_ids.tolist() == [100, 500, 700, 900]
    edges = {
        (int(graph.neuron_ids[pre]), int(graph.neuron_ids[post])): int(count)
        for pre, post, count in zip(graph.pre, graph.post, graph.signed_count, strict=True)
    }
    assert edges == {(900, 100): 3, (500, 100): 6, (100, 500): 6, (900, 500): -1, (500, 900): -6}
    assert graph.n_neurons == 4
    assert graph.n_edges == 5
    assert graph.n_synapses == 22


def test_cancelling_rows_are_dropped() -> None:
    graph = Connectome.from_edges(
        dataset="toy", version="1", pre_ids=[1, 1], post_ids=[2, 2], signed_count=[5, -5]
    )
    assert graph.n_neurons == 2
    assert graph.n_edges == 0


def test_index_of_rejects_unknown_ids() -> None:
    graph = _toy()
    assert graph.index_of([900, 100]).tolist() == [3, 0]
    with pytest.raises(KeyError, match="absent"):
        graph.index_of([101])


def test_explicit_neuron_ids_must_cover_edges_and_be_unique() -> None:
    with pytest.raises(KeyError, match="absent"):
        Connectome.from_edges(
            dataset="toy", version="1", pre_ids=[1], post_ids=[2], signed_count=[1], neuron_ids=[1]
        )
    with pytest.raises(ValueError, match="unique"):
        Connectome.from_edges(
            dataset="toy",
            version="1",
            pre_ids=[1],
            post_ids=[2],
            signed_count=[1],
            neuron_ids=[1, 2, 2],
        )


def test_constructor_rejects_unsorted_or_zero_edges() -> None:
    ids = torch.tensor([1, 2], dtype=torch.int64)
    with pytest.raises(ValueError, match="sorted"):
        Connectome(
            dataset="toy",
            version="1",
            neuron_ids=ids,
            pre=torch.tensor([1, 0]),
            post=torch.tensor([1, 0]),
            signed_count=torch.tensor([1, 1], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="zero"):
        Connectome(
            dataset="toy",
            version="1",
            neuron_ids=ids,
            pre=torch.tensor([0]),
            post=torch.tensor([1]),
            signed_count=torch.tensor([0], dtype=torch.int32),
        )
    with pytest.raises(TypeError, match="integers"):
        Connectome.from_edges(
            dataset="toy", version="1", pre_ids=torch.tensor([1.0]), post_ids=[2], signed_count=[1]
        )


def test_weight_matrix_is_post_by_pre_and_scaled() -> None:
    graph = _toy()
    dense = graph.weight_matrix(scale=0.5, device=torch.device("cpu")).to_dense()
    index = {root: i for i, root in enumerate(graph.neuron_ids.tolist())}
    assert dense[index[100], index[500]] == pytest.approx(3.0)
    assert dense[index[500], index[900]] == pytest.approx(-0.5)
    assert dense[index[700]].abs().sum() == 0
    spikes = torch.zeros(4, 1)
    spikes[index[500]] = 1.0
    drive = graph.weight_matrix(scale=1.0, device=torch.device("cpu")) @ spikes
    assert drive[:, 0].tolist() == [6.0, 0.0, 0.0, -6.0]


@pytest.mark.mps
def test_weight_matrix_matches_cpu_on_mps() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    graph = _toy()
    device = torch.device("mps")
    spikes = torch.eye(4)
    expected = graph.weight_matrix(scale=0.25, device=torch.device("cpu")) @ spikes
    actual = torch.sparse.mm(graph.weight_matrix(scale=0.25, device=device), spikes.to(device))
    torch.testing.assert_close(actual.cpu(), expected)
