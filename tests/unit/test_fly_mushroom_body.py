from __future__ import annotations

import numpy as np
import pytest
import torch

from neuromorphic.fly.continual import (
    AssociativeReadout,
    NearestClassMean,
    NearestNeighbor,
    PerceptronReadout,
    kc_code,
    make_multimodal_task,
    make_odor_task,
    run_class_incremental,
)
from neuromorphic.fly.mushroom_body import (
    PNKCCircuit,
    degree_preserving,
    erdos_renyi,
    extract_pn_kc,
)

pd = pytest.importorskip("pandas")


def _circuit(seed: int = 0, n_kc: int = 60, n_pn: int = 12, per_kc: int = 4) -> PNKCCircuit:
    rng = np.random.default_rng(seed)
    kc = np.repeat(np.arange(n_kc), per_kc)
    pn = np.concatenate([rng.choice(n_pn, per_kc, replace=False) for _ in range(n_kc)])
    return PNKCCircuit(
        pn_ids=np.arange(100, 100 + n_pn, dtype=np.int64),
        pn_glomerulus=tuple(f"G{i // 2}" for i in range(n_pn)),
        kc_ids=np.arange(1000, 1000 + n_kc, dtype=np.int64),
        kc_type=("KCg-m",) * n_kc,
        edge_kc=kc.astype(np.int64),
        edge_pn=pn.astype(np.int64),
        edge_weight=rng.integers(1, 20, kc.size).astype(np.float64),
    )


def test_degree_preserving_rewire_keeps_degrees_and_kc_weights() -> None:
    real = _circuit()
    null = degree_preserving(real, np.random.default_rng(1))
    assert null.topology == "degree_preserving"
    np.testing.assert_array_equal(null.kc_in_degree(), real.kc_in_degree())
    np.testing.assert_array_equal(null.pn_out_degree(), real.pn_out_degree())
    np.testing.assert_allclose(
        null.dense().sum(axis=1), real.dense().sum(axis=1)
    )  # incoming weight per KC
    assert not np.array_equal(null.dense(binary=True), real.dense(binary=True))


def test_erdos_renyi_keeps_edge_count_and_weight_multiset() -> None:
    real = _circuit()
    null = erdos_renyi(real, np.random.default_rng(2))
    assert null.n_edges == real.n_edges
    np.testing.assert_array_equal(np.sort(null.edge_weight), np.sort(real.edge_weight))


def test_circuit_rejects_duplicate_pairs_and_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    real = _circuit()
    path = tmp_path / "mb.npz"
    real.save(path)
    loaded = PNKCCircuit.load(path)
    np.testing.assert_array_equal(loaded.dense(), real.dense())
    assert loaded.pn_glomerulus == real.pn_glomerulus
    with pytest.raises(ValueError, match="unique"):
        PNKCCircuit(
            pn_ids=real.pn_ids,
            pn_glomerulus=real.pn_glomerulus,
            kc_ids=real.kc_ids,
            kc_type=real.kc_type,
            edge_kc=np.array([0, 0], dtype=np.int64),
            edge_pn=np.array([1, 1], dtype=np.int64),
            edge_weight=np.array([1.0, 2.0]),
        )


def test_extract_sums_neuropils_and_drops_unconnected_neurons() -> None:
    annotations = pd.DataFrame(
        {
            "root_id": [1, 2, 3, 10, 11, 12],
            "cell_class": ["ALPN", "ALPN", "ALPN", "Kenyon_Cell", "Kenyon_Cell", "Kenyon_Cell"],
            "cell_sub_class": ["uniglomerular", "uniglomerular", "multiglomerular", "", "", ""],
            "cell_type": ["DA1_lPN", "VA1v_adPN", "M_PN", "KCab", "KCg-m", "KCg-d"],
            "side": ["right", "right", "right", "right", "right", "left"],
        }
    )
    connections = pd.DataFrame(
        {
            "pre_pt_root_id": [1, 1, 2, 3, 1],
            "post_pt_root_id": [10, 10, 11, 10, 12],
            "syn_count": [4, 3, 6, 9, 5],
        }
    )
    circuit = extract_pn_kc(annotations, connections)
    assert circuit.pn_ids.tolist() == [1, 2]
    assert circuit.pn_glomerulus == ("DA1", "VA1v")
    assert circuit.kc_ids.tolist() == [10, 11]
    assert circuit.dense().tolist() == [[7.0, 0.0], [0.0, 6.0]]


def test_kc_code_is_k_sparse_binary_and_requires_positive_drive() -> None:
    weights = torch.eye(10)
    pn = torch.tensor([[5.0, 4.0, 3.0, 0, 0, 0, 0, 0, 0, 0], [0.0] * 10])
    code = kc_code(pn, weights, sparsity=0.2)
    assert code[0].tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert code[1].sum() == 0


def test_associative_readout_never_changes_other_classes() -> None:
    readout = AssociativeReadout(4, 3)
    readout.fit_task(torch.tensor([[1.0, 0, 1, 0]]), torch.tensor([0]))
    before = readout.weight.clone()
    readout.fit_task(torch.tensor([[0.0, 1, 1, 0]]), torch.tensor([2]))
    torch.testing.assert_close(readout.weight[:2], before[:2])
    assert readout.predict(torch.tensor([[0.0, 1, 0, 0]]), seen=3).item() == 2


def test_perceptron_depresses_the_wrong_winner() -> None:
    readout = PerceptronReadout(2, 2, epochs=1, seed=0)
    readout.fit_task(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
    readout.fit_task(torch.tensor([[1.0, 0.0]]), torch.tensor([1]))
    assert readout.weight[0, 0] < 1.0


def test_class_incremental_runner_tracks_forgetting() -> None:
    rng = np.random.default_rng(0)
    task = make_odor_task(
        np.arange(8) // 2,
        n_glomeruli=4,
        n_classes=4,
        n_train=20,
        n_test=20,
        sigma=0.1,
        rng=rng,
        active_probability=0.6,
    )
    result = run_class_incremental(
        NearestClassMean(8, 4),
        task.train_x,
        task.train_y,
        task.test_x,
        task.test_y,
        classes_per_task=2,
    )
    assert result.accuracy.shape == (2, 2)
    assert result.accuracy[1, 1] > 0
    assert result.final_accuracy == pytest.approx(result.accuracy[1].mean())
    assert result.forgetting == pytest.approx(result.accuracy[0, 0] - result.accuracy[1, 0])


def test_multimodal_task_labels_modes_and_classes() -> None:
    task = make_multimodal_task(
        np.arange(6) // 2,
        n_glomeruli=3,
        n_classes=2,
        modes=3,
        n_train_per_mode=4,
        n_test_per_mode=2,
        sigma=0.1,
        rng=np.random.default_rng(0),
    )
    assert task.train_mode is not None and task.test_mode is not None
    assert task.train_x.shape == (24, 6)
    assert torch.equal(task.train_y, task.train_mode // 3)
    assert task.test_y.bincount().tolist() == [6, 6]


def test_nearest_neighbor_restricts_to_seen_classes() -> None:
    learner = NearestNeighbor()
    learner.fit_task(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]))
    learner.fit_task(torch.tensor([[1.0, 0.1]]), torch.tensor([2]))
    query = torch.tensor([[1.0, 0.1]])
    assert learner.predict(query, seen=3).item() == 2
    assert learner.predict(query, seen=2).item() == 0
