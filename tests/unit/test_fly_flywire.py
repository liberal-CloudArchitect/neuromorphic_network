from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neuromorphic.fly.flywire import SIGNED_COUNT_COLUMN, load_shiu_connectome

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _write(tmp_path: Path, pre_index: list[int]) -> tuple[Path, Path]:
    roots = [700, 100, 500]
    completeness = tmp_path / "completeness.csv"
    pd.DataFrame({"Completed": [True] * 3}, index=roots).to_csv(completeness)
    post_index = [1, 2, 0]
    edges = pd.DataFrame(
        {
            "Presynaptic_ID": [roots[i] for i in [0, 1, 2]],
            "Postsynaptic_ID": [roots[i] for i in post_index],
            "Presynaptic_Index": pre_index,
            "Postsynaptic_Index": post_index,
            "Connectivity": [4, 2, 7],
            "Excitatory": [1, -1, 1],
            SIGNED_COUNT_COLUMN: [4, -2, 7],
        }
    )
    connectivity = tmp_path / "connectivity.parquet"
    edges.to_parquet(connectivity)
    return connectivity, completeness


def test_loader_keeps_root_ids_and_signs(tmp_path: Path) -> None:
    connectivity, completeness = _write(tmp_path, [0, 1, 2])
    graph = load_shiu_connectome(connectivity, completeness, version="toy")
    assert graph.neuron_ids.tolist() == [100, 500, 700]
    dense = graph.weight_matrix(scale=1.0, device=torch.device("cpu")).to_dense()
    row = {root: i for i, root in enumerate(graph.neuron_ids.tolist())}
    assert dense[row[100], row[700]] == 4
    assert dense[row[500], row[100]] == -2
    assert dense[row[700], row[500]] == 7


def test_loader_rejects_index_id_mismatch(tmp_path: Path) -> None:
    connectivity, completeness = _write(tmp_path, [1, 1, 2])
    with pytest.raises(ValueError, match="row order"):
        load_shiu_connectome(connectivity, completeness, version="toy")
