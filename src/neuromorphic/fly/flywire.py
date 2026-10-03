"""Readers for FlyWire-derived connectome tables.

``pandas`` and ``pyarrow`` come from the optional ``fly`` extra and are imported
lazily so the main network never depends on them.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch

from neuromorphic.fly.connectome import Connectome

SIGNED_COUNT_COLUMN = "Excitatory x Connectivity"


def _pandas() -> Any:
    try:
        return importlib.import_module("pandas")
    except ModuleNotFoundError as error:  # pragma: no cover - environment guard
        raise ModuleNotFoundError("install the optional extra: pip install -e '.[fly]'") from error


def load_shiu_connectome(
    connectivity: Path, completeness: Path, *, dataset: str = "flywire", version: str
) -> Connectome:
    """Load the signed connectome used by Shiu et al. (2024).

    ``completeness`` lists one FlyWire root ID per row; its row number is the
    neuron index referenced by ``Presynaptic_Index``/``Postsynaptic_Index``.
    Edge signs are the paper's per-neuron excitatory/inhibitory assignment.
    """

    pd = _pandas()
    neuron_table = pd.read_csv(completeness, index_col=0)
    root_ids = torch.as_tensor(neuron_table.index.to_numpy(dtype="int64"))
    edges = pd.read_parquet(
        connectivity,
        columns=[
            "Presynaptic_ID",
            "Postsynaptic_ID",
            "Presynaptic_Index",
            "Postsynaptic_Index",
            SIGNED_COUNT_COLUMN,
        ],
    )
    pre_index = torch.as_tensor(edges["Presynaptic_Index"].to_numpy(dtype="int64"))
    post_index = torch.as_tensor(edges["Postsynaptic_Index"].to_numpy(dtype="int64"))
    pre_ids = torch.as_tensor(edges["Presynaptic_ID"].to_numpy(dtype="int64"))
    post_ids = torch.as_tensor(edges["Postsynaptic_ID"].to_numpy(dtype="int64"))
    if not (
        torch.equal(root_ids[pre_index], pre_ids) and torch.equal(root_ids[post_index], post_ids)
    ):
        raise ValueError("connectivity indices disagree with completeness row order")
    return Connectome.from_edges(
        dataset=dataset,
        version=version,
        pre_ids=pre_ids,
        post_ids=post_ids,
        signed_count=torch.as_tensor(edges[SIGNED_COUNT_COLUMN].to_numpy(dtype="int64")),
        neuron_ids=root_ids,
    )
