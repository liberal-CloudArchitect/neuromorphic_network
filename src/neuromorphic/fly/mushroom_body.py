"""Mushroom-body PN->KC circuit extraction and topology null models.

The circuit is the olfactory input layer of one hemisphere: uniglomerular
antennal-lobe projection neurons (PNs) onto Kenyon cells (KCs).  Null models
keep chosen statistics of the real bipartite graph so that a "real wiring"
claim can be separated from degree or density effects (see CR-013).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neuromorphic.fly.flywire import _pandas

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PNKCCircuit:
    """Bipartite PN->KC graph as an edge list with synapse-count weights."""

    pn_ids: IntArray
    pn_glomerulus: tuple[str, ...]
    kc_ids: IntArray
    kc_type: tuple[str, ...]
    edge_kc: IntArray
    edge_pn: IntArray
    edge_weight: FloatArray
    topology: str = "real"

    def __post_init__(self) -> None:
        edges = self.edge_kc.shape[0]
        if self.edge_pn.shape != (edges,) or self.edge_weight.shape != (edges,):
            raise ValueError("edge arrays must have equal length")
        if len(self.pn_glomerulus) != self.n_pn or len(self.kc_type) != self.n_kc:
            raise ValueError("annotations must align with neuron ids")
        if edges and (self.edge_kc.max() >= self.n_kc or self.edge_pn.max() >= self.n_pn):
            raise ValueError("edge indices out of range")
        if np.any(self.edge_weight <= 0):
            raise ValueError("edge weights must be positive synapse counts")
        pairs = self.edge_kc * self.n_pn + self.edge_pn
        if np.unique(pairs).size != edges:
            raise ValueError("PN->KC edges must be unique pairs")

    @property
    def n_pn(self) -> int:
        return int(self.pn_ids.shape[0])

    @property
    def n_kc(self) -> int:
        return int(self.kc_ids.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_kc.shape[0])

    def kc_in_degree(self) -> IntArray:
        return np.bincount(self.edge_kc, minlength=self.n_kc).astype(np.int64)

    def pn_out_degree(self) -> IntArray:
        return np.bincount(self.edge_pn, minlength=self.n_pn).astype(np.int64)

    def dense(self, *, binary: bool = False) -> FloatArray:
        """Return ``W[kc, pn]`` with synapse counts (or 1 for binary)."""

        matrix = np.zeros((self.n_kc, self.n_pn), dtype=np.float64)
        matrix[self.edge_kc, self.edge_pn] = 1.0 if binary else self.edge_weight
        return matrix

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            pn_ids=self.pn_ids,
            pn_glomerulus=np.array(self.pn_glomerulus),
            kc_ids=self.kc_ids,
            kc_type=np.array(self.kc_type),
            edge_kc=self.edge_kc,
            edge_pn=self.edge_pn,
            edge_weight=self.edge_weight,
            topology=np.array(self.topology),
        )

    @classmethod
    def load(cls, path: Path) -> PNKCCircuit:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                pn_ids=data["pn_ids"].astype(np.int64),
                pn_glomerulus=tuple(str(item) for item in data["pn_glomerulus"]),
                kc_ids=data["kc_ids"].astype(np.int64),
                kc_type=tuple(str(item) for item in data["kc_type"]),
                edge_kc=data["edge_kc"].astype(np.int64),
                edge_pn=data["edge_pn"].astype(np.int64),
                edge_weight=data["edge_weight"].astype(np.float64),
                topology=str(data["topology"]),
            )


def extract_pn_kc(annotations: Any, connections: Any, *, side: str = "right") -> PNKCCircuit:
    """Build the PN->KC circuit of one hemisphere from FlyWire tables.

    ``annotations`` is the FlyWire neuron annotation table (Schlegel et al.
    2024 supplemental file 1) and ``connections`` the proofread edge table with
    ``pre_pt_root_id``, ``post_pt_root_id`` and ``syn_count`` columns.  Synapse
    counts are summed over neuropils.  Only KCs with at least one PN input and
    PNs with at least one KC output are kept.
    """

    kcs = annotations[(annotations.cell_class == "Kenyon_Cell") & (annotations.side == side)]
    pns = annotations[
        (annotations.cell_class == "ALPN") & (annotations.cell_sub_class == "uniglomerular")
    ]
    edges = connections[
        connections.pre_pt_root_id.isin(pns.root_id) & connections.post_pt_root_id.isin(kcs.root_id)
    ]
    pairs = edges.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False).syn_count.sum()
    pn_ids = np.sort(pairs.pre_pt_root_id.unique()).astype(np.int64)
    kc_ids = np.sort(pairs.post_pt_root_id.unique()).astype(np.int64)
    pn_type = pns.set_index("root_id").cell_type.astype(str)
    kc_type = kcs.set_index("root_id").cell_type.astype(str)
    glomeruli = tuple(pn_type.loc[pn_ids].str.replace(r"_.*$", "", regex=True))
    return PNKCCircuit(
        pn_ids=pn_ids,
        pn_glomerulus=glomeruli,
        kc_ids=kc_ids,
        kc_type=tuple(kc_type.loc[kc_ids]),
        edge_kc=np.searchsorted(kc_ids, pairs.post_pt_root_id.to_numpy()).astype(np.int64),
        edge_pn=np.searchsorted(pn_ids, pairs.pre_pt_root_id.to_numpy()).astype(np.int64),
        edge_weight=pairs.syn_count.to_numpy().astype(np.float64),
    )


def load_flywire_pn_kc(
    annotation_tsv: Path, connections_feather: Path, *, side: str = "right"
) -> PNKCCircuit:
    pd = _pandas()
    annotations = pd.read_csv(annotation_tsv, sep="\t", low_memory=False)
    connections = pd.read_feather(
        connections_feather, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"]
    )
    return extract_pn_kc(annotations, connections, side=side)


def degree_preserving(
    circuit: PNKCCircuit, rng: np.random.Generator, *, swaps_per_edge: int = 10
) -> PNKCCircuit:
    """Rewire by bipartite double-edge swaps.

    Each accepted swap exchanges the PN endpoints of two edges, so every KC's
    in-degree and incoming weights and every PN's out-degree are preserved
    exactly; swaps that would create a duplicate pair are rejected.
    """

    kc = circuit.edge_kc.copy()
    pn = circuit.edge_pn.copy()
    n_pn = circuit.n_pn
    existing = set((kc * n_pn + pn).tolist())
    attempts = swaps_per_edge * circuit.n_edges
    first = rng.integers(0, circuit.n_edges, attempts)
    second = rng.integers(0, circuit.n_edges, attempts)
    for a, b in zip(first.tolist(), second.tolist(), strict=True):
        if kc[a] == kc[b] or pn[a] == pn[b]:
            continue
        new_a = int(kc[a]) * n_pn + int(pn[b])
        new_b = int(kc[b]) * n_pn + int(pn[a])
        if new_a in existing or new_b in existing:
            continue
        existing.difference_update((int(kc[a]) * n_pn + int(pn[a]), int(kc[b]) * n_pn + int(pn[b])))
        existing.update((new_a, new_b))
        pn[a], pn[b] = pn[b], pn[a]
    return replace(circuit, edge_pn=pn, topology="degree_preserving")


def erdos_renyi(circuit: PNKCCircuit, rng: np.random.Generator) -> PNKCCircuit:
    """Same number of edges placed uniformly; weights permuted from the real graph."""

    chosen = rng.choice(circuit.n_kc * circuit.n_pn, size=circuit.n_edges, replace=False)
    return replace(
        circuit,
        edge_kc=(chosen // circuit.n_pn).astype(np.int64),
        edge_pn=(chosen % circuit.n_pn).astype(np.int64),
        edge_weight=rng.permutation(circuit.edge_weight),
        topology="er",
    )
