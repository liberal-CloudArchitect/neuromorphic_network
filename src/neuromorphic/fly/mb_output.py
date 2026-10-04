"""Mushroom-body output loop: DAN->MBON teaching and MBON->DAN feedback (F1e).

Protocol: docs/fly/F1e-feedback-loop-protocol.md.  Everything here is
structural; no learning dynamics are simulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

INHIBITORY = frozenset({"gaba", "glutamate"})


@dataclass(frozen=True, slots=True)
class OutputLoop:
    """``teach[d, m]``: DAN->MBON synapses; ``feedback[m, d]``: MBON->DAN synapses."""

    dan_ids: IntArray
    dan_type: tuple[str, ...]
    mbon_ids: IntArray
    mbon_type: tuple[str, ...]
    mbon_inhibitory: BoolArray
    teach: FloatArray
    feedback: FloatArray

    def __post_init__(self) -> None:
        d, m = self.dan_ids.size, self.mbon_ids.size
        if self.teach.shape != (d, m) or self.feedback.shape != (m, d):
            raise ValueError("teach must be [DAN, MBON] and feedback [MBON, DAN]")
        if self.mbon_inhibitory.shape != (m,):
            raise ValueError("one transmitter flag per MBON")


def extract_output_loop(annotations: Any, connections: Any, *, side: str = "right") -> OutputLoop:
    dans = annotations[(annotations.cell_class == "DAN") & (annotations.side == side)]
    mbons = annotations[(annotations.cell_class == "MBON") & (annotations.side == side)]
    dans = dans.sort_values("root_id")
    mbons = mbons.sort_values("root_id")
    dan_ids = dans.root_id.to_numpy().astype(np.int64)
    mbon_ids = mbons.root_id.to_numpy().astype(np.int64)

    def matrix(pre_ids: IntArray, post_ids: IntArray) -> FloatArray:
        edges = connections[
            connections.pre_pt_root_id.isin(pre_ids) & connections.post_pt_root_id.isin(post_ids)
        ]
        pairs = edges.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False).syn_count.sum()
        out = np.zeros((pre_ids.size, post_ids.size))
        out[
            np.searchsorted(pre_ids, pairs.pre_pt_root_id.to_numpy()),
            np.searchsorted(post_ids, pairs.post_pt_root_id.to_numpy()),
        ] = pairs.syn_count.to_numpy()
        return out

    return OutputLoop(
        dan_ids=dan_ids,
        dan_type=tuple(dans.cell_type.astype(str)),
        mbon_ids=mbon_ids,
        mbon_type=tuple(mbons.cell_type.astype(str)),
        mbon_inhibitory=mbons.top_nt.astype(str).str.lower().isin(INHIBITORY).to_numpy(),
        teach=matrix(dan_ids, mbon_ids),
        feedback=matrix(mbon_ids, dan_ids),
    )


def reciprocal_mask(loop: OutputLoop, threshold: float) -> BoolArray:
    """``mask[m, d]`` is True when DAN d teaches MBON m with >= ``threshold`` synapses."""

    return (loop.teach >= threshold).T


def reciprocal_fraction(feedback: FloatArray, mask: BoolArray) -> float:
    total = float(feedback.sum())
    return float(feedback[mask].sum()) / total if total > 0 else float("nan")


def rewire_feedback(
    feedback: FloatArray, rng: np.random.Generator, *, swaps_per_edge: int = 10
) -> FloatArray:
    """Degree-preserving bipartite rewiring of MBON->DAN edges (weights move with edges)."""

    src, dst = np.nonzero(feedback)
    weight = feedback[src, dst]
    dst = dst.copy()
    n_dst = feedback.shape[1]
    existing = set((src * n_dst + dst).tolist())
    edges = src.size
    first = rng.integers(0, edges, swaps_per_edge * edges)
    second = rng.integers(0, edges, swaps_per_edge * edges)
    for a, b in zip(first.tolist(), second.tolist(), strict=True):
        sa, sb, da, db = int(src[a]), int(src[b]), int(dst[a]), int(dst[b])
        if sa == sb or da == db:
            continue
        new_a, new_b = sa * n_dst + db, sb * n_dst + da
        if new_a in existing or new_b in existing:
            continue
        existing.difference_update((sa * n_dst + da, sb * n_dst + db))
        existing.update((new_a, new_b))
        dst[a], dst[b] = db, da
    out = np.zeros_like(feedback)
    out[src, dst] = weight
    return out


def inhibitory_enrichment(feedback: FloatArray, mask: BoolArray, inhibitory: BoolArray) -> float:
    """Inhibitory share of reciprocal minus non-reciprocal feedback synapses."""

    inhibitory_rows = np.broadcast_to(inhibitory[:, None], feedback.shape)
    reciprocal = feedback[mask].sum()
    other = feedback[~mask].sum()
    if reciprocal == 0 or other == 0:
        return float("nan")
    share_reciprocal = feedback[mask & inhibitory_rows].sum() / reciprocal
    share_other = feedback[~mask & inhibitory_rows].sum() / other
    return float(share_reciprocal - share_other)
