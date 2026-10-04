"""DoOR consensus odor responses mapped onto FlyWire glomeruli (F1d).

DoOR.data (Münch & Galizia 2016; CC BY-SA 4.0) ships semicolon-separated
R tables: the response matrix has odors (InChIKey) as row names and receptors
as columns; the mapping table links receptors to glomeruli.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neuromorphic.fly.flywire import _pandas


@dataclass(frozen=True, slots=True)
class DoorBlock:
    """Dense odor x glomerulus response block; missing values filled with 0."""

    glomeruli: tuple[str, ...]
    receptors: tuple[str, ...]
    odors: tuple[str, ...]
    responses: NDArray[np.float64]
    coverage: NDArray[np.float64]


def load_door_block(
    response_csv: Path,
    mappings_csv: Path,
    available_glomeruli: Iterable[str],
    *,
    n_glomeruli: int,
    min_coverage: float,
) -> DoorBlock:
    """Select the best-measured glomeruli and the odors that cover them.

    For each glomerulus in ``available_glomeruli`` the receptor with the most
    measured odors is used; the ``n_glomeruli`` best-measured glomeruli are kept,
    then odors measured on at least ``min_coverage`` of them.  The solvent
    reference row (``SFR``) is dropped.
    """

    pd = _pandas()
    responses = pd.read_csv(response_csv, sep=";", index_col=0).drop(index="SFR", errors="ignore")
    mapping = pd.read_csv(mappings_csv, sep=";", index_col=0)[["receptor", "glomerulus"]].dropna()
    available = set(available_glomeruli)
    mapping = mapping[mapping.receptor.isin(responses.columns) & mapping.glomerulus.isin(available)]
    measured = responses.notna().sum()
    mapping = mapping.assign(n=mapping.receptor.map(measured))
    best = (
        mapping.sort_values(["n", "receptor"], ascending=[False, True])
        .drop_duplicates("glomerulus")
        .head(n_glomeruli)
    )
    if len(best) < n_glomeruli:
        raise ValueError("not enough measured glomeruli")
    block = responses[best.receptor.to_list()]
    coverage = block.notna().mean(axis=1)
    block = block[coverage >= min_coverage]
    return DoorBlock(
        glomeruli=tuple(best.glomerulus.astype(str)),
        receptors=tuple(best.receptor.astype(str)),
        odors=tuple(block.index.astype(str)),
        responses=block.fillna(0.0).to_numpy(dtype=np.float64),
        coverage=coverage[coverage >= min_coverage].to_numpy(dtype=np.float64),
    )
