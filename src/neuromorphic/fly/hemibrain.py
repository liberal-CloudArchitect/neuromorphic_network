"""Hemibrain v1.2 tables and protocerebral-bridge (PB) glomerulus phases (F1l).

The traced-adjacency export names columnar central-complex neurons by PB
glomerulus, e.g. ``EPG(PB08)_L3`` or ``PEN_a(PB06a)_R4``.  Glomeruli lie on a line
L9..L1 R1..R9 (positions 0..17); glomeruli eight positions apart share a phase,
the period that Delta7 outputs follow (``Delta7(PB15)_L4R5``).  Phase index
``position % 8`` maps to the angle ``-pi + (index + 0.5) * pi / 4``.  The tables
are re-labelled with the FlyWire column names so the existing extractors apply.
"""

from __future__ import annotations

import math
import re
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neuromorphic.fly.central_complex import CXCircuit

FloatArray = NDArray[np.float64]
MEMBER = "exported-traced-adjacencies-v1.2/{}"
_GLOMERULUS = re.compile(r"\)_([LR])(\d)$")
_SOMA_SIDE = re.compile(r"_([LR])$")
N_PHASES = 8


def pb_glomerulus(instance: str) -> tuple[str, int] | None:
    """``("L", 3)`` for ``EPG(PB08)_L3``; None for multi-glomerulus or other names."""

    match = _GLOMERULUS.search(instance)
    return (match.group(1), int(match.group(2))) if match else None


def pb_position(side: str, index: int) -> int:
    """Linear PB position: L9 -> 0, ..., L1 -> 8, R1 -> 9, ..., R9 -> 17."""

    if side not in {"L", "R"} or not 1 <= index <= 9:
        raise ValueError(f"no PB glomerulus {side}{index}")
    return 9 - index if side == "L" else 8 + index


def phase_index(side: str, index: int) -> int:
    return pb_position(side, index) % N_PHASES


def phase_angle(index: int) -> float:
    return -math.pi + (index + 0.5) * 2.0 * math.pi / N_PHASES


def _side(instance: str) -> str:
    glomerulus = pb_glomerulus(instance)
    if glomerulus is not None:
        return "left" if glomerulus[0] == "L" else "right"
    match = _SOMA_SIDE.search(instance)
    if match is None:
        return "center"
    return "left" if match.group(1) == "L" else "right"


def read_tables(archive: Path, transmitters: Path) -> tuple[Any, Any, Any]:
    """FlyWire-style ``annotations``, ``connections`` and per-ROI ``roi_connections``.

    ``side`` is the PB glomerulus side for columnar neurons and the instance's
    soma side otherwise; ``top_nt`` is the hemibrain body-mean prediction.
    """

    from neuromorphic.fly.flywire import _pandas

    pd = _pandas()
    with tarfile.open(archive, "r:gz") as tar:

        def table(name: str) -> Any:
            member = tar.extractfile(MEMBER.format(name))
            if member is None:
                raise FileNotFoundError(name)
            return pd.read_csv(member)

        neurons = table("traced-neurons.csv")
        total = table("traced-total-connections.csv")
        roi = table("traced-roi-connections.csv")
    nt = pd.read_feather(transmitters)
    nt = nt.reset_index() if "body" not in nt.columns else nt
    predicted = dict(zip(nt["body"].astype(np.int64), nt["predicted_nt"].astype(str), strict=True))
    instance = neurons.instance.fillna("").astype(str)
    annotations = pd.DataFrame(
        {
            "root_id": neurons.bodyId.astype(np.int64),
            "cell_type": neurons.type,
            "instance": instance,
            "side": [_side(x) for x in instance],
            "top_nt": [predicted.get(int(b), "unknown") for b in neurons.bodyId],
        }
    )
    rename = {
        "bodyId_pre": "pre_pt_root_id",
        "bodyId_post": "post_pt_root_id",
        "weight": "syn_count",
    }
    return annotations, total.rename(columns=rename), roi.rename(columns=rename)


def circuit_phases(annotations: Any, circuit: CXCircuit) -> FloatArray:
    """PB phase angle per circuit neuron (NaN for neurons without one glomerulus)."""

    lookup = dict(zip(annotations.root_id.astype(np.int64), annotations.instance, strict=True))
    out = np.full(circuit.n, np.nan)
    for i, body in enumerate(circuit.ids.tolist()):
        glomerulus = pb_glomerulus(str(lookup.get(body, "")))
        if glomerulus is not None:
            out[i] = phase_angle(phase_index(*glomerulus))
    return out
