from __future__ import annotations

import io
import math
import tarfile
from pathlib import Path

import numpy as np
import pytest

from neuromorphic.fly.central_complex import extract_cx
from neuromorphic.fly.hemibrain import (
    MEMBER,
    circuit_phases,
    pb_glomerulus,
    pb_position,
    phase_angle,
    phase_index,
    read_tables,
)

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def test_glomerulus_parsing() -> None:
    assert pb_glomerulus("EPG(PB08)_L3") == ("L", 3)
    assert pb_glomerulus("PEN_a(PB06a)_R9") == ("R", 9)
    assert pb_glomerulus("Delta7(PB15)_L4R5_R") is None
    assert pb_glomerulus("GLNO(LAL-NO1)_L") is None


def test_glomeruli_eight_apart_share_a_phase() -> None:
    assert [pb_position("L", 9), pb_position("L", 1), pb_position("R", 1)] == [0, 8, 9]
    # Delta7 output pairs (L4,R5), (L1,R8), (L9,L1) are same-phase glomeruli.
    assert phase_index("L", 4) == phase_index("R", 5)
    assert phase_index("L", 1) == phase_index("R", 8) == phase_index("L", 9)
    assert phase_index("R", 1) != phase_index("R", 2)
    with pytest.raises(ValueError):
        pb_position("R", 0)
    assert phase_angle(0) == pytest.approx(-math.pi + math.pi / 8)


def _archive(tmp_path: Path) -> tuple[Path, Path]:
    neurons = pd.DataFrame(
        {
            "bodyId": [1, 2, 3, 4],
            "type": ["EPG", "EPG", "Delta7", "GLNO"],
            "instance": ["EPG(PB08)_L4", "EPG(PB08)_R5", "Delta7(PB15)_L4R5_R", "GLNO(LAL-NO1)_L"],
        }
    )
    total = pd.DataFrame({"bodyId_pre": [1, 3], "bodyId_post": [2, 1], "weight": [7, 9]})
    roi = pd.DataFrame(
        {"bodyId_pre": [1, 3], "bodyId_post": [2, 1], "roi": ["EB", "PB"], "weight": [7, 9]}
    )
    archive = tmp_path / "adjacencies.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, frame in (
            ("traced-neurons.csv", neurons),
            ("traced-total-connections.csv", total),
            ("traced-roi-connections.csv", roi),
        ):
            payload = frame.to_csv(index=False).encode()
            info = tarfile.TarInfo(MEMBER.format(name))
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    transmitters = tmp_path / "nt.feather"
    pd.DataFrame(
        {"predicted_nt": ["acetylcholine", "acetylcholine", "glutamate", "gaba"]},
        index=pd.Index([1, 2, 3, 4], name="body"),
    ).to_feather(transmitters)
    return archive, transmitters


def test_read_tables_relabels_for_flywire_extractors(tmp_path: Path) -> None:
    annotations, connections, roi = read_tables(*_archive(tmp_path))
    assert annotations.side.tolist() == ["left", "right", "right", "left"]
    assert annotations.top_nt.tolist() == ["acetylcholine", "acetylcholine", "glutamate", "gaba"]
    assert {"pre_pt_root_id", "post_pt_root_id", "syn_count"} <= set(connections.columns)
    assert roi.roi.tolist() == ["EB", "PB"]
    circuit = extract_cx(annotations, connections, types=("EPG", "Delta7"))
    assert circuit.sign.tolist() == [1.0, 1.0, -1.0]
    phases = circuit_phases(annotations, circuit)
    assert phases[0] == pytest.approx(phases[1])  # L4 and R5 share a phase
    assert np.isnan(phases[2])
