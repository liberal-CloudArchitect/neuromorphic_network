"""Create and load a checksum-verified network-mvp-v1 bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from neuromorphic.modules.network import ModularBrainNetwork, ModularBrainOutput
from neuromorphic.tasks.base import TaskBatch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(slots=True)
class NetworkMVP:
    model: ModularBrainNetwork
    manifest: dict[str, Any]
    device: torch.device

    def initial_state(self, batch_size: int, dtype: torch.dtype = torch.float32) -> Any:
        return self.model.initial_state(batch_size, device=self.device, dtype=dtype)

    def predict_batch(
        self,
        batch: TaskBatch,
        *,
        state: Any = None,
        telemetry: bool = False,
        routing_mode: str = "learned",
    ) -> ModularBrainOutput:
        if telemetry not in {True, False}:
            raise TypeError("telemetry must be boolean")
        self.model.eval()
        with torch.no_grad():
            return self.model.forward_batch(
                batch.to(self.device),
                state,
                phase="evaluate",
                telemetry_enabled=telemetry,
                routing_mode=routing_mode,  # type: ignore[arg-type]
            )


def create_network_mvp_bundle(
    directory: Path,
    *,
    model: ModularBrainNetwork,
    source_commit: str,
    gate_status: str,
    qualification_only: bool,
    model_config: dict[str, object] | None = None,
) -> Path:
    if gate_status != "PASSED" and not qualification_only:
        raise ValueError("a formal network MVP bundle requires GATE-NN-MVP PASSED")
    directory.mkdir(parents=True, exist_ok=False)
    weights = directory / "model.pt"
    torch.save(model.state_dict(), weights)
    manifest = {
        "schema_version": "network-mvp-v1",
        "qualification_only": qualification_only,
        "gate_status": gate_status,
        "source_commit": source_commit,
        "model_config": model_config or {},
        "weights": {"path": "model.pt", "sha256": _sha256(weights)},
        "limitations": [
            "Artificial computational model; no biological equivalence claim.",
            "Qualification bundles are fixtures and are not released network MVPs.",
        ],
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory


def load_network_mvp(path: Path, device: torch.device) -> NetworkMVP:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "network-mvp-v1":
        raise ValueError("unsupported network MVP bundle schema")
    weights_value = manifest.get("weights")
    if not isinstance(weights_value, dict):
        raise ValueError("network MVP bundle weights metadata is invalid")
    weights = path / str(weights_value.get("path"))
    if not weights.is_file() or _sha256(weights) != weights_value.get("sha256"):
        raise ValueError("network MVP bundle checksum mismatch")
    config = manifest.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("network MVP model config is invalid")
    allowed = {
        name: value
        for name, value in config.items()
        if name
        in {
            "feature_dim",
            "episodic_slots",
            "working_slots",
            "working_slot_dim",
            "action_embedding_dim",
            "task_embedding_dim",
            "router_top_k",
            "router_capacity_factor",
            "tbptt_interval",
        }
    }
    model = ModularBrainNetwork(**allowed).to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return NetworkMVP(model=model, manifest=manifest, device=device)


__all__ = ["NetworkMVP", "create_network_mvp_bundle", "load_network_mvp"]
