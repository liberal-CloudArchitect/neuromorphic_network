from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from neuromorphic.core import (
    MODULE_IDS,
    OPTIONAL_EXPERT_IDS,
    REQUIRED_PATH_IDS,
    BrainModule,
    BrainPacket,
    ModuleContext,
    ModuleOutput,
    ModuleState,
)
from neuromorphic.core.contracts import validate_module_context
from neuromorphic.core.registry import SENSORY_ENCODER


def _packet() -> BrainPacket:
    return BrainPacket(
        representation=torch.zeros(2, 3, 4),
        valid_mask=torch.tensor([[True, True, False], [True, True, True]]),
        modality="symbolic",
        step_index=torch.tensor([[0, 1, 2], [3, 4, 5]]),
        source_module=SENSORY_ENCODER,
        goal_context=torch.ones(2, 3, 2),
        metadata={"temperature": 0.5, "split": "train"},
    )


def test_frozen_module_registry_has_required_and_optional_partition() -> None:
    assert MODULE_IDS == (
        "sensory_encoder.v1",
        "episodic_memory.v1",
        "working_memory.v1",
        "predictive_adapter.v1",
        "action_selector.v1",
        "sparse_router.v1",
    )
    assert set(OPTIONAL_EXPERT_IDS).isdisjoint(REQUIRED_PATH_IDS)
    assert set(OPTIONAL_EXPERT_IDS) | set(REQUIRED_PATH_IDS) == set(MODULE_IDS)


def test_brain_packet_accepts_valid_time_aligned_tensors() -> None:
    packet = _packet()
    assert packet.representation.shape == (2, 3, 4)
    assert packet.valid_mask.dtype is torch.bool


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("representation", torch.zeros(2, 3, dtype=torch.float32), ValueError),
        ("representation", torch.zeros(2, 3, 4, dtype=torch.int64), TypeError),
        ("valid_mask", torch.ones(2, 3), TypeError),
        ("valid_mask", torch.ones(2, 4, dtype=torch.bool), ValueError),
        ("step_index", torch.ones(2, 3), TypeError),
        ("step_index", torch.full((2, 3), -1, dtype=torch.int64), ValueError),
        ("goal_context", torch.ones(2, 2, 4), ValueError),
        ("metadata", {"tensor": torch.tensor(1)}, TypeError),
        ("source_module", "unregistered.v1", ValueError),
    ],
)
def test_brain_packet_rejects_invalid_contract_fields(
    field: str, value: object, exception: type[Exception]
) -> None:
    arguments: dict[str, object] = {
        "representation": torch.zeros(2, 3, 4),
        "valid_mask": torch.ones(2, 3, dtype=torch.bool),
        "modality": "symbolic",
        "step_index": torch.arange(6).reshape(2, 3),
        "source_module": SENSORY_ENCODER,
    }
    arguments[field] = value
    with pytest.raises(exception):
        BrainPacket(**arguments)  # type: ignore[arg-type]


def test_context_state_and_output_validate_device_shape_and_scalar_losses() -> None:
    packet = _packet()
    context = ModuleContext(
        task_id="associative-recall-v1",
        phase="train",
        reset_mask=torch.tensor([[True, False, False], [True, False, True]]),
        eligible_modules=OPTIONAL_EXPERT_IDS,
        telemetry_enabled=True,
    )
    state = ModuleState(
        module_id=SENSORY_ENCODER,
        version="sensory-state-v1",
        tensors={"hidden": torch.zeros(2, 8)},
    )
    output = ModuleOutput(
        packet=packet,
        state=state,
        prediction_logits=torch.zeros(2, 3, 32),
        action_logits=torch.zeros(2, 3, 4),
        auxiliary_losses={"prediction": torch.tensor(0.25)},
    )
    assert context.reset_mask.tolist() == [[True, False, False], [True, False, True]]
    validate_module_context(context, batch_size=2, sequence_length=3)
    assert output.auxiliary_losses["prediction"].ndim == 0


def test_context_rejects_unknown_or_duplicate_modules() -> None:
    with pytest.raises(ValueError, match="unregistered"):
        ModuleContext(
            task_id="task-v1",
            phase="train",
            reset_mask=torch.zeros((2, 3), dtype=torch.bool),
            eligible_modules=("unknown.v1",),
        )
    with pytest.raises(ValueError, match="duplicates"):
        ModuleContext(
            task_id="task-v1",
            phase="train",
            reset_mask=torch.zeros((2, 3), dtype=torch.bool),
            eligible_modules=(OPTIONAL_EXPERT_IDS[0], OPTIONAL_EXPERT_IDS[0]),
        )


def test_output_rejects_non_scalar_or_nonfinite_auxiliary_loss() -> None:
    packet = _packet()
    state = ModuleState(SENSORY_ENCODER, "sensory-state-v1")
    with pytest.raises(ValueError, match="scalar"):
        ModuleOutput(packet, state, auxiliary_losses={"loss": torch.ones(1)})
    with pytest.raises(ValueError, match="finite"):
        ModuleOutput(packet, state, auxiliary_losses={"loss": torch.tensor(float("nan"))})


@dataclass
class _ExampleModule:
    module_id: str = SENSORY_ENCODER

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ModuleState:
        hidden = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        return ModuleState(self.module_id, "sensory-state-v1", {"h": hidden})

    def forward(
        self,
        packet: BrainPacket,
        state: ModuleState,
        context: ModuleContext,
    ) -> ModuleOutput:
        del context
        return ModuleOutput(packet, state)

    def reset_state(self, state: ModuleState, reset_mask: Tensor) -> ModuleState:
        del reset_mask
        return state


def test_brain_module_protocol_is_runtime_checkable() -> None:
    assert isinstance(_ExampleModule(), BrainModule)
