from __future__ import annotations

import pytest
import torch

from neuromorphic.core.contracts import BrainModule, BrainPacket, ModuleContext
from neuromorphic.core.module_registry import ModuleRegistry
from neuromorphic.core.registry import SENSORY_ENCODER, WORKING_MEMORY_V2
from neuromorphic.modules.working_memory import WorkingMemory
from neuromorphic.modules.working_memory_v2 import WorkingMemoryV2
from neuromorphic.transfer.surprise_wm import SurpriseGatedWorkingMemory

_FEATURES = 12


def _packet(representation: torch.Tensor, valid: torch.Tensor) -> BrainPacket:
    batch, steps, _ = representation.shape
    return BrainPacket(
        representation=representation,
        valid_mask=valid,
        modality="fixture",
        step_index=torch.arange(steps).expand(batch, steps).clone(),
        source_module=SENSORY_ENCODER,
    )


def _context(reset: torch.Tensor) -> ModuleContext:
    return ModuleContext(
        task_id="uncued_rule_switch.p6",
        phase="train",
        reset_mask=reset,
        eligible_modules=(WORKING_MEMORY_V2,),
    )


def _fixture(seed: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three items over seven steps with an episode reset and trailing padding."""

    generator = torch.Generator().manual_seed(seed)
    representation = torch.randn(3, 7, _FEATURES, generator=generator)
    valid = torch.ones(3, 7, dtype=torch.bool)
    valid[1, 5:] = False
    valid[2, 6] = False
    reset = torch.zeros(3, 7, dtype=torch.bool)
    reset[:, 0] = True
    reset[0, 4] = True
    surprise = torch.rand(3, 7, 1, generator=generator)
    return representation, valid, reset, surprise


def _v2_from(v1: WorkingMemory, *, surprise_gating: bool) -> WorkingMemoryV2:
    v2 = WorkingMemoryV2(
        _FEATURES, slots=4, slot_dim=6, write_source="current", surprise_gating=surprise_gating
    )
    missing, unexpected = v2.load_state_dict(v1.state_dict(), strict=False)
    assert missing == ["surprise_gate"] and not unexpected
    return v2


@pytest.mark.parametrize("surprise_gating", [False, True])
def test_v2_without_enrichment_and_with_zero_surprise_equals_v1(surprise_gating: bool) -> None:
    torch.manual_seed(11)
    v1 = WorkingMemory(_FEATURES, slots=4, slot_dim=6)
    v2 = _v2_from(v1, surprise_gating=surprise_gating)
    representation, valid, reset, _ = _fixture()
    packet, context = _packet(representation, valid), _context(reset)
    zero = torch.zeros(3, 7, 1)

    expected = v1(
        packet, v1.initial_state(3, device=torch.device("cpu"), dtype=torch.float32), context
    )
    actual = v2(
        packet,
        v2.initial_state(3, device=torch.device("cpu"), dtype=torch.float32),
        context,
        surprise=zero if surprise_gating else None,
    )

    assert torch.equal(actual.packet.representation, expected.packet.representation)
    assert torch.equal(actual.state.tensors["slots"], expected.state.tensors["slots"])
    assert actual.auxiliary_losses.keys() == expected.auxiliary_losses.keys()
    for name, loss in expected.auxiliary_losses.items():
        assert torch.equal(actual.auxiliary_losses[name], loss)


def test_v2_steps_match_the_t1b_enriched_gate_equations() -> None:
    torch.manual_seed(5)
    reference = SurpriseGatedWorkingMemory(
        _FEATURES, slots=4, slot_dim=6, mode="gate", write_source="enriched", gate_address="global"
    )
    with torch.no_grad():
        reference.surprise_gate.copy_(torch.tensor([2.0, -1.0, 0.5, 3.0]))
    v2 = WorkingMemoryV2(_FEATURES, slots=4, slot_dim=6)
    missing, unexpected = v2.load_state_dict(reference.state_dict(), strict=False)
    assert not missing and unexpected == ["surprise_input.weight", "surprise_input.bias"]

    representation, valid, _, surprise = _fixture(seed=8)
    no_reset = torch.zeros(3, 7, dtype=torch.bool)
    output = v2(
        _packet(representation, valid),
        v2.initial_state(3, device=torch.device("cpu"), dtype=torch.float32),
        _context(no_reset),
        surprise=surprise,
    )

    slots = reference.initial_slots(3, device=torch.device("cpu"))
    enriched_steps = []
    for step in range(7):
        slots, enriched = reference.step(
            slots, representation[:, step], valid[:, step], surprise[:, step, 0]
        )
        enriched_steps.append(enriched)
    torch.testing.assert_close(output.packet.representation, torch.stack(enriched_steps, dim=1))
    torch.testing.assert_close(output.state.tensors["slots"], slots)


def test_surprise_is_detached_and_trains_the_gate_weight_and_candidate() -> None:
    torch.manual_seed(1)
    module = WorkingMemoryV2(_FEATURES, slots=4, slot_dim=6)
    representation, valid, reset, surprise = _fixture()
    surprise.requires_grad_(True)
    output = module(
        _packet(representation, valid),
        module.initial_state(3, device=torch.device("cpu"), dtype=torch.float32),
        _context(reset),
        surprise=surprise,
    )
    auxiliary = torch.stack(list(output.auxiliary_losses.values())).sum()
    loss = output.packet.representation.square().sum() + auxiliary
    loss.backward()

    assert surprise.grad is None
    for parameter in (module.surprise_gate, module.update_gate.weight, module.candidate.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum().item() > 0.0


def test_surprise_on_padding_and_reset_steps_has_no_effect() -> None:
    torch.manual_seed(4)
    module = WorkingMemoryV2(_FEATURES, slots=4, slot_dim=6)
    representation, valid, reset, _ = _fixture()
    packet, context = _packet(representation, valid), _context(reset)
    masked_only = torch.where((~valid | reset).unsqueeze(-1), 5.0, 0.0)

    def run(surprise: torch.Tensor) -> torch.Tensor:
        state = module.initial_state(3, device=torch.device("cpu"), dtype=torch.float32)
        return module.forward(packet, state, context, surprise=surprise).state.tensors["slots"]

    torch.testing.assert_close(run(masked_only), run(torch.zeros_like(masked_only)))


def test_surprise_raises_write_strength_on_valid_steps() -> None:
    torch.manual_seed(6)
    module = WorkingMemoryV2(_FEATURES, slots=4, slot_dim=6)
    representation, valid, reset, _ = _fixture()
    packet, context = _packet(representation, valid), _context(reset)
    zero = torch.zeros(3, 7, 1)
    surprising = zero.clone()
    surprising[:, 2] = 1.0

    def write_change(surprise: torch.Tensor) -> float:
        state = module.initial_state(3, device=torch.device("cpu"), dtype=torch.float32)
        output = module.forward(packet, state, context, surprise=surprise)
        return float(output.auxiliary_losses["working_consistency"].detach())

    assert write_change(surprising) > write_change(zero)


def test_episode_reset_clears_slots_before_the_step() -> None:
    module = WorkingMemoryV2(_FEATURES, slots=4, slot_dim=6)
    representation, valid, reset, surprise = _fixture()
    packet, context = _packet(representation, valid), _context(reset)
    filled = module.initial_state(3, device=torch.device("cpu"), dtype=torch.float32)
    filled.tensors["slots"].fill_(0.7)
    empty = module.initial_state(3, device=torch.device("cpu"), dtype=torch.float32)

    from_filled = module(packet, filled, context, surprise=surprise)
    from_empty = module(packet, empty, context, surprise=surprise)
    torch.testing.assert_close(
        from_filled.state.tensors["slots"], from_empty.state.tensors["slots"]
    )

    cleared = module.reset_state(from_filled.state, torch.tensor([True, False, False]))
    assert torch.count_nonzero(cleared.tensors["slots"][0]) == 0
    torch.testing.assert_close(cleared.tensors["slots"][1:], from_filled.state.tensors["slots"][1:])


def test_gating_requires_a_well_formed_surprise() -> None:
    representation, valid, reset, surprise = _fixture()
    packet, context = _packet(representation, valid), _context(reset)
    gated = WorkingMemoryV2(_FEATURES, slots=4, slot_dim=6)
    state = gated.initial_state(3, device=torch.device("cpu"), dtype=torch.float32)

    with pytest.raises(ValueError, match="requires surprise"):
        gated(packet, state, context)
    with pytest.raises(ValueError, match=r"\[B, T, 1\]"):
        gated(packet, state, context, surprise=surprise[..., 0])
    with pytest.raises(TypeError, match="floating-point"):
        gated(packet, state, context, surprise=surprise.to(torch.long))

    ungated = WorkingMemoryV2(_FEATURES, slots=4, slot_dim=6, surprise_gating=False)
    ungated(
        packet, ungated.initial_state(3, device=torch.device("cpu"), dtype=torch.float32), context
    )
    with pytest.raises(ValueError, match="write source"):
        WorkingMemoryV2(_FEATURES, write_source="read")  # type: ignore[arg-type]


def test_v2_is_a_registered_brain_module_with_its_own_state_version() -> None:
    module = WorkingMemoryV2(_FEATURES, slots=4, slot_dim=6)
    registry = ModuleRegistry((module,))
    assert isinstance(module, BrainModule)
    assert registry.get(WORKING_MEMORY_V2) is module

    representation, valid, reset, surprise = _fixture()
    output = module(
        _packet(representation, valid),
        module.initial_state(3, device=torch.device("cpu"), dtype=torch.float32),
        _context(reset),
        surprise=surprise,
    )
    assert output.packet.source_module == WORKING_MEMORY_V2
    assert output.state.version == "working-state-v2"

    v1 = WorkingMemory(_FEATURES, slots=4, slot_dim=6)
    with pytest.raises(ValueError, match="working-state-v2"):
        module(
            _packet(representation, valid),
            v1.initial_state(3, device=torch.device("cpu"), dtype=torch.float32),
            _context(reset),
            surprise=surprise,
        )
