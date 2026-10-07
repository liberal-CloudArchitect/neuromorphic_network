from __future__ import annotations

import torch

from neuromorphic.transfer.multi_context import FEEDBACK_BIT, MultiContextSwitchTask
from neuromorphic.transfer.surprise_wm import (
    SurpriseGatedWorkingMemory,
    T1Network,
    parameter_count,
)


def _pair() -> tuple[SurpriseGatedWorkingMemory, SurpriseGatedWorkingMemory]:
    torch.manual_seed(5)
    flat = SurpriseGatedWorkingMemory(16, slots=4, slot_dim=8, mode="gate", write_source="enriched")
    addressed = SurpriseGatedWorkingMemory(
        16, slots=4, slot_dim=8, mode="gate", write_source="enriched", gate_address="read"
    )
    addressed.load_state_dict(flat.state_dict())
    return flat, addressed


def test_read_addressed_gate_equals_global_gate_under_uniform_attention() -> None:
    flat, addressed = _pair()
    slots = torch.zeros(3, 4, 8)  # empty memory -> uniform attention
    representation = torch.randn(3, 16)
    valid = torch.ones(3, dtype=torch.bool)
    surprise = torch.tensor([0.0, 0.4, 0.9])
    expected = flat.step(slots, representation, valid, surprise)
    actual = addressed.step(slots, representation, valid, surprise)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_read_addressed_gate_puts_surprise_on_the_attended_slot() -> None:
    _, addressed = _pair()
    with torch.no_grad():
        addressed.surprise_gate.fill_(4.0)
    representation = torch.randn(1, 16)
    key = addressed.read_query(representation)[0].detach()
    slots = torch.zeros(1, 4, 8)
    slots[0, 2] = 30 * key / key.norm()  # slot 2 matches the read query
    valid = torch.ones(1, dtype=torch.bool)
    with torch.no_grad():
        calm, _ = addressed.step(slots, representation, valid, torch.zeros(1))
        attention = addressed.last_attention
        surprised, _ = addressed.step(slots, representation, valid, torch.ones(1))
    assert int(attention.argmax()) == 2 and float(attention[0, 2]) > 0.9
    change_calm = (calm - slots).norm(dim=-1)[0]
    change_surprised = (surprised - slots).norm(dim=-1)[0]
    gain = change_surprised - change_calm
    assert int(gain.argmax()) == 2  # the surprise opened the attended slot most
    assert float(gain[2]) > 0 and bool((gain[[0, 1, 3]] < gain[2]).all())


def test_addressed_network_is_causal_and_parameter_matched() -> None:
    task = MultiContextSwitchTask()
    batch = task.generate("train", [3, 4])
    counts = set()
    for address in ("global", "read"):
        torch.manual_seed(6)
        net = T1Network(
            task.input_dim,
            mode="gate",
            write_source="enriched",
            feedback_bit=FEEDBACK_BIT,
            gate_address=address,
        )
        counts.add(parameter_count(net))
        net.trace_attention = True
        logits, _ = net.run(batch)
        assert len(net.attention_trace) == batch.inputs.shape[1]
        cut = 15
        future = batch.inputs.clone()
        future[:, cut:] = torch.rand_like(future[:, cut:])
        altered = net(
            type(batch)(
                **{**{f: getattr(batch, f) for f in batch.__dataclass_fields__}, "inputs": future}
            )
        )
        torch.testing.assert_close(altered[:, :cut], logits[:, :cut])
    assert counts == {28_746}
