from __future__ import annotations

import torch

from neuromorphic.transfer.surprise_wm import (
    T1GRU,
    SurpriseGatedWorkingMemory,
    T1Network,
    parameter_count,
)
from neuromorphic.transfer.uncued_switch import (
    FEEDBACK,
    UncuedRuleSwitchTask,
    answer,
)


def test_rule_target_matches_answers_and_consistent_rules_reproduce_oracle() -> None:
    task = UncuedRuleSwitchTask()
    batch = task.generate("train", list(range(128)))
    rule = batch.auxiliary_targets["rule"]
    assert bool((rule[batch.valid_mask] >= 0).all()) and bool((rule[~batch.valid_mask] == -1).all())
    rows, cols = torch.nonzero(batch.loss_mask, as_tuple=True)
    x = batch.inputs[rows, cols]
    expected = [
        answer(int(r), int(a), int(b))
        for r, a, b in zip(rule[rows, cols], x[:, 8], x[:, 9], strict=True)
    ]
    assert torch.equal(batch.targets[rows, cols], torch.tensor(expected))
    sets = task.consistent_rules(batch)
    oracle = task.oracle(batch)
    for b, t in zip(rows.tolist(), cols.tolist(), strict=True):
        members = torch.nonzero(sets[b, t]).flatten().tolist()
        votes = [answer(r, int(batch.inputs[b, t, 8]), int(batch.inputs[b, t, 9])) for r in members]
        assert abs(sum(votes) / len(votes) - float(oracle[b, t])) < 1e-6
    # Before any switch the observer's set after feedback always holds the true rule.
    events = batch.inputs[..., :4].argmax(dim=-1)
    feedback = events.eq(FEEDBACK) & batch.valid_mask
    pre = feedback & batch.auxiliary_targets["trials_since_switch"].roll(1, dims=1).eq(-1)
    held = sets.gather(-1, rule.clamp(min=0)[..., None]).squeeze(-1)
    assert bool(held[pre].all())


def test_enriched_write_combines_the_step_with_stored_content() -> None:
    torch.manual_seed(2)
    current = SurpriseGatedWorkingMemory(16, slots=4, slot_dim=8, write_source="current")
    enriched = SurpriseGatedWorkingMemory(16, slots=4, slot_dim=8, write_source="enriched")
    enriched.load_state_dict(current.state_dict())
    representation = torch.randn(1, 16).expand(2, 16)
    slots = torch.randn(2, 4, 8)  # same step, two different memories
    valid = torch.ones(2, dtype=torch.bool)
    zero = torch.zeros(2)
    after_current, _ = current.step(slots, representation, valid, zero)
    after_enriched, read_enriched = enriched.step(slots, representation, valid, zero)
    # Same gate in both; the candidate is what changes.
    gates = torch.sigmoid(current.update_gate(representation))[..., None]
    target_current = (after_current - (1 - gates) * slots) / gates
    target_enriched = (after_enriched - (1 - gates) * slots) / gates
    torch.testing.assert_close(target_current[0], target_current[1])  # ignores the memory
    assert not torch.allclose(target_enriched[0], target_enriched[1])  # reads the memory
    manual = torch.tanh(enriched.candidate(read_enriched))[:, None].expand(2, 4, 8)
    torch.testing.assert_close(target_enriched, manual)


def test_t1b_cells_are_causal_and_parameter_matched() -> None:
    task = UncuedRuleSwitchTask()
    batch = task.generate("train", [5, 6])
    counts = set()
    for source in ("current", "enriched"):
        for mode in ("plain", "gate"):
            torch.manual_seed(3)
            net = T1Network(task.input_dim, mode=mode, write_source=source)
            counts.add(parameter_count(net))
            logits, states = net.run(batch)
            assert states.shape == (*batch.valid_mask.shape, 4 * 32)
            torch.testing.assert_close(net(batch), logits)
            cut = 12
            future = batch.inputs.clone()
            future[:, cut:] = torch.rand_like(future[:, cut:])
            altered = net(
                type(batch)(
                    **{
                        **{f: getattr(batch, f) for f in batch.__dataclass_fields__},
                        "inputs": future,
                    }
                )
            )
            torch.testing.assert_close(altered[:, :cut], logits[:, :cut])
    assert counts == {28_554}
    gru = T1GRU(task.input_dim)
    _, hidden = gru.run(batch)
    assert hidden.shape == (*batch.valid_mask.shape, 64)
    assert abs(parameter_count(gru) / 28_554 - 1) < 0.06
