from __future__ import annotations

import torch

from neuromorphic.core.contracts import BrainPacket, ModuleContext
from neuromorphic.modules.working_memory import WorkingMemory
from neuromorphic.tasks.base import DatasetSplit
from neuromorphic.transfer.surprise_wm import SurpriseGatedWorkingMemory, T1Network
from neuromorphic.transfer.uncued_switch import (
    CUE,
    FEEDBACK,
    QUERY,
    UncuedRuleSwitchTask,
)


def test_task_is_deterministic_split_disjoint_and_cue_free_after_start() -> None:
    task = UncuedRuleSwitchTask()
    first = task.generate("train", [0, 1, 2])
    again = task.generate("train", [0, 1, 2])
    assert torch.equal(first.inputs, again.inputs)
    splits: tuple[DatasetSplit, ...] = ("train", "test", "ood")
    hashes = {s: {task.content_hash(s, i) for i in range(50)} for s in splits}
    assert not (hashes["train"] & hashes["test"]) and not (hashes["test"] & hashes["ood"])
    batch = task.generate("train", list(range(64)))
    events = batch.inputs[..., :4].argmax(dim=-1)
    cues = (events == CUE) & batch.valid_mask
    assert torch.equal(cues.sum(dim=1), torch.ones(64, dtype=torch.long))  # only the opening cue
    assert bool(cues[:, 0].all())
    # Every query is followed by feedback revealing its target.
    query = (events == QUERY) & batch.valid_mask
    rows, cols = torch.nonzero(query, as_tuple=True)
    assert bool((events[rows, cols + 1] == FEEDBACK).all())
    assert torch.equal(batch.inputs[rows, cols + 1, 10].long(), batch.targets[rows, cols])


def test_oracle_is_perfect_before_any_switch_and_recovers_after() -> None:
    task = UncuedRuleSwitchTask()
    batch = task.generate("train", list(range(256)))
    oracle = task.oracle(batch)
    since = batch.auxiliary_targets["trials_since_switch"]
    query = batch.loss_mask
    target = batch.targets.float()
    accuracy = oracle * target + (1 - oracle) * (1 - target)
    assert float(accuracy[query & (since == -1)].min()) == 1.0
    after = [float(accuracy[query & (since == k)].mean()) for k in range(4)]
    assert after == sorted(after) and after[0] < 0.6 and after[3] > 0.8


def test_plain_mode_reproduces_working_memory_v1() -> None:
    torch.manual_seed(0)
    feature = 16
    reference = WorkingMemory(feature_dim=feature, slots=4, slot_dim=8)
    mirror = SurpriseGatedWorkingMemory(feature, slots=4, slot_dim=8, mode="plain")
    for name in ("candidate", "update_gate", "read_query", "read_projection"):
        getattr(mirror, name).load_state_dict(getattr(reference, name).state_dict())
    representation = torch.randn(3, 5, feature)
    valid = torch.ones(3, 5, dtype=torch.bool)
    valid[2, 3:] = False
    reset = torch.zeros(3, 5, dtype=torch.bool)
    reset[:, 0] = True
    packet = BrainPacket(
        representation=representation,
        valid_mask=valid,
        modality="symbolic",
        step_index=torch.arange(5).expand(3, 5),
        source_module="sensory_encoder.v1",
    )
    context = ModuleContext(
        task_id="delayed_rule_switch.v1", phase="train", reset_mask=reset, eligible_modules=()
    )
    expected = reference(
        packet, reference.initial_state(3, device=torch.device("cpu"), dtype=torch.float32), context
    )
    slots = mirror.initial_slots(3, device=torch.device("cpu"))
    outputs = []
    for t in range(5):
        slots, enriched = mirror.step(slots, representation[:, t], valid[:, t], torch.zeros(3))
        outputs.append(enriched)
    torch.testing.assert_close(torch.stack(outputs, dim=1), expected.packet.representation)
    torch.testing.assert_close(slots, expected.state.tensors["slots"])


def test_network_is_causal_and_surprise_only_changes_gated_modes() -> None:
    task = UncuedRuleSwitchTask()
    batch = task.generate("train", [3, 4])
    torch.manual_seed(1)
    net = T1Network(task.input_dim, mode="gate")
    logits = net(batch)
    cut = 10
    future = batch.inputs.clone()
    future[:, cut:] = torch.rand_like(future[:, cut:])
    altered = net(
        type(batch)(
            **{**{f: getattr(batch, f) for f in batch.__dataclass_fields__}, "inputs": future}
        )
    )
    torch.testing.assert_close(altered[:, :cut], logits[:, :cut])
    plain = T1Network(task.input_dim, mode="plain")
    plain.load_state_dict(net.state_dict())
    assert not torch.allclose(plain(batch), logits)  # surprise opened the gate somewhere
