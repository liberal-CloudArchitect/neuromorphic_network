"""Surprise-gated working memory and compact T1 networks (CR-014).

``SurpriseGatedWorkingMemory`` reproduces the update and read equations of
``working_memory.v1`` step by step and adds an optional outcome-surprise
signal ``s_t = |feedback_t - p_t|``, where ``p_t`` is the network's own
(detached) probability for answer 1 at the preceding query.  Modes:

- ``plain``: the main-network working memory (surprise unused);
- ``gate``: surprise adds a learned per-slot bias to the update gate only
  (F1b: prediction error gates the write, not the representation);
- ``input``: the same surprise enters as an extra input feature instead;
- ``shuffled``: like ``gate`` but with surprise taken from another episode.

``write_source`` (T1b) chooses what the write candidate is computed from:
``current`` is ``working_memory.v1`` (the step's representation ``r_t``);
``enriched`` uses the read-enriched ``r_t + P·read_t``, so one write can
combine the current step with what memory already holds (e.g. feedback with
the stored query).  The update gate always sees ``r_t`` only.

``gate_address`` (T1d) chooses where the surprise bias lands: ``global``
adds ``s_t·w`` to every slot's gate logit (T1-T1c); ``read`` weights it by the
step's read attention, ``s_t·w ⊙ (S·a_t)``, so a surprising outcome opens the
slots that were just read.  With uniform attention the two are identical.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from neuromorphic.core.contracts import ModuleContext
from neuromorphic.modules.sensory_encoder import SensoryEncoder
from neuromorphic.tasks.base import TaskBatch
from neuromorphic.transfer.uncued_switch import FEEDBACK, QUERY

Mode = Literal["plain", "gate", "input", "shuffled"]
MODES: tuple[Mode, ...] = ("plain", "gate", "input", "shuffled")
WriteSource = Literal["current", "enriched"]
WRITE_SOURCES: tuple[WriteSource, ...] = ("current", "enriched")
GateAddress = Literal["global", "read"]
GATE_ADDRESSES: tuple[GateAddress, ...] = ("global", "read")
_FEEDBACK_BIT = 10


class SurpriseGatedWorkingMemory(nn.Module):
    """Four learned slots with ``working_memory.v1`` equations plus a surprise channel."""

    def __init__(
        self,
        feature_dim: int,
        *,
        slots: int = 4,
        slot_dim: int = 32,
        mode: Mode = "plain",
        surprise_init: float = 2.0,
        write_source: WriteSource = "current",
        gate_address: GateAddress = "global",
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode}")
        if write_source not in WRITE_SOURCES:
            raise ValueError(f"unknown write source: {write_source}")
        if gate_address not in GATE_ADDRESSES:
            raise ValueError(f"unknown gate address: {gate_address}")
        self.mode = mode
        self.write_source = write_source
        self.gate_address = gate_address
        self.last_attention: Tensor = torch.zeros(0)
        self.slots = slots
        self.slot_dim = slot_dim
        self.candidate = nn.Linear(feature_dim, slot_dim)
        self.update_gate = nn.Linear(feature_dim, slots)
        self.read_query = nn.Linear(feature_dim, slot_dim)
        self.read_projection = nn.Linear(slot_dim, feature_dim)
        self.surprise_gate = nn.Parameter(torch.full((slots,), surprise_init))
        self.surprise_input = nn.Linear(1, feature_dim)

    def initial_slots(self, batch: int, *, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.slots, self.slot_dim, device=device)

    def step(
        self, slots: Tensor, representation: Tensor, valid: Tensor, surprise: Tensor
    ) -> tuple[Tensor, Tensor]:
        """One causal step; ``surprise`` is ``[B]`` (zero where undefined)."""

        if self.mode == "input":
            representation = representation + self.surprise_input(surprise[:, None])
        attention = torch.softmax(
            torch.einsum("bsd,bd->bs", slots, self.read_query(representation)) / self.slot_dim**0.5,
            dim=-1,
        )
        self.last_attention = attention.detach()
        read = torch.einsum("bs,bsd->bd", attention, slots)
        enriched = representation + self.read_projection(read)
        logits = self.update_gate(representation)
        if self.mode in {"gate", "shuffled"}:
            bias = surprise[:, None] * self.surprise_gate[None, :]
            if self.gate_address == "read":
                bias = bias * attention * self.slots
            logits = logits + bias
        gates = torch.sigmoid(logits) * valid[:, None]
        source = enriched if self.write_source == "enriched" else representation
        candidate = torch.tanh(self.candidate(source)).unsqueeze(1)
        updated = slots + gates.unsqueeze(-1) * (candidate - slots)
        updated = torch.where(valid[:, None, None], updated, slots)
        return updated, torch.where(valid[:, None], enriched, representation)


def selector_head(feature_dim: int, classes: int = 2) -> nn.Sequential:
    """``action_selector.v1`` integration core plus a ``classes``-way head."""

    return nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, feature_dim),
        nn.GELU(),
        nn.Linear(feature_dim, classes),
    )


class T1Network(nn.Module):
    """Adapter -> ``sensory_encoder.v1`` -> surprise-gated working memory -> selector head."""

    def __init__(
        self,
        input_dim: int,
        *,
        feature_dim: int = 64,
        mode: Mode = "plain",
        write_source: WriteSource = "current",
        feedback_bit: int = _FEEDBACK_BIT,
        gate_address: GateAddress = "global",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.feedback_bit = feedback_bit
        self.trace_attention = False  # T1d analysis: keep each step's read attention
        self.attention_trace: list[Tensor] = []
        self.adapter = nn.Linear(input_dim, feature_dim)
        self.encoder = SensoryEncoder(feature_dim=feature_dim)
        self.memory = SurpriseGatedWorkingMemory(
            feature_dim, mode=mode, write_source=write_source, gate_address=gate_address
        )
        self.head = selector_head(feature_dim)

    def encode(self, batch: TaskBatch) -> Tensor:
        steps = batch.inputs.shape[1]
        reset = torch.zeros_like(batch.valid_mask)
        reset[:, 0] = True
        context = ModuleContext(
            task_id="uncued_rule_switch.v1", phase="train", reset_mask=reset, eligible_modules=()
        )
        index = torch.arange(steps, device=batch.inputs.device).expand_as(batch.valid_mask)
        output = self.encoder.forward_inputs(
            self.adapter(batch.inputs), batch.valid_mask, index, None, context
        )
        return output.packet.representation

    def forward(self, batch: TaskBatch) -> Tensor:
        """Logits ``[B, T, 2]``; surprise is computed causally from own past responses."""

        return self.run(batch)[0]

    def run(self, batch: TaskBatch) -> tuple[Tensor, Tensor]:
        """Logits ``[B, T, 2]`` and the flattened slots after each step ``[B, T, S·D]``."""

        representation = self.encode(batch)
        batch_size, steps, _ = representation.shape
        device = representation.device
        slots = self.memory.initial_slots(batch_size, device=device)
        last_answer = torch.full((batch_size,), 0.5, device=device)
        event = batch.inputs[..., :4].argmax(dim=-1)
        logits: list[Tensor] = []
        states: list[Tensor] = []
        self.attention_trace = []
        for t in range(steps):
            is_feedback = event[:, t].eq(FEEDBACK) & batch.valid_mask[:, t]
            surprise = torch.where(
                is_feedback,
                (batch.inputs[:, t, self.feedback_bit] - last_answer).abs(),
                torch.zeros_like(last_answer),
            )
            if self.mode == "shuffled" and batch_size > 1:
                surprise = surprise.roll(1, dims=0)
            slots, enriched = self.memory.step(
                slots, representation[:, t], batch.valid_mask[:, t], surprise
            )
            step_logits = self.head(enriched)
            logits.append(step_logits)
            states.append(slots.flatten(1))
            if self.trace_attention:
                self.attention_trace.append(self.memory.last_attention)
            is_query = event[:, t].eq(QUERY) & batch.valid_mask[:, t]
            probability = torch.softmax(step_logits, dim=-1)[:, 1].detach()
            last_answer = torch.where(is_query, probability, last_answer)
        return torch.stack(logits, dim=1), torch.stack(states, dim=1)


class T1GRU(nn.Module):
    """Shared-GRU main baseline: adapter -> GRU -> selector head (same inputs)."""

    def __init__(self, input_dim: int, *, hidden: int = 64) -> None:
        super().__init__()
        self.adapter = nn.Linear(input_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = selector_head(hidden)

    def forward(self, batch: TaskBatch) -> Tensor:
        return self.run(batch)[0]

    def run(self, batch: TaskBatch) -> tuple[Tensor, Tensor]:
        """Logits ``[B, T, 2]`` and the hidden state after each step ``[B, T, H]``."""

        output, _ = self.gru(self.adapter(batch.inputs))
        return self.head(output), output


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "GATE_ADDRESSES",
    "MODES",
    "T1GRU",
    "WRITE_SOURCES",
    "GateAddress",
    "Mode",
    "SurpriseGatedWorkingMemory",
    "T1Network",
    "WriteSource",
    "parameter_count",
    "selector_head",
]
