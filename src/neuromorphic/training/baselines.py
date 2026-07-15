"""Monolithic GRU and Transformer baselines for P1 tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class BaselineOutput:
    """Task logits plus an optional next-state prediction."""

    logits: Tensor
    next_state_logits: Tensor | None = None


@dataclass(frozen=True, slots=True)
class ParameterMatch:
    """Deterministic parameter-matching decision for a monolithic baseline."""

    target: int
    actual: int
    relative_error: float
    hidden_size: int
    feedforward_size: int | None


@dataclass(frozen=True, slots=True)
class MacProfile:
    """Supported-operation MAC estimate with explicit parameter coverage."""

    estimated_macs: int
    supported_parameters: int
    total_parameters: int
    coverage: float
    unsupported_parameters: tuple[str, ...]
    operators: tuple[MacOperatorRecord, ...]


@dataclass(frozen=True, slots=True)
class MacOperatorRecord:
    """One auditable analytic MAC contribution."""

    name: str
    operator: str
    calls: int
    input_shape: tuple[int, ...]
    estimated_macs: int


class GRUBaseline(nn.Module):
    """Single-stack recurrent baseline with a common input projection."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 128,
        layers: int = 1,
        dropout: float = 0.0,
        auxiliary_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.layers = layers
        self.input_projection = nn.Linear(input_dim, hidden_size)
        self.encoder = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output_head = nn.Linear(hidden_size, num_classes)
        self.next_state_head = (
            None if auxiliary_classes is None else nn.Linear(hidden_size, auxiliary_classes)
        )

    def forward(self, inputs: Tensor, valid_mask: Tensor) -> BaselineOutput:
        output, _ = self.forward_chunk(inputs, valid_mask)
        return output

    def forward_chunk(
        self, inputs: Tensor, valid_mask: Tensor, state: Tensor | None = None
    ) -> tuple[BaselineOutput, Tensor]:
        """Run one recurrent chunk and return its detachable hidden state."""
        if inputs.ndim != 3 or valid_mask.shape != inputs.shape[:2]:
            raise ValueError("invalid baseline input or mask shape")
        encoded, hidden_state = self.encoder(self.input_projection(inputs), state)
        logits = self.output_head(encoded)
        next_state_logits = None if self.next_state_head is None else self.next_state_head(encoded)
        return BaselineOutput(logits=logits, next_state_logits=next_state_logits), hidden_state


class TransformerBaseline(nn.Module):
    """Causal Transformer baseline with padding-mask support."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 128,
        layers: int = 2,
        heads: int = 4,
        feedforward_size: int = 512,
        dropout: float = 0.0,
        auxiliary_classes: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.layers = layers
        self.heads = heads
        self.feedforward_size = feedforward_size
        self.input_projection = nn.Linear(input_dim, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.output_head = nn.Linear(hidden_size, num_classes)
        self.next_state_head = (
            None if auxiliary_classes is None else nn.Linear(hidden_size, auxiliary_classes)
        )

    def forward(self, inputs: Tensor, valid_mask: Tensor) -> BaselineOutput:
        if inputs.ndim != 3 or valid_mask.shape != inputs.shape[:2]:
            raise ValueError("invalid baseline input or mask shape")
        length = inputs.shape[1]
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=inputs.device), diagonal=1
        )
        encoded = self.encoder(
            self.input_projection(inputs),
            mask=causal_mask,
            src_key_padding_mask=~valid_mask,
        )
        logits = self.output_head(encoded)
        next_state = None if self.next_state_head is None else self.next_state_head(encoded)
        return BaselineOutput(logits=logits, next_state_logits=next_state)


def trainable_parameter_count(model: nn.Module) -> int:
    """Count trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def validate_parameter_target(model: nn.Module, target: int | None, tolerance: float) -> None:
    """Reject a supposedly matched model outside the declared tolerance."""
    if target is None:
        return
    actual = trainable_parameter_count(model)
    relative_error = abs(actual - target) / target
    if relative_error > tolerance:
        raise ValueError(
            f"parameter target mismatch: target={target}, actual={actual}, "
            f"relative_error={relative_error:.3f}"
        )


def select_parameter_matched_baseline(
    *,
    kind: Literal["gru", "transformer"],
    input_dim: int,
    num_classes: int,
    target: int,
    tolerance: float = 0.05,
    layers: int | None = None,
    heads: int = 4,
    dropout: float = 0.0,
    auxiliary_classes: int | None = None,
) -> tuple[nn.Module, ParameterMatch]:
    """Search the frozen hidden/FFN space and return the closest legal model."""
    if target <= 0 or not 0.0 < tolerance <= 0.25:
        raise ValueError("target and tolerance must be positive")
    candidates: list[tuple[int, int, int | None, nn.Module]] = []
    if kind == "gru":
        resolved_layers = 1 if layers is None else layers
        for hidden_size in range(8, 513):
            candidate_model: nn.Module = GRUBaseline(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_size=hidden_size,
                layers=resolved_layers,
                dropout=dropout,
                auxiliary_classes=auxiliary_classes,
            )
            candidates.append(
                (
                    trainable_parameter_count(candidate_model),
                    hidden_size,
                    None,
                    candidate_model,
                )
            )
    else:
        resolved_layers = 2 if layers is None else layers
        for hidden_size in range(max(8, heads), 513):
            if hidden_size % heads:
                continue
            for multiplier in (2, 4, 8):
                feedforward_size = hidden_size * multiplier
                candidate_model = TransformerBaseline(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    hidden_size=hidden_size,
                    layers=resolved_layers,
                    heads=heads,
                    feedforward_size=feedforward_size,
                    dropout=dropout,
                    auxiliary_classes=auxiliary_classes,
                )
                candidates.append(
                    (
                        trainable_parameter_count(candidate_model),
                        hidden_size,
                        feedforward_size,
                        candidate_model,
                    )
                )
    actual, selected_hidden_size, selected_feedforward_size, selected = min(
        candidates,
        key=lambda candidate: (abs(candidate[0] - target), candidate[0], candidate[1]),
    )
    validate_parameter_target(selected, target, tolerance)
    return selected, ParameterMatch(
        target=target,
        actual=actual,
        relative_error=abs(actual - target) / target,
        hidden_size=selected_hidden_size,
        feedforward_size=selected_feedforward_size,
    )


def estimate_macs(model: nn.Module, sequence_length: int) -> int:
    """Estimate supported Linear/GRU/attention multiply-accumulates per sequence."""
    if isinstance(model, GRUBaseline):
        input_projection = model.input_dim * model.hidden_size
        recurrent = (
            model.layers
            * 3
            * (model.hidden_size * model.hidden_size + model.hidden_size * model.hidden_size)
        )
        heads = sum(
            module.in_features * module.out_features
            for module in (model.output_head, model.next_state_head)
            if isinstance(module, nn.Linear)
        )
        return sequence_length * (input_projection + recurrent + heads)
    if isinstance(model, TransformerBaseline):
        projection = model.input_dim * model.hidden_size
        attention = model.layers * (
            4 * model.hidden_size * model.hidden_size + 2 * sequence_length * model.hidden_size
        )
        feedforward = model.layers * 2 * model.hidden_size * model.feedforward_size
        heads = sum(
            module.in_features * module.out_features
            for module in (model.output_head, model.next_state_head)
            if isinstance(module, nn.Linear)
        )
        return sequence_length * (projection + attention + feedforward + heads)
    raise TypeError(f"unsupported model for MAC estimation: {type(model).__name__}")


def profile_macs(model: nn.Module, sequence_length: int) -> MacProfile:
    """Return the MAC estimate and disclose unsupported trainable parameters."""
    supported_ids: set[int] = set()
    supported_types = (nn.Linear, nn.GRU, nn.MultiheadAttention)
    for module in model.modules():
        if isinstance(module, supported_types):
            supported_ids.update(
                id(parameter)
                for parameter in module.parameters(recurse=False)
                if parameter.requires_grad
            )
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    supported = sum(
        parameter.numel() for parameter in trainable.values() if id(parameter) in supported_ids
    )
    total = sum(parameter.numel() for parameter in trainable.values())
    unsupported = tuple(
        name for name, parameter in trainable.items() if id(parameter) not in supported_ids
    )
    operators: list[MacOperatorRecord] = []
    if isinstance(model, GRUBaseline):
        operators.extend(
            [
                MacOperatorRecord(
                    name="input_projection",
                    operator="Linear",
                    calls=1,
                    input_shape=(1, sequence_length, model.input_dim),
                    estimated_macs=sequence_length * model.input_dim * model.hidden_size,
                ),
                MacOperatorRecord(
                    name="encoder",
                    operator="GRU",
                    calls=1,
                    input_shape=(1, sequence_length, model.hidden_size),
                    estimated_macs=sequence_length
                    * model.layers
                    * 3
                    * (2 * model.hidden_size * model.hidden_size),
                ),
            ]
        )
        for name, head in (
            ("output_head", model.output_head),
            ("next_state_head", model.next_state_head),
        ):
            if isinstance(head, nn.Linear):
                operators.append(
                    MacOperatorRecord(
                        name=name,
                        operator="Linear",
                        calls=1,
                        input_shape=(1, sequence_length, head.in_features),
                        estimated_macs=sequence_length * head.in_features * head.out_features,
                    )
                )
    elif isinstance(model, TransformerBaseline):
        operators.extend(
            [
                MacOperatorRecord(
                    name="input_projection",
                    operator="Linear",
                    calls=1,
                    input_shape=(1, sequence_length, model.input_dim),
                    estimated_macs=sequence_length * model.input_dim * model.hidden_size,
                ),
                MacOperatorRecord(
                    name="encoder.self_attention",
                    operator="Attention",
                    calls=model.layers,
                    input_shape=(1, sequence_length, model.hidden_size),
                    estimated_macs=sequence_length
                    * model.layers
                    * (
                        4 * model.hidden_size * model.hidden_size
                        + 2 * sequence_length * model.hidden_size
                    ),
                ),
                MacOperatorRecord(
                    name="encoder.feedforward",
                    operator="Linear",
                    calls=2 * model.layers,
                    input_shape=(1, sequence_length, model.hidden_size),
                    estimated_macs=sequence_length
                    * model.layers
                    * 2
                    * model.hidden_size
                    * model.feedforward_size,
                ),
            ]
        )
        for name, head in (
            ("output_head", model.output_head),
            ("next_state_head", model.next_state_head),
        ):
            if isinstance(head, nn.Linear):
                operators.append(
                    MacOperatorRecord(
                        name=name,
                        operator="Linear",
                        calls=1,
                        input_shape=(1, sequence_length, head.in_features),
                        estimated_macs=sequence_length * head.in_features * head.out_features,
                    )
                )
    return MacProfile(
        estimated_macs=estimate_macs(model, sequence_length),
        supported_parameters=supported,
        total_parameters=total,
        coverage=supported / total if total else 1.0,
        unsupported_parameters=unsupported,
        operators=tuple(operators),
    )
