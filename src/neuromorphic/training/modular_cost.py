"""Auditable MAC accounting for boundary, required, and sparse P2 paths."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Literal

from torch import nn

from neuromorphic.core.registry import OPTIONAL_EXPERT_IDS, REQUIRED_PATH_IDS
from neuromorphic.modules.network import ModularBrainNetwork
from neuromorphic.tasks.control import TASK_IDS

type MacCategory = Literal["boundary", "required", "optional"]


@dataclass(frozen=True, slots=True)
class ModuleMacRecord:
    """One supported operator contribution with dense and actual call counts."""

    module_id: str
    category: MacCategory
    operator: str
    macs_per_call: int
    dense_calls: int
    active_calls: int
    supported_parameters: int
    total_parameters: int

    def __post_init__(self) -> None:
        if not self.module_id or not self.operator:
            raise ValueError("module_id and operator must be non-empty")
        if (
            min(
                self.macs_per_call,
                self.dense_calls,
                self.active_calls,
                self.supported_parameters,
                self.total_parameters,
            )
            < 0
        ):
            raise ValueError("MAC counts and parameter counts must be non-negative")
        if self.active_calls > self.dense_calls:
            raise ValueError("active calls cannot exceed dense calls")
        if self.supported_parameters > self.total_parameters:
            raise ValueError("supported parameters cannot exceed total parameters")
        if self.category == "required" and self.module_id not in REQUIRED_PATH_IDS:
            raise ValueError("required MAC records must use a required-path module ID")
        if self.category == "optional" and self.module_id not in OPTIONAL_EXPERT_IDS:
            raise ValueError("optional MAC records must use an optional expert ID")

    @property
    def dense_macs(self) -> int:
        return self.macs_per_call * self.dense_calls

    @property
    def active_macs(self) -> int:
        return self.macs_per_call * self.active_calls


@dataclass(frozen=True, slots=True)
class ModularMacProfile:
    """Aggregate cost split required by the GATE-2 report."""

    boundary_macs: int
    required_macs: int
    dense_optional_macs: int
    active_optional_macs: int
    active_total_macs: int
    dense_total_macs: int
    parameter_coverage: float
    supported_parameters: int
    total_parameters: int
    records: tuple[ModuleMacRecord, ...]

    @property
    def sparse_optional_saving(self) -> int:
        return self.dense_optional_macs - self.active_optional_macs

    def to_dict(self) -> dict[str, object]:
        """Return an explicit JSON-compatible profile."""

        return {
            "boundary_macs": self.boundary_macs,
            "required_macs": self.required_macs,
            "dense_optional_macs": self.dense_optional_macs,
            "active_optional_macs": self.active_optional_macs,
            "active_total_macs": self.active_total_macs,
            "dense_total_macs": self.dense_total_macs,
            "sparse_optional_saving": self.sparse_optional_saving,
            "parameter_coverage": self.parameter_coverage,
            "supported_parameters": self.supported_parameters,
            "total_parameters": self.total_parameters,
            "records": [asdict(record) for record in self.records],
        }


def linear_macs(layer: nn.Linear, *, token_count: int) -> int:
    """Return multiply-accumulates for a linear layer over a token count."""

    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    return token_count * layer.in_features * layer.out_features


def build_modular_mac_profile(records: Iterable[ModuleMacRecord]) -> ModularMacProfile:
    """Aggregate operator records without hiding unsupported parameters."""

    frozen_records = tuple(records)
    boundary = sum(record.active_macs for record in frozen_records if record.category == "boundary")
    required = sum(record.active_macs for record in frozen_records if record.category == "required")
    dense_optional = sum(
        record.dense_macs for record in frozen_records if record.category == "optional"
    )
    active_optional = sum(
        record.active_macs for record in frozen_records if record.category == "optional"
    )
    supported = sum(record.supported_parameters for record in frozen_records)
    total = sum(record.total_parameters for record in frozen_records)
    coverage = 1.0 if total == 0 else supported / total
    return ModularMacProfile(
        boundary_macs=boundary,
        required_macs=required,
        dense_optional_macs=dense_optional,
        active_optional_macs=active_optional,
        active_total_macs=boundary + required + active_optional,
        dense_total_macs=boundary + required + dense_optional,
        parameter_coverage=coverage,
        supported_parameters=supported,
        total_parameters=total,
        records=frozen_records,
    )


def _module_record(
    *,
    module_id: str,
    category: MacCategory,
    module: nn.Module,
    dense_calls: int,
    active_calls: int,
) -> ModuleMacRecord:
    linears = tuple(layer for layer in module.modules() if isinstance(layer, nn.Linear))
    macs_per_call = sum(layer.in_features * layer.out_features for layer in linears)
    supported_parameters = sum(
        parameter.numel() for layer in linears for parameter in layer.parameters(recurse=False)
    )
    total_parameters = sum(parameter.numel() for parameter in module.parameters())
    return ModuleMacRecord(
        module_id=module_id,
        category=category,
        operator="Linear",
        macs_per_call=macs_per_call,
        dense_calls=dense_calls,
        active_calls=active_calls,
        supported_parameters=supported_parameters,
        total_parameters=total_parameters,
    )


def profile_modular_execution(
    model: ModularBrainNetwork,
    *,
    task_token_counts: Mapping[str, int],
    expert_active_counts: Mapping[str, int],
) -> ModularMacProfile:
    """Profile supported linear MACs using observed sparse call counts."""

    if set(task_token_counts) != set(TASK_IDS):
        raise ValueError("task token counts must cover the three frozen P2 tasks")
    if set(expert_active_counts) != set(OPTIONAL_EXPERT_IDS):
        raise ValueError("expert call counts must cover all optional experts")
    valid_tokens = sum(task_token_counts.values())
    records: list[ModuleMacRecord] = []
    for task_id, adapter in zip(TASK_IDS, model.boundary_adapters.adapters, strict=True):
        calls = task_token_counts[task_id]
        records.append(
            _module_record(
                module_id=f"boundary.{task_id}",
                category="boundary",
                module=adapter,
                dense_calls=calls,
                active_calls=calls,
            )
        )
    for module_id in REQUIRED_PATH_IDS:
        module = model.registry.get(module_id)
        if not isinstance(module, nn.Module):
            raise TypeError("registered required module is not torch.nn.Module")
        records.append(
            _module_record(
                module_id=module_id,
                category="required",
                module=module,
                dense_calls=valid_tokens,
                active_calls=valid_tokens,
            )
        )
    for module_id in OPTIONAL_EXPERT_IDS:
        module = model.registry.get(module_id)
        if not isinstance(module, nn.Module):
            raise TypeError("registered optional module is not torch.nn.Module")
        records.append(
            _module_record(
                module_id=module_id,
                category="optional",
                module=module,
                dense_calls=valid_tokens,
                active_calls=expert_active_counts[module_id],
            )
        )
    return build_modular_mac_profile(records)


__all__ = [
    "MacCategory",
    "ModularMacProfile",
    "ModuleMacRecord",
    "build_modular_mac_profile",
    "linear_macs",
    "profile_modular_execution",
]
