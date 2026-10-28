"""Strict configuration and cell registry for p6-protocol-v1 (CR-015, errata E1-E2).

Every number frozen by ``docs/p6_implementation_spec.md`` or an erratum is
validated here, so a run cannot start with a budget, seed or preset that the
protocol does not allow.  The registries say which cells exist and how each
one is built; runners consume them and never invent cells of their own.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuromorphic.training.p5_config import P5_PILOT_PRESETS

P6_PROTOCOL_VERSION: Final = "p6-protocol-v1"
P6_DEVELOPMENT_SEED: Final = 7
P6_FORMAL_SEEDS: Final[tuple[int, ...]] = (17, 29, 43, 59, 71)

type P6PresetId = Literal["preset-0", "preset-1", "preset-2", "preset-3", "preset-4"]
P6_PILOT_PRESETS: Final[dict[str, tuple[float, float, float]]] = {
    **P5_PILOT_PRESETS,
    "preset-4": (1.0e-3, 0.10, 0.01),
}
P6_BASELINE_LEARNING_RATES: Final[tuple[float, ...]] = tuple(
    sorted({learning_rate for learning_rate, _, _ in P6_PILOT_PRESETS.values()})
)

type P6Model = Literal["modular-v4", "shared-gru", "shared-transformer"]
type P6BaselineModel = Literal["shared-gru", "shared-transformer"]
P6_BASELINE_MODELS: Final[tuple[P6BaselineModel, ...]] = ("shared-gru", "shared-transformer")

type P6MechanismVariant = Literal[
    "full", "wm-v1", "gate-off", "enrich-off", "shuffle-surprise", "surprise-zero"
]

_STRUCTURAL = {
    "weight_decay": 0.01,
    "dual_budget_weight": 0.001,
    "dual_route_fraction": 0.25,
    "gradient_clip_norm": 1.0,
}
_TASK_COUNT = 4


@dataclass(frozen=True, slots=True)
class P6VariantSpec:
    """How one mechanism cell builds and runs ``ModularBrainNetworkV4``."""

    retrained: bool
    working_memory: Literal["v2", "v1"]
    write_source: Literal["enriched", "current"]
    surprise_gating: bool
    working_surprise: Literal["full", "shuffle", "zero"]
    source_variant: P6MechanismVariant | None

    def network_kwargs(self) -> dict[str, object]:
        return {
            "working_memory": self.working_memory,
            "write_source": self.write_source,
            "surprise_gating": self.surprise_gating,
        }


P6_MECHANISM_VARIANTS: Final[dict[P6MechanismVariant, P6VariantSpec]] = {
    "full": P6VariantSpec(True, "v2", "enriched", True, "full", None),
    "wm-v1": P6VariantSpec(True, "v1", "enriched", True, "full", None),
    "gate-off": P6VariantSpec(True, "v2", "enriched", False, "full", None),
    "enrich-off": P6VariantSpec(True, "v2", "current", True, "full", None),
    "shuffle-surprise": P6VariantSpec(False, "v2", "enriched", True, "shuffle", "full"),
    "surprise-zero": P6VariantSpec(False, "v2", "enriched", True, "zero", "full"),
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _HashedConfig(_StrictModel):
    """Hash only the fields that define the experiment, never local paths or IDs."""

    def compatible_dict(self) -> dict[str, object]:
        value = self.model_dump(
            mode="json",
            exclude={
                "run_id",
                "resume",
                "output_root",
                "control_root",
                "telemetry_enabled",
                "expected_git_commit",
            },
        )
        matrix = getattr(self, "matrix", None)
        if callable(matrix):
            value["matrix"] = [cell.model_dump(mode="json") for cell in matrix()]
        return value

    def config_hash(self) -> str:
        encoded = json.dumps(self.compatible_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _require_structural(config: BaseModel, label: str) -> None:
    for name, frozen in _STRUCTURAL.items():
        if name in type(config).model_fields and getattr(config, name) != frozen:
            raise ValueError(f"{label} {name} is frozen at {frozen}")


class P6QualificationConfig(_HashedConfig):
    """Engineering qualification at the P5 scale (E2, part one)."""

    schema_version: Literal["p6-qualification-v1"] = "p6-qualification-v1"
    protocol_version: Literal["p6-protocol-v1"] = "p6-protocol-v1"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    seed: Literal[7] = 7
    train_samples: Literal[64] = 64
    validation_samples: Literal[32] = 32
    batch_size: Literal[8] = 8
    steps_per_task: Literal[4] = 4
    learning_rate: float = Field(default=3.0e-4, gt=0.0)
    weight_decay: float = 0.01
    temporal_loss_weight: float = 0.10
    semantic_loss_weight: float = 0.01
    dual_budget_weight: float = 0.001
    dual_route_fraction: float = 0.25
    gradient_clip_norm: float = 1.0
    telemetry_enabled: Literal[False] = False
    output_root: Path = Path("artifacts/runs")
    control_root: Path = Path("artifacts/p6/control")
    run_id: str | None = None

    @model_validator(mode="after")
    def validate_frozen(self) -> P6QualificationConfig:
        _require_structural(self, "P6 qualification")
        if (self.learning_rate, self.temporal_loss_weight, self.semantic_loss_weight) != (
            3.0e-4,
            0.10,
            0.01,
        ):
            raise ValueError("P6 qualification optimizer settings are frozen at preset-1")
        return self


class P6SurpriseCheckConfig(_HashedConfig):
    """Surprise-validity training on seed 7 and train/validation data only (E2)."""

    schema_version: Literal["p6-surprise-check-v1"] = "p6-surprise-check-v1"
    protocol_version: Literal["p6-protocol-v1"] = "p6-protocol-v1"
    profile: Literal["ci", "formal"]
    device: Literal["cpu", "mps", "cuda"]
    seed: Literal[7] = 7
    preset: Literal["preset-1"] = "preset-1"
    train_samples: int = Field(ge=1)
    validation_samples: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    steps_per_task: int = Field(ge=1)
    correlation_threshold: float = 0.5
    weight_decay: float = 0.01
    dual_budget_weight: float = 0.001
    dual_route_fraction: float = 0.25
    gradient_clip_norm: float = 1.0
    telemetry_enabled: Literal[False] = False
    output_root: Path = Path("artifacts/runs")
    control_root: Path = Path("artifacts/p6/control")
    run_id: str | None = None
    expected_git_commit: str | None = None

    @model_validator(mode="after")
    def validate_frozen(self) -> P6SurpriseCheckConfig:
        _require_structural(self, "P6 surprise check")
        if self.correlation_threshold != 0.5:
            raise ValueError("P6 surprise correlation threshold is frozen at 0.5")
        budget = (self.train_samples, self.validation_samples, self.batch_size, self.steps_per_task)
        if self.profile == "ci" and (budget != (64, 32, 8, 4) or self.device != "cpu"):
            raise ValueError("P6 surprise-check CI budget is frozen on CPU")
        if self.profile == "formal" and budget != (8192, 2048, 64, 1000):
            raise ValueError("P6 formal surprise-check budget is frozen")
        return self

    @property
    def max_steps(self) -> int:
        return self.steps_per_task * _TASK_COUNT


class P6PilotCandidate(_StrictModel):
    candidate_id: str
    model: P6Model
    learning_rate: float = Field(gt=0.0)
    temporal_loss_weight: float | None
    semantic_loss_weight: float | None
    max_steps: int = Field(ge=1)


class P6PilotConfig(_HashedConfig):
    """Validation-only pilot: five V4 presets plus three learning rates per baseline (E2)."""

    schema_version: Literal["p6-pilot-v1"] = "p6-pilot-v1"
    protocol_version: Literal["p6-protocol-v1"] = "p6-protocol-v1"
    device: Literal["mps", "cuda"]
    seed: Literal[7] = 7
    train_samples: Literal[8192] = 8192
    validation_samples: Literal[2048] = 2048
    batch_size: Literal[64] = 64
    steps_per_candidate: Literal[1000] = 1000
    validation_interval: Literal[100] = 100
    checkpoint_interval: Literal[100] = 100
    weight_decay: float = 0.01
    dual_budget_weight: float = 0.001
    dual_route_fraction: float = 0.25
    gradient_clip_norm: float = 1.0
    telemetry_enabled: Literal[False] = False
    output_root: Path = Path("artifacts/runs")
    control_root: Path
    qualification_report: Path
    run_id: str | None = None
    resume: Path | None = None
    expected_git_commit: str | None = None

    @model_validator(mode="after")
    def validate_frozen(self) -> P6PilotConfig:
        _require_structural(self, "P6 pilot")
        return self

    def candidates(self) -> tuple[P6PilotCandidate, ...]:
        steps = self.steps_per_candidate * _TASK_COUNT
        modular = tuple(
            P6PilotCandidate(
                candidate_id=preset_id,
                model="modular-v4",
                learning_rate=learning_rate,
                temporal_loss_weight=temporal,
                semantic_loss_weight=semantic,
                max_steps=steps,
            )
            for preset_id, (learning_rate, temporal, semantic) in P6_PILOT_PRESETS.items()
        )
        baselines = tuple(
            P6PilotCandidate(
                candidate_id=f"{model}__lr{learning_rate:.0e}",
                model=model,
                learning_rate=learning_rate,
                temporal_loss_weight=None,
                semantic_loss_weight=None,
                max_steps=steps,
            )
            for model in P6_BASELINE_MODELS
            for learning_rate in P6_BASELINE_LEARNING_RATES
        )
        return modular + baselines


class P6MechanismCell(_StrictModel):
    cell_id: str
    seed: int
    variant: P6MechanismVariant
    retrained: bool
    max_steps: int = Field(ge=0)
    source_cell_id: str | None


class _FormalBudget(_HashedConfig):
    device: Literal["cpu", "mps", "cuda"]
    seeds: tuple[int, ...]
    train_samples: int = Field(ge=1)
    validation_samples: int = Field(ge=1)
    analysis_samples: int = Field(ge=1)
    test_samples: int = Field(ge=1)
    ood_samples: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    steps_per_task: int = Field(ge=1)
    validation_interval: int = Field(ge=1)
    checkpoint_interval: int = Field(ge=1)
    patience: int = Field(ge=1)
    bootstrap_samples: int = Field(ge=1)
    wall_clock_hours: float = Field(gt=0.0)
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    telemetry_enabled: Literal[False] = False
    output_root: Path = Path("artifacts/runs")
    control_root: Path
    run_id: str | None = None
    resume: Path | None = None
    expected_git_commit: str | None = None

    def _validate_budget(self, profile: str, label: str, wall_clock_hours: float) -> None:
        _require_structural(self, label)
        samples = (
            self.train_samples,
            self.validation_samples,
            self.analysis_samples,
            self.test_samples,
            self.ood_samples,
        )
        if profile == "qualification":
            if (
                self.seeds != (P6_DEVELOPMENT_SEED,)
                or samples != (64, 32, 32, 32, 32)
                or self.batch_size != 8
                or self.steps_per_task != 2
                or self.bootstrap_samples != 200
            ):
                raise ValueError(f"{label} qualification budget is frozen")
            return
        if (
            self.device != "cuda"
            or self.seeds != P6_FORMAL_SEEDS
            or samples != (8192, 2048, 512, 2048, 2048)
            or self.batch_size != 64
            or self.steps_per_task != 5000
            or self.validation_interval != 100
            or self.checkpoint_interval != 100
            or self.patience != 30
            or self.bootstrap_samples != 10000
            or self.wall_clock_hours != wall_clock_hours
        ):
            raise ValueError(f"{label} formal budget is frozen")


class P6MechanismConfig(_FormalBudget):
    schema_version: Literal["p6-mechanism-v1"] = "p6-mechanism-v1"
    protocol_version: Literal["p6-protocol-v1"] = "p6-protocol-v1"
    profile: Literal["qualification", "mechanism"]
    selected_preset: P6PresetId
    dual_budget_weight: float = 0.001
    dual_route_fraction: float = 0.25
    qualification_report: Path | None = None
    pilot_lock: Path | None = None

    @model_validator(mode="after")
    def validate_frozen(self) -> P6MechanismConfig:
        self._validate_budget(self.profile, "P6 mechanism", 48.0)
        if self.profile == "mechanism" and (
            self.qualification_report is None or self.pilot_lock is None
        ):
            raise ValueError("P6 formal mechanism requires qualification and pilot locks")
        return self

    def matrix(self) -> tuple[P6MechanismCell, ...]:
        steps = self.steps_per_task * _TASK_COUNT
        return tuple(
            P6MechanismCell(
                cell_id=f"{variant}__s{seed}",
                seed=seed,
                variant=variant,
                retrained=spec.retrained,
                max_steps=steps if spec.retrained else 0,
                source_cell_id=(
                    None if spec.source_variant is None else f"{spec.source_variant}__s{seed}"
                ),
            )
            for seed in self.seeds
            for variant, spec in P6_MECHANISM_VARIANTS.items()
        )


class P6FullCell(_StrictModel):
    cell_id: str
    seed: int
    model: P6Model
    retrained: bool
    max_steps: int = Field(ge=0)
    learning_rate: float | None
    source_cell_id: str | None


class P6FullConfig(_FormalBudget):
    """Train the two baselines; the modular cells are the mechanism run's ``full`` cells."""

    schema_version: Literal["p6-full-v1"] = "p6-full-v1"
    protocol_version: Literal["p6-protocol-v1"] = "p6-protocol-v1"
    profile: Literal["qualification", "full"]
    baseline_learning_rates: dict[P6BaselineModel, float]
    pilot_lock: Path | None = None
    mechanism_lock: Path | None = None

    @model_validator(mode="after")
    def validate_frozen(self) -> P6FullConfig:
        self._validate_budget(self.profile, "P6 full", 24.0)
        if set(self.baseline_learning_rates) != set(P6_BASELINE_MODELS):
            raise ValueError("P6 full needs a learning rate for each baseline")
        if any(
            rate not in P6_BASELINE_LEARNING_RATES for rate in self.baseline_learning_rates.values()
        ):
            raise ValueError("P6 baseline learning rates must come from the P6 grid")
        if self.profile == "full" and (self.pilot_lock is None or self.mechanism_lock is None):
            raise ValueError("P6 full requires pilot and mechanism locks")
        return self

    def matrix(self) -> tuple[P6FullCell, ...]:
        steps = self.steps_per_task * _TASK_COUNT
        cells: list[P6FullCell] = []
        for seed in self.seeds:
            cells.append(
                P6FullCell(
                    cell_id=f"modular-v4__s{seed}",
                    seed=seed,
                    model="modular-v4",
                    retrained=False,
                    max_steps=0,
                    learning_rate=None,
                    source_cell_id=f"full__s{seed}",
                )
            )
            cells.extend(
                P6FullCell(
                    cell_id=f"{model}__s{seed}",
                    seed=seed,
                    model=model,
                    retrained=True,
                    max_steps=steps,
                    learning_rate=self.baseline_learning_rates[model],
                    source_cell_id=None,
                )
                for model in P6_BASELINE_MODELS
            )
        return tuple(cells)


def _load[Config: BaseModel](path: Path, model: type[Config]) -> Config:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return model.model_validate(value)


def load_p6_qualification_config(path: Path) -> P6QualificationConfig:
    return _load(path, P6QualificationConfig)


def load_p6_surprise_check_config(path: Path) -> P6SurpriseCheckConfig:
    return _load(path, P6SurpriseCheckConfig)


def load_p6_pilot_config(path: Path) -> P6PilotConfig:
    return _load(path, P6PilotConfig)


def load_p6_mechanism_config(path: Path) -> P6MechanismConfig:
    return _load(path, P6MechanismConfig)


def load_p6_full_config(path: Path) -> P6FullConfig:
    return _load(path, P6FullConfig)


__all__ = [
    "P6_BASELINE_LEARNING_RATES",
    "P6_BASELINE_MODELS",
    "P6_DEVELOPMENT_SEED",
    "P6_FORMAL_SEEDS",
    "P6_MECHANISM_VARIANTS",
    "P6_PILOT_PRESETS",
    "P6_PROTOCOL_VERSION",
    "P6BaselineModel",
    "P6FullCell",
    "P6FullConfig",
    "P6MechanismCell",
    "P6MechanismConfig",
    "P6MechanismVariant",
    "P6Model",
    "P6PilotCandidate",
    "P6PilotConfig",
    "P6PresetId",
    "P6QualificationConfig",
    "P6SurpriseCheckConfig",
    "P6VariantSpec",
    "load_p6_full_config",
    "load_p6_mechanism_config",
    "load_p6_pilot_config",
    "load_p6_qualification_config",
    "load_p6_surprise_check_config",
]
