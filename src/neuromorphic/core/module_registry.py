"""Runtime registry for the frozen artificial computation module identifiers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import cast

from torch import nn

from neuromorphic.core.contracts import BrainModule
from neuromorphic.core.registry import MODULE_IDS, OPTIONAL_EXPERT_IDS, REQUIRED_PATH_IDS


class ModuleRegistry(nn.Module):
    """Own module implementations while preserving their frozen public identifiers.

    ``nn.ModuleDict`` cannot use identifiers containing dots, so implementations
    are held by a ``ModuleList`` and resolved through an explicit immutable-name
    index. Registration order never changes routing tie-breaking: callers use the
    frozen registry partitions exposed below.
    """

    def __init__(self, modules: Iterable[BrainModule] = ()) -> None:
        super().__init__()
        self.implementations = nn.ModuleList()
        self._index_by_id: dict[str, int] = {}
        for module in modules:
            self.register(module)

    def register(self, module: BrainModule) -> None:
        """Register one PyTorch implementation under its frozen ``module_id``."""

        module_id = module.module_id
        if module_id not in MODULE_IDS:
            raise ValueError(f"unregistered module identifier: {module_id}")
        if module_id in self._index_by_id:
            raise ValueError(f"duplicate module identifier: {module_id}")
        if not isinstance(module, nn.Module):
            raise TypeError("registered BrainModule implementations must inherit nn.Module")
        if not isinstance(module, BrainModule):
            raise TypeError("registered implementation does not satisfy BrainModule")
        self._index_by_id[module_id] = len(self.implementations)
        self.implementations.append(module)

    def get(self, module_id: str) -> BrainModule:
        """Return a registered implementation or raise a precise lookup error."""

        try:
            index = self._index_by_id[module_id]
        except KeyError as error:
            raise KeyError(f"module implementation is not registered: {module_id}") from error
        return cast(BrainModule, self.implementations[index])

    @property
    def ids(self) -> tuple[str, ...]:
        """Return registered IDs in the frozen global order, not insertion order."""

        return tuple(module_id for module_id in MODULE_IDS if module_id in self._index_by_id)

    @property
    def required_ids(self) -> tuple[str, ...]:
        return tuple(module_id for module_id in REQUIRED_PATH_IDS if module_id in self._index_by_id)

    @property
    def optional_ids(self) -> tuple[str, ...]:
        return tuple(
            module_id for module_id in OPTIONAL_EXPERT_IDS if module_id in self._index_by_id
        )

    def require_complete(self) -> None:
        """Reject a registry that cannot execute the frozen six-module graph."""

        missing = tuple(module_id for module_id in MODULE_IDS if module_id not in self._index_by_id)
        if missing:
            raise ValueError(f"module registry is incomplete; missing: {missing}")

    def __contains__(self, module_id: object) -> bool:
        return isinstance(module_id, str) and module_id in self._index_by_id

    def __len__(self) -> int:
        return len(self._index_by_id)

    def __iter__(self) -> Iterator[BrainModule]:
        for module_id in self.ids:
            yield self.get(module_id)


__all__ = ["ModuleRegistry"]
