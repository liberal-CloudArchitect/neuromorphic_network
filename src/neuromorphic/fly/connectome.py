"""Device-neutral container for signed, synapse-count connectomes.

The container stores one aggregated edge per ``(pre, post)`` neuron pair.  Its
weights are signed synapse counts: positive for excitatory and negative for
inhibitory presynaptic transmitters.  Signs are dataset annotations (often
predicted), not measured synaptic efficacies, so every downstream model must
treat them as structural priors rather than physiological ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

_INDEX_DTYPE = torch.int64
_COUNT_DTYPE = torch.int32


def _as_index_tensor(values: Tensor | Sequence[int], *, name: str) -> Tensor:
    tensor = torch.as_tensor(values)
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if tensor.is_floating_point() or tensor.is_complex() or tensor.dtype == torch.bool:
        raise TypeError(f"{name} must contain integers")
    return tensor.to(device="cpu", dtype=_INDEX_DTYPE)


@dataclass(frozen=True, slots=True)
class Connectome:
    """Aggregated directed graph with signed synapse counts.

    ``neuron_ids`` are dataset root IDs sorted ascending; ``pre`` and ``post``
    index into it.  ``signed_count`` is never zero, and ``(pre, post)`` pairs are
    unique and sorted lexicographically by ``(post, pre)``.
    """

    dataset: str
    version: str
    neuron_ids: Tensor
    pre: Tensor
    post: Tensor
    signed_count: Tensor

    def __post_init__(self) -> None:
        if not self.dataset or not self.version:
            raise ValueError("dataset and version must be non-empty")
        for name in ("neuron_ids", "pre", "post"):
            tensor = getattr(self, name)
            if tensor.dtype != _INDEX_DTYPE or tensor.dim() != 1 or tensor.device.type != "cpu":
                raise ValueError(f"{name} must be a one-dimensional CPU int64 tensor")
        if self.signed_count.dtype != _COUNT_DTYPE or self.signed_count.dim() != 1:
            raise ValueError("signed_count must be a one-dimensional int32 tensor")
        edges = self.pre.numel()
        if self.post.numel() != edges or self.signed_count.numel() != edges:
            raise ValueError("pre, post and signed_count must have equal length")
        neurons = self.neuron_ids.numel()
        if neurons == 0:
            raise ValueError("connectome must contain at least one neuron")
        if neurons > 1 and not bool(torch.all(self.neuron_ids[1:] > self.neuron_ids[:-1])):
            raise ValueError("neuron_ids must be strictly increasing")
        if edges == 0:
            return
        if int(self.pre.min()) < 0 or int(self.post.min()) < 0:
            raise ValueError("edge indices must be non-negative")
        if int(self.pre.max()) >= neurons or int(self.post.max()) >= neurons:
            raise ValueError("edge indices must address neuron_ids")
        if bool(torch.any(self.signed_count == 0)):
            raise ValueError("signed_count must not contain zero-weight edges")
        keys = self.post * neurons + self.pre
        if edges > 1 and not bool(torch.all(keys[1:] > keys[:-1])):
            raise ValueError("edges must be unique and sorted by (post, pre)")

    @classmethod
    def from_edges(
        cls,
        *,
        dataset: str,
        version: str,
        pre_ids: Tensor | Sequence[int],
        post_ids: Tensor | Sequence[int],
        signed_count: Tensor | Sequence[int],
        neuron_ids: Tensor | Sequence[int] | None = None,
    ) -> Connectome:
        """Build a connectome from root-ID edge rows.

        Duplicate ``(pre, post)`` rows, such as per-neuropil rows of the same
        neuron pair, are summed.  Pairs whose signed counts cancel to zero are
        dropped.  When ``neuron_ids`` is given it must cover every edge endpoint
        and also keeps isolated neurons.
        """

        pre_root = _as_index_tensor(pre_ids, name="pre_ids")
        post_root = _as_index_tensor(post_ids, name="post_ids")
        counts = _as_index_tensor(signed_count, name="signed_count")
        if pre_root.numel() != post_root.numel() or counts.numel() != pre_root.numel():
            raise ValueError("pre_ids, post_ids and signed_count must have equal length")
        endpoints = torch.cat((pre_root, post_root))
        if neuron_ids is None:
            ids = torch.unique(endpoints)
        else:
            ids = torch.unique(_as_index_tensor(neuron_ids, name="neuron_ids"))
            if ids.numel() != len(neuron_ids):
                raise ValueError("neuron_ids must be unique")
        if ids.numel() == 0:
            raise ValueError("connectome must contain at least one neuron")
        pre = _lookup(ids, pre_root, name="pre_ids")
        post = _lookup(ids, post_root, name="post_ids")
        neurons = ids.numel()
        keys, inverse = torch.unique(post * neurons + pre, return_inverse=True)
        summed = torch.zeros(keys.numel(), dtype=_INDEX_DTYPE).index_add_(0, inverse, counts)
        keep = summed != 0
        keys = keys[keep]
        summed = summed[keep]
        limit = torch.iinfo(_COUNT_DTYPE).max
        if summed.numel() and int(summed.abs().max()) > limit:
            raise OverflowError("aggregated synapse count exceeds int32")
        return cls(
            dataset=dataset,
            version=version,
            neuron_ids=ids,
            pre=keys % neurons,
            post=torch.div(keys, neurons, rounding_mode="floor"),
            signed_count=summed.to(_COUNT_DTYPE),
        )

    @property
    def n_neurons(self) -> int:
        return self.neuron_ids.numel()

    @property
    def n_edges(self) -> int:
        return self.pre.numel()

    @property
    def n_synapses(self) -> int:
        """Total absolute synapse count over retained edges."""

        return int(self.signed_count.abs().to(_INDEX_DTYPE).sum())

    def index_of(self, root_ids: Tensor | Sequence[int]) -> Tensor:
        """Map dataset root IDs to row indices, rejecting unknown IDs."""

        return _lookup(
            self.neuron_ids, _as_index_tensor(root_ids, name="root_ids"), name="root_ids"
        )

    def weight_matrix(
        self,
        *,
        scale: float,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Return sparse ``W[post, pre] = scale * signed_count``.

        MPS lacks compressed sparse layouts, so accelerator matrices use COO; the
        CPU path uses CSR, which is faster for the repeated ``W @ spikes`` product.
        """

        indices = torch.stack((self.post, self.pre))
        values = self.signed_count.to(dtype) * scale
        size = (self.n_neurons, self.n_neurons)
        # Invariants were validated in __post_init__; skip the per-call re-check.
        matrix = torch.sparse_coo_tensor(
            indices, values, size, is_coalesced=True, check_invariants=False
        )
        if device.type == "cpu":
            return matrix.to_sparse_csr()
        return matrix.to(device)


def _lookup(sorted_ids: Tensor, query: Tensor, *, name: str) -> Tensor:
    if query.numel() == 0:
        return query.clone()
    position = torch.searchsorted(sorted_ids, query)
    clipped = position.clamp(max=sorted_ids.numel() - 1)
    missing = sorted_ids[clipped] != query
    if bool(torch.any(missing)):
        example = int(query[missing][0])
        raise KeyError(f"{name} contains an ID absent from the connectome: {example}")
    return clipped
