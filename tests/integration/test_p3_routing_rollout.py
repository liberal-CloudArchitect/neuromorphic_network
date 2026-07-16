import torch

from neuromorphic.core.contracts import BrainPacket
from neuromorphic.modules.sparse_router import SparseRouter
from neuromorphic.tasks import SmallGraphTask


def _packet() -> BrainPacket:
    return BrainPacket(
        representation=torch.zeros(4, 2, 16),
        valid_mask=torch.ones(4, 2, dtype=torch.bool),
        modality="fixture",
        step_index=torch.arange(2).expand(4, 2),
        source_module="sensory_encoder.v1",
    )


def test_dense_fixed_and_random_routing_are_explicit() -> None:
    router = SparseRouter(feature_dim=16)
    dense = router.route(_packet(), mode="dense")
    fixed = router.route(_packet(), mode="fixed")
    random_first = router.route(_packet(), mode="random")
    random_second = router.route(_packet(), mode="random")
    assert dense.executed_mask.sum(dim=-1).eq(3).all()
    assert fixed.executed_mask.sum(dim=-1).eq(2).all()
    assert torch.equal(random_first.executed_mask, random_second.executed_mask)


def test_small_graph_rollout_passes_explicit_reset() -> None:
    task = SmallGraphTask(profile="smoke")
    resets: list[bool] = []

    def policy(observation: torch.Tensor, reset: bool) -> int:
        resets.append(reset)
        mask = observation[-4:].bool()
        return int(torch.nonzero(mask, as_tuple=False)[0].item())

    records = task.rollout_records(policy, "test", [0, 1])
    assert len(records) == 2
    assert sum(resets) == 2
    assert all("success_rate" in record for record in records)
