from __future__ import annotations

from collections import deque

import torch
from torch import Tensor

from neuromorphic.tasks import SmallGraphTask


def _shortest_action_from_observation(observation: Tensor) -> int:
    adjacency = observation[:256].reshape(16, 16).bool()
    current = int(observation[256:272].argmax().item())
    goal = int(observation[272:288].argmax().item())
    action_region = observation[288:352].reshape(4, 16)
    valid_slots = observation[352:356].bool()
    distances = [-1] * 16
    distances[goal] = 0
    queue: deque[int] = deque([goal])
    while queue:
        node = queue.popleft()
        for neighbor in torch.nonzero(adjacency[node], as_tuple=False).flatten().tolist():
            if distances[neighbor] < 0:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    for slot in range(4):
        if bool(valid_slots[slot].item()):
            neighbor = int(action_region[slot].argmax().item())
            if distances[neighbor] == distances[current] - 1:
                return slot
    raise AssertionError("connected graph must expose an optimal action")


def test_small_graph_shapes_degree_bound_and_oracle_targets() -> None:
    task = SmallGraphTask()
    batch = task.generate("train", list(range(32)))
    adjacency = batch.auxiliary_targets["adjacency"][:, 0]
    node_count = batch.auxiliary_targets["node_count"][:, 0]
    for graph, count_tensor in zip(adjacency, node_count, strict=True):
        count = int(count_tensor.item())
        assert 6 <= count <= 10
        assert torch.equal(graph, graph.T)
        assert int(graph[:count, :count].sum(dim=1).max().item()) <= 4
    oracle = task.oracle(batch)
    assert oracle.shape == (*batch.targets.shape, 4)
    chosen = torch.nn.functional.one_hot(batch.targets.clamp_min(0), num_classes=4).bool()
    assert torch.all((chosen & oracle).any(dim=-1)[batch.loss_mask])
    assert torch.equal(
        batch.auxiliary_targets["next_state"][batch.loss_mask],
        batch.auxiliary_targets["action_nodes"][chosen & batch.loss_mask.unsqueeze(-1)],
    )


def test_small_graph_id_and_ood_node_ranges() -> None:
    task = SmallGraphTask()
    train = task.generate("train", list(range(24)))
    ood = task.generate("ood", list(range(24)))
    train_counts = train.auxiliary_targets["node_count"][:, 0]
    ood_counts = ood.auxiliary_targets["node_count"][:, 0]
    assert torch.all((train_counts >= 6) & (train_counts <= 10))
    assert torch.all((ood_counts >= 11) & (ood_counts <= 16))


def test_small_graph_canonical_actions_and_live_rollout_succeed() -> None:
    task = SmallGraphTask()
    batch = task.generate("test", list(range(12)))
    metrics = task.evaluate_actions(batch, batch.targets)
    assert metrics == {
        "optimal_action_rate": 1.0,
        "success_rate": 1.0,
        "path_excess": 0.0,
    }
    rollout = task.rollout(_shortest_action_from_observation, "test", list(range(12)))
    assert rollout == {
        "optimal_action_rate": 1.0,
        "success_rate": 1.0,
        "path_excess": 0.0,
    }


def test_small_graph_contains_set_valued_shortest_path_targets() -> None:
    task = SmallGraphTask()
    batch = task.generate("train", list(range(128)))
    optimal_counts = batch.auxiliary_targets["optimal_action_mask"].sum(dim=-1)
    assert torch.any(optimal_counts > 1)
