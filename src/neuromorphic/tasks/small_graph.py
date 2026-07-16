"""Built-in deterministic shortest-path task with at most four action slots."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor

from neuromorphic.tasks.base import (
    CPU_DEVICE,
    SPLIT_SEEDS,
    DatasetSplit,
    TaskBatch,
    deterministic_seed,
    make_generator,
    stable_content_hash,
)

type GraphPolicy = Callable[[Tensor], int | Tensor]
type StatefulGraphPolicy = Callable[[Tensor, bool], int | Tensor]


@dataclass(frozen=True, slots=True)
class _Graph:
    adjacency: Tensor
    node_count: int
    start: int
    goal: int


@dataclass(frozen=True, slots=True)
class _Sample:
    inputs: Tensor
    targets: Tensor
    loss_mask: Tensor
    optimal_action_mask: Tensor
    next_state: Tensor
    current_state: Tensor
    action_nodes: Tensor
    graph: _Graph
    optimal_distance: int


class SmallGraphTask:
    """Shortest-path behavior cloning over deterministic connected graphs."""

    task_id = "small_graph.v1"
    task_version = "small-graph-v1"
    max_nodes = 16
    max_actions = 4
    input_dim = 356
    num_classes = 4

    def __init__(self, *, profile: str = "smoke", distribution: str = "v1") -> None:
        if distribution not in {"v1", "scale", "topology", "joint"}:
            raise ValueError(f"unknown small-graph distribution: {distribution}")
        self.profile = profile
        self.distribution = distribution
        self.task_version = (
            "small-graph-v1" if distribution == "v1" else f"small-graph-p3-{distribution}-v1"
        )

    @staticmethod
    def _randint(generator: torch.Generator, low: int, high: int) -> int:
        return int(torch.randint(low, high, (), generator=generator).item())

    def _make_graph(self, split: DatasetSplit, sample_index: int) -> _Graph:
        generator = make_generator(self.task_version, split, sample_index)
        if split == "ood":
            node_count = (
                self._randint(generator, 11, 17)
                if self.distribution in {"v1", "scale", "joint"}
                else self._randint(generator, 6, 11)
            )
        else:
            node_count = self._randint(generator, 6, 11)
        adjacency = torch.zeros((self.max_nodes, self.max_nodes), dtype=torch.bool)
        degrees = [0] * node_count

        if split == "ood" and self.distribution in {"topology", "joint"}:
            for node in range(node_count):
                neighbor = (node + 1) % node_count
                adjacency[node, neighbor] = True
                adjacency[neighbor, node] = True
                degrees[node] += 1
                degrees[neighbor] += 1
            for node in range(0, node_count - 2, 3):
                neighbor = node + 2
                adjacency[node, neighbor] = True
                adjacency[neighbor, node] = True
                degrees[node] += 1
                degrees[neighbor] += 1
        else:
            self._add_random_edges(adjacency, degrees, node_count, generator)

        start = self._randint(generator, 0, node_count)
        goal = self._randint(generator, 0, node_count - 1)
        if goal >= start:
            goal += 1
        return _Graph(adjacency, node_count, start, goal)

    def _add_random_edges(
        self,
        adjacency: Tensor,
        degrees: list[int],
        node_count: int,
        generator: torch.Generator,
    ) -> None:

        # A bounded-degree random tree guarantees connectivity.
        for node in range(1, node_count):
            candidates = [parent for parent in range(node) if degrees[parent] < self.max_actions]
            parent = candidates[self._randint(generator, 0, len(candidates))]
            adjacency[node, parent] = True
            adjacency[parent, node] = True
            degrees[node] += 1
            degrees[parent] += 1

        # Extra edges create cycles and, in some samples, multiple optimal actions.
        for _ in range(node_count * 2):
            first = self._randint(generator, 0, node_count)
            second = self._randint(generator, 0, node_count)
            if (
                first != second
                and not bool(adjacency[first, second].item())
                and degrees[first] < self.max_actions
                and degrees[second] < self.max_actions
            ):
                adjacency[first, second] = True
                adjacency[second, first] = True
                degrees[first] += 1
                degrees[second] += 1

    @staticmethod
    def _distances(adjacency: Tensor, goal: int, node_count: int) -> list[int]:
        distances = [-1] * node_count
        distances[goal] = 0
        frontier = [goal]
        while frontier:
            current = frontier.pop(0)
            neighbors = torch.nonzero(adjacency[current, :node_count], as_tuple=False).flatten()
            for neighbor_tensor in neighbors:
                neighbor = int(neighbor_tensor.item())
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[current] + 1
                    frontier.append(neighbor)
        if any(distance < 0 for distance in distances):
            raise RuntimeError("generated graph is disconnected")
        return distances

    def _action_nodes(self, graph: _Graph, current: int) -> Tensor:
        neighbors = torch.nonzero(
            graph.adjacency[current, : graph.node_count], as_tuple=False
        ).flatten()
        if self.distribution in {"topology", "joint"}:
            neighbors = neighbors.flip(0)
        action_nodes = torch.full((self.max_actions,), -1, dtype=torch.long)
        action_nodes[: neighbors.numel()] = neighbors
        return action_nodes

    def _observation(self, graph: _Graph, current: int) -> Tensor:
        action_nodes = self._action_nodes(graph, current)
        observation = torch.zeros(self.input_dim, dtype=torch.float32)
        offset = 0
        observation[offset : offset + 256] = graph.adjacency.to(torch.float32).flatten()
        offset += 256
        observation[offset + current] = 1.0
        offset += 16
        observation[offset + graph.goal] = 1.0
        offset += 16
        for slot, neighbor_tensor in enumerate(action_nodes):
            neighbor = int(neighbor_tensor.item())
            if neighbor >= 0:
                observation[offset + slot * 16 + neighbor] = 1.0
        offset += 64
        observation[offset : offset + self.max_actions] = action_nodes.ge(0).to(torch.float32)
        return observation

    @lru_cache(maxsize=32_768)  # noqa: B019 - bounded task instances live for one run
    def _make_sample(self, split: DatasetSplit, sample_index: int) -> _Sample:
        graph = self._make_graph(split, sample_index)
        distances = self._distances(graph.adjacency, graph.goal, graph.node_count)
        rows: list[Tensor] = []
        labels: list[int] = []
        selected: list[bool] = []
        optimal_masks: list[Tensor] = []
        next_states: list[int] = []
        current_states: list[int] = []
        action_rows: list[Tensor] = []
        current = graph.start

        while current != graph.goal:
            action_nodes = self._action_nodes(graph, current)
            optimal_mask = torch.zeros(self.max_actions, dtype=torch.bool)
            for slot, neighbor_tensor in enumerate(action_nodes):
                neighbor = int(neighbor_tensor.item())
                if neighbor >= 0 and distances[neighbor] == distances[current] - 1:
                    optimal_mask[slot] = True
            canonical_action = int(torch.nonzero(optimal_mask, as_tuple=False)[0].item())
            next_node = int(action_nodes[canonical_action].item())
            rows.append(self._observation(graph, current))
            labels.append(canonical_action)
            selected.append(True)
            optimal_masks.append(optimal_mask)
            next_states.append(next_node)
            current_states.append(current)
            action_rows.append(action_nodes)
            current = next_node

        # A terminal observation makes episode completion explicit but carries no loss.
        rows.append(self._observation(graph, current))
        labels.append(-100)
        selected.append(False)
        optimal_masks.append(torch.zeros(self.max_actions, dtype=torch.bool))
        next_states.append(-100)
        current_states.append(current)
        action_rows.append(self._action_nodes(graph, current))
        return _Sample(
            inputs=torch.stack(rows),
            targets=torch.tensor(labels, dtype=torch.long),
            loss_mask=torch.tensor(selected, dtype=torch.bool),
            optimal_action_mask=torch.stack(optimal_masks),
            next_state=torch.tensor(next_states, dtype=torch.long),
            current_state=torch.tensor(current_states, dtype=torch.long),
            action_nodes=torch.stack(action_rows),
            graph=graph,
            optimal_distance=distances[graph.start],
        )

    def content_hash(self, split: DatasetSplit, sample_index: int) -> str:
        sample = self._make_sample(split, sample_index)
        return stable_content_hash(
            sample.graph.adjacency,
            sample.inputs,
            sample.targets,
            sample.optimal_action_mask,
            prefix=self.task_version,
        )

    def generate(
        self,
        split: DatasetSplit,
        sample_indices: Sequence[int],
        *,
        device: torch.device = CPU_DEVICE,
    ) -> TaskBatch:
        if not sample_indices:
            raise ValueError("sample_indices cannot be empty")
        samples = [self._make_sample(split, index) for index in sample_indices]
        max_steps = max(sample.inputs.shape[0] for sample in samples)
        batch_size = len(samples)
        inputs = torch.zeros((batch_size, max_steps, self.input_dim), dtype=torch.float32)
        targets = torch.full((batch_size, max_steps), -100, dtype=torch.long)
        valid_mask = torch.zeros((batch_size, max_steps), dtype=torch.bool)
        loss_mask = torch.zeros((batch_size, max_steps), dtype=torch.bool)
        episode_ids = torch.full((batch_size, max_steps), -1, dtype=torch.long)
        optimal_action_mask = torch.zeros(
            (batch_size, max_steps, self.max_actions), dtype=torch.bool
        )
        next_state = torch.full((batch_size, max_steps), -100, dtype=torch.long)
        current_state = torch.full((batch_size, max_steps), -1, dtype=torch.long)
        action_nodes = torch.full((batch_size, max_steps, self.max_actions), -1, dtype=torch.long)
        adjacency = torch.zeros(
            (batch_size, max_steps, self.max_nodes, self.max_nodes), dtype=torch.bool
        )
        start_node = torch.full((batch_size, max_steps), -1, dtype=torch.long)
        goal_node = torch.full((batch_size, max_steps), -1, dtype=torch.long)
        optimal_distance = torch.full((batch_size, max_steps), -1, dtype=torch.long)
        node_count = torch.full((batch_size, max_steps), -1, dtype=torch.long)
        hashes: list[str] = []
        lengths: list[int] = []
        indexed_samples = zip(sample_indices, samples, strict=True)
        for batch_index, (sample_index, sample) in enumerate(indexed_samples):
            steps = sample.inputs.shape[0]
            inputs[batch_index, :steps] = sample.inputs
            targets[batch_index, :steps] = sample.targets
            valid_mask[batch_index, :steps] = True
            loss_mask[batch_index, :steps] = sample.loss_mask
            episode_ids[batch_index, :steps] = deterministic_seed(
                self.task_version, split, sample_index
            )
            optimal_action_mask[batch_index, :steps] = sample.optimal_action_mask
            next_state[batch_index, :steps] = sample.next_state
            current_state[batch_index, :steps] = sample.current_state
            action_nodes[batch_index, :steps] = sample.action_nodes
            adjacency[batch_index, :steps] = sample.graph.adjacency
            start_node[batch_index, :steps] = sample.graph.start
            goal_node[batch_index, :steps] = sample.graph.goal
            optimal_distance[batch_index, :steps] = sample.optimal_distance
            node_count[batch_index, :steps] = sample.graph.node_count
            hashes.append(self.content_hash(split, sample_index))
            lengths.append(steps)

        return TaskBatch(
            inputs=inputs,
            targets=targets,
            valid_mask=valid_mask,
            loss_mask=loss_mask,
            episode_ids=episode_ids,
            metadata={
                "task_id": self.task_id,
                "task_version": self.task_version,
                "split": split,
                "split_seed": SPLIT_SEEDS[split],
                "sample_indices": tuple(sample_indices),
                "content_hashes": tuple(hashes),
                "sequence_lengths": tuple(lengths),
                "profile": self.profile,
                "distribution": self.distribution,
            },
            auxiliary_targets={
                "optimal_action_mask": optimal_action_mask,
                "next_state": next_state,
                "current_state": current_state,
                "action_nodes": action_nodes,
                "adjacency": adjacency,
                "start_node": start_node,
                "goal_node": goal_node,
                "optimal_distance": optimal_distance,
                "node_count": node_count,
            },
        ).to(device)

    def oracle(self, batch: TaskBatch) -> Tensor:
        batch.validate()
        try:
            return batch.auxiliary_targets["optimal_action_mask"]
        except KeyError as error:
            raise ValueError("SmallGraph batch is missing optimal_action_mask") from error

    @staticmethod
    def _action_id(action: int | Tensor) -> int:
        if isinstance(action, int):
            return action
        if action.numel() == 1:
            return int(action.item())
        if action.shape[-1] != 4:
            raise ValueError("policy output must be an action ID or four action logits")
        return int(action.reshape(-1, 4)[-1].argmax().item())

    def evaluate_actions(self, batch: TaskBatch, actions: Tensor) -> Mapping[str, float]:
        """Execute supplied action sequences against the batch graphs."""
        batch.validate()
        if actions.ndim == 3:
            if actions.shape != (batch.batch_size, batch.sequence_length, self.max_actions):
                raise ValueError("action logits must have shape [B, T, 4]")
            action_ids = actions.argmax(dim=-1)
        elif actions.shape == (batch.batch_size, batch.sequence_length):
            action_ids = actions
        else:
            raise ValueError("actions must have shape [B, T] or [B, T, 4]")

        successes = 0
        total_excess = 0.0
        optimal_actions = 0
        attempted_actions = 0
        aux = batch.auxiliary_targets
        for batch_index in range(batch.batch_size):
            adjacency = aux["adjacency"][batch_index, 0].to("cpu")
            current = int(aux["start_node"][batch_index, 0].item())
            goal = int(aux["goal_node"][batch_index, 0].item())
            count = int(aux["node_count"][batch_index, 0].item())
            shortest = int(aux["optimal_distance"][batch_index, 0].item())
            distances = self._distances(adjacency, goal, count)
            steps_taken = 0
            for step in range(batch.sequence_length):
                if current == goal:
                    break
                if not bool(batch.valid_mask[batch_index, step].item()):
                    break
                action = int(action_ids[batch_index, step].item())
                neighbors = (
                    torch.nonzero(adjacency[current, :count], as_tuple=False).flatten().tolist()
                )
                attempted_actions += 1
                steps_taken += 1
                if 0 <= action < len(neighbors):
                    next_node = int(neighbors[action])
                    if distances[next_node] == distances[current] - 1:
                        optimal_actions += 1
                    current = next_node
            if current == goal:
                successes += 1
            total_excess += max(steps_taken - shortest, 0)
        return {
            "optimal_action_rate": optimal_actions / max(attempted_actions, 1),
            "success_rate": successes / batch.batch_size,
            "path_excess": total_excess / batch.batch_size,
        }

    def rollout(
        self,
        policy: GraphPolicy,
        split: DatasetSplit,
        sample_indices: Sequence[int],
        *,
        device: torch.device = CPU_DEVICE,
        max_steps: int | None = None,
    ) -> Mapping[str, float]:
        """Roll out a policy on live graph states and compute graph metrics."""
        if not sample_indices:
            raise ValueError("sample_indices cannot be empty")
        successes = 0
        total_excess = 0.0
        optimal_actions = 0
        attempted_actions = 0
        for sample_index in sample_indices:
            graph = self._make_graph(split, sample_index)
            distances = self._distances(graph.adjacency, graph.goal, graph.node_count)
            shortest = distances[graph.start]
            horizon = (
                max_steps if max_steps is not None else max(graph.node_count * 2, shortest * 2)
            )
            current = graph.start
            steps_taken = 0
            while current != graph.goal and steps_taken < horizon:
                observation = self._observation(graph, current).to(device)
                action = self._action_id(policy(observation))
                action_nodes = self._action_nodes(graph, current)
                attempted_actions += 1
                steps_taken += 1
                if 0 <= action < self.max_actions:
                    next_node = int(action_nodes[action].item())
                    if next_node >= 0:
                        if distances[next_node] == distances[current] - 1:
                            optimal_actions += 1
                        current = next_node
            if current == graph.goal:
                successes += 1
            total_excess += max(steps_taken - shortest, 0)
        return {
            "optimal_action_rate": optimal_actions / max(attempted_actions, 1),
            "success_rate": successes / len(sample_indices),
            "path_excess": total_excess / len(sample_indices),
        }

    def rollout_records(
        self,
        policy: StatefulGraphPolicy,
        split: DatasetSplit,
        sample_indices: Sequence[int],
        *,
        device: torch.device = CPU_DEVICE,
        max_steps: int | None = None,
    ) -> list[dict[str, object]]:
        """Return one auditable live-rollout record per sample with explicit reset."""

        if not sample_indices:
            raise ValueError("sample_indices cannot be empty")
        records: list[dict[str, object]] = []
        for sample_index in sample_indices:
            graph = self._make_graph(split, sample_index)
            distances = self._distances(graph.adjacency, graph.goal, graph.node_count)
            shortest = distances[graph.start]
            horizon = max_steps or max(graph.node_count * 2, shortest * 2)
            current = graph.start
            steps = 0
            optimal = 0
            invalid = 0
            reset = True
            while current != graph.goal and steps < horizon:
                action = self._action_id(
                    policy(self._observation(graph, current).to(device), reset)
                )
                reset = False
                action_nodes = self._action_nodes(graph, current)
                steps += 1
                if 0 <= action < self.max_actions:
                    next_node = int(action_nodes[action].item())
                    if next_node >= 0:
                        if distances[next_node] == distances[current] - 1:
                            optimal += 1
                        current = next_node
                    else:
                        invalid += 1
                else:
                    invalid += 1
            records.append(
                {
                    "schema_version": "p3-small-graph-rollout-v1",
                    "task_id": self.task_id,
                    "task_version": self.task_version,
                    "distribution": self.distribution,
                    "split": split,
                    "sample_index": sample_index,
                    "success_rate": float(current == graph.goal),
                    "path_excess": float(max(steps - shortest, 0)),
                    "optimal_action_rate": optimal / max(steps, 1),
                    "invalid_actions": invalid,
                    "node_count": graph.node_count,
                    "shortest_distance": shortest,
                    "bootstrap_stratum": f"nodes-{graph.node_count}/distance-{shortest}",
                }
            )
        return records
