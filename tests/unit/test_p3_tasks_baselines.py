import torch

from neuromorphic.tasks import create_task
from neuromorphic.training.p3_baselines import (
    SharedGRUBaseline,
    SharedTransformerBaseline,
    select_shared_parameter_match,
)


def test_analysis_split_and_p3_distributions_are_isolated() -> None:
    distributions = {
        "associative_recall.v1": ("capacity", "interference", "joint"),
        "delayed_rule_switch.v1": ("delay", "composition", "joint"),
        "small_graph.v1": ("scale", "topology", "joint"),
    }
    for task_id, variants in distributions.items():
        base = create_task(task_id, profile="smoke")
        analysis = base.generate("analysis", [0, 1])
        assert analysis.metadata["split_seed"] == 5501
        hashes = {base.content_hash("test", 0), base.content_hash("analysis", 0)}
        for variant in variants:
            task = create_task(task_id, profile="smoke", distribution=variant)
            hashes.add(task.content_hash("ood", 0))
            assert task.generate("ood", [0]).metadata["distribution"] == variant
        assert len(hashes) == 5


def test_shared_baselines_cover_three_task_boundaries() -> None:
    for model in (SharedGRUBaseline(hidden_size=32), SharedTransformerBaseline(hidden_size=32)):
        for task_id in (
            "associative_recall.v1",
            "delayed_rule_switch.v1",
            "small_graph.v1",
        ):
            batch = create_task(task_id, profile="smoke").generate("train", [0, 1])
            output = model(batch)
            assert output.logits.shape[:2] == batch.targets.shape
            assert torch.isfinite(output.logits).all()


def test_shared_parameter_matching_is_within_five_percent() -> None:
    target = 230_365
    for kind in ("gru", "transformer"):
        match = select_shared_parameter_match(kind, target)
        assert match.relative_error <= 0.05
