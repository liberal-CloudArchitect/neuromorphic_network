import torch

from neuromorphic.evaluation.p3_records import (
    linear_cka,
    linear_probe_accuracy,
    p3_sample_records,
    rsa_spearman,
)
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


def test_bounded_representation_analysis_and_model_agnostic_records() -> None:
    features = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1])
    assert linear_cka(features, features) == 1.0
    assert rsa_spearman(features, features) == 1.0
    assert linear_probe_accuracy(features, labels, features, labels) == 1.0
    task = create_task("associative_recall.v1", profile="smoke")
    batch = task.generate("test", [0])
    output = SharedGRUBaseline(hidden_size=32)(batch)
    records = p3_sample_records(output, batch, run_seed=17, model_id="gru", variant_id="full")
    assert records[0]["sample_index"] == 0
    assert records[0]["stratum"] == records[0]["bootstrap_stratum"]
