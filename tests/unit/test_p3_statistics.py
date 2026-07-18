import pytest

from neuromorphic.evaluation.p3_statistics import (
    chance_normalized,
    forgetting,
    normalized_aulc,
    normalized_ood,
    paired_hierarchical_bootstrap,
)


def _records(offset: float, variant: str = "reference") -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "task_id": "task",
            "split": "test",
            "distribution": "v1",
            "model_id": "modular",
            "variant_id": variant,
            "sample_index": sample,
            "stratum": f"s-{sample % 2}",
            "value": offset + seed / 1000 + sample / 100,
        }
        for seed in (17, 29, 43)
        for sample in range(8)
    ]


def test_paired_bootstrap_preserves_known_difference() -> None:
    result = paired_hierarchical_bootstrap(
        _records(1.0), _records(0.75, "comparison"), samples=500, rng_seed=7
    )
    assert result.estimate == pytest.approx(0.25)
    assert result.lower == pytest.approx(0.25)
    assert result.upper == pytest.approx(0.25)
    assert result.p_value < 0.01


def test_paired_bootstrap_rejects_duplicate_or_mismatched_samples() -> None:
    reference = _records(1.0)
    with pytest.raises(ValueError, match="duplicate"):
        paired_hierarchical_bootstrap(
            [*reference, reference[0]], _records(0.5, "comparison"), samples=10
        )
    with pytest.raises(ValueError, match="identical"):
        paired_hierarchical_bootstrap(reference[:-1], _records(0.5, "comparison"), samples=10)
    mixed = _records(1.0)
    mixed[0]["variant_id"] = "unexpected"
    with pytest.raises(ValueError, match="one model_id/variant_id"):
        paired_hierarchical_bootstrap(mixed, _records(0.5, "comparison"), samples=10)


def test_p3_curve_and_normalization_formulas() -> None:
    assert normalized_aulc([(0, 0.0), (5, 1.0)], maximum_step=10) == pytest.approx(0.75)
    assert chance_normalized(0.75, 0.5) == pytest.approx(0.5)
    assert normalized_ood(0.6, 0.8, 0.5) == pytest.approx(1 / 3)
    assert forgetting(0.9, 0.7) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="exceed chance"):
        normalized_ood(0.6, 0.5, 0.5)
