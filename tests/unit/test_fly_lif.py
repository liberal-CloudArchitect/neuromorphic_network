from __future__ import annotations

import dataclasses

import pytest
import torch

from neuromorphic.fly import Connectome
from neuromorphic.fly.lif import LIFParameters, simulate

# Root IDs: 1 = stimulated source, 2 = excitatory target.
_PARAMS = LIFParameters(duration_ms=200.0)


def _graph(excitatory: int = 40) -> Connectome:
    return Connectome.from_edges(
        dataset="toy", version="1", pre_ids=[1], post_ids=[2], signed_count=[excitatory]
    )


def _rates(graph: Connectome, **kwargs: object) -> dict[int, float]:
    result = simulate(graph, graph.index_of([1]), parameters=_PARAMS, trials=4, **kwargs)  # type: ignore[arg-type]
    rates = result.rates_hz()
    return {int(root): float(rates[i]) for i, root in enumerate(graph.neuron_ids)}


def test_unstimulated_network_stays_silent() -> None:
    graph = _graph()
    result = simulate(graph, torch.tensor([], dtype=torch.int64), parameters=_PARAMS, trials=2)
    assert int(result.spike_counts.sum()) == 0
    assert result.spike_counts.shape == (2, 2)


def test_excitation_propagates_and_silencing_blocks_it() -> None:
    graph = _graph()
    rates = _rates(graph)
    assert rates[1] > 50.0
    assert rates[2] > 10.0
    silenced = _rates(graph, silenced=graph.index_of([1]))
    assert silenced[1] == rates[1]
    assert silenced[2] == 0.0


def test_inhibition_reduces_downstream_rate() -> None:
    # Driven neuron 1 excites neuron 2; driven neuron 3 inhibits it.
    def target_rate(inhibitory: int) -> float:
        graph = Connectome.from_edges(
            dataset="toy",
            version="1",
            pre_ids=[1, 3],
            post_ids=[2, 2],
            signed_count=[40, -inhibitory],
        )
        result = simulate(graph, graph.index_of([1, 3]), parameters=_PARAMS, trials=4)
        return float(result.rates_hz()[1])

    baseline = target_rate(1)
    assert baseline > 10.0
    assert target_rate(200) < 0.5 * baseline


def test_same_seed_is_bitwise_reproducible_on_cpu() -> None:
    graph = _graph()
    stim = graph.index_of([1])
    first = simulate(graph, stim, parameters=_PARAMS, trials=3, seed=11).spike_counts
    second = simulate(graph, stim, parameters=_PARAMS, trials=3, seed=11).spike_counts
    other = simulate(graph, stim, parameters=_PARAMS, trials=3, seed=12).spike_counts
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_refractory_period_caps_rate() -> None:
    # Neuron 2 is not driven, so it keeps the 2.2 ms refractory period even
    # when every presynaptic spike alone exceeds threshold.
    graph = _graph(excitatory=1000)
    saturating = dataclasses.replace(_PARAMS, poisson_rate_hz=10_000.0)
    result = simulate(graph, graph.index_of([1]), parameters=saturating)
    interval = saturating.refractory_steps + 1
    assert 0 < int(result.spike_counts[0, 1]) <= saturating.steps // interval + 1
    driven_limited = dataclasses.replace(saturating, driven_refractory_ms=2.2)
    capped = simulate(graph, graph.index_of([1]), parameters=driven_limited)
    assert int(capped.spike_counts[0, 0]) < int(result.spike_counts[0, 0])


def test_parameter_and_index_validation() -> None:
    with pytest.raises(ValueError, match="distinct"):
        LIFParameters(membrane_tau_ms=5.0, synaptic_tau_ms=5.0)
    with pytest.raises(ValueError, match="threshold"):
        LIFParameters(v_threshold_mv=-60.0)
    graph = _graph()
    with pytest.raises(ValueError, match="outside"):
        simulate(graph, torch.tensor([5]))
    with pytest.raises(ValueError, match="duplicates"):
        simulate(graph, torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="too high"):
        simulate(graph, torch.tensor([0]), parameters=LIFParameters(poisson_rate_hz=20_000.0))


@pytest.mark.mps
def test_deterministic_drive_matches_cpu_on_mps() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    graph = Connectome.from_edges(
        dataset="toy", version="1", pre_ids=[1, 1], post_ids=[2, 3], signed_count=[40, -3]
    )
    # Probability-one Poisson drive removes RNG-stream differences between devices.
    params = dataclasses.replace(_PARAMS, poisson_rate_hz=10_000.0, poisson_weight_factor=0.5)
    stim = graph.index_of([1, 3])
    cpu = simulate(graph, stim, parameters=params, trials=2).spike_counts
    mps = simulate(graph, stim, parameters=params, trials=2, device=torch.device("mps"))
    assert int(cpu.sum()) > 0
    assert torch.equal(mps.spike_counts, cpu)


def test_input_during_refractory_is_discarded_by_default() -> None:
    # A strong source makes the target fire at its refractory ceiling; input
    # arriving inside refractory windows only counts in "accumulate" mode.
    graph = _graph(excitatory=30)
    stim = graph.index_of([1])
    params = dataclasses.replace(_PARAMS, poisson_rate_hz=400.0)
    discard = simulate(graph, stim, parameters=params, trials=8).rates_hz()[1]
    accumulate = simulate(
        graph,
        stim,
        parameters=dataclasses.replace(params, refractory_input="accumulate"),
        trials=8,
    ).rates_hz()[1]
    assert float(discard) < float(accumulate)
