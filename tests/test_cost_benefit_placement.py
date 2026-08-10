import pytest

from agentshift.controller.placement import (
    CostBenefitPlacementPolicy,
    EngineLoad,
    MobilityCandidate,
    order_admissible_first,
)


def candidate(agent_id="a", **overrides):
    values = {
        "agent_id": agent_id,
        "source_engine": "source",
        "prefix_tokens": 1000,
        "kv_bytes": 1024**3,
        "remaining_gap_seconds": 0.2,
        "recompute_seconds": 0.5,
        "future_service_seconds": 0.1,
    }
    values.update(overrides)
    return MobilityCandidate(**values)


def loads():
    return [
        EngineLoad("source", queue_depth=8, free_kv_tokens=10000, kv_pressure=0.9),
        EngineLoad("target", queue_depth=1, free_kv_tokens=1500, kv_pressure=0.2),
    ]


def test_cost_benefit_policy_selects_useful_hidden_move():
    policy = CostBenefitPlacementPolicy(
        bandwidth_bytes_per_second=10 * 1024**3,
        max_concurrent_migrations=1,
    )
    selected = policy.select([candidate()], loads())
    assert len(selected) == 1
    assert selected[0].destination_engine == "target"
    assert selected[0].exposed_seconds == 0


def test_cost_benefit_policy_rejects_unknown_effect_and_excess_exposure():
    policy = CostBenefitPlacementPolicy(
        bandwidth_bytes_per_second=1024**3,
        acceptable_exposure_seconds=0.01,
    )
    assert policy.select([candidate(unknown_effect=True)], loads()) == []
    assert policy.select([candidate(remaining_gap_seconds=0.1)], loads()) == []


def test_cost_benefit_policy_bounds_concurrency_and_capacity():
    policy = CostBenefitPlacementPolicy(
        bandwidth_bytes_per_second=10 * 1024**3,
        max_concurrent_migrations=3,
    )
    selected = policy.select([candidate("a"), candidate("b")], loads())
    assert len(selected) == 1


def test_cost_benefit_policy_selects_only_disjoint_engine_pairs():
    policy = CostBenefitPlacementPolicy(
        bandwidth_bytes_per_second=10 * 1024**3,
        max_concurrent_migrations=3,
    )
    candidates = [
        candidate("from-a", source_engine="a"),
        candidate("from-c", source_engine="c"),
    ]
    engine_loads = [
        EngineLoad("a", queue_depth=9, free_kv_tokens=10_000, kv_pressure=0.9),
        EngineLoad("b", queue_depth=0, free_kv_tokens=10_000, kv_pressure=0.1),
        EngineLoad("c", queue_depth=8, free_kv_tokens=10_000, kv_pressure=0.8),
        EngineLoad("d", queue_depth=1, free_kv_tokens=10_000, kv_pressure=0.2),
    ]
    selected = policy.select(candidates, engine_loads)
    used_engines = [
        engine
        for result in selected
        for engine in (result.source_engine, result.destination_engine)
    ]
    assert len(selected) == 2
    assert len(used_engines) == len(set(used_engines))


def test_cost_benefit_policy_validates_configuration():
    with pytest.raises(ValueError):
        CostBenefitPlacementPolicy(bandwidth_bytes_per_second=0)


def test_admissible_first_defers_the_longest_deadline_breaker():
    jobs = [
        {"id": "short", "deadline": 0.06, "duration": 0.03, "size": 4},
        {"id": "long", "deadline": 0.10, "duration": 0.09, "size": 32},
        {"id": "medium", "deadline": 0.20, "duration": 0.06, "size": 16},
    ]
    ordered = order_admissible_first(
        jobs,
        deadline_seconds=lambda item: item["deadline"],
        duration_seconds=lambda item: item["duration"],
        size=lambda item: item["size"],
    )
    assert [item["id"] for item in ordered] == ["short", "medium", "long"]
