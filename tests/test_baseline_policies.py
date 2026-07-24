import pytest

from agentshift.controller.baselines import calibrate_ttl, agentix_style_route


def test_calibrated_ttl_retains_when_recompute_dominates():
    result = calibrate_ttl(
        [0.01, 0.05, 0.1],
        recompute_seconds=1.0,
        kv_gib=1.0,
        hbm_cost_per_gib_second=0.01,
    )
    assert result.ttl_seconds == 0.1


def test_calibrated_ttl_evicts_under_high_memory_pressure():
    result = calibrate_ttl(
        [0.01, 0.05, 0.1],
        recompute_seconds=0.01,
        kv_gib=4.0,
        hbm_cost_per_gib_second=1.0,
    )
    assert result.ttl_seconds == 0.0


def test_agentix_style_routing_preserves_locality_when_queues_are_equal():
    decision = agentix_style_route(
        source_engine="a",
        destination_engine="b",
        source_queue_seconds=0.0,
        destination_queue_seconds=0.0,
        next_call_service_seconds=0.1,
        destination_reprefill_seconds=0.5,
    )
    assert decision.engine == "a"


def test_agentix_style_routing_trades_locality_for_large_queue_relief():
    decision = agentix_style_route(
        source_engine="a",
        destination_engine="b",
        source_queue_seconds=1.0,
        destination_queue_seconds=0.0,
        next_call_service_seconds=0.1,
        destination_reprefill_seconds=0.5,
    )
    assert decision.engine == "b"


def test_baseline_policies_reject_negative_costs():
    with pytest.raises(ValueError):
        calibrate_ttl(
            [0.1],
            recompute_seconds=-1,
            kv_gib=1,
            hbm_cost_per_gib_second=1,
        )
    with pytest.raises(ValueError):
        agentix_style_route(
            source_engine="a",
            destination_engine="b",
            source_queue_seconds=-1,
            destination_queue_seconds=0,
            next_call_service_seconds=0,
            destination_reprefill_seconds=0,
        )
