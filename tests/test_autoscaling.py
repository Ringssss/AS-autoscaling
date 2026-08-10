from agentshift.controller.autoscaling import (
    EngineLifecycle,
    EngineObservation,
    WarmPoolAutoscaler,
)


def observations(requests=(0, 0, 0), owners=(0, 0, 0)):
    return [
        EngineObservation(f"e{index}", request, 0, owner)
        for index, (request, owner) in enumerate(zip(requests, owners))
    ]


def test_sustained_load_scales_out_one_warm_engine():
    autoscaler = WarmPoolAutoscaler(
        ["e0", "e1", "e2"],
        min_active=1,
        scale_out_requests_per_engine=2,
        scale_out_windows=2,
        cooldown_seconds=0,
    )
    assert autoscaler.tick(0, observations((3, 0, 0))) == []
    actions = autoscaler.tick(1, observations((3, 0, 0)))
    assert [(item.action, item.engine_id) for item in actions] == [
        ("SCALE_OUT", "e1")
    ]
    assert autoscaler.states["e1"] == EngineLifecycle.ACTIVE


def test_scale_in_waits_for_drain_to_become_empty():
    autoscaler = WarmPoolAutoscaler(
        ["e0", "e1"],
        min_active=1,
        scale_in_windows=2,
        cooldown_seconds=0,
    )
    autoscaler.states["e1"] = EngineLifecycle.ACTIVE
    assert autoscaler.tick(0, observations(owners=(0, 2))) == []
    actions = autoscaler.tick(1, observations(owners=(0, 2)))
    assert actions[0].action == "SCALE_IN_START"
    draining = actions[0].engine_id
    owner_counts = (1, 0) if draining == "e0" else (0, 1)
    assert autoscaler.tick(2, observations(owners=owner_counts)) == []
    final = autoscaler.tick(3, observations())
    assert [(item.action, item.engine_id) for item in final] == [
        ("SCALE_IN_COMPLETE", draining)
    ]


def test_cooldown_prevents_immediate_second_scale_out():
    autoscaler = WarmPoolAutoscaler(
        ["e0", "e1", "e2"],
        min_active=1,
        scale_out_windows=1,
        cooldown_seconds=10,
    )
    assert autoscaler.tick(0, observations((3, 0, 0)))[0].engine_id == "e1"
    assert autoscaler.tick(1, observations((3, 3, 0))) == []
    assert autoscaler.tick(10, observations((3, 3, 0)))[0].engine_id == "e2"


def test_missing_observation_is_rejected():
    autoscaler = WarmPoolAutoscaler(["e0", "e1"], min_active=1)
    try:
        autoscaler.tick(0, [EngineObservation("e0", 0, 0, 0)])
    except KeyError as exc:
        assert "e1" in str(exc)
    else:
        raise AssertionError("missing observation should fail")
