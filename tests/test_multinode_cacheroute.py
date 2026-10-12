import random

import pytest

from experiments.multinode.cacheroute import build_plan, dispatch


def test_top_rate_admission_and_lpt_preserve_total_expected_load():
    plan = build_plan({"a": 9, "b": 8, "c": 7, "d": 6, "tail": 1}, 2, 10, 4)
    assert plan.assignments == {"a": (0,), "b": (1,), "c": (1,), "d": (0,)}
    assert plan.expected_load == (15.5, 15.5)
    assert plan.slots_used == 4
    assert "tail" not in plan.assignments


def test_hot_key_has_distinct_destinations_with_per_assignment_load_cap():
    plan = build_plan({"hot": 25, "cool": 4}, 4, 10, 4)
    assert len(set(plan.assignments["hot"])) == 3
    assert max(plan.expected_load) <= 10
    rng = random.Random(0)
    assert dispatch(plan, "hot", [8, 3, 5, 0], rng) == 1


def test_plan_is_stable_when_live_load_moves():
    plan = build_plan({"warm": 1}, 3, 10, 1)
    assert dispatch(plan, "warm", [100, 0, 0], random.Random(0)) == 0
    assert plan.assignments["warm"] == (0,)


def test_unadmitted_keys_use_only_two_sampled_choices():
    plan = build_plan({}, 5, 10, 0)
    expected_rng, actual_rng = random.Random(13), random.Random(13)
    loads = [8, 4, 7, 1, 9]
    for _ in range(20):
        choices = expected_rng.sample(range(5), 2)
        assert dispatch(plan, "cold", loads, actual_rng) == min(choices, key=lambda i: loads[i])


def test_insufficient_slots_do_not_silently_skip_a_higher_rate_key():
    plan = build_plan({"first": 18, "second": 12, "third": 1}, 3, 10, 3)
    assert set(plan.assignments) == {"first"}
    assert sum(plan.expected_load) == pytest.approx(31)


@pytest.mark.parametrize("rate", [-1, float("nan"), float("inf"), 31])
def test_invalid_or_infeasible_rates_fail_explicitly(rate):
    with pytest.raises(ValueError):
        build_plan({"key": rate}, 3, 10, 30)
