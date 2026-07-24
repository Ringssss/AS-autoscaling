from scripts.benchmark_migration_policy import (
    PolicyAgent,
    order_agents,
    projected_cost,
)


def candidate(name: str, prefix: int, gap: float, copy: float, ordinal: int):
    return PolicyAgent(name, prefix, (), gap, copy, ordinal)


def test_order_agents_basic_policies():
    agents = [
        candidate("a", 8, 0.3, 0.1, 0),
        candidate("b", 4, 0.1, 0.05, 1),
        candidate("c", 16, 0.6, 0.2, 2),
    ]
    assert [item.agent_id for item in order_agents("shortest-kv", agents)] == [
        "b",
        "a",
        "c",
    ]
    assert [item.agent_id for item in order_agents("earliest-return", agents)] == [
        "b",
        "a",
        "c",
    ]
    assert [item.agent_id for item in order_agents("largest-kv", agents)] == [
        "c",
        "a",
        "b",
    ]


def test_oracle_minimizes_projected_exposure():
    agents = [
        candidate("a", 8, 0.10, 0.08, 0),
        candidate("b", 4, 0.09, 0.03, 1),
        candidate("c", 16, 0.30, 0.12, 2),
    ]
    oracle = order_agents("oracle", agents)
    all_costs = [projected_cost(order) for order in __import__("itertools").permutations(agents)]
    assert projected_cost(oracle) == min(all_costs)


def test_agentshift_defers_long_copy_that_breaks_a_deadline():
    agents = [
        candidate("short-urgent", 4, 0.06, 0.03, 0),
        candidate("long-urgent", 32, 0.10, 0.09, 1),
        candidate("medium", 16, 0.20, 0.06, 2),
    ]
    order = order_agents("agentshift-score", agents)
    assert [item.agent_id for item in order] == [
        "short-urgent",
        "medium",
        "long-urgent",
    ]
    assert sum(value > 0 for value in _exposure(order)) == 1


def _exposure(order):
    elapsed = 0.0
    values = []
    for item in order:
        elapsed += item.estimated_copy_seconds
        values.append(max(0.0, elapsed - item.gap_seconds))
    return values
