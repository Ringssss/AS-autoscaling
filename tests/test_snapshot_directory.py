import pytest

from agentshift.controller.snapshot_directory import (
    SnapshotDirectory,
    best_reusable_prefix,
    validated_snapshots,
)


def response(query_tokens, boundaries, *, matched_tokens=None, **overrides):
    result = {
        "query_tokens": query_tokens,
        "matched_tokens": query_tokens if matched_tokens is None else matched_tokens,
        "snapshot_boundaries": boundaries,
        "available_token_slots": 5,
        "evictable_token_slots": 20,
        "available_snapshot_slots": 1,
        "evictable_snapshot_slots": 2,
    }
    result.update(overrides)
    return result


def test_only_confirmed_boundaries_no_rounding_or_invented_prompt_end():
    tokens = tuple(range(131))
    assert validated_snapshots(tokens, response(131, [17, 64, 129])) == (
        tokens[:17], tokens[:64], tokens[:129])
    assert validated_snapshots(tokens, response(131, [], matched_tokens=100)) == ()


@pytest.mark.parametrize("overrides", [
    {"query_tokens": 5},
    {"query_tokens": True},
    {"matched_tokens": 7},
    {"matched_tokens": -1},
    {"matched_tokens": 3, "snapshot_boundaries": [4]},
    {"snapshot_boundaries": [0]},
    {"snapshot_boundaries": [2, 2]},
    {"snapshot_boundaries": [3, 2]},
    {"snapshot_boundaries": [True]},
    {"snapshot_boundaries": [2.0]},
    {"snapshot_boundaries": [7]},
    {"available_token_slots": -1},
    {"evictable_token_slots": -1},
    {"available_snapshot_slots": -1},
    {"evictable_snapshot_slots": False},
])
def test_rejects_malformed_or_overclaimed_engine_response(overrides):
    value = response(6, [2, 4])
    value.update(overrides)
    with pytest.raises(ValueError):
        validated_snapshots(range(6), value)


def test_missing_resources_are_not_defaulted_to_zero():
    value = response(6, [2])
    del value["available_snapshot_slots"]
    with pytest.raises(ValueError):
        validated_snapshots(range(6), value)


def test_best_match_requires_confirmed_identity_and_suffix_by_default():
    tokens = (10, 11, 12, 13)
    observed = [(10, 11), tokens, (10, 99, 12), tokens + (14,)]
    assert best_reusable_prefix(tokens, observed) == 2
    assert best_reusable_prefix(tokens, observed, require_suffix=False) == 4
    assert best_reusable_prefix((20, 21), observed) == 0
    assert best_reusable_prefix((), observed) == 0


def test_atomic_update_removes_stale_query_path_and_preserves_other_paths():
    directory = SnapshotDirectory()
    directory.update(2, (1, 2, 3, 4), response(4, [2, 4]))
    directory.update(2, (8, 9, 10), response(3, [3]))
    directory.update(2, (1, 2, 3, 4), response(4, [2], matched_tokens=2))
    assert set(directory.prefixes(2)) == {(1, 2), (8, 9, 10)}
    previous = directory.prefixes(2)
    with pytest.raises(ValueError):
        directory.update(2, (1, 2, 3, 4), response(4, [5]))
    assert directory.prefixes(2) == previous
    directory.update(2, (1, 2, 3, 4), response(4, [], matched_tokens=0))
    assert directory.prefixes(2) == ((8, 9, 10),)


def test_short_query_does_not_disprove_a_longer_untested_snapshot():
    directory = SnapshotDirectory()
    directory.update(7, (1, 2, 3, 4), response(4, [4]))
    directory.update(7, (1, 2), response(2, [], matched_tokens=2))
    assert directory.prefixes(7) == ((1, 2, 3, 4),)


def test_physical_engines_stay_separate_and_resources_are_not_a_reservation():
    directory = SnapshotDirectory()
    directory.update(7, (1, 2), response(2, [2], available_token_slots=0))
    directory.update(3, (8, 9), response(2, [2]))
    assert directory.best_reusable_prefix(7, (1, 2, 3)) == 2
    assert directory.best_reusable_prefix(3, (1, 2, 3)) == 0
    resources = directory.resources(7)
    assert resources.available_token_slots == 0
    assert resources.evictable_token_slots == 20
    with pytest.raises((AttributeError, TypeError)):
        resources.available_token_slots = 20
    directory.discard_engine(7)
    assert directory.prefixes(7) == () and directory.resources(7) is None
    assert directory.best_reusable_prefix(3, (8, 9, 10)) == 2


def test_lru_bounds_and_mutable_caller_input_are_safe():
    directory = SnapshotDirectory(max_snapshots_per_engine=2, max_tokens_per_engine=5)
    tokens = [1, 2]
    directory.update(0, tokens, response(2, [2]))
    tokens[0] = 99
    directory.update(0, (3, 4), response(2, [2]))
    directory.best_reusable_prefix(0, (1, 2, 7))  # Touch successful match.
    directory.update(0, (5, 6), response(2, [2]))
    assert set(directory.prefixes(0)) == {(1, 2), (5, 6)}
    directory.update(0, tuple(range(6)), response(6, [6]))
    assert sum(map(len, directory.prefixes(0))) <= 5


def test_engine_lru_budget_and_clear():
    directory = SnapshotDirectory(max_engines=2)
    for engine in (10, 20):
        directory.update(engine, (engine, 1), response(2, [2]))
    directory.best_reusable_prefix(10, (10, 1, 2))
    directory.update(30, (30, 1), response(2, [2]))
    assert directory.prefixes(20) == ()
    assert directory.resources(20) is None
    assert directory.prefixes(10) == ((10, 1),)
    directory.clear()
    assert directory.prefixes(10) == directory.prefixes(30) == ()


def test_response_and_query_mutation_cannot_rewrite_confirmed_state():
    directory = SnapshotDirectory()
    tokens = [7, 8, 9]
    value = response(3, [2])
    confirmed = directory.update("physical-A", tokens, value)
    tokens[:] = [1, 2, 3]
    value["snapshot_boundaries"].append(3)
    value["available_token_slots"] = 999
    assert confirmed == ((7, 8),)
    assert directory.prefixes("physical-A") == confirmed
    assert directory.resources("physical-A").available_token_slots == 5


@pytest.mark.parametrize("tokens", [[1, True], [1, -1], "123", [1.0, 2]])
def test_invalid_token_identity_is_rejected(tokens):
    with pytest.raises(ValueError):
        validated_snapshots(tokens, response(2, [1]))


def test_token_budget_drops_old_entries_without_evicting_engine_state():
    directory = SnapshotDirectory(max_snapshots_per_engine=10, max_tokens_per_engine=5)
    directory.update(0, (1, 2, 3), response(3, [3]))
    directory.update(0, (4, 5, 6), response(3, [3]))
    assert directory.prefixes(0) == ((4, 5, 6),)
    directory.update(0, tuple(range(7)), response(7, [7]))
    assert directory.prefixes(0) == ((4, 5, 6),)
    assert directory.resources(0).available_token_slots == 5
