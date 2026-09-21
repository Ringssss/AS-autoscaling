import pytest

from experiments.local_pipeline.metrics import (
    align_gaps_with_windows, interval_intersection, interval_union,
    summarize_emissions,
)


def stream(*rows):
    return [{"time": time, "tokens": tokens} for time, tokens in rows]


def test_single_token_intervals_and_thresholds():
    result = summarize_emissions(stream((0, 1), (.01, 2), (.04, 3), (.10, 4)), 4)
    assert result["complete"]
    assert result["gap_p50_seconds"] == pytest.approx(.03)
    assert result["gap_p95_seconds"] == pytest.approx(.057)
    assert result["gap_max_seconds"] == pytest.approx(.06)
    assert result["gaps_over_20ms"] == 2
    assert result["gaps_over_50ms"] == 1


def test_duplicate_metadata_does_not_hide_long_gap():
    result = summarize_emissions(stream((0, 1), (.049, 1), (.060, 2), (.070, 2)), 2)
    assert result["duplicate_count"] == 2
    assert result["valid_gap_count"] == 1
    assert result["gap_max_seconds"] == .060
    assert result["gaps_over_50ms"] == 1


def test_chunks_are_not_expanded_and_initial_chunk_has_no_gap():
    result = summarize_emissions(stream((0, 3), (.02, 6), (.03, 7)), 7)
    assert result["valid_gap_count"] == 1
    assert result["gap_p50_seconds"] == pytest.approx(.01)
    assert result["coalesced_emission_count"] == 2
    assert [chunk["token_delta"] for chunk in result["coalesced_emissions"]] == [3, 3]


def test_reset_breaks_series_and_invalidates_completeness():
    result = summarize_emissions(stream((0, 1), (.01, 2), (.02, 0), (.05, 1), (.07, 2)), 2)
    assert result["reset_count"] == 1
    assert not result["complete"]
    assert result["valid_gap_count"] == 2
    assert [gap["seconds"] for gap in result["gaps"]] == pytest.approx([.01, .02])


def test_missing_terminal_tokens_explicit():
    result = summarize_emissions(stream((0, 1), (.01, 2)), 4)
    assert result["missing_tokens"] == 2
    assert not result["complete"]
    assert result["maximum_tokens"] == 2
    assert summarize_emissions([], 2)["gap_p99_seconds"] is None
    assert summarize_emissions(stream((0, 3)), 2)["extra_tokens"] == 1


def test_zero_count_is_not_first_token_and_time_regression_breaks_gap():
    result = summarize_emissions(stream((0, 0), (.1, 1), (.09, 2), (.11, 3)), 3)
    assert result["first_token_time"] == .1
    assert result["timestamp_error_count"] == 1
    assert not result["complete"]
    assert result["valid_gap_count"] == 1
    assert result["gap_max_seconds"] == pytest.approx(.02)


@pytest.mark.parametrize("emissions", [stream((float("nan"), 1)), stream((0, -1)), stream((0, 1.2))])
def test_invalid_observations_fail(emissions):
    with pytest.raises(ValueError):
        summarize_emissions(emissions, 1)


def test_interval_algebra_counts_overlap_once():
    assert interval_union([(1, 3), (2, 4), (4, 5), (8, 8)]) == [(1, 5)]
    assert interval_intersection([(1, 5), (6, 9)], [(2, 7)]) == [(2, 5), (6, 7)]
    assert interval_intersection([(1, 2)], [(2, 3)]) == []
    with pytest.raises(ValueError):
        interval_union([(3, 2)])


def test_phase_associations_are_clipped_and_union_avoids_double_count():
    gaps = [{"start": 1, "end": 5, "seconds": 4}]
    windows = [{"start": 0, "end": 3, "name": "rpc"},
               {"start": 2, "end": 4, "name": "install"},
               {"start": 5, "end": 6, "name": "after"}]
    result = align_gaps_with_windows(gaps, windows)[0]
    assert result["overlap_seconds"] == 3
    assert [window["name"] for window in result["windows"]] == ["rpc", "install"]
    assert [window["overlap_seconds"] for window in result["windows"]] == [2, 2]
