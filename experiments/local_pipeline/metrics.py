"""Client-observed stream cadence; chunked emissions are never invented tokens.

All times must share one monotonic clock. These are HTTP emission gaps, not
device execution times. A count jump prevents inference of individual gaps.
"""

import math


def percentile(values, percentile_value):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * percentile_value / 100
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize_emissions(emissions, expected_tokens):
    """Summarize cumulative counts, retaining integrity and coalescing evidence.

    Duplicate counts do not move the last token's observation time. Resets break
    the series: the first observation after a reset supplies no inferred gap.
    Initial chunks have no preceding observed token, even if their count is one.
    """
    if not isinstance(expected_tokens, int) or expected_tokens < 0:
        raise ValueError("expected_tokens must be a nonnegative integer")
    gaps, chunks = [], []
    previous = None
    last_time = None
    duplicates = resets = timestamp_errors = 0
    maximum = final_count = 0
    first_token_time = None
    for emission in emissions:
        time, count = float(emission["time"]), emission["tokens"]
        if not math.isfinite(time) or not isinstance(count, int) or count < 0:
            raise ValueError("finite times and nonnegative integer counts required")
        final_count = count
        maximum = max(maximum, count)
        if count and first_token_time is None:
            first_token_time = time
        if last_time is not None and time < last_time:
            timestamp_errors += 1
            previous = None
        last_time = time
        if previous is not None and count == previous[1]:
            duplicates += 1
            continue
        if previous is not None and count < previous[1]:
            resets += 1
            previous = (time, count)
            continue
        delta = count - previous[1] if previous else count
        if delta > 1:
            chunks.append({"time": time, "token_delta": delta,
                           "previous_time": previous[0] if previous else None})
        if previous is not None and previous[1] > 0 and delta == 1:
            gaps.append({"start": previous[0], "end": time,
                         "seconds": time - previous[0], "token_count": count})
        previous = (time, count)
    seconds = [gap["seconds"] for gap in gaps]
    return {
        "emission_count": len(emissions), "expected_tokens": expected_tokens,
        "final_tokens": final_count, "maximum_tokens": maximum,
        "missing_tokens": max(0, expected_tokens - final_count),
        "extra_tokens": max(0, final_count - expected_tokens),
        "duplicate_count": duplicates, "reset_count": resets,
        "timestamp_error_count": timestamp_errors,
        "complete": final_count == expected_tokens and not resets and not timestamp_errors,
        "first_token_time": first_token_time,
        "valid_gap_count": len(gaps), "gaps": gaps,
        "coalesced_emission_count": len(chunks), "coalesced_emissions": chunks,
        "gap_p50_seconds": percentile(seconds, 50),
        "gap_p95_seconds": percentile(seconds, 95),
        "gap_p99_seconds": percentile(seconds, 99),
        "gap_max_seconds": max(seconds) if seconds else None,
        "gaps_over_20ms": sum(value > 0.020 for value in seconds),
        "gaps_over_50ms": sum(value > 0.050 for value in seconds),
    }


def interval_union(intervals):
    """Sorted disjoint half-open intervals; touching intervals are merged."""
    checked = []
    for start, end in intervals:
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("invalid interval")
        if end > start:
            checked.append((start, end))
    result = []
    for start, end in sorted(checked):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def interval_intersection(left, right):
    left, right = interval_union(left), interval_union(right)
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def align_gaps_with_windows(gaps, windows):
    """Associate each gap with overlapping RPC/phase envelopes, not causality.

    windows: dictionaries with start/end and optional name. Overlapping envelopes
    are counted once in total overlap, while individual names remain available.
    """
    interval_union([(window["start"], window["end"]) for window in windows])
    result = []
    for gap in gaps:
        associations = []
        overlaps = []
        for index, window in enumerate(windows):
            intersection = interval_intersection(
                [(gap["start"], gap["end"])], [(window["start"], window["end"])])
            if intersection:
                overlaps.extend(intersection)
                associations.append({"window_index": index, "name": window.get("name"),
                                     "overlap_seconds": sum(b-a for a, b in intersection)})
        result.append({**gap, "windows": associations,
                       "overlap_seconds": sum(b-a for a, b in interval_union(overlaps))})
    return result
