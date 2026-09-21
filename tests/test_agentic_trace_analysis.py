from experiments.agentic_profile.analyze_engine_traces import clipped, merged, union


def test_interval_union_prevents_graph_kernel_double_counting():
    # One graph envelope contains multiple kernels and overlaps a copy.
    intervals = [(1, 5), (2, 3), (4, 4.5), (4.8, 6), (7, 8)]
    assert merged(intervals) == [[1, 6], [7, 8]]
    assert union(intervals) == 6
    assert union(clipped(intervals, 2, 7.5)) == 4.5


def test_intersection_merges_overlapping_tools_before_counting():
    from experiments.agentic_profile.analyze_workflow_capture import intersection
    assert union(intersection([(0, 4), (2, 6)], [(1, 2), (3, 8)])) == 4
