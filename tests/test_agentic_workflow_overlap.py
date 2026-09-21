from experiments.agentic_profile.analyze_workflow_overlap import intersection, union_duration


def test_union_does_not_double_count_simultaneous_requests():
    assert intersection((2, 8), [(1, 5), (3, 7), (9, 10)]) == 5
    assert union_duration([(1, 4), (2, 3), (4, 6)]) == 5


def test_non_overlapping_and_zero_width_intervals():
    assert intersection((2, 3), [(0, 2), (3, 5)]) == 0
    assert union_duration([(4, 3), (2, 2)]) == 0
