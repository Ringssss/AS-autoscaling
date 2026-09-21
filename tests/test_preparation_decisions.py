import pytest

from agentshift.controller.preparation_decisions import CostCurve, PreparationEstimate, choose_preparation


def test_arrived_child_uses_residual_ready_time_and_includes_wait():
    cold = PreparationEstimate('cold', 0, 0.3, 0.15)
    copy = PreparationEstimate('copy', 0.8, 0.05, 0.0)
    assert choose_preparation([cold, copy], deadline_seconds=2).method == 'cold'
    ready = PreparationEstimate('copy', 0.04, 0.05, 0.02)
    assert choose_preparation([cold, ready], deadline_seconds=2).method == 'copy'
    assert ready.finish_seconds == pytest.approx(0.09)


def test_impossible_capacity_and_slo_do_not_make_copy_feasible():
    cold = PreparationEstimate('cold', 0, 0.4, 0)
    copy = PreparationEstimate('copy', 0.03, 0.05, 0, capacity_feasible=False)
    assert choose_preparation([cold, copy], deadline_seconds=0.2).method == 'cold'


def test_margin_avoids_tiny_uncertain_gain_and_prep_has_interference_cost():
    cold = PreparationEstimate('cold', 0, 0.2, 0)
    copy = PreparationEstimate('copy', 0.1, 0.05, 0, interference_seconds=0.04)
    assert choose_preparation([cold, copy], deadline_seconds=1, minimum_gain_seconds=0.03).method == 'cold'


def test_curve_interpolates_measurements_and_clamps_negative_savings():
    curve = CostCurve(((1024, 0.12), (4096, 0.24), (8192, 0.4)))
    assert curve.seconds(2560) == pytest.approx(0.18)
    assert curve.saved_seconds(4096, 8192) == pytest.approx(0.24)
    assert curve.saved_seconds(4096, -1) == 0
    with pytest.raises(ValueError):
        CostCurve(((1024, -0.1),))
