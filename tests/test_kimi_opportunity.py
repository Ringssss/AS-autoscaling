import numpy as np
import pytest

from experiments.multinode.analyze_kimi_opportunity import oracle_budget


def test_cohort_ceiling_uses_latency_mass_not_request_share_and_excludes_invalid():
    result=oracle_budget(np.array([1.,9.,1000.]),np.array([10.,90.,0.]),
                         [True,False,True],[True,True,False])
    assert result['removed_ttft_seconds']==1
    assert result['ttft_sum_share']==pytest.approx(.1)
    assert result['cohort_only_zero_ttft_mean_ttft_speedup']==pytest.approx(10/9)
    assert result['cohort_only_zero_ttft_mean_e2e_speedup']==pytest.approx(100/99)


def test_empty_cohort_has_no_gain_and_zero_baseline_is_not_fabricated_speedup():
    result=oracle_budget(np.array([2.,3.]),np.array([10.,10.]),[False,False],[True,True])
    assert result['cohort_only_zero_ttft_mean_ttft_speedup']==1
    assert result['cohort_only_zero_ttft_mean_e2e_speedup']==1
    zero=oracle_budget(np.array([0.]),np.array([0.]),[True],[True])
    assert zero['ttft_sum_share'] is None
    assert zero['cohort_only_zero_ttft_mean_ttft_speedup'] is None
