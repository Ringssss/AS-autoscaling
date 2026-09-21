import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "hybrid_state_20260912"))

from analyze_hybrid_state import (  # noqa: E402
    HybridStateBudget,
    LinkCost,
    summarize_trace,
)


def test_hybrid_state_budget_matches_qwen_architecture():
    budget = HybridStateBudget.from_config(
        {
            "text_config": {
                "dtype": "bfloat16",
                "full_attention_interval": 4,
                "head_dim": 256,
                "linear_conv_kernel_dim": 4,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_num_value_heads": 32,
                "linear_value_head_dim": 128,
                "num_hidden_layers": 40,
                "num_key_value_heads": 2,
                "mamba_ssm_dtype": "float32",
            }
        }
    )

    assert budget.full_layers == 10
    assert budget.linear_layers == 30
    assert budget.full_kv_bytes_per_token == 10 * 2 * 2 * 256 * 2
    assert budget.linear_state_bytes == 30 * (49_152 + 2_097_152)
    assert budget.linear_state_mib > 61.4
    assert budget.linear_state_mib < 61.5
    assert budget.crossover_tokens == math.ceil(
        budget.linear_state_bytes / budget.full_kv_bytes_per_token
    )
    assert budget.crossover_tokens == 3144


def test_link_cost_uses_one_snapshot_and_bandwidths():
    budget = HybridStateBudget(
        full_layers=1,
        linear_layers=1,
        full_kv_bytes_per_token=20_480,
        linear_state_bytes=61_440,
    )
    link = LinkCost.from_link(
        prefix_tokens=1000,
        gap_seconds=0.1,
        budget=budget,
        bandwidths_bytes_per_second=(1024**3,),
    )

    assert link.full_kv_bytes == 20_480_000
    assert link.linear_state_bytes == 61_440
    assert link.full_only_fits_gap is True
    assert link.full_plus_linear_fits_gap is True
    assert link.transfer_times_seconds[0] == (20_541_440 / 1024**3)


def test_summarize_trace_preserves_counts_and_components(tmp_path):
    budget = HybridStateBudget(
        full_layers=1,
        linear_layers=1,
        full_kv_bytes_per_token=10,
        linear_state_bytes=100,
    )
    path = tmp_path / "synthetic.npz"
    np.savez(
        path,
        input_length=np.array([100, 150, 20, 300]),
        output_length=np.ones(4, dtype=np.int64),
        timestamp=np.array([0.0, 1.0, 2.0, 3.0]),
        parent_index=np.array([-1, 0, -1, 2]),
        child_count=np.array([1, 0, 1, 0]),
        root_index=np.array([0, 0, 2, 2]),
        arrival_gap_s=np.array([np.nan, 1.0, np.nan, 2.0]),
        parent_prefix_tokens=np.array([0, 100, 0, 20]),
        usable_parent_link=np.array([False, True, False, True]),
        next_child_arrival_gap_s=np.array([1.0, np.nan, 2.0, np.nan]),
    )

    summary = summarize_trace(
        path,
        budget,
        bandwidths_bytes_per_second=(100,),
        baseline_mha_bytes_per_token=20,
    )

    assert summary.request_count == 4
    assert summary.usable_link_count == 2
    assert summary.link_component_count == 2
    assert summary.link_prefix_tokens_p50 == 60.0
    assert summary.drain_component_count == 2
    assert summary.max_component_requests == 2
    assert summary.hybrid_vs_mha_bytes_p50 < 1.0
    assert summary.all_metrics_finite is True
