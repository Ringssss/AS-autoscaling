import pytest

from experiments.fluidkv.analyze_decision_workloads import (
    analyze_kimi_rows,
    analyze_qwen_rows,
    parent_overlap,
    select_continuous_window,
)


def row(chat, parent, timestamp, length=32, hashes=None, output=3):
    return {"chat_id": chat, "parent_chat_id": parent, "timestamp": timestamp,
            "input_length": length, "output_length": output,
            "hash_ids": hashes if hashes is not None else [chat] * ((length + 15) // 16),
            "type": "text", "turn": 1}


def test_id_join_partial_boundary_and_reuse_distance():
    parent = row(20, -1, 1, 18, [1, 2])
    child = row(90, 20, 4, 35, [1, 9, 3])
    summary = analyze_qwen_rows([child, row(7, -1, 2), parent])
    assert summary["linked_requests"] == 1
    assert summary["parent_input_overlap_tokens"] == 16
    assert summary["snapshot_eligible_children"] == 0
    assert summary["arrival_deltas_seconds"]["p50"] == 3
    assert summary["intervening_requests"]["p50"] == 1
    assert parent_overlap(parent, child)["parent_partial_block"]


def test_exact_parent_extension_and_equal_length_distinguished():
    parent = row(40, -1, 1, 32, [1, 2])
    same = row(41, 40, 2, 32, [1, 2])
    extension = row(42, 40, 3, 35, [1, 2, 3])
    summary = analyze_qwen_rows([parent, same, extension])
    assert summary["exact_block_aligned_parent_matches"] == 2
    assert summary["snapshot_eligible_children"] == 1


def test_window_half_open_keeps_background_and_does_not_add_ancestors():
    rows = [row(1, -1, 0), row(2, -1, 5), row(3, 2, 10),
            row(4, 1, 11), row(5, -1, 19), row(6, -1, 20)]
    selected, audit = select_continuous_window(
        list(reversed(rows)), start_seconds=10, duration_seconds=10, warmup_seconds=5)
    assert [r["chat_id"] for r in selected] == [2, 3, 4, 5]
    assert audit["warmup"]["requests"] == 1
    assert audit["measurement"]["requests"] == 3
    assert audit["measurement"]["offered_requests_per_second"] == 0.3
    assert audit["measurement"]["parent_location"] == {"warmup": 1, "before_warmup": 1, "root": 1}
    assert audit["selection"][0]["hash_ids"] == rows[1]["hash_ids"]
    assert audit["selection"][0]["input_identity_sha256"]


def test_context_exclusions_break_links_and_are_not_output_truncation():
    rows = [row(1, -1, 0), row(2, -1, 5, 32, [1, 2], output=9),
            row(3, 2, 10, 36, [1, 2, 3], output=4)]
    selected, audit = select_continuous_window(
        rows, start_seconds=10, duration_seconds=10, context_length=40)
    assert [r["chat_id"] for r in selected] == [1, 3]
    assert selected[-1]["output_length"] == 4
    assert audit["excluded"] == [{"chat_id": 2, "phase": "warmup", "reason": "input_plus_output_exceeds_context", "input_length": 32, "output_length": 9}]
    assert audit["measurement"]["parent_location"] == {"excluded_context": 1}
    assert audit["measurement"]["snapshot_eligible_parent_in_replay"] == 0


def test_no_request_cap_and_missing_parent_reporting():
    rows = [row(0, -1, 0)] + [row(i, 9999, 10 + i / 100) for i in range(1, 101)]
    selected, audit = select_continuous_window(rows, start_seconds=10, duration_seconds=10)
    assert len(selected) == 101
    assert audit["measurement"]["requests"] == 100
    assert audit["measurement"]["parent_location"] == {"missing_source": 100}


def test_duplicate_shape_and_noncausal_parents_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        analyze_qwen_rows([row(2, -1, 1)] * 2)
    with pytest.raises(ValueError, match="block"):
        analyze_qwen_rows([row(2, -1, 1, 17, [1])])
    with pytest.raises(ValueError, match="parent timestamp"):
        analyze_qwen_rows([row(2, -1, 2), row(4, 2, 1)])


def test_kimi_observation_only_no_lineage_and_invalid_timing_counted():
    rows = [dict(request_id="x", input_length=100, output_length=4, cached_tokens=80,
                 arrival_seconds=10, finished_seconds=12, ttft_seconds=0.5,
                 conversation_id=None, input_ids_redacted=True),
            dict(request_id="y", input_length=200, output_length=8, cached_tokens=None,
                 arrival_seconds=11, finished_seconds=10, ttft_seconds=1,
                 conversation_id=None, input_ids_redacted=True)]
    audit = analyze_kimi_rows(rows)
    assert audit["requests"] == 2
    assert audit["cached_token_sum"] == 80
    assert audit["missing_cached_tokens"] == 1
    assert audit["invalid_e2e_records"] == 1
    assert audit["ttft_over_e2e"]["p50"] == 0.25
    assert audit["native_lineage_replay_supported"] is False
