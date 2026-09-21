import pytest

from agentshift.workloads.production import (
    block_tokens,
    inspect_qwen_rows,
    parse_kimi_line,
    surrogate_tokens,
    select_exact_parent_cohort,
)


def row(chat, parent, length, hashes, ts=1.0):
    return dict(chat_id=chat, parent_chat_id=parent, input_length=length,
                hash_ids=hashes, timestamp=ts, output_length=3, type="text", turn=1)


def test_parent_join_uses_ids_and_partial_blocks_are_not_complete():
    rows = [row(10, -1, 18, [4, 5]), row(30, 10, 35, [4, 6, 7], 3.0)]
    audit, pairs = inspect_qwen_rows(rows)
    assert audit["linked_requests"] == 1
    assert audit["verified_parent_input_overlap_tokens"] == 16
    assert pairs == []


def test_exact_pair_keeps_original_lengths_and_arrivals():
    rows = [row(11, -1, 32, [2, 3]), row(51, 11, 37, [2, 3, 4], 8.0)]
    audit, pairs = inspect_qwen_rows(rows)
    assert len(pairs) == 1
    assert pairs[0]["parent_id"] == 11
    assert pairs[0]["arrival_delta_seconds"] == 7.0
    assert len(surrogate_tokens(rows[1])) == 37
    assert surrogate_tokens(rows[1])[:32] == surrogate_tokens(rows[0])
    assert audit["exact_parent_prefix_pairs"] == 1


def test_missing_parent_is_reported_not_synthesized():
    audit, pairs = inspect_qwen_rows([row(8, 100, 16, [1])])
    assert audit["missing_parents"] == 1
    assert not pairs


def test_block_mapping_is_deterministic_and_not_single_token_repetition():
    assert block_tokens(57) == block_tokens(57)
    assert block_tokens(57) != block_tokens(58)
    assert len(set(block_tokens(57))) > 1
    assert all(1000 <= token < 31000 for token in block_tokens(57))


def test_invalid_shape_and_duplicate_id_rejected():
    with pytest.raises(ValueError):
        inspect_qwen_rows([row(0, -1, 17, [1])])
    with pytest.raises(ValueError):
        inspect_qwen_rows([row(0, -1, 16, [1])] * 2)


def test_kimi_redaction_does_not_create_identity_or_tokens():
    line = ("Finish: input_ids_len=1088, obj=GenerateReqInput(rid='a', "
            "input_ids=[<REDACTED_INPUT_IDS>], conversation_id=None), "
            "out={'meta_info': {'completion_tokens': 8, 'cached_tokens': 900, "
            "'request_received_ts': 12.5, 'request_finished_ts': 13.5, "
            "'dp_rank': 2}}")
    result = parse_kimi_line(line)
    assert result["input_length"] == 1088
    assert result["conversation_id"] is None
    assert result["input_ids_redacted"]
    assert "hash_ids" not in result
    assert result["cached_tokens"] == 900


def test_exact_cohort_preserves_parent_ids_tokens_and_original_times():
    rows = [row(1, -1, 18, [1, 2], 1), row(2, 1, 33, [1, 3, 4], 2),
            row(90, -1, 32, [9, 10], 3), row(100, 90, 35, [9, 10, 11], 5)]
    selected = select_exact_parent_cohort(rows, 2)
    assert selected == rows[2:]
    assert selected[1]["parent_chat_id"] == selected[0]["chat_id"]
    assert selected[0]["timestamp"] == 3
    assert selected[1]["timestamp"] == 5
