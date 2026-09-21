from experiments.fluidkv.audit_identity_ambiguity import classify_pair, summarize_pairs


def row(chat, length, hashes, parent=-1):
    return dict(chat_id=chat, parent_chat_id=parent, input_length=length,
                hash_ids=hashes, timestamp=chat, output_length=1)


def test_partial_last_block_mismatch_does_not_prove_token_divergence():
    parent, child = row(1, 18, [1, 2]), row(2, 35, [1, 9, 3], 1)
    evidence = classify_pair(parent, child)
    assert evidence["category"] == "append_compatible_unknown_tail"
    assert evidence["confirmed_shared_complete_tokens"] == 16
    assert evidence["unknown_parent_tail_tokens"] == 2
    assert not evidence["complete_parent_equality_proven"]


def test_mismatch_between_two_complete_blocks_is_confirmed_divergence():
    parent, child = row(1, 32, [1, 2]), row(2, 48, [1, 9, 3], 1)
    evidence = classify_pair(parent, child)
    assert evidence["category"] == "confirmed_complete_block_divergence"
    assert evidence["confirmed_shared_complete_tokens"] == 16


def test_unknown_whole_short_parent_and_shorter_child_are_not_proven_matches():
    parent = row(1, 8, [1])
    result = classify_pair(parent, row(2, 16, [9], 1))
    assert result["category"] == "append_compatible_unknown_tail"
    assert result["confirmed_shared_complete_tokens"] == 0
    assert result["no_complete_parent_blocks"]
    parent = row(1, 32, [1, 2])
    assert classify_pair(parent, row(2, 18, [1, 9], 1))["category"] == "shorter_child_no_confirmed_divergence"


def test_aligned_checkpoint_candidate_uses_only_confirmed_blocks():
    parent = row(1, 70, [1, 2, 3, 4, 5])
    child = row(2, 90, [1, 2, 3, 4, 99, 100], 1)
    result = classify_pair(parent, child)
    assert result["floor64_tokens"] == 64
    assert result["floor64_identity_supported"]
    assert not result["checkpoint_availability_proven"]
    summary = summarize_pairs([child], {1: parent, 2: child})
    assert summary["counts"]["append_compatible_unknown_tail"] == 1
    assert summary["floor64_identity_supported_pairs"] == 1
