import json

from experiments.fluidkv.analyze_reroute_blind_spots import (
    VerifiedPrefixRing,
    analyze_qwen,
    matched_prefix,
    state_bytes,
)


def row(chat, parent, length, hashes, ts=1.0):
    return {"chat_id": chat, "parent_chat_id": parent, "input_length": length,
            "hash_ids": hashes, "timestamp": ts, "output_length": 3}


def test_partial_parent_block_is_not_claimed():
    parent = row(1, -1, 18, [4, 5], 1.0)
    child = row(2, 1, 35, [4, 5, 6], 8.0)
    matched, overlap, exact = matched_prefix(parent, child)
    assert (matched, overlap, exact) == (2, 16, False)


def test_exact_parent_boundary_is_claimed_only_when_complete():
    parent = row(1, -1, 32, [4, 5], 1.0)
    child = row(2, 1, 37, [4, 5, 6], 8.0)
    assert matched_prefix(parent, child) == (2, 32, True)


def test_hybrid_has_fixed_state_and_dense_is_token_linear():
    assert state_bytes(0, "qwen3.5-35b-hybrid") == 0
    assert state_bytes(1, "qwen3.5-35b-hybrid") == 20_480 + 64_389_120
    assert state_bytes(1, "qwen3-8b-dense") == 147_456


def test_candidate_projection_has_two_distinct_engines_and_respects_membership():
    ring = VerifiedPrefixRing(range(8))
    candidates = ring.candidates("fluidkv:known-prefix")
    assert len(candidates) == 2 and candidates[0] != candidates[1]
    active = [engine for engine in range(8) if engine not in candidates]
    after = ring.candidates("fluidkv:known-prefix", active)
    assert len(after) == 2 and set(after).issubset(active)


def test_owner_retirement_creates_reroute_blind_spot(tmp_path):
    path = tmp_path/"trace.jsonl"
    rows = [row(1, -1, 32, [900, 901], 1.0), row(2, 1, 37, [900, 901, 902], 61.0)]
    path.write_text("\n".join(json.dumps(item) for item in rows))
    summary, links, scenarios = analyze_qwen(path)
    assert len(links) == 1 and links[0]["overlap_tokens"] == 32
    assert summary["single_retirement_scenario_links_not_unique"] >= 1
    assert any(item["retired_engine"] in item["parent_candidates_before"] for item in scenarios)
