import json

from agentshift.workloads.traces import iter_flowprefill_turns, iter_kimi_requests


def test_flowprefill_parent_chain(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [
        {
            "chat_id": 0,
            "parent_chat_id": -1,
            "timestamp": 1.0,
            "input_length": 32,
            "output_length": 4,
            "turn": 1,
            "hash_ids": [1, 2],
        },
        {
            "chat_id": 1,
            "parent_chat_id": 0,
            "timestamp": 4.5,
            "input_length": 99,
            "output_length": 3,
            "turn": 2,
            "hash_ids": [1, 2, 3],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    turns = list(iter_flowprefill_turns(path))
    assert turns[1].agent_id == turns[0].agent_id
    assert turns[1].incremental_input_tokens == 16
    assert turns[1].tool_gap_seconds == 3.5


def test_kimi_length_trace(tmp_path):
    path = tmp_path / "kimi.csv"
    path.write_text("TIMESTAMP,ContextTokens,GeneratedTokens\nnow,1200,42\n")
    assert list(iter_kimi_requests(path)) == [("now", 1200, 42)]
